"""Phase 3: NUTS posterior sampling around the trained (MAP) network."""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cosmopinn.bayes.packer import make_packer, sampled_scalar_names
from cosmopinn.bayes.posterior import make_log_posterior, propagate_samples_to_bands
from cosmopinn.bayes.sampler import ess_per_parameter, gelman_rubin, run_nuts_chains
from cosmopinn.config import as_jnp, wde_label
from cosmopinn.data.sn import SNData, build_sn_data


def run_bayesian_phase(physics, model, cfg: Any, z_data: Array, H_data: Array,
                       sigma_data: Array, z_bao: Array, sn_data: Optional[SNData] = None) -> None:
    sep = "=" * 70
    print(f"\n{sep}\nPhase 3: NUTS posterior sampling\n{sep}")
    dt = as_jnp(cfg.HMC_DTYPE or cfg.DTYPE)
    model = jax.tree_util.tree_map(
        lambda x: x.astype(dt) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
        else x, model)
    z_data, H_data, sigma_data, z_bao = (a.astype(dt) for a in (z_data, H_data, sigma_data, z_bao))
    if cfg.USE_SN and cfg.HMC_USE_SN:
        sn_data = build_sn_data(cfg, dtype=cfg.HMC_DTYPE or cfg.DTYPE)
    sn_label = cfg.SN_DATASET if (cfg.HMC_USE_SN and sn_data is not None) else "off"

    packer = make_packer(model, cfg, cfg.HMC_SCOPE)
    theta_map = packer.theta_map.astype(dt)
    packer = dataclasses.replace(
        packer, theta_map=theta_map, scalar_mask=packer.scalar_mask.astype(dt),
        w_mask=packer.w_mask.astype(dt), w_order=packer.w_order.astype(dt),
        w_lcdm=packer.w_lcdm.astype(dt), w_sigma=packer.w_sigma.astype(dt))
    print(f"  scope={cfg.HMC_SCOPE}  dim={packer.dim} ({packer.n_sampled_scalars()} scalars, "
          f"{packer.n_sampled_w()} w_DE head params)  w_DE network: {wde_label(cfg)}")
    print(f"  likelihood: CC={cfg.HMC_USE_CC} BAO={cfg.HMC_USE_BAO} SN={sn_label} "
          f"PDE={cfg.HMC_USE_PDE}  w_DE head prior: {packer.prior_note}")

    z_pde = (jnp.linspace(0.0, cfg.Z_MAX, int(cfg.HMC_PDE_N_COLLOC), dtype=dt).reshape(-1, 1)
             if cfg.HMC_USE_PDE else None)
    logdensity_fn = make_log_posterior(physics, model, cfg, z_data, H_data, sigma_data, z_bao,
                                       sn_data, packer, z_pde_grid=z_pde)

    n_chains = int(cfg.HMC_NUM_CHAINS)
    base_key = jax.random.PRNGKey(int(cfg.HMC_SEED))
    keys = jax.vmap(lambda c: jax.random.fold_in(base_key, c))(jnp.arange(n_chains))
    theta_inits = theta_map[None, :] + cfg.HMC_JITTER * jax.vmap(
        lambda k: jax.random.normal(k, theta_map.shape, dtype=dt))(keys)

    res = run_nuts_chains(
        logdensity_fn=logdensity_fn, theta_inits=theta_inits,
        num_warmup=cfg.HMC_NUM_WARMUP, num_samples=cfg.HMC_NUM_SAMPLES,
        target_accept=cfg.HMC_TARGET_ACCEPT, max_tree_depth=cfg.HMC_MAX_TREE_DEPTH,
        step_size=cfg.HMC_INIT_STEP_SIZE, seed=int(cfg.HMC_SEED), log_every=cfg.HMC_LOG_EVERY,
        dense_mass=bool(cfg.HMC_DENSE_MASS), vectorise=bool(cfg.HMC_VECTORISE_CHAINS))

    chains = res.samples
    rhat = gelman_rubin(chains)
    ess = ess_per_parameter(chains)
    n_div = int(np.sum(res.divergences))
    print(f"  divergences={n_div}  R-hat {np.nanmin(rhat):.3f}..{np.nanmax(rhat):.3f}  "
          f"bulk-ESS {np.nanmin(ess):.0f}..{np.nanmax(ess):.0f}")

    samples = chains.reshape(-1, chains.shape[-1])[::max(1, int(cfg.HMC_THIN))]

    z_grid = jnp.linspace(0.0, cfg.Z_MAX, cfg.HMC_BAND_N_Z, dtype=dt).reshape(-1, 1)
    bands = propagate_samples_to_bands(cfg, samples, packer, z_grid)

    @jax.jit
    def scalars(thetas):
        def one(theta):
            m = packer.reconstruct(theta)
            return jnp.stack([m.Omega_m0(), m.H0(), m.R_drag()])
        return jax.vmap(one)(thetas)

    n_c, n_s, _ = chains.shape
    traces = np.asarray(scalars(jnp.asarray(chains.reshape(n_c * n_s, -1), dtype=dt))).reshape(n_c, n_s, 3)

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    np.savez(os.path.join(cfg.SAVE_DIR, "bayes_bands.npz"), **bands)
    if cfg.HMC_SAVE_CHAIN:
        np.savez(os.path.join(cfg.SAVE_DIR, "hmc_chain.npz"),
                 chains=chains, log_probs=res.log_probs, rhat=rhat, ess=ess,
                 step_size=res.step_size, divergences=res.divergences,
                 tree_depths=res.tree_depths, accept_stats=res.accept_stats,
                 theta_map=np.asarray(theta_map), scalar_mask=np.asarray(packer.scalar_mask),
                 w_mask=np.asarray(packer.w_mask), w_order=np.asarray(packer.w_order),
                 w_lcdm=np.asarray(packer.w_lcdm), w_sigma=np.asarray(packer.w_sigma),
                 scalar_traces=traces, scalar_names=np.array(["Omega_m0", "H0", "R_drag"]),
                 prior_mu=np.array([cfg.OMEGA_M0_PRIOR_MU, cfg.H0_PRIOR_MU, cfg.R_DRAG_PRIOR_MU]))

    _write_summary(cfg, packer, res, rhat, ess, n_div, bands, sn_label)
    print(f"  Posterior outputs saved to {cfg.SAVE_DIR}\n{sep}")


def _write_summary(cfg, packer, res, rhat, ess, n_div, bands, sn_label):
    n_c, n_s, dim = res.samples.shape
    lines = [
        "Cosmo-PINN NUTS summary",
        "=" * 60,
        f"w_DE network      : {wde_label(cfg)}",
        f"Scope             : {cfg.HMC_SCOPE} (dim {dim}: {packer.n_sampled_scalars()} scalars, "
        f"{packer.n_sampled_w()} w_DE head params)",
        f"Chains x samples  : {n_c} x {n_s} (warmup {cfg.HMC_NUM_WARMUP})",
        f"Likelihood        : CC={cfg.HMC_USE_CC} BAO={cfg.HMC_USE_BAO} SN={sn_label} "
        f"PDE={cfg.HMC_USE_PDE} (sigma={cfg.HMC_PDE_SIGMA}, n={cfg.HMC_PDE_N_COLLOC})",
        f"w_DE head prior   : {packer.prior_note}",
        "",
        f"Divergences       : {n_div} / {n_c * n_s}",
        f"Step size         : {float(np.mean(res.step_size)):.3e}",
        f"Acceptance        : {float(np.mean(res.accept_stats)):.3f}",
        f"R-hat             : {np.nanmin(rhat):.4f} .. {np.nanmax(rhat):.4f}",
        f"Bulk ESS          : {np.nanmin(ess):.0f} .. {np.nanmax(ess):.0f}",
        "",
    ]
    scalar_idx = np.where(np.asarray(packer.scalar_mask) > 0)[0]
    for name, i in zip(sampled_scalar_names(cfg), scalar_idx):
        lines.append(f"  {name:10s} R-hat={rhat[i]:.4f}  ESS={ess[i]:.0f}")
    w_idx = np.where(np.asarray(packer.w_mask) > 0)[0]
    order = np.asarray(packer.w_order)
    for i in w_idx[np.argsort(order[w_idx])]:
        lines.append(f"  {packer.coef_label(order[i]):10s} R-hat={rhat[i]:.4f}  ESS={ess[i]:.0f}")

    lines.append("")
    for key, label in (("Omega_m0_scalar", "Omega_m0"), ("H0_scalar", "H0"), ("R_drag_scalar", "R_drag")):
        a = np.asarray(bands[key])
        p = np.percentile(a, [2.5, 16.0, 50.0, 84.0, 97.5])
        lines.append(f"{label:10s} median={p[2]:.4f}  68%=[{p[1]:.4f}, {p[3]:.4f}]  "
                     f"95%=[{p[0]:.4f}, {p[4]:.4f}]")

    with open(os.path.join(cfg.SAVE_DIR, "hmc_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
