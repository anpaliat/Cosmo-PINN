"""Training: Adam with GradNorm balancing and RAD resampling, then L-BFGS, then HMC."""

from __future__ import annotations

import os
import time
from math import gcd
from typing import Any, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from cosmopinn.config import as_jnp, wde_label
from cosmopinn.data.bao import bao_redshifts
from cosmopinn.data.cc import load_cc_data
from cosmopinn.data.sn import build_sn_data, dataset_name
from cosmopinn.physics.hubble import deceleration
from cosmopinn.training.grids import collocation_points, integration_grid, resample_collocation
from cosmopinn.training.loss import compute_loss
from cosmopinn.training.model import consts_from_cfg, deceleration_autodiff

LOSS_KEYS = ["pde", "data", "bao", "sn", "smooth", "dwde", "omega_smooth", "smooth_h",
             "dh", "bounds", "wde_bound", "smooth_max", "smooth_h_max", "omega_smooth_max"]


def _cast_tree(tree, dtype):
    return jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, tree)


def _schedule(cfg: Any):
    if cfg.SCHEDULER == "cosine":
        return optax.cosine_decay_schedule(init_value=cfg.ADAM_LR, decay_steps=cfg.ADAM_EPOCHS,
                                           alpha=cfg.ADAM_LR_MIN / cfg.ADAM_LR)
    return optax.exponential_decay(init_value=cfg.ADAM_LR, transition_steps=cfg.SCHEDULER_STEP,
                                   decay_rate=cfg.SCHEDULER_GAMMA, staircase=True)


def _trainable_filter(model, cfg: Any):
    spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    spec = eqx.tree_at(lambda m: m.Omega_m0_raw, spec, bool(cfg.OMEGA_M0_TRAINABLE))
    spec = eqx.tree_at(lambda m: m.H0_raw, spec, bool(cfg.H0_TRAINABLE))
    return eqx.tree_at(lambda m: m.R_drag_raw, spec, bool(cfg.R_DRAG_TRAINABLE))


def _weight_decay_mask(model):
    mask = jax.tree_util.tree_map(lambda _: False, model)
    layers = list(model.layers) + list(model.w_layers)
    n_main = len(model.layers)

    def leaves(m):
        out = []
        for i in range(len(layers)):
            lin = m.layers[i] if i < n_main else m.w_layers[i - n_main]
            out += [lin.weight, lin.bias]
        return out

    return eqx.tree_at(leaves, mask, [True] * (2 * len(layers)))


def _gradnorm_update(lam_old, grads, cfg):
    active = [g for g in grads if g > 1e-30]
    if len(active) < 2:
        return lam_old
    g_target = sum(active) / len(active)
    return tuple(old if g <= 1e-30 else float(np.clip(old * (g_target / g) ** cfg.GRADNORM_ALPHA,
                                                       cfg.GRADNORM_LAM_LO, cfg.GRADNORM_LAM_HI))
                 for old, g in zip(lam_old, grads))


class Trainer:
    def __init__(self, physics, cfg: Any):
        self.physics = physics
        self.cfg = cfg
        self.key = jax.random.PRNGKey(int(cfg.SEED))
        self.history: Dict[str, list] = {k: [] for k in (
            ["epoch", "total"] + LOSS_KEYS
            + ["cc_chi2_dof", "bao_chi2_dof", "sn_chi2_dof",
               "lam_pde", "lam_data", "lam_bao", "lam_sn",
               "share_pde", "share_data", "share_bao", "share_sn",
               "Omega_m0", "H0", "R_drag", "omega_m0_prior", "h0_prior", "r_drag_prior",
               "gn_epoch", "gn_pde", "gn_data", "gn_bao", "gn_sn"])}

    def run(self):
        cfg, physics = self.cfg, self.physics
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)
        adam_dt = as_jnp(cfg.ADAM_DTYPE)

        z_data, H_data, sigma_data = load_cc_data(cfg, dtype=cfg.ADAM_DTYPE)
        z_colloc = collocation_points(cfg, dtype=cfg.ADAM_DTYPE)
        z_bao = integration_grid(cfg, bao_redshifts(cfg), dtype=cfg.ADAM_DTYPE)
        sn_data = build_sn_data(cfg, dtype=cfg.ADAM_DTYPE)

        self.key, mkey = jax.random.split(self.key)
        model = _cast_tree(physics.build_model(cfg, mkey), adam_dt)
        filter_spec = _trainable_filter(model, cfg)
        n_params = sum(int(np.prod(p.shape)) for p in
                       jax.tree_util.tree_leaves(eqx.filter(model, filter_spec)))
        print(f"Trainable parameters: {n_params} | w_DE: {wde_label(cfg)} | "
              f"backend: {jax.default_backend()}")
        print(f"[SN] {dataset_name(cfg)}: N = {0 if sn_data is None else sn_data.n_sn}")

        wd_mask, _ = eqx.partition(_weight_decay_mask(model), filter_spec)
        sched = _schedule(cfg)
        opt = optax.chain(
            optax.clip_by_global_norm(cfg.MAX_GRAD_NORM),
            optax.add_decayed_weights(weight_decay=cfg.WEIGHT_DECAY, mask=wd_mask),
            optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
            optax.scale_by_schedule(sched),
            optax.scale(-1.0),
        )
        diff, static = eqx.partition(model, filter_spec)
        opt_state = opt.init(diff)

        lam_pde, lam_data, lam_bao, lam_sn = (cfg.LAMBDA_PDE, cfg.LAMBDA_DATA,
                                              cfg.LAMBDA_BAO, cfg.LAMBDA_SN)
        lam_next = jnp.asarray([lam_pde, lam_data, lam_bao, cfg.LAMBDA_SMOOTH, cfg.LAMBDA_DWDE,
                                cfg.LAMBDA_SMOOTH_H, cfg.LAMBDA_DH, cfg.LAMBDA_BOUNDS, lam_sn],
                               dtype=adam_dt)
        dh_ema = jnp.zeros((z_colloc.shape[0],), dtype=adam_dt)
        ema_a = jnp.asarray(cfg.DH_EMA_ALPHA, dtype=adam_dt)
        ema_1a = jnp.asarray(1.0 - cfg.DH_EMA_ALPHA, dtype=adam_dt)

        def loss_terms(diff, z_colloc, lam, dh_ema):
            return compute_loss(physics, eqx.combine(diff, static), z_colloc, z_data, H_data,
                                sigma_data, z_bao, sn_data, cfg, lambdas=lam, dh_dx_ema=dh_ema)

        def adam_update(diff, opt_state, z_colloc, lam, dh_ema):
            (_, aux), grads = jax.value_and_grad(
                lambda d: (lambda o: (o["total"], o))(loss_terms(d, z_colloc, lam, dh_ema)),
                has_aux=True)(diff)
            updates, opt_state = opt.update(grads, opt_state, diff)
            return (optax.apply_updates(diff, updates), opt_state, aux,
                    ema_a * dh_ema + ema_1a * aux["dh_dx_detached"])

        step = eqx.filter_jit(adam_update)

        @eqx.filter_jit
        def run_block(diff, opt_state, z_colloc, lam, dh_ema, n_steps):
            def body(carry, _):
                d, s, e = carry
                d, s, _, e = adam_update(d, s, z_colloc, lam, e)
                return (d, s, e), None
            (diff, opt_state, dh_ema), _ = jax.lax.scan(
                body, (diff, opt_state, dh_ema), None, length=int(n_steps))
            return diff, opt_state, dh_ema

        eval_terms = eqx.filter_jit(loss_terms)

        @eqx.filter_jit
        def grad_norms(diff, z_colloc, lam, dh_ema):
            return tuple(optax.global_norm(jax.grad(
                lambda d, k=k: loss_terms(d, z_colloc, lam, dh_ema)[k])(diff))
                for k in ("pde", "data", "bao", "sn"))

        print("\n--- Phase 1: Adam ---")
        t0 = time.time()
        block = max(gcd(gcd(int(cfg.LOG_EVERY), int(cfg.GRADNORM_EVERY)),
                        int(cfg.RESAMPLE_EVERY)), 1)
        epoch = 0
        while epoch < cfg.ADAM_EPOCHS:
            lam = lam_next
            target = 1 if epoch == 0 else min((epoch // block + 1) * block, cfg.ADAM_EPOCHS)
            if target - epoch == 1:
                diff, opt_state, aux, dh_ema = step(diff, opt_state, z_colloc, lam, dh_ema)
            else:
                diff, opt_state, dh_ema = run_block(diff, opt_state, z_colloc, lam, dh_ema,
                                                    target - epoch)
                aux = None
            epoch = target

            if (cfg.BALANCING_MODE == "gradnorm" and epoch <= cfg.GRADNORM_FREEZE_AFTER
                    and epoch % cfg.GRADNORM_EVERY == 0):
                gn = tuple(float(g) for g in grad_norms(diff, z_colloc, lam, dh_ema))
                lam_pde, lam_data, lam_bao, lam_sn = _gradnorm_update(
                    (lam_pde, lam_data, lam_bao, lam_sn), gn, cfg)
                lam_next = lam.at[0].set(lam_pde).at[1].set(lam_data).at[2].set(lam_bao).at[8].set(lam_sn)
                for k, g in zip(("gn_pde", "gn_data", "gn_bao", "gn_sn"), gn):
                    self.history[k].append(g)
                self.history["gn_epoch"].append(epoch)

            if (epoch % cfg.RESAMPLE_EVERY == 0 and cfg.RESAMPLE_START < epoch < cfg.ADAM_EPOCHS):
                self.key, rkey = jax.random.split(self.key)
                z_colloc = resample_collocation(eqx.combine(diff, static), cfg, rkey,
                                                dtype=cfg.ADAM_DTYPE)
                dh_ema = jnp.zeros((z_colloc.shape[0],), dtype=adam_dt)

            if epoch % cfg.LOG_EVERY == 0 or epoch == 1:
                if aux is None:
                    aux = eval_terms(diff, z_colloc, lam, dh_ema)
                self._log(epoch, aux, (lam_pde, lam_data, lam_bao, lam_sn),
                          f"lr={float(sched(epoch - 1)):.2e}  t={time.time() - t0:.0f}s")

        final_dt = as_jnp(cfg.DTYPE)
        if final_dt != adam_dt:
            diff, static = _cast_tree(diff, final_dt), _cast_tree(static, final_dt)
            z_data, H_data, sigma_data, z_colloc, z_bao, dh_ema = (
                a.astype(final_dt) for a in (z_data, H_data, sigma_data, z_colloc, z_bao, dh_ema))
            sn_data = build_sn_data(cfg, dtype=cfg.DTYPE)

        lam = jnp.asarray([lam_pde, lam_data, lam_bao, cfg.LAMBDA_SMOOTH, cfg.LAMBDA_DWDE,
                           cfg.LAMBDA_SMOOTH_H, cfg.LAMBDA_DH, cfg.LAMBDA_BOUNDS, lam_sn],
                          dtype=final_dt)

        def lbfgs_terms(d):
            return compute_loss(physics, eqx.combine(d, static), z_colloc, z_data, H_data,
                                sigma_data, z_bao, sn_data, cfg, lambdas=lam, dh_dx_ema=dh_ema)

        diff = self._lbfgs(diff, lbfgs_terms, (lam_pde, lam_data, lam_bao, lam_sn))
        model = eqx.combine(diff, static)

        eqx.tree_serialise_leaves(os.path.join(cfg.SAVE_DIR, "model.eqx"), model)
        self._export(model, final_dt, z_data, H_data, sigma_data)

        if cfg.USE_HMC:
            from cosmopinn.bayes.driver import run_bayesian_phase
            run_bayesian_phase(physics, model, cfg, z_data, H_data, sigma_data, z_bao, sn_data)
        return model

    def _lbfgs(self, diff, terms_fn, lams):
        cfg = self.cfg
        if cfg.LBFGS_EPOCHS <= 0:
            return diff
        print("\n--- Phase 2: L-BFGS ---")

        def value_fn(d):
            return terms_fn(d)["total"]

        opt = optax.lbfgs()
        state = opt.init(diff)
        value_and_grad = optax.value_and_grad_from_state(value_fn)

        @eqx.filter_jit
        def lbfgs_block(d, state):
            def body(carry, _):
                d, state = carry
                val, grad = value_and_grad(d, state=state)
                updates, state = opt.update(grad, state, d, value=val, grad=grad,
                                            value_fn=value_fn)
                return (optax.apply_updates(d, updates), state), val
            (d, state), vals = jax.lax.scan(body, (d, state), None, length=int(cfg.LBFGS_MAX_ITER))
            return d, state, vals

        eval_terms = eqx.filter_jit(terms_fn)
        prev, patience = None, 0
        for outer in range(1, cfg.LBFGS_EPOCHS + 1):
            diff, state, vals = lbfgs_block(diff, state)
            stop = False
            for v in np.asarray(vals):
                cur = float(v)
                if prev is not None and prev > 0:
                    patience = patience + 1 if abs(prev - cur) / prev < cfg.LBFGS_TOL_REL else 0
                prev = cur
                if cfg.LBFGS_EARLY_STOP and patience >= cfg.LBFGS_TOL_PATIENCE:
                    stop = True
                    break
            self._log(cfg.ADAM_EPOCHS + outer, eval_terms(diff), lams, "")
            if stop:
                print(f"L-BFGS converged at outer iteration {outer}")
                break
        return diff

    def _log(self, epoch, aux, lams, extra):
        g = lambda k: float(aux[k])
        values = np.array([g("pde"), g("data"), g("bao"), g("sn")])
        weighted = np.asarray(lams) * values
        shares = 100.0 * weighted / (weighted.sum() + 1e-30)
        print(f"Epoch {epoch:6d}  L={g('total'):.5e}  PDE={values[0]:.3e} CC={values[1]:.3e} "
              f"BAO={values[2]:.3e} SN={values[3]:.3e}  "
              f"chi2/dof[CC={g('data_raw_chi2_dof'):.2f} BAO={g('bao_raw_chi2_dof'):.2f} "
              f"SN={g('sn_raw_chi2_dof'):.2f}]  Om0={g('Omega_m0_value'):.4f} "
              f"H0={g('H0_value'):.3f} rd={g('R_drag_value'):.3f}  {extra}")

        h = self.history
        h["epoch"].append(epoch)
        h["total"].append(g("total"))
        for k in LOSS_KEYS:
            h[k].append(g(k))
        for k, src in (("cc_chi2_dof", "data_raw_chi2_dof"), ("bao_chi2_dof", "bao_raw_chi2_dof"),
                       ("sn_chi2_dof", "sn_raw_chi2_dof"), ("Omega_m0", "Omega_m0_value"),
                       ("H0", "H0_value"), ("R_drag", "R_drag_value"),
                       ("omega_m0_prior", "omega_m0_prior"), ("h0_prior", "h0_prior"),
                       ("r_drag_prior", "r_drag_prior")):
            h[k].append(g(src))
        for i, name in enumerate(("pde", "data", "bao", "sn")):
            h[f"lam_{name}"].append(float(lams[i]))
            h[f"share_{name}"].append(float(shares[i]))

    def _export(self, model, dtype, z_data, H_data, sigma_data):
        cfg = self.cfg
        z = jnp.linspace(0.0, cfg.Z_MAX, 1000, dtype=dtype).reshape(-1, 1)
        f = self.physics.compute_physical_quantities(model, z, cfg)
        flat = lambda a: np.asarray(a).reshape(-1)

        H0 = float(model.H0())
        Om0 = float(model.Omega_m0())
        Ob0 = float(cfg.OMEGA_B) * float(cfg.H0) ** 2 / H0 ** 2
        path = os.path.join(cfg.SAVE_DIR, "run_results.npz")
        np.savez(
            path,
            z=flat(z), H=flat(f["H"]), w_DE=flat(f["w_DE"]), rho_DE=flat(f["rho_DE"]),
            Omega_DE=flat(f["Omega_DE"]), Omega_m=flat(f["Omega_m"]), Omega_b=flat(f["Omega_b"]),
            q=flat(deceleration(f)),
            q_autodiff=flat(deceleration_autodiff(model, consts_from_cfg(cfg), z)),
            z_data=flat(z_data), H_data=flat(H_data), sigma_data=flat(sigma_data),
            H0_learned=H0, Omega_m0_learned=Om0, Omega_b0_learned=Ob0,
            Omega_DE0_learned=1.0 - Om0 - Ob0, R_drag_learned=float(model.R_drag()),
            wde_label=np.str_(wde_label(cfg)), sn_dataset=np.str_(dataset_name(cfg)),
            **{f"hist_{k}": np.asarray(v) for k, v in self.history.items()},
        )
        print(f"\nModel and results saved to {cfg.SAVE_DIR}")
