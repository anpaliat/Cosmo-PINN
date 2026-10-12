"""Log-posterior for NUTS and propagation of samples to credible bands."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cosmopinn.bayes.packer import Packer
from cosmopinn.data.bao import compute_bao_chi2_total
from cosmopinn.data.cc import cc_whitened_residuals
from cosmopinn.data.sn import SNData, compute_sn_chi2_total
from cosmopinn.physics.hubble import deceleration
from cosmopinn.training.model import consts_from_cfg, fields_single, normalised_residual_at


def make_log_posterior(physics, model, cfg: Any, z_data: Array, H_data: Array,
                       sigma_data: Array, z_bao_grid: Array, sn_data: Optional[SNData],
                       packer: Packer, z_pde_grid: Optional[Array] = None
                       ) -> Callable[[Array], Array]:
    """log p(theta | data) up to a constant: CC + BAO + SN + PDE likelihoods and priors."""
    dtype = z_bao_grid.dtype
    use_cc = bool(cfg.HMC_USE_CC)
    use_bao = bool(cfg.HMC_USE_BAO)
    use_sn = bool(cfg.HMC_USE_SN) and sn_data is not None
    use_pde = bool(cfg.HMC_USE_PDE)

    consts = consts_from_cfg(cfg)
    z_bao_1d = z_bao_grid.reshape(-1)
    n_cc = z_data.shape[0] if use_cc else 0

    if use_pde:
        z_pde = (jnp.linspace(0.0, cfg.Z_MAX, int(cfg.HMC_PDE_N_COLLOC), dtype=dtype).reshape(-1, 1)
                 if z_pde_grid is None else z_pde_grid.astype(dtype))
        pde_inv_sig2 = jnp.asarray(1.0 / float(cfg.HMC_PDE_SIGMA) ** 2, dtype=dtype)

    theta_map = packer.theta_map
    if packer.lcdm_prior:
        w_inv_var = packer.w_mask / packer.w_sigma ** 2
        ridge_mask = (1.0 - packer.scalar_mask) * (1.0 - packer.w_mask)
    else:
        w_inv_var = None
        ridge_mask = 1.0 - packer.scalar_mask
    w_sig = float(cfg.HMC_WEIGHT_PRIOR_SIGMA)
    inv_w_sig2 = 1.0 / w_sig ** 2 if w_sig > 0.0 else 0.0

    if n_cc > 0:
        z_combined = jnp.concatenate([z_data, z_bao_grid], axis=0)
    else:
        z_combined = z_bao_grid

    def logdensity_fn(theta: Array) -> Array:
        m = packer.reconstruct(theta)
        H_all = physics.compute_physical_quantities(m, z_combined, cfg)["H"]
        H_b = H_all[n_cc:].reshape(-1)
        R_drag = m.R_drag()

        log_L = jnp.zeros((), dtype=dtype)
        if n_cc > 0:
            log_L = log_L - 0.5 * jnp.sum(cc_whitened_residuals(H_all[:n_cc], H_data, sigma_data, cfg) ** 2)
        if use_bao:
            log_L = log_L - 0.5 * compute_bao_chi2_total(H_b, z_bao_1d, R_drag, cfg)[0]
        if use_sn:
            log_L = log_L - 0.5 * compute_sn_chi2_total(H_b, z_bao_1d, sn_data, cfg)[0]
        if use_pde:
            r = normalised_residual_at(m, consts, z_pde)
            log_L = log_L - 0.5 * pde_inv_sig2 * jnp.sum(r ** 2)

        Om0, H0 = m.Omega_m0(), m.H0()
        log_P = jnp.zeros((), dtype=dtype)
        log_P = (log_P - 0.5 * ((Om0 - cfg.OMEGA_M0_PRIOR_MU) / cfg.OMEGA_M0_PRIOR_SIGMA) ** 2
                 + jnp.log(Om0 * (1.0 - Om0)))
        log_P = (log_P - 0.5 * ((H0 - cfg.H0_PRIOR_MU) / cfg.H0_PRIOR_SIGMA) ** 2
                 + jax.nn.log_sigmoid(m.H0_raw))
        log_P = (log_P - 0.5 * ((R_drag - cfg.R_DRAG_PRIOR_MU) / cfg.R_DRAG_PRIOR_SIGMA) ** 2
                 + jax.nn.log_sigmoid(m.R_drag_raw))
        if w_inv_var is not None:
            log_P = log_P - 0.5 * jnp.sum(w_inv_var * (theta - packer.w_lcdm) ** 2)
        if inv_w_sig2 > 0.0:
            log_P = log_P - 0.5 * jnp.sum(((theta - theta_map) * ridge_mask) ** 2) * inv_w_sig2
        return log_L + log_P

    return logdensity_fn


def propagate_samples_to_bands(cfg: Any, samples: np.ndarray, packer: Packer,
                               z_grid: Array) -> Dict[str, np.ndarray]:
    """Median and 68/95% bands of H, w_DE, Omega_DE, rho_DE and q on z_grid."""
    dtype = z_grid.dtype
    consts = consts_from_cfg(cfg)
    z_flat = z_grid.reshape(-1)

    @jax.jit
    def per_sample(theta):
        m = packer.reconstruct(theta)
        d = jax.vmap(lambda z: fields_single(m, consts, z))(z_flat)
        q = deceleration(d)
        return d, q, jnp.stack([m.Omega_m0(), m.H0(), m.R_drag()])

    d, q, scal = jax.vmap(per_sample)(jnp.asarray(samples, dtype=dtype))
    fields = {"H": d["H"], "w_DE": d["w_DE"], "Omega_DE": d["Omega_DE"],
              "rho_DE": d["rho_DE"], "q": q}

    out: Dict[str, np.ndarray] = {"z": np.asarray(z_flat)}
    for k, arr in fields.items():
        arr = np.asarray(arr)
        out[f"{k}_samples"] = arr
        out[f"{k}_median"] = np.median(arr, axis=0)
        out[f"{k}_lo68"], out[f"{k}_hi68"] = np.percentile(arr, [16.0, 84.0], axis=0)
        out[f"{k}_lo95"], out[f"{k}_hi95"] = np.percentile(arr, [2.5, 97.5], axis=0)
    scal = np.asarray(scal)
    out["Omega_m0_scalar"], out["H0_scalar"], out["R_drag_scalar"] = scal[:, 0], scal[:, 1], scal[:, 2]
    return out
