"""PINN training loss: physics residual, data chi^2, smoothness and prior terms."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from cosmopinn.data.bao import compute_bao_chi2_total
from cosmopinn.data.cc import cc_whitened_residuals
from cosmopinn.data.sn import SNData, compute_sn_chi2_total
from cosmopinn.physics.residual import pde_residual
from cosmopinn.training.model import gaussian_penalty

DIFF_KEYS = ("rho_DE", "w_DE", "h")
N_SECOND_DERIV_MAX = 500


def soft_cap(L: Array, c: float) -> Array:
    """c^2 log(1 + L/c^2): identity for small L, logarithmic above c^2."""
    return (c * c) * jnp.log1p(L / (c * c))


def compute_loss(physics, model, z_colloc: Array, z_data: Array, H_data: Array,
                 sigma_data: Array, z_bao: Array, sn_data: Optional[SNData], cfg: Any,
                 lambdas: Optional[Tuple] = None,
                 dh_dx_ema: Optional[Array] = None) -> Dict[str, Array]:
    """Return the total loss and its individual terms.

    lambdas = (pde, data, bao, smooth, dwde, smooth_h, dh, bounds, sn).
    """
    if lambdas is None:
        lambdas = (cfg.LAMBDA_PDE, cfg.LAMBDA_DATA, cfg.LAMBDA_BAO, cfg.LAMBDA_SMOOTH,
                   cfg.LAMBDA_DWDE, cfg.LAMBDA_SMOOTH_H, cfg.LAMBDA_DH,
                   cfg.LAMBDA_BOUNDS, cfg.LAMBDA_SN)
    (lam_pde, lam_data, lam_bao, lam_smooth, lam_dwde,
     lam_smooth_h, lam_dh, lam_bounds, lam_sn) = lambdas

    n_c = z_colloc.shape[0]
    n_d = z_data.shape[0]

    fields_db = physics.compute_physical_quantities(
        model, jnp.concatenate([z_data, z_bao], axis=0), cfg)
    H_d = fields_db["H"][:n_d]
    H_b = fields_db["H"][n_d:].reshape(-1)

    def fields_at(z):
        d = physics.compute_physical_quantities(model, jnp.reshape(z, (1, 1)), cfg)
        return {k: v[0, 0] for k, v in d.items()}

    def diff_vector(z):
        d = fields_at(z)
        return jnp.stack([d[k] for k in DIFF_KEYS])

    z_flat = jnp.reshape(z_colloc, (-1,))
    f_c, df_c = jax.vmap(lambda z: jax.jvp(fields_at, (z,), (jnp.ones_like(z),)))(z_flat)
    d1 = jnp.stack([df_c[k] for k in DIFF_KEYS], axis=-1)

    n2 = min(n_c, N_SECOND_DERIV_MAX)
    z_sub = z_flat[jnp.linspace(0, n_c - 1, n2).astype(jnp.int32)] if n2 < n_c else z_flat
    d2 = jax.vmap(lambda z: jax.jvp(jax.jacfwd(diff_vector), (z,), (jnp.ones_like(z),))[1])(z_sub)

    w_DE = f_c["w_DE"]
    L_PDE = jnp.sum(pde_residual(z_flat, f_c, df_c) ** 2) / n_c

    Z = cfg.Z_MAX
    d_wde_dx = d1[:, 1] * Z
    dh_dx = d1[:, 2] * Z
    d2_rho_dx2 = d2[:, 0] * Z * Z
    d2_wde_dx2 = d2[:, 1] * Z * Z
    d2_h_dx2 = d2[:, 2] * Z * Z

    L_SMOOTH = jnp.mean(d2_wde_dx2 ** 2)
    L_SMOOTH_H = jnp.mean(d2_h_dx2 ** 2)
    L_OMEGA_SMOOTH = jnp.mean(d2_rho_dx2 ** 2)
    L_DWDE = jnp.mean(d_wde_dx ** 2)
    L_SMOOTH_MAX = jnp.max(d2_wde_dx2 ** 2)
    L_SMOOTH_H_MAX = jnp.max(d2_h_dx2 ** 2)
    L_OMEGA_SMOOTH_MAX = jnp.max(d2_rho_dx2 ** 2)
    L_DH = jnp.mean((dh_dx - (0.0 if dh_dx_ema is None else dh_dx_ema)) ** 2)

    Omega_DE = f_c["Omega_DE"]
    low = jnp.maximum(-Omega_DE, 0.0)
    high = jnp.maximum(Omega_DE - 1.0, 0.0)
    L_BOUNDS = jnp.mean(low ** 2 + low) + jnp.mean(high ** 2 + high)

    w_lo = jnp.maximum(-(1.0 + w_DE), 0.0)
    w_hi = jnp.maximum(w_DE - 1.0, 0.0)
    L_WDE_BOUND = jnp.mean(w_lo ** 2 + w_lo) + jnp.mean(w_hi ** 2 + w_hi)

    r_cc = cc_whitened_residuals(H_d, H_data, sigma_data, cfg)
    L_DATA_RAW = jnp.mean(r_cc ** 2) if n_d > 0 else jnp.zeros((), dtype=H_d.dtype)
    L_DATA = L_DATA_RAW if (cfg.USE_CC and n_d > 0) else jnp.zeros((), dtype=H_d.dtype)

    z_bao_1d = z_bao.reshape(-1)
    Omega_m0, H0, r_drag = model.Omega_m0(), model.H0(), model.R_drag()
    chi2_bao, n_bao = compute_bao_chi2_total(H_b, z_bao_1d, r_drag, cfg)
    L_BAO_RAW = chi2_bao / n_bao
    L_BAO = soft_cap(L_BAO_RAW, cfg.SOFT_CAP_C_BAO)

    chi2_sn, n_sn = compute_sn_chi2_total(H_b, z_bao_1d, sn_data, cfg)
    L_SN_RAW = chi2_sn / (n_sn - 1) if n_sn > 1 else chi2_sn
    L_SN = soft_cap(L_SN_RAW, cfg.SOFT_CAP_C_SN) if n_sn > 0 else L_SN_RAW

    L_total = (lam_pde * L_PDE + lam_data * L_DATA + lam_bao * L_BAO + lam_sn * L_SN
               + lam_smooth * L_SMOOTH + lam_dwde * L_DWDE + lam_smooth_h * L_SMOOTH_H
               + lam_dh * L_DH + cfg.LAMBDA_OMEGA_SMOOTH * L_OMEGA_SMOOTH
               + lam_bounds * L_BOUNDS + cfg.LAMBDA_WDE_BOUND * L_WDE_BOUND
               + cfg.LAMBDA_SMOOTH_MAX * L_SMOOTH_MAX
               + cfg.LAMBDA_SMOOTH_H_MAX * L_SMOOTH_H_MAX
               + cfg.LAMBDA_OMEGA_SMOOTH_MAX * L_OMEGA_SMOOTH_MAX
               + physics.prior_loss(model, cfg))

    return {
        "total": L_total,
        "pde": L_PDE, "data": L_DATA, "bao": L_BAO, "sn": L_SN,
        "smooth": L_SMOOTH, "dwde": L_DWDE, "omega_smooth": L_OMEGA_SMOOTH,
        "smooth_h": L_SMOOTH_H, "dh": L_DH, "bounds": L_BOUNDS, "wde_bound": L_WDE_BOUND,
        "smooth_max": L_SMOOTH_MAX, "smooth_h_max": L_SMOOTH_H_MAX,
        "omega_smooth_max": L_OMEGA_SMOOTH_MAX,
        "omega_m0_prior": cfg.LAMBDA_OMEGA_M0_PRIOR * gaussian_penalty(
            Omega_m0, cfg.OMEGA_M0_PRIOR_MU, cfg.OMEGA_M0_PRIOR_SIGMA),
        "h0_prior": cfg.LAMBDA_H0_PRIOR * gaussian_penalty(H0, cfg.H0_PRIOR_MU, cfg.H0_PRIOR_SIGMA),
        "r_drag_prior": cfg.LAMBDA_R_DRAG_PRIOR * gaussian_penalty(
            r_drag, cfg.R_DRAG_PRIOR_MU, cfg.R_DRAG_PRIOR_SIGMA),
        "data_raw_chi2_dof": L_DATA_RAW,
        "bao_raw_chi2_dof": L_BAO_RAW,
        "sn_raw_chi2_dof": L_SN_RAW,
        "Omega_m0_value": Omega_m0,
        "H0_value": H0,
        "R_drag_value": r_drag,
        "dh_dx_detached": dh_dx,
    }
