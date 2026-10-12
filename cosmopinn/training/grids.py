"""Collocation grid, BAO/SN integration grid and residual-based resampling."""

from __future__ import annotations

from functools import partial
from typing import Any, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cosmopinn.config import as_jnp
from cosmopinn.training.model import consts_from_cfg, normalised_residual_at


def collocation_points(cfg: Any, n: Optional[int] = None, dtype=None) -> Array:
    """Uniform grid on [0, Z_MAX], shape (n, 1)."""
    n = cfg.N_COLLOCATION if n is None else n
    jdt = as_jnp(dtype if dtype is not None else cfg.DTYPE)
    return jnp.linspace(0.0, cfg.Z_MAX, n, dtype=jdt).reshape(-1, 1)


def integration_grid(cfg: Any, bao_redshifts: List[float], dtype=None) -> Array:
    """Uniform grid on [0, Z_MAX] refined around each BAO redshift, shape (n, 1)."""
    jdt = as_jnp(dtype if dtype is not None else cfg.DTYPE)
    z_max = float(cfg.Z_MAX)
    refine = [np.linspace(max(0.0, zb - 1e-2), min(z_max, zb + 1e-2), 5) for zb in bao_redshifts]
    refine.append(np.asarray(bao_redshifts, dtype=float))
    all_z = np.unique(np.concatenate([np.linspace(0.0, z_max, cfg.N_BAO_GRID)] + refine))
    if all_z[0] > 0.0:
        all_z = np.concatenate([[0.0], all_z])
    if all_z[-1] < z_max:
        all_z = np.concatenate([all_z, [z_max]])
    return jnp.asarray(all_z, dtype=jdt).reshape(-1, 1)


@partial(jax.jit, static_argnums=(2,))
def _residual_score(model, z_pool: Array, consts) -> Array:
    score = (normalised_residual_at(model, consts, z_pool) ** 2).reshape(-1)
    return jnp.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def resample_collocation(model, cfg: Any, key: Array, dtype=None) -> Array:
    """Residual-based adaptive distribution (RAD) of the collocation points."""
    jdt = as_jnp(dtype if dtype is not None else cfg.DTYPE)
    n_pool, n_keep = cfg.RESAMPLE_POOL, cfg.N_COLLOCATION
    z_pool = jnp.linspace(0.0, cfg.Z_MAX, n_pool, dtype=jdt).reshape(-1, 1)
    score = _residual_score(model, z_pool, consts_from_cfg(cfg))

    k_resid, k_unif = jax.random.split(key)
    if float(jnp.sum(score)) <= 0.0:
        idx = jax.random.permutation(k_resid, n_pool)[:n_keep]
    else:
        weights = score ** cfg.RESAMPLE_K
        weights = weights / jnp.sum(weights)
        n_resid = int(round(n_keep * (1.0 - cfg.RESAMPLE_UNIFORM_FRAC)))
        idx = jnp.concatenate([
            jax.random.choice(k_resid, n_pool, shape=(n_resid,), replace=True, p=weights),
            jax.random.randint(k_unif, (n_keep - n_resid,), 0, n_pool),
        ])
    return jnp.sort(z_pool[idx], axis=0)
