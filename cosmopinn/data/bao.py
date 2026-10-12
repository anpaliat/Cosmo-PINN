"""BAO likelihood (DESI DR2 by default) and distance integrals."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass(frozen=True)
class BAOData:
    z: Tuple[float, ...]
    kinds: Tuple[str, ...]
    obs: np.ndarray
    inv_cov: np.ndarray


@lru_cache(maxsize=4)
def _load(data_path: str, cov_path: str) -> BAOData:
    table = np.genfromtxt(data_path, comments="#", dtype=None, encoding="utf-8")
    z = tuple(float(row[0]) for row in table)
    obs = np.array([float(row[1]) for row in table])
    kinds = tuple(str(row[2]).split("_")[0].upper() for row in table)
    unknown = set(kinds) - {"DV", "DM", "DH"}
    if unknown:
        raise ValueError(f"{data_path}: unknown BAO quantities {sorted(unknown)}")
    cov = np.loadtxt(cov_path, comments="#")
    if cov.shape != (len(z), len(z)):
        raise ValueError(f"{cov_path}: covariance shape {cov.shape} does not match {len(z)} data points")
    return BAOData(z=z, kinds=kinds, obs=obs, inv_cov=np.linalg.inv(cov))


def load_bao_data(cfg: Any) -> BAOData:
    """BAO measurements and inverse covariance from cfg.BAO_DATA_PATH / cfg.BAO_COV_PATH."""
    return _load(str(cfg.BAO_DATA_PATH), str(cfg.BAO_COV_PATH))


def bao_redshifts(cfg: Any) -> list:
    return list(load_bao_data(cfg).z)


def cumulative_comoving_distance(z_grid: Array, H_grid: Array, c_kms: float) -> Array:
    """D_M(z) = c int_0^z dz'/H(z') with the cumulative trapezoidal rule."""
    dz = z_grid[1:] - z_grid[:-1]
    cumint = jnp.cumsum(dz * 0.5 * (1.0 / H_grid[1:] + 1.0 / H_grid[:-1]), axis=0)
    return c_kms * jnp.concatenate([jnp.zeros((1,), dtype=H_grid.dtype), cumint])


def interp(z_query: Array, z_grid: Array, values: Array) -> Array:
    """Linear interpolation on a sorted grid."""
    idx = jnp.clip(jnp.searchsorted(z_grid, z_query), 1, z_grid.shape[0] - 1)
    z_lo, z_hi = z_grid[idx - 1], z_grid[idx]
    v_lo, v_hi = values[idx - 1], values[idx]
    return v_lo + (z_query - z_lo) / (z_hi - z_lo) * (v_hi - v_lo)


def compute_bao_chi2_total(H_grid: Array, z_grid: Array, r_drag: Array,
                           cfg: Any) -> Tuple[Array, int]:
    """Return (chi^2, N) for the BAO measurements (D_V, D_M, D_H over r_drag)."""
    data = load_bao_data(cfg)
    c_kms = cfg.C_KMS
    DM_grid = cumulative_comoving_distance(z_grid, H_grid, c_kms)

    theory = []
    for z_bao, obs_type in zip(data.z, data.kinds):
        z_q = jnp.asarray(z_bao, dtype=H_grid.dtype)
        H_q = interp(z_q, z_grid, H_grid)
        DM_q = interp(z_q, z_grid, DM_grid)
        if obs_type == "DH":
            theory.append(c_kms / (H_q * r_drag))
        elif obs_type == "DM":
            theory.append(DM_q / r_drag)
        else:
            DA_q = DM_q / (1.0 + z_q)
            DV = ((1.0 + z_q) ** 2 * DA_q ** 2 * c_kms * z_q / H_q) ** (1.0 / 3.0)
            theory.append(DV / r_drag)

    residuals = jnp.asarray(data.obs, dtype=H_grid.dtype) - jnp.stack(theory)
    return residuals @ jnp.asarray(data.inv_cov, dtype=H_grid.dtype) @ residuals, len(data.z)
