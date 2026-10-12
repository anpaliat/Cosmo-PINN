"""Cosmic chronometers: Moresco et al. H(z) data and their covariance (Moresco et al. 2020)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from cosmopinn.config import as_jnp

SYS_COLUMNS = {"imf": 1, "slib": 2, "sps": 3, "spsooo": 4}

MORESCO_BC03_Z = np.array([
    0.1791, 0.1993, 0.3519, 0.3802, 0.4004, 0.4247, 0.4497, 0.4783,
    0.5929, 0.6797, 0.7812, 0.8754, 1.037, 1.363, 1.965,
])


def moresco_mask(z: np.ndarray) -> np.ndarray:
    """True for the Moresco 2012/2015/2016 (BC03) measurements."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    return np.any(np.abs(z[:, None] - MORESCO_BC03_Z[None, :]) < 5e-5, axis=1)


def build_cc_covariance(z: np.ndarray, H: np.ndarray, sigma: np.ndarray, sys_path: str,
                        components: Sequence[str] = ("imf", "spsooo")) -> np.ndarray:
    """Cov_ij = diag(sigma^2) + sum_X (H_i eta_X(z_i)) (H_j eta_X(z_j)) on the Moresco points."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    H = np.asarray(H, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)

    cov = np.diag(sigma ** 2)
    table = np.loadtxt(sys_path, comments="#")
    mask = moresco_mask(z)
    for name in components:
        key = str(name).lower()
        if key not in SYS_COLUMNS:
            raise ValueError(f"Unknown CC systematic '{name}'; choose from {sorted(SYS_COLUMNS)}")
        eta = np.interp(z, table[:, 0], table[:, SYS_COLUMNS[key]]) / 100.0
        v = np.where(mask, H * eta, 0.0)
        cov += np.outer(v, v)
    return cov


def _read_table(path: str) -> np.ndarray:
    """First three columns (z, H, sigma); whitespace- or comma-separated, '#' comments."""
    with open(path, encoding="utf-8") as f:
        first = next(line for line in f if line.strip() and not line.lstrip().startswith("#"))
    raw = np.genfromtxt(path, comments="#", usecols=(0, 1, 2),
                        delimiter="," if "," in first else None)
    return raw.reshape(1, -1) if raw.ndim == 1 else raw


@lru_cache(maxsize=8)
def _whitener(data_path: str, sys_path: str,
              components: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    raw = _read_table(data_path)
    cov = build_cc_covariance(raw[:, 0], raw[:, 1], raw[:, 2], sys_path, components)
    L = np.linalg.cholesky(cov)
    return np.linalg.solve(L, np.eye(len(cov))), np.sqrt(np.diag(cov))


def cc_whitener(cfg: Any) -> Tuple[np.ndarray, np.ndarray]:
    """(L^{-1}, sqrt(diag C)) with C = L L^T the full CC covariance."""
    comps = tuple(str(c).lower() for c in cfg.CC_COV_COMPONENTS)
    return _whitener(str(cfg.CC_DATA_PATH), str(cfg.CC_SYS_PATH), comps)


def load_cc_data(cfg: Any, dtype=None) -> Tuple[Array, Array, Array]:
    """Return (z, H, sigma) as (N, 1) arrays; sigma includes systematics if CC_USE_COV."""
    jdt = as_jnp(dtype if dtype is not None else cfg.DTYPE)
    raw = _read_table(cfg.CC_DATA_PATH)
    sigma = cc_whitener(cfg)[1].reshape(-1, 1) if cfg.CC_USE_COV else raw[:, 2:3]
    print(f"[CC] {len(raw)} points, z in [{raw[:, 0].min():.4f}, {raw[:, 0].max():.4f}], "
          f"covariance: {'Moresco+20 ' + str(tuple(cfg.CC_COV_COMPONENTS)) if cfg.CC_USE_COV else 'diagonal'}")
    return (jnp.asarray(raw[:, 0:1], dtype=jdt), jnp.asarray(raw[:, 1:2], dtype=jdt),
            jnp.asarray(sigma, dtype=jdt))


def cc_whitened_residuals(H_model: Array, H_data: Array, sigma_data: Array, cfg: Any) -> Array:
    """Residuals r_w with chi^2 = sum(r_w^2)."""
    resid = H_model - H_data
    if not cfg.CC_USE_COV:
        return resid / sigma_data
    W = jnp.asarray(cc_whitener(cfg)[0], dtype=resid.dtype)
    if W.shape[0] != resid.shape[0]:
        raise ValueError(f"CC covariance is {W.shape[0]}x{W.shape[0]} but got "
                         f"{resid.shape[0]} residuals")
    return W @ resid
