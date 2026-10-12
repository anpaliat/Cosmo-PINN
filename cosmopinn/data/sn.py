"""Type Ia supernova likelihoods with analytic marginalisation over M."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax import Array

from cosmopinn.config import as_jnp
from cosmopinn.data.bao import cumulative_comoving_distance, interp


def _read_square_matrix(path: str) -> np.ndarray:
    with open(path, "r") as f:
        n = int(f.readline().strip())
        flat = np.loadtxt(f)
    if flat.size != n * n:
        raise ValueError(f"{path}: expected {n * n} entries, found {flat.size}")
    return flat.reshape(n, n)


def _z_mask(z: np.ndarray, cfg: Any) -> np.ndarray:
    return (z > 0.0) & (z < 0.999 * cfg.Z_MAX)


def _load_pantheon_plus(cfg: Any):
    raw = np.genfromtxt(cfg.SN_PANTHEON_DATA, names=True, dtype=None, encoding="utf-8")
    z = np.asarray(raw["zHD"], dtype=np.float64)
    mu = np.asarray(raw["MU_SH0ES"], dtype=np.float64)
    is_cal = np.asarray(raw["IS_CALIBRATOR"], dtype=np.int32)
    cov = _read_square_matrix(cfg.SN_PANTHEON_COV)
    if cov.shape[0] != len(z):
        raise ValueError(f"Pantheon+ covariance has dim {cov.shape[0]}, data has {len(z)}")
    mask = (is_cal == 0) & _z_mask(z, cfg)
    idx = np.where(mask)[0]
    print(f"[SN] Pantheon+: {mask.sum()}/{len(z)} SNe (calibrators and z >= Z_MAX removed)")
    return z[mask], mu[mask], cov[np.ix_(idx, idx)], "cov", None


def _load_pantheon_plus_sh0es(cfg: Any):
    raw = np.genfromtxt(cfg.SN_PANTHEON_DATA, names=True, dtype=None, encoding="utf-8")
    z = np.asarray(raw["zHD"], dtype=np.float64)
    mb = np.asarray(raw["m_b_corr"], dtype=np.float64)
    is_cal = np.asarray(raw["IS_CALIBRATOR"], dtype=np.int32).astype(bool)
    mu_ceph = np.asarray(raw["CEPH_DIST"], dtype=np.float64)
    cov = _read_square_matrix(cfg.SN_PANTHEON_COV)
    if cov.shape[0] != len(z):
        raise ValueError(f"Pantheon+ covariance has dim {cov.shape[0]}, data has {len(z)}")
    keep = is_cal | ((~is_cal) & _z_mask(z, cfg))
    idx = np.where(keep)[0]
    print(f"[SN] Pantheon+&SH0ES: {is_cal.sum()} calibrators + "
          f"{keep.sum() - is_cal.sum()} Hubble-flow SNe")
    extra = {"is_calib": is_cal[keep], "mu_ceph": mu_ceph[keep]}
    return z[keep], mb[keep], cov[np.ix_(idx, idx)], "cov", extra


def _load_union3(cfg: Any):
    arr = np.loadtxt(cfg.SN_UNION3_DATA, comments="#", usecols=(1, 4)).reshape(-1, 2)
    z, mu = arr[:, 0], arr[:, 1]
    cov = _read_square_matrix(cfg.SN_UNION3_COV)
    if cov.shape[0] != len(z):
        raise ValueError(f"Union3 covariance has dim {cov.shape[0]}, data has {len(z)}")
    mask = _z_mask(z, cfg)
    idx = np.where(mask)[0]
    print(f"[SN] Union3: {mask.sum()}/{len(z)} bins")
    return z[mask], mu[mask], cov[np.ix_(idx, idx)], "cov", None


def _load_des_dovekie(cfg: Any):
    arr = np.genfromtxt(cfg.SN_DES_DATA, delimiter=",", names=True, dtype=None, encoding="utf-8")
    z = np.asarray(arr["zHD"], dtype=np.float64)
    mu = np.asarray(arr["MU"], dtype=np.float64)
    npz = np.load(cfg.SN_DES_INVCOV)
    n = int(npz["nsn"][0])
    if n != len(z):
        raise ValueError(f"DES-Dovekie covariance has nsn={n}, data has {len(z)}")
    invcov = np.zeros((n, n), dtype=np.float64)
    invcov[np.triu_indices(n)] = np.asarray(npz["cov"], dtype=np.float64)
    invcov = invcov + invcov.T - np.diag(np.diag(invcov))
    mask = _z_mask(z, cfg)
    print(f"[SN] DES-Dovekie: {mask.sum()}/{len(z)} SNe")
    if mask.all():
        return z, mu, invcov, "invcov", None
    idx = np.where(mask)[0]
    invcov = np.linalg.inv(np.linalg.inv(invcov)[np.ix_(idx, idx)])
    return z[mask], mu[mask], invcov, "invcov", None


_LOADERS = {
    "pantheon_plus": _load_pantheon_plus,
    "pantheon_plus_sh0es": _load_pantheon_plus_sh0es,
    "union3": _load_union3,
    "des_dovekie": _load_des_dovekie,
}

_CACHE: dict = {}


@dataclass(frozen=True)
class SNData:
    z_sn: Array
    y_obs: Array
    M: Array
    mode: str
    is_calib: Optional[Array]
    mu_ceph: Optional[Array]
    n_sn: int


def dataset_name(cfg: Any) -> str:
    return (cfg.SN_DATASET or "none").strip().lower()


def build_sn_data(cfg: Any, dtype=None) -> Optional[SNData]:
    """Load the SN catalogue selected by cfg.SN_DATASET (None if 'none')."""
    name = dataset_name(cfg)
    if name == "none":
        return None
    if name not in _LOADERS:
        raise ValueError(f"Unknown SN dataset {name!r}; choose from {list(_LOADERS) + ['none']}")

    key = (name, cfg.SN_PANTHEON_DATA, cfg.SN_UNION3_DATA, cfg.SN_DES_DATA, float(cfg.Z_MAX))
    if key not in _CACHE:
        _CACHE[key] = _LOADERS[name](cfg)
    z, y, mat, mode, extra = _CACHE[key]
    jdt = as_jnp(dtype if dtype is not None else cfg.DTYPE)
    mat_t = jnp.asarray(mat, dtype=jdt)
    if mode == "cov":
        M = jnp.linalg.cholesky(mat_t)
        if not bool(jnp.all(jnp.isfinite(M))):
            M = jnp.linalg.cholesky(mat_t + cfg.SN_JITTER * jnp.eye(mat_t.shape[0], dtype=jdt))
    else:
        M = mat_t

    return SNData(
        z_sn=jnp.asarray(z, dtype=jdt), y_obs=jnp.asarray(y, dtype=jdt), M=M, mode=mode,
        is_calib=None if extra is None else jnp.asarray(extra["is_calib"], dtype=bool),
        mu_ceph=None if extra is None else jnp.asarray(extra["mu_ceph"], dtype=jdt),
        n_sn=int(len(y)))


def _apply_cinv(sn: SNData, vec: Array) -> Array:
    if sn.mode == "cov":
        return jsl.cho_solve((sn.M, True), vec)
    return sn.M @ vec


def compute_sn_chi2_total(H_grid: Array, z_grid: Array, sn: Optional[SNData],
                          cfg: Any) -> Tuple[Array, int]:
    """chi^2 marginalised over the absolute magnitude: A - B^2 / C."""
    if sn is None:
        return jnp.zeros((), dtype=H_grid.dtype), 0

    DM_grid = cumulative_comoving_distance(z_grid, H_grid, cfg.C_KMS)
    DL = jnp.clip((1.0 + sn.z_sn) * interp(sn.z_sn, z_grid, DM_grid), 1e-10, None)
    mu_th = 5.0 * jnp.log10(DL) + 25.0
    mu_model = mu_th if sn.is_calib is None else jnp.where(sn.is_calib, sn.mu_ceph, mu_th)

    delta = sn.y_obs - mu_model
    ones = jnp.ones_like(sn.y_obs)
    inv_delta = _apply_cinv(sn, delta)
    A = jnp.dot(delta, inv_delta)
    B = jnp.dot(ones, inv_delta)
    C = jnp.dot(ones, _apply_cinv(sn, ones))
    return A - B * B / C, sn.n_sn
