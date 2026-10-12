"""Configuration: defaults, YAML loader and run summary."""

from __future__ import annotations

import math
import os

import jax.numpy as jnp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DTYPES = {
    "float64": "float64", "f64": "float64", "64": "float64",
    "float32": "float32", "f32": "float32", "32": "float32",
}

_PATH_KEYS = (
    "CC_DATA_PATH", "CC_SYS_PATH", "BAO_DATA_PATH", "BAO_COV_PATH",
    "SN_PANTHEON_DATA", "SN_PANTHEON_COV",
    "SN_UNION3_DATA", "SN_UNION3_COV",
    "SN_DES_DATA", "SN_DES_INVCOV",
)


def as_jnp(spec):
    """Map a dtype string ("float32"/"float64") to a jnp dtype."""
    if spec is None:
        return jnp.float64
    if isinstance(spec, str):
        return jnp.float64 if _DTYPES.get(spec.lower().strip()) != "float32" else jnp.float32
    return jnp.dtype(spec)


class Cfg:
    """Run configuration. Every attribute can be overridden from the YAML file."""

    H0: float = 69.5
    OMEGA_M0: float = 0.30
    R_DRAG: float = 147.1
    Z_MAX: float = 2.34
    C_KMS: float = 299792.458
    OMEGA_B_H2: float = 0.0224

    OMEGA_M0_TRAINABLE: bool = True
    OMEGA_M0_PRIOR_MU: float = 0.30
    OMEGA_M0_PRIOR_SIGMA: float = 0.03
    LAMBDA_OMEGA_M0_PRIOR: float = 0.01

    H0_TRAINABLE: bool = True
    H0_PRIOR_MU: float = 68.0
    H0_PRIOR_SIGMA: float = 1.0
    LAMBDA_H0_PRIOR: float = 1e-4

    R_DRAG_TRAINABLE: bool = True
    R_DRAG_PRIOR_MU: float = 147.1
    R_DRAG_PRIOR_SIGMA: float = 2.0
    LAMBDA_R_DRAG_PRIOR: float = 1e-4

    HIDDEN_LAYERS: list = None
    FOURIER_FREQS: int = 4

    WDE_NET_HIDDEN_LAYERS: list = None
    WDE_NET_FOURIER_FREQS: int = 0

    N_COLLOCATION: int = 1000
    N_BAO_GRID: int = 1000

    ADAM_LR: float = 1e-3
    ADAM_LR_MIN: float = 1e-4
    ADAM_EPOCHS: int = 3000
    SCHEDULER: str = "cosine"
    SCHEDULER_STEP: int = 2000
    SCHEDULER_GAMMA: float = 0.5
    MAX_GRAD_NORM: float = 1.0
    WEIGHT_DECAY: float = 1e-4

    LBFGS_EPOCHS: int = 100
    LBFGS_MAX_ITER: int = 20
    LBFGS_EARLY_STOP: bool = False
    LBFGS_TOL_REL: float = 1e-7
    LBFGS_TOL_PATIENCE: int = 5

    LOG_EVERY: int = 500
    SEED: int = 42

    LAMBDA_PDE: float = 1.0
    LAMBDA_DATA: float = 1.0
    LAMBDA_BAO: float = 1.0
    LAMBDA_SN: float = 1.0
    LAMBDA_SMOOTH: float = 5e-4
    LAMBDA_DWDE: float = 5e-4
    LAMBDA_SMOOTH_H: float = 5e-4
    LAMBDA_DH: float = 5e-4
    LAMBDA_OMEGA_SMOOTH: float = 5e-4
    LAMBDA_SMOOTH_MAX: float = 1e-6
    LAMBDA_SMOOTH_H_MAX: float = 1e-5
    LAMBDA_OMEGA_SMOOTH_MAX: float = 1e-5
    LAMBDA_BOUNDS: float = 10.0
    LAMBDA_WDE_BOUND: float = 0.0
    DH_EMA_ALPHA: float = 0.99

    SOFT_CAP_C_SN: float = 4.0
    SOFT_CAP_C_BAO: float = 4.0

    BALANCING_MODE: str = "gradnorm"
    GRADNORM_EVERY: int = 500
    GRADNORM_ALPHA: float = 0.3
    GRADNORM_LAM_LO: float = 0.5
    GRADNORM_LAM_HI: float = 5.0
    GRADNORM_FREEZE_AFTER: int = 3000

    RESAMPLE_EVERY: int = 500
    RESAMPLE_START: int = 1000
    RESAMPLE_POOL: int = 5000
    RESAMPLE_K: float = 1.0
    RESAMPLE_UNIFORM_FRAC: float = 0.2

    USE_CC: bool = True
    CC_DATA_PATH: str = "data/cc/HzTable_MM_BC03.dat"
    CC_USE_COV: bool = True
    CC_SYS_PATH: str = "data/cc/data_MM20.dat"
    CC_COV_COMPONENTS: tuple = ("imf", "spsooo")

    BAO_DATA_PATH: str = "data/bao/desi_gaussian_bao_ALL_GCcomb_mean.txt"
    BAO_COV_PATH: str = "data/bao/desi_gaussian_bao_ALL_GCcomb_cov.txt"

    SN_DATASET: str = "pantheon_plus"
    SN_PANTHEON_DATA: str = "data/sn/PantheonPlus/Pantheon+SH0ES.dat"
    SN_PANTHEON_COV: str = "data/sn/PantheonPlus/Pantheon+SH0ES_STAT+SYS.cov"
    SN_UNION3_DATA: str = "data/sn/Union3/lcparam_full.txt"
    SN_UNION3_COV: str = "data/sn/Union3/mag_covmat.txt"
    SN_DES_DATA: str = "data/sn/DES-Dovekie/DES-Dovekie_HD.csv"
    SN_DES_INVCOV: str = "data/sn/DES-Dovekie/covtot_inv_000.npz"
    SN_JITTER: float = 1e-8

    DTYPE: str = "float64"
    ADAM_DTYPE: str = "float32"

    USE_HMC: bool = True
    HMC_SCOPE: str = "wde"
    HMC_DTYPE: str = "float64"
    HMC_NUM_WARMUP: int = 3000
    HMC_NUM_SAMPLES: int = 3000
    HMC_NUM_CHAINS: int = 4
    HMC_VECTORISE_CHAINS: bool = True
    HMC_TARGET_ACCEPT: float = 0.85
    HMC_MAX_TREE_DEPTH: int = 10
    HMC_DENSE_MASS: bool = True
    HMC_INIT_STEP_SIZE: float = 0.02
    HMC_JITTER: float = 1e-2
    HMC_SEED: int = 42
    HMC_LOG_EVERY: int = 100
    HMC_WEIGHT_PRIOR_SIGMA: float = 5.0
    HMC_USE_CC: bool = True
    HMC_USE_BAO: bool = True
    HMC_USE_SN: bool = True
    HMC_USE_PDE: bool = True
    HMC_PDE_SIGMA: float = 0.03
    HMC_PDE_N_COLLOC: int = 200
    HMC_WNET_PRIOR_SIGMA: float = 0.16
    HMC_BAND_N_Z: int = 500
    HMC_THIN: int = 1
    HMC_SAVE_CHAIN: bool = True

    SAVE_DIR: str = "results/run"

    def __init__(self) -> None:
        self.HIDDEN_LAYERS = [64, 64, 64, 64]
        self.WDE_NET_HIDDEN_LAYERS = [32, 32]
        self.update_derived()

    def update_derived(self) -> None:
        """Recompute quantities that depend on other settings."""
        self.OMEGA_B = round(self.OMEGA_B_H2 * 1e4, 10) / self.H0 ** 2
        self.LOGIT_OMEGA_M0 = math.log(self.OMEGA_M0 / (1.0 - self.OMEGA_M0))
        self.USE_SN = (self.SN_DATASET or "none").lower() != "none"
        for key in _PATH_KEYS:
            path = getattr(self, key)
            if path and not os.path.isabs(path):
                setattr(self, key, os.path.join(REPO_ROOT, path))


_YAML_MAP = {
    "cosmology": {
        "H0": "H0", "OMEGA_M0": "OMEGA_M0", "R_DRAG": "R_DRAG",
        "Z_MAX": "Z_MAX", "omega_b_h2": "OMEGA_B_H2",
    },
    "wde_representation": {
        "hidden_layers": "WDE_NET_HIDDEN_LAYERS",
        "fourier_freqs": "WDE_NET_FOURIER_FREQS",
    },
    "network": {"hidden_layers": "HIDDEN_LAYERS", "fourier_freqs": "FOURIER_FREQS"},
    "training": {
        "adam_lr": "ADAM_LR", "adam_lr_min": "ADAM_LR_MIN",
        "adam_epochs": "ADAM_EPOCHS", "scheduler": "SCHEDULER",
        "max_grad_norm": "MAX_GRAD_NORM", "weight_decay": "WEIGHT_DECAY",
        "lbfgs_epochs": "LBFGS_EPOCHS", "lbfgs_max_iter": "LBFGS_MAX_ITER",
        "lbfgs_early_stop": "LBFGS_EARLY_STOP",
        "n_collocation": "N_COLLOCATION", "n_bao_grid": "N_BAO_GRID",
        "log_every": "LOG_EVERY", "seed": "SEED",
        "resample_every": "RESAMPLE_EVERY", "resample_start": "RESAMPLE_START",
    },
    "loss_weights": {
        "lambda_pde": "LAMBDA_PDE", "lambda_data": "LAMBDA_DATA",
        "lambda_bao": "LAMBDA_BAO", "lambda_sn": "LAMBDA_SN",
        "lambda_smooth": "LAMBDA_SMOOTH", "lambda_dwde": "LAMBDA_DWDE",
        "lambda_smooth_h": "LAMBDA_SMOOTH_H", "lambda_dh": "LAMBDA_DH",
        "lambda_omega_smooth": "LAMBDA_OMEGA_SMOOTH",
        "lambda_smooth_max": "LAMBDA_SMOOTH_MAX",
        "lambda_smooth_h_max": "LAMBDA_SMOOTH_H_MAX",
        "lambda_omega_smooth_max": "LAMBDA_OMEGA_SMOOTH_MAX",
        "lambda_bounds": "LAMBDA_BOUNDS", "lambda_wde_bound": "LAMBDA_WDE_BOUND",
    },
    "robust": {"soft_cap_c_sn": "SOFT_CAP_C_SN", "soft_cap_c_bao": "SOFT_CAP_C_BAO"},
    "balancing": {
        "mode": "BALANCING_MODE", "gradnorm_every": "GRADNORM_EVERY",
        "gradnorm_alpha": "GRADNORM_ALPHA", "gradnorm_lam_lo": "GRADNORM_LAM_LO",
        "gradnorm_lam_hi": "GRADNORM_LAM_HI",
        "gradnorm_freeze_after": "GRADNORM_FREEZE_AFTER",
    },
    "cosmic_chronometers": {
        "use_cc": "USE_CC", "data_path": "CC_DATA_PATH", "use_cov": "CC_USE_COV",
        "sys_path": "CC_SYS_PATH",
    },
    "bao": {"data_path": "BAO_DATA_PATH", "cov_path": "BAO_COV_PATH"},
    "supernovae": {"dataset": "SN_DATASET"},
    "output": {"save_dir": "SAVE_DIR"},
    "bayesian": {
        "use_hmc": "USE_HMC", "scope": "HMC_SCOPE",
        "num_warmup": "HMC_NUM_WARMUP", "num_samples": "HMC_NUM_SAMPLES",
        "num_chains": "HMC_NUM_CHAINS", "vectorise_chains": "HMC_VECTORISE_CHAINS",
        "log_every": "HMC_LOG_EVERY", "target_accept": "HMC_TARGET_ACCEPT",
        "max_tree_depth": "HMC_MAX_TREE_DEPTH", "dense_mass": "HMC_DENSE_MASS",
        "init_step_size": "HMC_INIT_STEP_SIZE", "jitter": "HMC_JITTER",
        "seed": "HMC_SEED", "weight_prior_sigma": "HMC_WEIGHT_PRIOR_SIGMA",
        "use_cc": "HMC_USE_CC", "use_bao": "HMC_USE_BAO", "use_sn": "HMC_USE_SN",
        "use_pde": "HMC_USE_PDE", "pde_sigma": "HMC_PDE_SIGMA",
        "pde_n_colloc": "HMC_PDE_N_COLLOC", "wnet_prior_sigma": "HMC_WNET_PRIOR_SIGMA",
        "band_n_z": "HMC_BAND_N_Z",
        "thin": "HMC_THIN", "save_chain": "HMC_SAVE_CHAIN",
    },
}

_PRIOR_MAP = {
    "omega_m0": ("OMEGA_M0_TRAINABLE", "OMEGA_M0_PRIOR_MU",
                 "OMEGA_M0_PRIOR_SIGMA", "LAMBDA_OMEGA_M0_PRIOR"),
    "H0": ("H0_TRAINABLE", "H0_PRIOR_MU", "H0_PRIOR_SIGMA", "LAMBDA_H0_PRIOR"),
    "r_drag": ("R_DRAG_TRAINABLE", "R_DRAG_PRIOR_MU",
               "R_DRAG_PRIOR_SIGMA", "LAMBDA_R_DRAG_PRIOR"),
}


def load_cfg(yaml_path: str) -> Cfg:
    """Read a YAML file and return the resolved configuration."""
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = Cfg()
    for section, mapping in _YAML_MAP.items():
        values = raw.get(section) or {}
        for key, attr in mapping.items():
            if values.get(key) is not None:
                setattr(cfg, attr, values[key])

    priors = raw.get("priors") or {}
    for name, attrs in _PRIOR_MAP.items():
        values = priors.get(name) or {}
        for key, attr in zip(("trainable", "prior_mu", "prior_sigma", "lambda_prior"), attrs):
            if values.get(key) is not None:
                setattr(cfg, attr, values[key])

    cc = raw.get("cosmic_chronometers") or {}
    if cc.get("cov_components") is not None:
        cfg.CC_COV_COMPONENTS = tuple(cc["cov_components"])

    bay = raw.get("bayesian") or {}
    tr = raw.get("training") or {}
    if tr.get("adam_dtype") is not None:
        cfg.ADAM_DTYPE = _DTYPES[str(tr["adam_dtype"]).lower()]
    if tr.get("dtype") is not None:
        cfg.DTYPE = _DTYPES[str(tr["dtype"]).lower()]
    if bay.get("dtype") is not None:
        cfg.HMC_DTYPE = _DTYPES[str(bay["dtype"]).lower()]

    cfg.update_derived()
    return cfg


def wde_label(cfg: Cfg) -> str:
    return f"hidden layers {list(cfg.WDE_NET_HIDDEN_LAYERS)}"


def print_summary(cfg: Cfg) -> None:
    sep = "=" * 70
    print(sep)
    print("Cosmo-PINN: direct w_DE(z) reconstruction")
    print(sep)
    print(f"  Cosmology    : H0={cfg.H0}  Om0={cfg.OMEGA_M0}  r_drag={cfg.R_DRAG}  "
          f"Z_MAX={cfg.Z_MAX}")
    print(f"  Priors       : Om0 N({cfg.OMEGA_M0_PRIOR_MU}, {cfg.OMEGA_M0_PRIOR_SIGMA})  "
          f"H0 N({cfg.H0_PRIOR_MU}, {cfg.H0_PRIOR_SIGMA})  "
          f"r_drag N({cfg.R_DRAG_PRIOR_MU}, {cfg.R_DRAG_PRIOR_SIGMA})")
    print(f"  Network      : layers={cfg.HIDDEN_LAYERS}  fourier={cfg.FOURIER_FREQS}")
    print(f"  w_DE network : {wde_label(cfg)}"
          + ("  (quintessence bound)" if cfg.LAMBDA_WDE_BOUND > 0 else ""))
    print(f"  Training     : Adam {cfg.ADAM_EPOCHS} ep @ lr={cfg.ADAM_LR}  "
          f"L-BFGS {cfg.LBFGS_EPOCHS}x{cfg.LBFGS_MAX_ITER}  N_coll={cfg.N_COLLOCATION}")
    print(f"  Data         : CC={cfg.USE_CC} (cov={cfg.CC_USE_COV} "
          f"{tuple(cfg.CC_COV_COMPONENTS)})  BAO={os.path.basename(cfg.BAO_DATA_PATH)}  SN={cfg.SN_DATASET!r}")
    print(f"  HMC          : use={cfg.USE_HMC}  scope={cfg.HMC_SCOPE}  "
          f"warmup/samples={cfg.HMC_NUM_WARMUP}/{cfg.HMC_NUM_SAMPLES}  "
          f"chains={cfg.HMC_NUM_CHAINS}")
    print(f"  Precision    : Adam={cfg.ADAM_DTYPE}  final={cfg.DTYPE}  HMC={cfg.HMC_DTYPE}")
    print(f"  Output dir   : {cfg.SAVE_DIR}")
    print(sep)
