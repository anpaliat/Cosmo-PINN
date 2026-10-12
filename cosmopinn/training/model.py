"""PINN model for the direct reconstruction of w_DE(z) and the background physics."""

from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.nn import softplus

from cosmopinn.config import as_jnp
from cosmopinn.physics.hubble import background
from cosmopinn.physics.residual import pde_residual_normalised


class PhysConsts(NamedTuple):
    Z_MAX: float
    omega_b_phys: float


def consts_from_cfg(cfg: Any) -> PhysConsts:
    return PhysConsts(Z_MAX=float(cfg.Z_MAX),
                      omega_b_phys=float(cfg.OMEGA_B) * float(cfg.H0) ** 2)


def _xavier_normal(key: Array, out_dim: int, in_dim: int, dtype) -> Array:
    std = math.sqrt(2.0 / (in_dim + out_dim))
    return std * jax.random.normal(key, (out_dim, in_dim), dtype=dtype)


def _make_linear(key: Array, in_dim: int, out_dim: int, dtype) -> eqx.nn.Linear:
    lin = eqx.nn.Linear(in_dim, out_dim, use_bias=True, key=key)
    lin = eqx.tree_at(lambda m: m.weight, lin, _xavier_normal(key, out_dim, in_dim, dtype))
    return eqx.tree_at(lambda m: m.bias, lin, jnp.zeros((out_dim,), dtype=dtype))


def _make_linear_wnet(key: Array, in_dim: int, out_dim: int, dtype) -> eqx.nn.Linear:
    kw, kb = jax.random.split(key)
    lin = eqx.nn.Linear(in_dim, out_dim, use_bias=True, key=kw)
    bound = 1.0 / math.sqrt(in_dim)
    bias = jax.random.uniform(kb, (out_dim,), dtype=dtype, minval=-bound, maxval=bound)
    lin = eqx.tree_at(lambda m: m.weight, lin, _xavier_normal(kw, out_dim, in_dim, dtype))
    return eqx.tree_at(lambda m: m.bias, lin, bias)


def _encode(x: Array, n_freqs: int, centre: bool = False) -> Array:
    lin = 2.0 * x - 1.0 if centre else x
    if n_freqs == 0:
        return jnp.reshape(lin, (1,))
    ang = x * jnp.arange(1, n_freqs + 1, dtype=x.dtype) * 2.0 * math.pi
    return jnp.concatenate([jnp.reshape(lin, (1,)), jnp.sin(ang), jnp.cos(ang)])


class WdePINN(eqx.Module):
    """Networks x = z/Z_MAX -> N_m(x) and x -> w_DE(x), plus the cosmological scalars."""

    layers: List[eqx.nn.Linear]
    w_layers: List[eqx.nn.Linear]
    Omega_m0_raw: Array
    H0_raw: Array
    R_drag_raw: Array

    n_freqs: int = eqx.field(static=True)
    wde_bounded: bool = eqx.field(static=True)
    w_n_freqs: int = eqx.field(static=True)

    def __init__(self, cfg: Any, key: Array):
        keys = jax.random.split(key, 16)
        dt = as_jnp(cfg.DTYPE)

        self.n_freqs = int(cfg.FOURIER_FREQS)
        layers = []
        d = 1 + 2 * self.n_freqs
        for i, h_dim in enumerate(cfg.HIDDEN_LAYERS):
            layers.append(_make_linear(keys[i], d, h_dim, dt))
            d = h_dim
        layers.append(_make_linear(keys[len(cfg.HIDDEN_LAYERS)], d, 1, dt))
        self.layers = layers

        self.wde_bounded = float(cfg.LAMBDA_WDE_BOUND) > 0.0
        b_init = -4.0 if self.wde_bounded else -1.0

        self.w_n_freqs = int(cfg.WDE_NET_FOURIER_FREQS)
        hidden = list(cfg.WDE_NET_HIDDEN_LAYERS)
        wkeys = jax.random.split(jax.random.fold_in(key, 0x77DE), len(hidden) + 1)
        w_layers = []
        d = 1 + 2 * self.w_n_freqs
        for i, h_dim in enumerate(hidden):
            w_layers.append(_make_linear_wnet(wkeys[i], d, h_dim, dt))
            d = h_dim
        head = eqx.nn.Linear(d, 1, use_bias=True, key=wkeys[-1])
        head = eqx.tree_at(lambda m: m.weight, head, jnp.zeros((1, d), dtype=dt))
        head = eqx.tree_at(lambda m: m.bias, head, jnp.full((1,), b_init, dtype=dt))
        w_layers.append(head)
        self.w_layers = w_layers

        self.Omega_m0_raw = jnp.asarray(cfg.LOGIT_OMEGA_M0, dtype=dt)
        self.H0_raw = jnp.asarray(math.log(math.expm1(float(cfg.H0))), dtype=dt)
        self.R_drag_raw = jnp.asarray(math.log(math.expm1(float(cfg.R_DRAG_PRIOR_MU))), dtype=dt)

    def Omega_m0(self) -> Array:
        return jax.nn.sigmoid(self.Omega_m0_raw)

    def H0(self) -> Array:
        return softplus(self.H0_raw)

    def R_drag(self) -> Array:
        return softplus(self.R_drag_raw)

    def w_features(self, x: Array) -> Array:
        """Learned features phi_j(x) of the w_DE network (last hidden layer)."""
        h = _encode(x, self.w_n_freqs, centre=True)
        for lin in self.w_layers[:-1]:
            h = jnp.tanh(lin(h))
        return h

    def forward_scalar(self, x: Array) -> Tuple[Array, Array]:
        h = _encode(x, self.n_freqs)
        for lin in self.layers[:-1]:
            h = jnp.tanh(lin(h))
        N_m = self.layers[-1](h)[0]
        s = self.w_layers[-1](self.w_features(x))[0]
        w_DE = -1.0 + 2.0 * jax.nn.sigmoid(s) if self.wde_bounded else s
        return N_m, w_DE


def fields_single(model: WdePINN, consts: PhysConsts, z: Array) -> Dict[str, Array]:
    """All background fields at a scalar redshift z (defined in hubble.py)."""
    x = z / consts.Z_MAX
    N_m, w_DE = model.forward_scalar(x)
    H0 = model.H0()
    return background(z, x, N_m, w_DE, model.Omega_m0(), consts.omega_b_phys / H0 ** 2, H0)


def compute_physical_quantities(model: WdePINN, z: Array, cfg: Any) -> Dict[str, Array]:
    """Batched fields: (N, 1) redshifts -> dict of (N, 1) arrays."""
    consts = consts_from_cfg(cfg)
    fields = jax.vmap(lambda zz: fields_single(model, consts, zz))(jnp.reshape(z, (-1,)))
    return {k: jnp.reshape(v, (-1, 1)) for k, v in fields.items()}


def fields_and_derivatives(model: WdePINN, consts: PhysConsts, z: Array):
    """Fields and their d/dz at the redshifts z (flattened): two dicts of (N,) arrays."""
    return jax.vmap(lambda zz: jax.jvp(lambda z2: fields_single(model, consts, z2),
                                       (zz,), (jnp.ones_like(zz),)))(jnp.reshape(z, (-1,)))


def normalised_residual_at(model: WdePINN, consts: PhysConsts, z: Array) -> Array:
    """pde_residual_normalised (residual.py) at the redshifts z, shape (N, 1)."""
    f, df = fields_and_derivatives(model, consts, z)
    return jnp.reshape(pde_residual_normalised(jnp.reshape(z, (-1,)), f, df), (-1, 1))


def deceleration_autodiff(model: WdePINN, consts: PhysConsts, z: Array) -> Array:
    """q(z) = (1+z) H'/H - 1 by automatic differentiation of H(z)."""
    def q(zz):
        H, dH = jax.value_and_grad(lambda z2: fields_single(model, consts, z2)["H"])(zz)
        return (1.0 + zz) / H * dH - 1.0
    return jnp.reshape(jax.vmap(q)(jnp.reshape(z, (-1,))), (-1, 1))


class WdePhysics:
    """Physics interface used by the loss, trainer and HMC modules."""

    name = "wde_direct"

    def build_model(self, cfg: Any, key: Array) -> WdePINN:
        return WdePINN(cfg, key)

    def compute_physical_quantities(self, model, z: Array, cfg: Any) -> Dict[str, Array]:
        return compute_physical_quantities(model, z, cfg)

    def prior_loss(self, model, cfg: Any) -> Array:
        return (cfg.LAMBDA_OMEGA_M0_PRIOR * gaussian_penalty(model.Omega_m0(), cfg.OMEGA_M0_PRIOR_MU, cfg.OMEGA_M0_PRIOR_SIGMA)
                + cfg.LAMBDA_H0_PRIOR * gaussian_penalty(model.H0(), cfg.H0_PRIOR_MU, cfg.H0_PRIOR_SIGMA)
                + cfg.LAMBDA_R_DRAG_PRIOR * gaussian_penalty(model.R_drag(), cfg.R_DRAG_PRIOR_MU, cfg.R_DRAG_PRIOR_SIGMA))


def gaussian_penalty(value: Array, mu: float, sigma: float) -> Array:
    return 0.5 * ((value - mu) / sigma) ** 2
