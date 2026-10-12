"""Selection of the sampled parameters and flat-vector <-> model conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

SCALAR_FIELDS = ("Omega_m0_raw", "H0_raw", "R_drag_raw")
SCALAR_NAMES = ("Omega_m0", "H0", "R_drag")
_SCALAR_FLAGS = ("OMEGA_M0_TRAINABLE", "H0_TRAINABLE", "R_DRAG_TRAINABLE")


def sampled_scalar_names(cfg: Any) -> list:
    return [n for n, flag in zip(SCALAR_NAMES, _SCALAR_FLAGS) if bool(getattr(cfg, flag))]


def _whead(m):
    """Output layer of the w_DE network: w_DE = b + sum_j a_j phi_j(x)."""
    return (m.w_layers[-1].weight, m.w_layers[-1].bias)


def select_sampled_spec(model, cfg: Any, scope: str):
    """Boolean filter marking the sampled leaves for scope scalars | wnet | wde | full."""
    scope = scope.lower()
    if scope == "full":
        spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
        for name, flag in zip(SCALAR_FIELDS, _SCALAR_FLAGS):
            if not bool(getattr(cfg, flag)):
                spec = eqx.tree_at(lambda m, n=name: getattr(m, n), spec, False)
        return spec
    if scope not in ("scalars", "wnet", "wde"):
        raise ValueError(f"Unknown HMC scope {scope!r}; use scalars, wnet, wde or full")

    spec = jax.tree_util.tree_map(lambda _: False, model)
    for name, flag in zip(SCALAR_FIELDS, _SCALAR_FLAGS):
        if bool(getattr(cfg, flag)):
            spec = eqx.tree_at(lambda m, n=name: getattr(m, n), spec, True)
    if scope in ("wnet", "wde"):
        spec = eqx.tree_at(_whead, spec, (True, True))
    if scope == "wde":
        spec = eqx.tree_at(lambda m: m.layers[-1].weight, spec, True)
        spec = eqx.tree_at(lambda m: m.layers[-1].bias, spec, True)
    return spec


@dataclass
class Packer:
    """theta <-> model conversion and per-position information on the w_DE network head."""
    reconstruct: Callable[[Array], Any]
    theta_map: Array
    scalar_mask: Array
    w_mask: Array
    w_order: Array
    w_lcdm: Array
    w_sigma: Array
    dim: int
    bounded: bool = False
    lcdm_prior: bool = True
    prior_note: str = ""

    def n_sampled_scalars(self) -> int:
        return int(jnp.sum(self.scalar_mask))

    def n_sampled_w(self) -> int:
        return int(jnp.sum(self.w_mask))

    @staticmethod
    def coef_label(order: int) -> str:
        return "b" if int(order) == 0 else f"a{int(order)}"


def make_packer(model, cfg: Any, scope: str) -> Packer:
    spec = select_sampled_spec(model, cfg, scope)
    sampled, frozen = eqx.partition(model, spec)
    theta_map, unravel = ravel_pytree(sampled)

    def reconstruct(theta: Array):
        return eqx.combine(unravel(theta), frozen)

    scalar_full = jax.tree_util.tree_map(jnp.zeros_like, model)
    for name in SCALAR_FIELDS:
        scalar_full = eqx.tree_at(lambda m, n=name: getattr(m, n), scalar_full,
                                  jnp.ones_like(getattr(model, name)))
    scalar_mask = ravel_pytree(eqx.partition(scalar_full, spec)[0])[0]

    def flat_at_head(values):
        full = eqx.tree_at(_whead, jax.tree_util.tree_map(jnp.zeros_like, model), tuple(values))
        return ravel_pytree(eqx.partition(full, spec)[0])[0]

    W, b = _whead(model)
    bounded = bool(model.wde_bounded)
    s = float(cfg.HMC_WNET_PRIOR_SIGMA)
    lcdm_prior = s > 0.0 and not bounded
    if lcdm_prior:
        prior_note = f"LCDM-centred N((a, b) = (0, -1), sigma^2), sigma = {s:.4g}"
    elif bounded:
        prior_note = "MAP-centred ridge (bounded mode: LCDM at s -> -inf)"
    else:
        prior_note = "MAP-centred ridge (wnet_prior_sigma <= 0)"

    w_mask = flat_at_head((jnp.ones_like(W), jnp.ones_like(b)))
    return Packer(
        reconstruct=reconstruct, theta_map=theta_map, scalar_mask=scalar_mask, w_mask=w_mask,
        w_order=flat_at_head((jnp.arange(1, W.shape[1] + 1, dtype=W.dtype).reshape(1, -1),
                              jnp.zeros_like(b))),
        w_lcdm=flat_at_head((jnp.zeros_like(W), jnp.full_like(b, -1.0))),
        w_sigma=flat_at_head((jnp.full_like(W, s), jnp.full_like(b, s))) + (1.0 - w_mask),
        dim=int(theta_map.shape[0]), bounded=bounded, lcdm_prior=lcdm_prior,
        prior_note=prior_note)
