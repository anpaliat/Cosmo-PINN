"""NUTS sampling with BlackJAX window adaptation and convergence diagnostics."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Callable

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass
class NUTSResult:
    """Chains stacked on the leading axis: samples (C, N, D), the rest (C, N) or (C,)."""
    samples: np.ndarray
    log_probs: np.ndarray
    step_size: np.ndarray
    accept_stats: np.ndarray
    tree_depths: np.ndarray
    divergences: np.ndarray


def _warmup_progress(num_warmup: int, n_chains: int, log_every: int) -> dict:
    if "adaptation_info_fn" not in inspect.signature(blackjax.window_adaptation).parameters:
        return {}
    every = max(1, int(log_every))
    t0 = time.time()
    tally = {"calls": 0, "acc": 0.0}

    def tick(acc):
        tally["calls"] += 1
        tally["acc"] += float(acc)
        step, rem = divmod(tally["calls"], n_chains)
        if rem == 0 and (step % every == 0 or step == num_warmup):
            n_block = every if step % every == 0 else step % every
            print(f"  [NUTS warmup {step:5d}/{num_warmup}]  elapsed {time.time() - t0:.0f}s  "
                  f"acc={tally['acc'] / (n_block * n_chains):.3f}", flush=True)
            tally["acc"] = 0.0

    def info_fn(state, info, adaptation_state):
        jax.debug.callback(tick, info.acceptance_rate)
        return info.acceptance_rate

    return {"adaptation_info_fn": info_fn}


def _to_dtype(tree, dtype):
    return jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
        else x, tree)


def _window_adaptation(logdensity_fn, n_chains, num_warmup, target_accept, max_tree_depth,
                       step_size, log_every, dense_mass):
    return blackjax.window_adaptation(
        blackjax.nuts, logdensity_fn, is_mass_matrix_diagonal=not dense_mass,
        target_acceptance_rate=target_accept, initial_step_size=step_size,
        max_num_doublings=max_tree_depth,
        **_warmup_progress(num_warmup, n_chains, log_every))


def _collect(chunks, num_samples, label, t0):
    pos, ld, acc, ns, div = (np.concatenate(x, axis=0) for x in zip(*chunks))
    print(f"  [NUTS] done  mean acc={float(np.mean(acc)):.3f}  "
          f"divergences={int(np.sum(div))}/{div.size}  ({time.time() - t0:.0f}s, {label})")
    return pos, ld, acc, np.ceil(np.log2(ns + 1.0)).astype(np.int32), div


def _run_single(logdensity_fn, theta_init, num_warmup, num_samples, target_accept,
                max_tree_depth, step_size, seed, log_every, dense_mass):
    key_warmup, key_sample = jax.random.split(jax.random.PRNGKey(seed))
    warmup = _window_adaptation(logdensity_fn, 1, num_warmup, target_accept, max_tree_depth,
                                step_size, log_every, dense_mass)
    (state, params), _ = jax.jit(warmup.run, static_argnums=(2,))(key_warmup, theta_init, num_warmup)
    params = _to_dtype(params, theta_init.dtype)
    state = _to_dtype(state, theta_init.dtype)
    kernel = blackjax.nuts(logdensity_fn, **params).step

    @jax.jit
    def scan_chunk(state, keys):
        def body(s, k):
            s, info = kernel(k, s)
            return s, (s.position, s.logdensity, info.acceptance_rate,
                       info.num_integration_steps, info.is_divergent)
        return jax.lax.scan(body, state, keys)

    chunk = min(log_every, num_samples)
    n_chunks = -(-num_samples // chunk)
    keys = jax.random.split(key_sample, num_samples)
    pad = n_chunks * chunk - num_samples
    if pad:
        keys = jnp.concatenate([keys, jnp.zeros((pad, 2), dtype=keys.dtype)])
    keys = keys.reshape(n_chunks, chunk, 2)

    t0, chunks = time.time(), []
    for ci in range(n_chunks):
        state, out = scan_chunk(state, keys[ci])
        real = min(chunk, num_samples - ci * chunk)
        chunks.append(tuple(np.asarray(o[:real]) for o in out))
        print(f"  [NUTS sample {ci * chunk + real:5d}/{num_samples}]  "
              f"acc={float(np.mean(chunks[-1][2])):.3f}  div={int(np.sum(chunks[-1][4]))}", flush=True)
    pos, ld, acc, depth, div = _collect(chunks, num_samples, "1 chain", t0)
    return pos, ld, float(params["step_size"]), acc, depth, div


def _run_vmapped(logdensity_fn, theta_inits, num_warmup, num_samples, target_accept,
                 max_tree_depth, step_size, seed, log_every, dense_mass) -> NUTSResult:
    n_chains = int(theta_inits.shape[0])
    key_warmup, key_sample = jax.random.split(jax.random.PRNGKey(seed))
    warmup = _window_adaptation(logdensity_fn, n_chains, num_warmup, target_accept,
                                max_tree_depth, step_size, log_every, dense_mass)

    @jax.jit
    def run_warmup(keys, positions):
        def one(k, p):
            (state, params), _ = warmup.run(k, p, num_warmup)
            return state, params
        return jax.vmap(one)(keys, positions)

    t0 = time.time()
    states, params = run_warmup(jax.random.split(key_warmup, n_chains), theta_inits)
    jax.block_until_ready(states.position)
    print(f"  [NUTS] warmup ({n_chains} chains) done in {time.time() - t0:.0f}s", flush=True)
    params = _to_dtype(params, theta_inits.dtype)
    states = _to_dtype(states, theta_inits.dtype)
    step_sizes = params["step_size"]
    inv_masses = params["inverse_mass_matrix"]

    def one_step(state, k, ss, im):
        new, info = blackjax.nuts(logdensity_fn, ss, im, max_num_doublings=int(max_tree_depth)).step(k, state)
        return new, (new.position, new.logdensity, info.acceptance_rate,
                     info.num_integration_steps, info.is_divergent)

    @jax.jit
    def scan_chunk(states, keys_tc):
        return jax.lax.scan(lambda c, k: jax.vmap(one_step)(c, k, step_sizes, inv_masses),
                            states, keys_tc)

    chunk = min(log_every, num_samples)
    n_chunks = -(-num_samples // chunk)
    keys = jax.random.split(key_sample, num_samples * n_chains).reshape(num_samples, n_chains, 2)
    pad = n_chunks * chunk - num_samples
    if pad:
        keys = jnp.concatenate([keys, jnp.zeros((pad, n_chains, 2), dtype=keys.dtype)])

    t0, chunks = time.time(), []
    for ci in range(n_chunks):
        states, out = scan_chunk(states, keys[ci * chunk:(ci + 1) * chunk])
        real = min(chunk, num_samples - ci * chunk)
        chunks.append(tuple(np.asarray(o[:real]) for o in out))
        print(f"  [NUTS sample {ci * chunk + real:5d}/{num_samples}]  "
              f"acc={float(np.mean(chunks[-1][2])):.3f}  div={int(np.sum(chunks[-1][4]))}", flush=True)
    pos, ld, acc, depth, div = (a.swapaxes(0, 1) for a in
                                _collect(chunks, num_samples, f"{n_chains} chains", t0))
    return NUTSResult(samples=pos.astype(np.float64), log_probs=ld.astype(np.float64),
                      step_size=np.asarray(step_sizes, dtype=np.float64),
                      accept_stats=acc.astype(np.float64), tree_depths=depth, divergences=div)


def run_nuts_chains(logdensity_fn: Callable[[Array], Array], theta_inits: Array,
                    num_warmup: int, num_samples: int, target_accept: float = 0.8,
                    max_tree_depth: int = 10, step_size: float = 1e-3, seed: int = 0,
                    log_every: int = 100, dense_mass: bool = False,
                    vectorise: bool = True) -> NUTSResult:
    """Run one NUTS chain per row of theta_inits, in parallel (vmap) or sequentially."""
    args = (num_warmup, num_samples, target_accept, max_tree_depth, step_size)
    if vectorise:
        return _run_vmapped(logdensity_fn, theta_inits, *args, seed, log_every, dense_mass)
    runs = [_run_single(logdensity_fn, theta_inits[c], *args, seed + c, log_every, dense_mass)
            for c in range(int(theta_inits.shape[0]))]
    pos, ld, ss, acc, depth, div = zip(*runs)
    return NUTSResult(samples=np.stack(pos).astype(np.float64),
                      log_probs=np.stack(ld).astype(np.float64),
                      step_size=np.asarray(ss, dtype=np.float64),
                      accept_stats=np.stack(acc).astype(np.float64),
                      tree_depths=np.stack(depth), divergences=np.stack(div))


def gelman_rubin(chains: np.ndarray) -> np.ndarray:
    """R-hat for chains of shape (n_chains, n_samples, n_params)."""
    m, n, d = chains.shape
    if m < 2 or n < 2:
        return np.full(d, np.nan)
    B = n * np.var(chains.mean(axis=1), axis=0, ddof=1)
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    var_hat = (1.0 - 1.0 / n) * W + B / n
    return np.sqrt(np.maximum(var_hat, 1e-30) / np.maximum(W, 1e-30))


def _geyer_ess(chain: np.ndarray) -> float:
    x = chain - chain.mean()
    n = len(x)
    if n < 4:
        return float(n)
    f = np.fft.fft(x, n=2 * n)
    acf = np.real(np.fft.ifft(f * np.conj(f)))[:n] / ((np.var(x) + 1e-30) * n)
    tau = 1.0
    for k in range(1, n - 1, 2):
        s = acf[k] + acf[k + 1]
        if s < 0:
            break
        tau += 2.0 * s
    return float(n / max(tau, 1.0))


def ess_per_parameter(chains: np.ndarray) -> np.ndarray:
    """Bulk ESS (ArviZ, rank-normalised); summed per-chain Geyer ESS if ArviZ is missing."""
    n_c, _, d = chains.shape
    try:
        import arviz as az
        return np.array([float(np.asarray(az.ess(chains[:, :, k], method="bulk"))) for k in range(d)])
    except Exception:
        return np.array([sum(_geyer_ess(chains[c, :, k]) for c in range(n_c)) for k in range(d)])
