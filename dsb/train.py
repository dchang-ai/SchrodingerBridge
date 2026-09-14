"""Driver for Algorithm 1 (with the cached trajectories of Algorithm 3).

The driver is dataset agnostic: it needs a network constructor, a sampler for
p_data and a sampler for p_prior, and calls back an ``eval_fn`` after every
backward half-step (i.e. whenever a new generative model B^n is available).
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import ipf
from .schedule import OUReference, constant_gammas, symmetric_gammas


@dataclasses.dataclass
class DSBConfig:
    # chain / reference dynamics
    num_steps: int = 20                 # N
    gamma_schedule: str = "symmetric"   # "symmetric" (App. J.2) or "constant"
    gamma_min: float = 1e-5
    gamma_max: float = 1e-1
    gamma: float = 1e-2                 # used by the constant schedule
    prior_mean: float = 0.0
    prior_std: float = 1.0
    ref_alpha: float | None = None      # drift -alpha(x - prior_mean); None -> 1/prior_var (OU), 0 -> Brownian
    # IPF
    num_dsb_iters: int = 10             # L
    train_steps: int = 5000             # gradient steps per half-step
    first_train_steps: int | None = None  # for the very first (SGM-like) backward half-step
    batch_size: int = 128
    cache_size: int = 512               # M trajectories per cache refresh
    cache_refresh: int = 200            # gradient steps between refreshes
    # optimisation
    lr: float = 1e-4
    warmup: int = 0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    reset_optimizer: bool = True
    loss_weighting: str = "none"        # "none" (paper) or "gamma" (weight 1/(2 gamma_k))
    use_dropout: bool = False
    # misc
    seed: int = 0
    log_every: int = 100
    eval_samples: int = 2000

    def reference(self) -> OUReference:
        if self.gamma_schedule == "symmetric":
            g = symmetric_gammas(self.num_steps, self.gamma_min, self.gamma_max)
        elif self.gamma_schedule == "constant":
            g = constant_gammas(self.num_steps, self.gamma)
        else:
            raise ValueError(self.gamma_schedule)
        alpha = (1.0 / self.prior_std**2) if self.ref_alpha is None else self.ref_alpha
        return OUReference(g, alpha=alpha, mean=self.prior_mean)


class Nets(eqx.Module):
    """Container for the four parameter trees that make up a checkpoint."""

    f_params: eqx.Module
    f_ema: eqx.Module
    b_params: eqx.Module
    b_ema: eqx.Module


def step_weights(cfg: DSBConfig, ref: OUReference):
    """Per-step loss weights indexed by the *input* step of each network."""
    g = np.asarray(ref.gammas, dtype=np.float64)
    if cfg.loss_weighting == "none":
        return None, None
    if cfg.loss_weighting == "gamma":
        wf = 1.0 / (2.0 * g)                       # f_net input k in 0..N-1 -> transition gammas[k]
        wb = np.concatenate([[0.0], 1.0 / (2.0 * g)])  # b_net input k in 1..N -> transition gammas[k-1]
        wf = wf / wf.mean()
        wb = wb / wb[1:].mean()
        return jnp.asarray(wf, jnp.float32), jnp.asarray(wb, jnp.float32)
    raise ValueError(cfg.loss_weighting)


def run_dsb(
    cfg: DSBConfig,
    *,
    make_net: Callable[[jax.Array], eqx.Module],
    sample_data: Callable[[jax.Array, int], jax.Array],
    sample_prior: Callable[[jax.Array, int], jax.Array],
    outdir: str,
    eval_fn: Callable | None = None,
    logger=None,
):
    os.makedirs(outdir, exist_ok=True)
    ckdir = os.path.join(outdir, "ckpt")
    os.makedirs(ckdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w") as fh:
        json.dump(dataclasses.asdict(cfg), fh, indent=2)
    log = logger.log if logger is not None else (lambda **kw: print(kw, flush=True))

    ref = cfg.reference()
    gammas = jnp.asarray(ref.gammas)
    n_steps = ref.num_steps
    key = jax.random.PRNGKey(cfg.seed)
    key, kf, kb = jax.random.split(key, 3)
    f_net, b_net = make_net(kf), make_net(kb)
    _, static = eqx.partition(f_net, eqx.is_array)
    optimizer = ipf.make_optimizer(cfg.lr, cfg.grad_clip, cfg.warmup)
    f_state = ipf.init_train_state(f_net, optimizer)
    b_state = ipf.init_train_state(b_net, optimizer)
    train_step = ipf.make_train_step(optimizer, static, cfg.ema_decay, cfg.use_dropout)
    wf, wb = step_weights(cfg, ref)

    # ---- resume -----------------------------------------------------------------
    start_n, start_phase = 0, "backward"
    latest = os.path.join(ckdir, "latest.json")
    if os.path.exists(latest):
        with open(latest) as fh:
            meta = json.load(fh)
        nets = eqx.tree_deserialise_leaves(
            os.path.join(ckdir, meta["file"]),
            Nets(f_state.params, f_state.ema_params, b_state.params, b_state.ema_params),
        )
        f_state = f_state._replace(params=nets.f_params, ema_params=nets.f_ema)
        b_state = b_state._replace(params=nets.b_params, ema_params=nets.b_ema)
        if meta["phase"] == "backward":
            start_n, start_phase = meta["n"], "forward"
        else:
            start_n, start_phase = meta["n"] + 1, "backward"
        key = jax.random.PRNGKey(cfg.seed + 1000 * (2 * start_n + (start_phase == "forward")))
        log(event="resume", n=start_n, phase=start_phase, file=meta["file"])

    def save(n, phase):
        fname = f"iter{n:03d}_{phase}.eqx"
        eqx.tree_serialise_leaves(
            os.path.join(ckdir, fname),
            Nets(f_state.params, f_state.ema_params, b_state.params, b_state.ema_params),
        )
        with open(latest, "w") as fh:
            json.dump({"n": n, "phase": phase, "file": fname}, fh)

    # ---- jitted pieces ------------------------------------------------------------
    @eqx.filter_jit
    def reference_batch(key):
        k1, k2 = jax.random.split(key)
        x0 = sample_data(k1, cfg.batch_size)
        return ipf.reference_pairs(ref, k2, x0)

    @eqx.filter_jit
    def forward_cache(f_ema, key):
        k1, k2 = jax.random.split(key)
        x0 = sample_data(k1, cfg.cache_size)
        return ipf.simulate(eqx.combine(f_ema, static), x0, gammas, k2, forward=True)

    @eqx.filter_jit
    def backward_cache(b_ema, key):
        k1, k2 = jax.random.split(key)
        xN = sample_prior(k1, cfg.cache_size)
        return ipf.simulate(eqx.combine(b_ema, static), xN, gammas, k2, forward=False)

    sample_cache = eqx.filter_jit(ipf.sample_cache)

    def weights_for(w, ks):
        return None if w is None else w[ks]

    # ---- IPF loop ------------------------------------------------------------------
    for n in range(start_n, cfg.num_dsb_iters):
        # ---------------- backward half-step: learn B^n from forward trajectories of F^n
        if not (n == start_n and start_phase == "forward"):
            steps = cfg.first_train_steps if (n == 0 and cfg.first_train_steps) else cfg.train_steps
            if cfg.reset_optimizer:
                b_state = b_state._replace(opt_state=optimizer.init(b_state.params), step=jnp.zeros((), jnp.int32))
            cache, t0, losses = None, time.time(), []
            for it in range(steps):
                key, k1, k2, k3 = jax.random.split(key, 4)
                if n == 0:
                    ks, xs, tg = reference_batch(k1)  # Technique 2: closed-form pairs
                else:
                    if it % cfg.cache_refresh == 0:
                        cache = forward_cache(f_state.ema_params, k1)
                    ks, xs, tg = sample_cache(cache, k2, cfg.batch_size)
                b_state, loss = train_step(b_state, ks, xs, tg, weights_for(wb, ks), k3)
                losses.append(loss)
                if (it + 1) % cfg.log_every == 0 or it == steps - 1:
                    l = float(jnp.mean(jnp.stack(losses)))
                    log(n=n, phase="backward", step=it + 1, loss=l, it_per_s=len(losses) / (time.time() - t0))
                    losses, t0 = [], time.time()
            save(n, "backward")
            if eval_fn is not None:
                key, ke = jax.random.split(key)
                metrics = eval_fn(n, eqx.combine(b_state.ema_params, static), eqx.combine(f_state.ema_params, static), ke)
                if metrics:
                    log(event="eval", n=n, **metrics)

        # ---------------- forward half-step: learn F^{n+1} from backward trajectories of B^n
        if cfg.reset_optimizer:
            f_state = f_state._replace(opt_state=optimizer.init(f_state.params), step=jnp.zeros((), jnp.int32))
        cache, t0, losses = None, time.time(), []
        for it in range(cfg.train_steps):
            key, k1, k2, k3 = jax.random.split(key, 4)
            if it % cfg.cache_refresh == 0:
                cache = backward_cache(b_state.ema_params, k1)
            ks, xs, tg = sample_cache(cache, k2, cfg.batch_size)
            f_state, loss = train_step(f_state, ks, xs, tg, weights_for(wf, ks), k3)
            losses.append(loss)
            if (it + 1) % cfg.log_every == 0 or it == cfg.train_steps - 1:
                l = float(jnp.mean(jnp.stack(losses)))
                log(n=n, phase="forward", step=it + 1, loss=l, it_per_s=len(losses) / (time.time() - t0))
                losses, t0 = [], time.time()
        save(n, "forward")

    return eqx.combine(f_state.ema_params, static), eqx.combine(b_state.ema_params, static)
