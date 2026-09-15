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
from .networks import ReferenceResidual, ScaledIncrement
from .schedule import OUReference, constant_gammas, geometric_gammas, symmetric_gammas


@dataclasses.dataclass
class DSBConfig:
    # chain / reference dynamics
    num_steps: int = 20                 # N
    gamma_schedule: str = "symmetric"   # "symmetric" (power ramp, App. J.2), "geometric" or "constant"
    gamma_min: float = 1e-5
    gamma_max: float = 1e-1
    gamma_power: float = 1.0            # ramp exponent of the symmetric schedule (1 = paper)
    gamma_legacy_ramp: bool = False     # pre-2026-09-15 ramp (gamma_max not reached); only for the first four CIFAR runs
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
    scaled_output: bool = False         # increment = sqrt(2 gamma_k) * net(k, x) (epsilon-style parametrisation)
    ref_residual: bool = False          # forward net = reference drift + network (untrained net == F^0)
    use_dropout: bool = False
    # misc
    seed: int = 0
    log_every: int = 100
    eval_samples: int = 2000

    def reference(self) -> OUReference:
        if self.gamma_schedule == "symmetric":
            g = symmetric_gammas(self.num_steps, self.gamma_min, self.gamma_max, self.gamma_power, self.gamma_legacy_ramp)
        elif self.gamma_schedule == "geometric":
            g = geometric_gammas(self.num_steps, self.gamma_min, self.gamma_max)
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


def output_scales(ref: OUReference):
    """sqrt(2 gamma) of the transition attached to each input index: f_net input k in 0..N-1
    uses gammas[k]; b_net input k in 1..N uses gammas[k-1] (index 0 unused)."""
    g = np.sqrt(2.0 * np.asarray(ref.gammas, dtype=np.float64))
    return tuple(g), tuple(np.concatenate([[g[0]], g]))


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
    sample_data: Callable[[jax.Array, int, jax.Array], jax.Array],
    sample_prior: Callable[[jax.Array, int], jax.Array],
    data: jax.Array,
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
    if cfg.scaled_output:
        sf, sb = output_scales(ref)
        f_net, b_net = ScaledIncrement(f_net, sf), ScaledIncrement(b_net, sb)
    if cfg.ref_residual:
        f_net = ReferenceResidual(f_net, ref.gammas, ref.alpha, ref.mean)
    _, f_static = eqx.partition(f_net, eqx.is_array)
    _, b_static = eqx.partition(b_net, eqx.is_array)
    optimizer = ipf.make_optimizer(cfg.lr, cfg.grad_clip, cfg.warmup)
    f_state = ipf.init_train_state(f_net, optimizer)
    b_state = ipf.init_train_state(b_net, optimizer)
    train_step_f = ipf.make_train_step(optimizer, f_static, cfg.ema_decay, cfg.use_dropout)
    train_step_b = ipf.make_train_step(optimizer, b_static, cfg.ema_decay, cfg.use_dropout)
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
        f_state = f_state._replace(params=nets.f_params, ema_params=nets.f_ema, step=jnp.asarray(meta.get("f_step", 0), jnp.int32))
        b_state = b_state._replace(params=nets.b_params, ema_params=nets.b_ema, step=jnp.asarray(meta.get("b_step", 0), jnp.int32))
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
            json.dump({"n": n, "phase": phase, "file": fname, "f_step": int(f_state.step), "b_step": int(b_state.step)}, fh)

    # ---- jitted pieces ------------------------------------------------------------
    # the dataset is passed as an argument (not closed over) so that it is not baked into
    # the compiled executables as a constant
    @eqx.filter_jit
    def reference_batch(key, data):
        k1, k2 = jax.random.split(key)
        x0 = sample_data(k1, cfg.batch_size, data)
        return ipf.reference_pairs(ref, k2, x0)

    @eqx.filter_jit
    def forward_cache(f_ema, key, data):
        k1, k2 = jax.random.split(key)
        x0 = sample_data(k1, cfg.cache_size, data)
        return ipf.simulate(eqx.combine(f_ema, f_static), x0, gammas, k2, forward=True)

    @eqx.filter_jit
    def backward_cache(b_ema, key):
        k1, k2 = jax.random.split(key)
        xN = sample_prior(k1, cfg.cache_size)
        return ipf.simulate(eqx.combine(b_ema, b_static), xN, gammas, k2, forward=False)

    sample_cache = eqx.filter_jit(ipf.sample_cache)

    def weights_for(w, ks):
        return None if w is None else w[ks]

    # ---- IPF loop ------------------------------------------------------------------
    for n in range(start_n, cfg.num_dsb_iters):
        # ---------------- backward half-step: learn B^n from forward trajectories of F^n
        if not (n == start_n and start_phase == "forward"):
            steps = cfg.first_train_steps if (n == 0 and cfg.first_train_steps) else cfg.train_steps
            b_state = b_state._replace(params=b_state.ema_params)  # warm start from the teacher
            if cfg.reset_optimizer:
                b_state = b_state._replace(opt_state=optimizer.init(b_state.params))
            cache, t0, losses = None, time.time(), []
            for it in range(steps):
                key, k1, k2, k3 = jax.random.split(key, 4)
                if n == 0:
                    ks, xs, tg = reference_batch(k1, data)  # Technique 2: closed-form pairs
                else:
                    if it % cfg.cache_refresh == 0:
                        cache = forward_cache(f_state.ema_params, k1, data)
                    ks, xs, tg = sample_cache(cache, k2, cfg.batch_size)
                b_state, loss = train_step_b(b_state, ks, xs, tg, weights_for(wb, ks), k3)
                losses.append(loss)
                if (it + 1) % cfg.log_every == 0 or it == steps - 1:
                    l = float(jnp.mean(jnp.stack(losses)))
                    log(n=n, phase="backward", step=it + 1, loss=l, it_per_s=len(losses) / (time.time() - t0))
                    losses, t0 = [], time.time()
            save(n, "backward")
            if eval_fn is not None:
                key, ke = jax.random.split(key)
                metrics = eval_fn(n, eqx.combine(b_state.ema_params, b_static), eqx.combine(f_state.ema_params, f_static), ke)
                if metrics:
                    log(event="eval", n=n, **metrics)

        # ---------------- forward half-step: learn F^{n+1} from backward trajectories of B^n
        f_state = f_state._replace(params=f_state.ema_params)  # warm start from the teacher
        if cfg.reset_optimizer:
            f_state = f_state._replace(opt_state=optimizer.init(f_state.params))
        cache, t0, losses = None, time.time(), []
        for it in range(cfg.train_steps):
            key, k1, k2, k3 = jax.random.split(key, 4)
            if it % cfg.cache_refresh == 0:
                cache = backward_cache(b_state.ema_params, k1)
            ks, xs, tg = sample_cache(cache, k2, cfg.batch_size)
            f_state, loss = train_step_f(f_state, ks, xs, tg, weights_for(wf, ks), k3)
            losses.append(loss)
            if (it + 1) % cfg.log_every == 0 or it == cfg.train_steps - 1:
                l = float(jnp.mean(jnp.stack(losses)))
                log(n=n, phase="forward", step=it + 1, loss=l, it_per_s=len(losses) / (time.time() - t0))
                losses, t0 = [], time.time()
        save(n, "forward")

    return eqx.combine(f_state.ema_params, f_static), eqx.combine(b_state.ema_params, b_static)


# ----------------------------------------------------------------------------- multi-GPU
def run_dsb_parallel(
    cfg: DSBConfig,
    *,
    make_net: Callable[[jax.Array], eqx.Module],
    sample_data: Callable[[jax.Array, int, jax.Array], jax.Array],
    sample_prior: Callable[[jax.Array, int], jax.Array],
    data: jax.Array,
    outdir: str,
    eval_fn: Callable | None = None,
    logger=None,
    devices=None,
    compute_dtype=None,
):
    """Data-parallel version of :func:`run_dsb` (see :mod:`dsb.parallel`).

    Same algorithm, same checkpoints (single-copy parameters), but the batch of every
    gradient step and the trajectories of every cache refresh are split over the devices of
    one node; ``cfg.batch_size`` and ``cfg.cache_size`` are *global* sizes.  Even on one
    device this path is faster than :func:`run_dsb` because cache sampling and the gradient
    step are fused into one jitted call.
    """
    from . import parallel as par

    os.makedirs(outdir, exist_ok=True)
    ckdir = os.path.join(outdir, "ckpt")
    os.makedirs(ckdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w") as fh:
        json.dump(dataclasses.asdict(cfg), fh, indent=2)
    log = logger.log if logger is not None else (lambda **kw: print(kw, flush=True))

    dp = par.DataParallel(devices)
    ref = cfg.reference()
    gammas = jnp.asarray(ref.gammas)
    key = jax.random.PRNGKey(cfg.seed)
    key, kf, kb = jax.random.split(key, 3)
    f_net, b_net = make_net(kf), make_net(kb)
    if cfg.scaled_output:
        sf, sb = output_scales(ref)
        f_net, b_net = ScaledIncrement(f_net, sf), ScaledIncrement(b_net, sb)
    if cfg.ref_residual:
        f_net = ReferenceResidual(f_net, ref.gammas, ref.alpha, ref.mean)
    _, f_static = eqx.partition(f_net, eqx.is_array)
    _, b_static = eqx.partition(b_net, eqx.is_array)
    optimizer = ipf.make_optimizer(cfg.lr, cfg.grad_clip, cfg.warmup)
    f_state = par.init_train_state(dp, f_net, optimizer)
    b_state = par.init_train_state(dp, b_net, optimizer)
    wf, wb = step_weights(cfg, ref)
    log(event="devices", ndev=dp.ndev, per_device_batch=dp.local(cfg.batch_size), per_device_cache=dp.local(cfg.cache_size))

    # ---- resume -----------------------------------------------------------------
    start_n, start_phase = 0, "backward"
    latest = os.path.join(ckdir, "latest.json")
    if os.path.exists(latest):
        with open(latest) as fh:
            meta = json.load(fh)
        like = Nets(dp.unreplicate(f_state.params), dp.unreplicate(f_state.ema_params),
                    dp.unreplicate(b_state.params), dp.unreplicate(b_state.ema_params))
        nets = eqx.tree_deserialise_leaves(os.path.join(ckdir, meta["file"]), like)
        f_state = f_state._replace(params=dp.replicate(nets.f_params), ema_params=dp.replicate(nets.f_ema),
                                   step=dp.replicate(jnp.asarray(meta.get("f_step", 0), jnp.int32)))
        b_state = b_state._replace(params=dp.replicate(nets.b_params), ema_params=dp.replicate(nets.b_ema),
                                   step=dp.replicate(jnp.asarray(meta.get("b_step", 0), jnp.int32)))
        if meta["phase"] == "backward":
            start_n, start_phase = meta["n"], "forward"
        else:
            start_n, start_phase = meta["n"] + 1, "backward"
        key = jax.random.PRNGKey(cfg.seed + 1000 * (2 * start_n + (start_phase == "forward")))
        log(event="resume", n=start_n, phase=start_phase, file=meta["file"])

    def save(n, phase):
        fname = f"iter{n:03d}_{phase}.eqx"
        nets = Nets(dp.unreplicate(f_state.params), dp.unreplicate(f_state.ema_params),
                    dp.unreplicate(b_state.params), dp.unreplicate(b_state.ema_params))
        eqx.tree_serialise_leaves(os.path.join(ckdir, fname), nets)
        with open(latest, "w") as fh:
            json.dump({"n": n, "phase": phase, "file": fname, "f_step": int(f_state.step), "b_step": int(b_state.step)}, fh)

    # ---- jitted, sharded pieces -------------------------------------------------------
    reference_batch = par.make_reference_batch(dp, ref, sample_data, cfg.batch_size, sample_args=(data,))
    forward_cache = par.make_cache_fn(dp, f_static, gammas, sample_data, cfg.cache_size, forward=True,
                                      compute_dtype=compute_dtype, sample_args=(data,))
    backward_cache = par.make_cache_fn(dp, b_static, gammas, sample_prior, cfg.cache_size, forward=False,
                                       compute_dtype=compute_dtype)
    step_b_ref = par.make_train_step(dp, optimizer, b_static, cfg.ema_decay, cfg.use_dropout, weight_table=wb,
                                     compute_dtype=compute_dtype)
    step_b = par.make_train_step(dp, optimizer, b_static, cfg.ema_decay, cfg.use_dropout, weight_table=wb,
                                 compute_dtype=compute_dtype, fused_cache_sampling=cfg.batch_size)
    step_f = par.make_train_step(dp, optimizer, f_static, cfg.ema_decay, cfg.use_dropout, weight_table=wf,
                                 compute_dtype=compute_dtype, fused_cache_sampling=cfg.batch_size)

    def warm_start(state):
        state = state._replace(params=state.ema_params)  # student starts from the teacher
        if cfg.reset_optimizer:
            state = state._replace(opt_state=dp.replicate(optimizer.init(state.params)))
        return state

    def combined(state, static):
        return eqx.combine(dp.unreplicate(state.ema_params), static)

    # ---- IPF loop ------------------------------------------------------------------
    for n in range(start_n, cfg.num_dsb_iters):
        if not (n == start_n and start_phase == "forward"):
            steps = cfg.first_train_steps if (n == 0 and cfg.first_train_steps) else cfg.train_steps
            b_state = warm_start(b_state)
            cache, t0, losses = None, time.time(), []
            for it in range(steps):
                key, k1, k2, k3 = jax.random.split(key, 4)
                if n == 0:
                    ks, xs, tg = reference_batch(k1)
                    b_state, loss = step_b_ref(b_state, ks, xs, tg, k3)
                else:
                    if it % cfg.cache_refresh == 0:
                        cache = forward_cache(f_state.ema_params, k1)
                    b_state, loss = step_b(b_state, cache, k2)
                losses.append(loss)
                if (it + 1) % cfg.log_every == 0 or it == steps - 1:
                    l = float(jnp.mean(jnp.stack(losses)))
                    log(n=n, phase="backward", step=it + 1, loss=l, it_per_s=len(losses) / (time.time() - t0))
                    losses, t0 = [], time.time()
            save(n, "backward")
            if eval_fn is not None:
                key, ke = jax.random.split(key)
                metrics = eval_fn(n, combined(b_state, b_static), combined(f_state, f_static), ke)
                if metrics:
                    log(event="eval", n=n, **metrics)

        f_state = warm_start(f_state)
        cache, t0, losses = None, time.time(), []
        for it in range(cfg.train_steps):
            key, k1, k2, k3 = jax.random.split(key, 4)
            if it % cfg.cache_refresh == 0:
                cache = backward_cache(b_state.ema_params, k1)
            f_state, loss = step_f(f_state, cache, k2)
            losses.append(loss)
            if (it + 1) % cfg.log_every == 0 or it == cfg.train_steps - 1:
                l = float(jnp.mean(jnp.stack(losses)))
                log(n=n, phase="forward", step=it + 1, loss=l, it_per_s=len(losses) / (time.time() - t0))
                losses, t0 = [], time.time()
        save(n, "forward")

    return combined(f_state, f_static), combined(b_state, b_static)
