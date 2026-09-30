"""Driver for Algorithm 1 (with the cached trajectories of Algorithm 3).

The driver is dataset agnostic: it needs a network constructor, a sampler for
p_data and a sampler for p_prior, and calls back an ``eval_fn`` after every
backward half-step (i.e. whenever a new generative model B^n is available).

Exact snapshots are committed every ``checkpoint_every`` gradient steps within
each half-step and at phase boundaries. Cache trajectories are reconstructed
from their saved refresh key and the frozen opposite EMA, never serialized.
Recovery rolls back uncommitted rows in ``metrics.jsonl`` and ``logger.path``;
evaluation callbacks must overwrite other per-iteration artifacts on replay.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
import tempfile
import uuid
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
    checkpoint_every: int = 1000       # exact snapshots per half-step; also save at phase boundaries
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


def _atomic_write(path, write):
    """Publish a complete, durable file without invalidating its predecessor."""
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        directory = os.open(os.path.dirname(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_json(path, value):
    _atomic_write(path, lambda fh: fh.write(json.dumps(value, indent=2).encode()))


class _Checkpoints:
    def __init__(self, cfg, outdir, logger, runtime, states, statics):
        if cfg.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        self.outdir = os.fspath(outdir)
        self.ckdir = os.path.join(outdir, "ckpt")
        os.makedirs(self.ckdir, exist_ok=True)
        self.path = os.path.join(self.ckdir, "resume.json")
        self.config = dataclasses.asdict(cfg)
        self.runtime = runtime
        self.model = {
            "tree": str(jax.tree_util.tree_structure(tuple(s.params for s in states))),
            "static_leaves": [repr(x) for x in jax.tree_util.tree_leaves(statics)],
            "leaves": [[list(x.shape), str(x.dtype)] for s in states for x in jax.tree_util.tree_leaves(s.params)],
        }
        self.log_paths = {os.path.join(self.outdir, "metrics.jsonl")}
        if logger is not None and hasattr(logger, "path"):
            self.log_paths.add(os.fspath(logger.path))
        self.previous = None

    def restore(self, f_state, b_state, key, log):
        config_path = os.path.join(self.outdir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as fh:
                previous_config = dataclasses.asdict(DSBConfig(**json.load(fh)))
            if any(previous_config[name] != value for name, value in self.config.items()
                   if name != "checkpoint_every"):
                raise ValueError("Incompatible saved config. Use a new output directory.")
        cursor = dict(n=0, phase="backward", it=0, losses=[])
        cache_key = jnp.zeros_like(key)
        if os.path.exists(self.path):
            with open(self.path) as fh:
                meta = json.load(fh)
            if meta.get("version") != 1:
                raise ValueError("Unsupported resume checkpoint version")
            for name, value in (("config", self.config), ("runtime", self.runtime), ("model", self.model)):
                previous = meta[name]
                if name == "config":
                    previous = {k: v for k, v in previous.items() if k != "checkpoint_every"}
                    value = {k: v for k, v in value.items() if k != "checkpoint_every"}
                if previous != value:
                    raise ValueError(f"Incompatible {name} for exact resume; use a new output directory")
            f_state, b_state, key, cache_key = eqx.tree_deserialise_leaves(
                os.path.join(self.ckdir, meta["file"]), (f_state, b_state, key, cache_key),
            )
            cursor = meta["cursor"]
            # Evaluation callbacks append metrics before returning. Roll back uncommitted
            # rows (including a torn last line), then replay with the saved evaluation RNG.
            for name, size in meta["logs"].items():
                path = os.path.join(self.outdir, name)
                if not os.path.exists(path):
                    if size:
                        raise ValueError(f"Missing committed log: {path}")
                    continue
                if os.path.getsize(path) < size:
                    raise ValueError(f"Truncated committed log: {path}")
                with open(path, "r+b") as fh:
                    fh.truncate(size)
            self.previous = meta["file"]
            log(event="resume", exact=True, n=cursor["n"], phase=cursor["phase"], step=cursor["it"])
        else:
            latest = os.path.join(self.ckdir, "latest.json")
            if os.path.exists(latest):
                with open(latest) as fh:
                    meta = json.load(fh)
                nets = eqx.tree_deserialise_leaves(
                    os.path.join(self.ckdir, meta["file"]),
                    Nets(f_state.params, f_state.ema_params, b_state.params, b_state.ema_params),
                )
                f_state = f_state._replace(params=nets.f_params, ema_params=nets.f_ema,
                                          step=jnp.asarray(meta.get("f_step", 0), jnp.int32))
                b_state = b_state._replace(params=nets.b_params, ema_params=nets.b_ema,
                                          step=jnp.asarray(meta.get("b_step", 0), jnp.int32))
                cursor.update(n=meta["n"] + (meta["phase"] == "forward"),
                              phase="eval" if meta["phase"] == "backward" else "backward")
                metrics_path = os.path.join(self.outdir, "metrics.jsonl")
                if cursor["phase"] == "eval" and os.path.exists(metrics_path):
                    with open(metrics_path) as fh:
                        if any(json.loads(line).get("n") == cursor["n"] for line in fh if line.strip()):
                            cursor["phase"] = "forward"
                key = jax.random.PRNGKey(self.config["seed"] + 1000 * (2 * cursor["n"] + (cursor["phase"] in ("eval", "forward"))))
                log(event="resume", exact=False, n=cursor["n"], phase=cursor["phase"], file=meta["file"])
        if not os.path.exists(config_path):
            _atomic_json(config_path, self.config)
        return f_state, b_state, key, cache_key, cursor

    def save(self, f_state, b_state, key, cache_key, cursor):
        fname = f"resume-{uuid.uuid4().hex}.eqx"
        _atomic_write(os.path.join(self.ckdir, fname),
                      lambda fh: eqx.tree_serialise_leaves(fh, (f_state, b_state, key, cache_key)))
        offsets = {}
        for path in self.log_paths:
            size = 0
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    os.fsync(fh.fileno())
                    size = os.fstat(fh.fileno()).st_size
            offsets[os.path.relpath(path, self.outdir)] = size
        _atomic_json(self.path, dict(version=1, file=fname, config=self.config, runtime=self.runtime,
                                    model=self.model, cursor=cursor, logs=offsets))
        old, self.previous = self.previous, fname
        if old is not None:
            os.unlink(os.path.join(self.ckdir, old))

    def save_models(self, f_state, b_state, n, phase):
        fname = f"iter{n:03d}_{phase}.eqx"
        nets = Nets(f_state.params, f_state.ema_params, b_state.params, b_state.ema_params)
        _atomic_write(os.path.join(self.ckdir, fname), lambda fh: eqx.tree_serialise_leaves(fh, nets))
        _atomic_json(os.path.join(self.ckdir, "latest.json"),
                     dict(n=n, phase=phase, file=fname, f_step=int(f_state.step), b_step=int(b_state.step)))


@eqx.filter_jit
def _state_is_finite(state):
    return jnp.all(jnp.asarray([
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves(state) if eqx.is_inexact_array(leaf)
    ]))


def _run_loop(cfg, f_state, b_state, key, *, checkpoints, optimizer, f_static, b_static,
              reference_step, backward_step, forward_step, forward_cache, backward_cache,
              eval_fn, log, replicate=lambda x: x, unreplicate=lambda x: x):
    f_state, b_state, key, cache_key, cursor = checkpoints.restore(
        unreplicate(f_state), unreplicate(b_state), key, log,
    )
    f_state, b_state = replicate(f_state), replicate(b_state)

    def save():
        checkpoints.save(unreplicate(f_state), unreplicate(b_state), key, cache_key, cursor)

    def combined(state, static):
        return eqx.combine(unreplicate(state.ema_params), static)

    def checked_losses(losses, n, phase, step):
        values = np.asarray(jnp.stack(losses)) if losses else np.empty(0)
        invalid = np.flatnonzero(~np.isfinite(values))
        if invalid.size:
            failed_step = step - len(losses) + int(invalid[0]) + 1
            raise FloatingPointError(f"Nonfinite loss: n={n} phase={phase} step={failed_step}")
        return values.tolist()

    def check_state(n, phase, step):
        if not bool(_state_is_finite((f_state, b_state))):
            raise FloatingPointError(f"Nonfinite state: n={n} phase={phase} step={step}")

    checked_losses(cursor["losses"], cursor["n"], cursor["phase"], cursor["it"])
    check_state(cursor["n"], cursor["phase"], cursor["it"])

    # A snapshot at it=0 is before warm-start/reset; all other snapshots are after
    # that initialization. The opposite EMA stays fixed throughout a half-step.
    if checkpoints.previous is None:
        save()
    while cursor["n"] < cfg.num_dsb_iters:
        n, phase, first_it = cursor["n"], cursor["phase"], cursor["it"]
        if phase == "eval":
            if eval_fn is not None:
                key, ke = jax.random.split(key)
                try:
                    metrics = eval_fn(n, combined(b_state, b_static), combined(f_state, f_static), ke)
                except FloatingPointError as exc:
                    raise FloatingPointError(f"Nonfinite evaluation: n={n} phase=eval: {exc}") from exc
                if metrics:
                    for name, value in metrics.items():
                        value = np.asarray(value)
                        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                            raise FloatingPointError(f"Nonfinite metric: n={n} phase=eval metric={name}")
                    log(event="eval", n=n, **metrics)
            checkpoints.save_models(unreplicate(f_state), unreplicate(b_state), n, "backward")
            cursor = dict(n=n, phase="forward", it=0, losses=[])
            save()
            continue

        backward = phase == "backward"
        steps = cfg.first_train_steps if backward and n == 0 and cfg.first_train_steps else cfg.train_steps
        state = b_state if backward else f_state
        if first_it == 0:
            state = state._replace(params=state.ema_params)
            if cfg.reset_optimizer:
                state = state._replace(opt_state=replicate(optimizer.init(state.params)))
        cache = None
        if first_it % cfg.cache_refresh and not (backward and n == 0):
            cache = forward_cache(f_state.ema_params, cache_key) if backward else backward_cache(b_state.ema_params, cache_key)
            jax.block_until_ready(cache)
        losses = [jnp.asarray(loss) for loss in cursor["losses"]]
        t0 = time.time()
        for it in range(first_it, steps):
            key, k1, k2, k3 = jax.random.split(key, 4)
            if backward and n == 0:
                state, loss = reference_step(state, k1, k3)
            else:
                if it % cfg.cache_refresh == 0:
                    # Cache overlap can exceed device memory at refresh boundaries.
                    jax.block_until_ready(state)
                    cache = None
                    cache_key = k1
                    cache = forward_cache(f_state.ema_params, k1) if backward else backward_cache(b_state.ema_params, k1)
                    jax.block_until_ready(cache)
                step = backward_step if backward else forward_step
                state, loss = step(state, cache, k2, k3)
            if backward:
                b_state = state
            else:
                f_state = state
            losses.append(loss)
            should_log = (it + 1) % cfg.log_every == 0 or it == steps - 1
            should_save = (it + 1) % cfg.checkpoint_every == 0 and it + 1 < steps
            if should_log or should_save:
                pending_losses = checked_losses(losses, n, phase, it + 1)
            if should_log:
                log(n=n, phase=phase, step=it + 1, loss=float(jnp.mean(jnp.stack(losses))),
                    it_per_s=len(losses) / (time.time() - t0))
                losses, pending_losses, t0 = [], [], time.time()
            if should_save:
                check_state(n, phase, it + 1)
                cursor = dict(n=n, phase=phase, it=it + 1, losses=pending_losses)
                save()
        check_state(n, phase, steps)
        cache = None
        if not backward:
            checkpoints.save_models(unreplicate(f_state), unreplicate(b_state), n, phase)
        cursor = dict(n=n if backward else n + 1, phase="eval" if backward else "backward", it=0, losses=[])
        save()
    return combined(f_state, f_static), combined(b_state, b_static)


def run_dsb(
    cfg: DSBConfig,
    *,
    make_forward_net: Callable[[jax.Array], eqx.Module],
    make_backward_net: Callable[[jax.Array], eqx.Module],
    sample_data: Callable[[jax.Array, int, jax.Array], jax.Array],
    sample_prior: Callable[[jax.Array, int], jax.Array],
    data: jax.Array,
    outdir: str,
    eval_fn: Callable | None = None,
    logger=None,
):
    log = logger.log if logger is not None else (lambda **kw: print(kw, flush=True))

    ref = cfg.reference()
    gammas = jnp.asarray(ref.gammas)
    key = jax.random.PRNGKey(cfg.seed)
    key, kf, kb = jax.random.split(key, 3)
    f_net, b_net = make_forward_net(kf), make_backward_net(kb)
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

    checkpoints = _Checkpoints(cfg, outdir, logger, {"driver": "single"},
                               (f_state, b_state), (f_static, b_static))

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

    def reference_step(state, k1, k3):
        ks, xs, tg = reference_batch(k1, data)
        return train_step_b(state, ks, xs, tg, weights_for(wb, ks), k3)

    def backward_step(state, cache, k2, k3):
        ks, xs, tg = sample_cache(cache, k2, cfg.batch_size)
        return train_step_b(state, ks, xs, tg, weights_for(wb, ks), k3)

    def forward_step(state, cache, k2, k3):
        ks, xs, tg = sample_cache(cache, k2, cfg.batch_size)
        return train_step_f(state, ks, xs, tg, weights_for(wf, ks), k3)

    return _run_loop(
        cfg, f_state, b_state, key, checkpoints=checkpoints, optimizer=optimizer,
        f_static=f_static, b_static=b_static, reference_step=reference_step,
        backward_step=backward_step, forward_step=forward_step,
        forward_cache=lambda params, rng: forward_cache(params, rng, data),
        backward_cache=backward_cache, eval_fn=eval_fn, log=log,
    )


# ----------------------------------------------------------------------------- multi-GPU
def run_dsb_parallel(
    cfg: DSBConfig,
    *,
    make_forward_net: Callable[[jax.Array], eqx.Module],
    make_backward_net: Callable[[jax.Array], eqx.Module],
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

    log = logger.log if logger is not None else (lambda **kw: print(kw, flush=True))

    dp = par.DataParallel(devices)
    ref = cfg.reference()
    gammas = jnp.asarray(ref.gammas)
    key = jax.random.PRNGKey(cfg.seed)
    key, kf, kb = jax.random.split(key, 3)
    f_net, b_net = make_forward_net(kf), make_backward_net(kb)
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

    runtime = {"driver": "parallel", "ndev": dp.ndev,
               "compute_dtype": None if compute_dtype is None else str(jnp.dtype(compute_dtype))}
    checkpoints = _Checkpoints(cfg, outdir, logger, runtime, (f_state, b_state), (f_static, b_static))

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

    def reference_step(state, k1, k3):
        ks, xs, tg = reference_batch(k1)
        return step_b_ref(state, ks, xs, tg, k3)

    return _run_loop(
        cfg, f_state, b_state, key, checkpoints=checkpoints, optimizer=optimizer,
        f_static=f_static, b_static=b_static, reference_step=reference_step,
        backward_step=lambda state, cache, k2, k3: step_b(state, cache, k2),
        forward_step=lambda state, cache, k2, k3: step_f(state, cache, k2),
        forward_cache=forward_cache, backward_cache=backward_cache,
        eval_fn=eval_fn, log=log, replicate=dp.replicate, unreplicate=dp.unreplicate,
    )
