"""Data-parallel building blocks for DSB on several GPUs of one node.

Design
------
Every DSB half-step is a regression on i.i.d. samples, so the natural parallelism is
*data parallelism*: the network parameters (and Adam / EMA state) are replicated on
every device and each device works on 1/ndev of the batch.  We use ``jax.shard_map``
over a one-dimensional device mesh (axis ``"data"``): the per-device function is the
ordinary single-device code of :mod:`dsb.ipf`, gradients are averaged with
``jax.lax.pmean`` (one NCCL all-reduce of the 9M-parameter gradient per step), and the
updated state comes out replicated (``PartitionSpec()``), so the same arrays can be
fed back in without any data movement.  ``shard_map`` was preferred over ``pmap``
(deprecated in favour of the sharding APIs and adds a leading device axis to every
array, which breaks the existing ``TrainState``/checkpoint layout) and over letting
``jit`` propagate ``NamedSharding`` annotations (works, but the placement of the
all-reduce and of the RNG is then left to XLA's SPMD partitioner; shard_map makes
both explicit).

Replicated arrays are ordinary ``jax.Array`` objects with a fully replicated sharding:
``np.asarray`` / ``eqx.tree_serialise_leaves`` see a single copy, so checkpoints written
from a parallel run load with the existing single-device code and vice versa
(:meth:`DataParallel.replicate` / :meth:`DataParallel.unreplicate`).

Trajectory cache (Algorithm 3): each device simulates ``M / ndev`` trajectories of its
own (``Cache.xs`` and ``Cache.targets`` are sharded along the trajectory axis, the step
indices ``ks`` are replicated) and each training step draws ``B / ndev`` (k, j) pairs
from the *local* shard of the cache -- a stratified version of the uniform sampling of
Algorithm 3 with no cross-device gather.

RNG: every per-device function folds ``jax.lax.axis_index("data")`` into the key it
receives, so devices use independent streams; with one device this reduces to
``fold_in(key, 0)`` (the sequence of random numbers differs from :mod:`dsb.train`, the
distribution does not).

Mixed precision (optional): :func:`mixed_precision` casts the float parameters and the
input to ``bfloat16`` for the network evaluation while the master weights, Adam, EMA
and the loss stay in float32; GroupNorm already normalises in float32 internally.
"""
from __future__ import annotations

from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from . import ipf
from .ipf import Cache, TrainState
from .networks import TimeEmbedding
from .schedule import OUReference


# --------------------------------------------------------------------------- mesh helpers
class DataParallel:
    """A 1-D mesh over ``devices`` plus the shardings / placement helpers used below."""

    def __init__(self, devices: Sequence[jax.Device] | None = None, axis: str = "data"):
        devices = list(jax.devices() if devices is None else devices)
        self.devices = devices
        self.ndev = len(devices)
        self.axis = axis
        self.mesh = Mesh(np.array(devices), (axis,))
        self.replicated = NamedSharding(self.mesh, P())
        self.sharded = NamedSharding(self.mesh, P(axis))

    # ---- placement -----------------------------------------------------------------
    def replicate(self, tree):
        """Put every array of ``tree`` (params, optimizer state, datasets) on all devices."""
        return jax.device_put(tree, self.replicated)

    def unreplicate(self, tree):
        """Single-device copy (device 0) of a replicated tree: what the single-GPU code
        produces, e.g. for ``eqx.tree_serialise_leaves`` or for ``jax.jit`` on one GPU."""
        return jax.device_put(tree, self.devices[0])

    def shard(self, x):
        """Shard a batch (leading axis) across the devices."""
        return jax.device_put(x, self.sharded)

    def local(self, n: int) -> int:
        if n % self.ndev:
            raise ValueError(f"{n} is not divisible by the number of devices {self.ndev}")
        return n // self.ndev

    def local_key(self, key):
        """Per-device PRNG key inside a shard_mapped function."""
        return jax.random.fold_in(key, jax.lax.axis_index(self.axis))

    def shard_map(self, f, in_specs, out_specs, check_vma: bool = True):
        return jax.shard_map(f, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs, check_vma=check_vma)

    @property
    def cache_specs(self):
        return Cache(xs=P(None, self.axis), targets=P(None, self.axis), ks=P())


def to_host(tree):
    """numpy copy of every array leaf (for checkpoints / inspection)."""
    return jax.tree_util.tree_map(lambda x: np.asarray(x) if eqx.is_array(x) else x, tree)


# --------------------------------------------------------------------------- mixed precision
class _CastOutput(eqx.Module):
    mod: eqx.Module
    dtype: jnp.dtype = eqx.field(static=True)

    def __call__(self, *args, **kwargs):
        return self.mod(*args, **kwargs).astype(self.dtype)


class MixedPrecision(eqx.Module):
    """Evaluate ``net`` with ``dtype`` activations; returns float32 (see :func:`mixed_precision`)."""

    net: eqx.Module
    dtype: jnp.dtype = eqx.field(static=True)

    def __call__(self, k, x, **kwargs):
        return self.net(k, x.astype(self.dtype), **kwargs).astype(jnp.float32)


def mixed_precision(net: eqx.Module, dtype=jnp.bfloat16) -> eqx.Module:
    """Cast the float parameters of ``net`` to ``dtype`` and make the time embedding emit
    ``dtype`` (otherwise ``bf16 @ f32`` promotes the whole U-Net back to float32).  The
    returned module maps float32 inputs to float32 increments.  Apply it *inside* the
    loss/simulation function to the combined network, so that master weights stay f32."""
    is_float = lambda a: eqx.is_array(a) and jnp.issubdtype(a.dtype, jnp.floating)
    net = jax.tree_util.tree_map(lambda a: a.astype(dtype) if is_float(a) else a, net)
    is_temb = lambda m: isinstance(m, TimeEmbedding)
    net = jax.tree_util.tree_map(lambda m: _CastOutput(m, dtype) if is_temb(m) else m, net, is_leaf=is_temb)
    return MixedPrecision(net, dtype)


def _combine(params, static, compute_dtype):
    net = eqx.combine(params, static)
    return net if compute_dtype is None else mixed_precision(net, compute_dtype)


# --------------------------------------------------------------------------- train step
def make_train_step(
    dp: DataParallel,
    optimizer,
    static,
    ema_decay: float,
    use_dropout: bool,
    weight_table=None,
    compute_dtype=None,
    fused_cache_sampling: int | None = None,
    ravel_grads: bool = True,
    steps_per_call: int = 1,
):
    """Data-parallel version of :func:`dsb.ipf.make_train_step`.

    Returns a jitted ``step(state, ks, xs, targets, key) -> (state, loss)`` where ``state``
    is replicated (see :meth:`DataParallel.replicate`) and ``ks, xs, targets`` are
    batch-sharded (global batch B, B/ndev per device; outputs of :func:`make_sample_cache`
    or :func:`make_reference_batch` already are).  ``weight_table`` is the per-step loss
    weight table indexed by the network's input step (``None`` = unweighted, as in the
    paper).  With ``fused_cache_sampling=B`` the signature becomes
    ``step(state, cache, key)``: each device draws its ``B/ndev`` pairs from its cache
    shard inside the same jitted call (one dispatch per training step instead of two).
    ``ravel_grads=True`` reduces all gradients in a single all-reduce of one flat vector;
    ``False`` leaves one all-reduce per parameter leaf to XLA's combiner/scheduler.
    ``steps_per_call=K`` (fused sampling only) runs K consecutive training steps, each on a
    fresh batch drawn from the cache, inside one jitted call (``lax.scan``) and returns the
    K losses as a vector: the host-side cost of dispatching the ~1200-leaf train state
    (several ms per call, see scripts/bench_collective.py) is then paid once per K steps.
    """
    d = dp.axis

    def loss_fn(params, ks, xs, targets, key):
        net = _combine(params, static, compute_dtype)
        weights = None if weight_table is None else jnp.asarray(weight_table)[ks]
        return ipf.mean_matching_loss(net, ks, xs, targets, weights, key if use_dropout else None)

    def update(state: TrainState, ks, xs, targets, key):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, ks, xs, targets, key)
        if ravel_grads:
            # One all-reduce per step: ravel the ~300 gradient leaves (+ the loss) into a
            # single vector and average it over the devices (identical on every device).
            flat, unravel = jax.flatten_util.ravel_pytree(grads)
            flat = jax.lax.pmean(jnp.concatenate([flat, loss[None].astype(flat.dtype)]), d)
            grads, loss = unravel(flat[:-1]), flat[-1]
        else:
            grads, loss = jax.lax.pmean((grads, loss), d)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = eqx.apply_updates(state.params, updates)
        decay = jnp.minimum(ema_decay, (1.0 + state.step) / (10.0 + state.step))
        ema = jax.tree_util.tree_map(lambda e, p: decay * e + (1.0 - decay) * p, state.ema_params, params)
        return TrainState(params, ema, opt_state, state.step + 1), loss

    if fused_cache_sampling is None:

        def local_step(state, ks, xs, targets, key):
            return update(state, ks, xs, targets, dp.local_key(key))

        in_specs = (P(), P(d), P(d), P(d), P())
    else:
        b_local = dp.local(fused_cache_sampling)

        def one_step(state, cache, key):
            k1, k2 = jax.random.split(key)
            ks, xs, targets = ipf.sample_cache(cache, k1, b_local)
            return update(state, ks, xs, targets, k2)

        if steps_per_call == 1:

            def local_step(state, cache, key):
                return one_step(state, cache, dp.local_key(key))

        else:

            def local_step(state, cache, key):
                keys = jax.random.split(dp.local_key(key), steps_per_call)
                return jax.lax.scan(lambda st, k: one_step(st, cache, k), state, keys)

        in_specs = (P(), dp.cache_specs, P())

    # check_vma=False: with the vma check on, jax.grad of the *invariant* params inside the
    # body inserts an implicit psum (transpose of the broadcast) so the local gradients would
    # already be summed over devices; we keep the classic pmap semantics instead, where the
    # body sees purely local gradients and the reduction is the explicit pmean above.
    return jax.jit(dp.shard_map(local_step, in_specs=in_specs, out_specs=(P(), P()), check_vma=False))


# --------------------------------------------------------------------------- cache
def make_cache_fn(
    dp: DataParallel,
    static,
    gammas,
    sample_init: Callable[..., jax.Array],
    cache_size: int,
    *,
    forward: bool,
    compute_dtype=None,
    last_step_noise: bool = True,
    sample_args=(),
):
    """Jitted ``refresh(ema_params, key) -> Cache`` simulating ``cache_size`` trajectories,
    ``cache_size / ndev`` per device (``Cache.xs``/``targets`` sharded along axis 1).

    ``sample_init(key, n, *sample_args)`` draws ``n`` initial states (data for the forward
    chain, prior for the backward chain).  Large arrays it needs (the CIFAR-10 images)
    should be passed in ``sample_args``: they are replicated on every device and handed to
    ``sample_init`` as plain per-device arrays.  (A jax array *closed over* by
    ``sample_init`` must not carry a mesh sharding -- pass an uncommitted array or use
    ``sample_args``.)
    """
    m_local = dp.local(cache_size)
    gammas = jnp.asarray(gammas)
    sample_args = tuple(dp.replicate(a) for a in sample_args)

    def local(ema_params, key, *args):
        k1, k2 = jax.random.split(dp.local_key(key))
        x_init = sample_init(k1, m_local, *args)
        net = _combine(ema_params, static, compute_dtype)
        return ipf.simulate(net, x_init, gammas, k2, forward=forward, last_step_noise=last_step_noise)

    fn = jax.jit(dp.shard_map(local, in_specs=(P(), P()) + (P(),) * len(sample_args), out_specs=dp.cache_specs))
    return lambda ema_params, key: fn(ema_params, key, *sample_args)


def make_sample_cache(dp: DataParallel, batch_size: int):
    """Jitted ``sample(cache, key) -> (ks, xs, targets)``: ``batch_size / ndev`` uniform draws
    from each device's cache shard; outputs are batch-sharded."""
    b_local = dp.local(batch_size)

    def local(cache, key):
        return ipf.sample_cache(cache, dp.local_key(key), b_local)

    return jax.jit(dp.shard_map(local, in_specs=(dp.cache_specs, P()), out_specs=P(dp.axis)))


def make_reference_batch(dp: DataParallel, ref: OUReference, sample_data: Callable[..., jax.Array], batch_size: int, sample_args=()):
    """Jitted ``batch(key) -> (ks, xs, targets)`` for the first (closed-form) half-step:
    each device samples ``batch_size / ndev`` data points (``sample_data(key, n, *sample_args)``,
    see :func:`make_cache_fn`) and reference pairs."""
    b_local = dp.local(batch_size)
    sample_args = tuple(dp.replicate(a) for a in sample_args)

    def local(key, *args):
        k1, k2 = jax.random.split(dp.local_key(key))
        return ipf.reference_pairs(ref, k2, sample_data(k1, b_local, *args))

    fn = jax.jit(dp.shard_map(local, in_specs=(P(),) * (1 + len(sample_args)), out_specs=P(dp.axis)))
    return lambda key: fn(key, *sample_args)


def make_generate(dp: DataParallel, static, gammas, sample_prior, num_samples: int, *, compute_dtype=None):
    """Jitted ``generate(b_ema, key) -> x0`` (num_samples / ndev samples per device, output
    sharded along the batch) for the evaluation / FID sample dumps."""
    n_local = dp.local(num_samples)
    gammas = jnp.asarray(gammas)

    def local(b_params, key):
        k1, k2 = jax.random.split(dp.local_key(key))
        net = _combine(b_params, static, compute_dtype)
        return ipf.generate(net, sample_prior(k1, n_local), gammas, k2, last_step_noise=False)

    return jax.jit(dp.shard_map(local, in_specs=(P(), P()), out_specs=P(dp.axis)))


# --------------------------------------------------------------------------- state helpers
def init_train_state(dp: DataParallel, net: eqx.Module, optimizer) -> TrainState:
    """Replicated :class:`TrainState` (single-copy semantics, ``ndev`` physical copies)."""
    return dp.replicate(ipf.init_train_state(net, optimizer))

