"""Core of Diffusion Schrödinger Bridge (Algorithm 1 / cached Algorithm 3).

Notation (see ``schedule.py``): ``gammas[k]`` is the step size of the transition
k -> k+1.  Two networks are learnt:

* forward mean  F(k, x) = x + f_net(k, x), k in {0..N-1}: mean of X_{k+1} given X_k = x,
* backward mean B(k, x) = x + b_net(k, x), k in {1..N}:   mean of X_{k-1} given X_k = x.

DSB iteration n:
  backward half-step (12): simulate forward trajectories X_0 ~ p_data,
      X_{k+1} = F^n(k, X_k) + sqrt(2 gammas[k]) Z, and regress
      b_net(k+1, X_{k+1})  ->  F^n(k, X_k) - F^n(k, X_{k+1});
  forward half-step (13): simulate backward trajectories X_N ~ p_prior,
      X_{k-1} = B^n(k, X_k) + sqrt(2 gammas[k-1]) Z, and regress
      f_net(k-1, X_{k-1})  ->  B^n(k, X_k) - B^n(k, X_{k-1}).
For n = 0 the forward mean is the reference F^0(k, x) = x + gammas[k] f(x) and the
pairs (X_k, X_{k+1}) are sampled in closed form (Technique 2).
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .schedule import OUReference


# --------------------------------------------------------------------------- helpers
def _bcast(v, x):
    """Reshape a per-sample vector ``v`` (B,) to broadcast against ``x`` (B, ...)."""
    return jnp.reshape(v, v.shape + (1,) * (x.ndim - v.ndim))


def batched_increment(net: eqx.Module, ks, xs):
    """Apply an (unbatched) Equinox network to a batch of (k, x) pairs."""
    return jax.vmap(net)(ks, xs)


def batched_mean(net: eqx.Module, ks, xs):
    return xs + batched_increment(net, ks, xs)


class Cache(NamedTuple):
    """Cached transitions: for each stored point ``x`` with step index ``k``, the network
    being trained should map (k, x) -> x + target."""

    xs: jax.Array       # (N, M, ...)
    targets: jax.Array  # (N, M, ...)
    ks: jax.Array       # (N,) int32


# --------------------------------------------------------------------------- simulation
def simulate(
    mean_net: eqx.Module,
    x_init: jax.Array,
    gammas: jax.Array,
    key,
    *,
    forward: bool,
    last_step_noise: bool = True,
    shared_noise: bool = False,
) -> Cache:
    """Run the chain driven by ``mean_net`` for N steps and record training pairs.

    forward=True : k = 0..N-1, X_{k+1} = F(k, X_k) + sqrt(2 gammas[k]) Z.
    forward=False: k = N..1,   X_{k-1} = B(k, X_k) + sqrt(2 gammas[k-1]) Z.

    Returns a :class:`Cache` whose ``xs[i]`` is the state *after* step i, ``ks[i]`` its
    step index and ``targets[i] = M(k_in, X_in) - M(k_in, X_out)`` (Prop. 3 targets).
    ``xs[-1]`` is the final state (X_N or X_0).  ``shared_noise`` uses the same Gaussian
    noise for every element of the batch (deterministic latent embedding, Figure 13).
    """
    n = gammas.shape[0]
    if forward:
        ks_in = jnp.arange(n, dtype=jnp.int32)
        ks_out = ks_in + 1
        gam = gammas
    else:
        ks_in = jnp.arange(n, 0, -1, dtype=jnp.int32)
        ks_out = ks_in - 1
        gam = gammas[::-1]
    noise_scale = jnp.sqrt(2.0 * gam)
    if not last_step_noise:
        noise_scale = noise_scale.at[-1].set(0.0)
    keys = jax.random.split(key, n)

    def step(x, inp):
        k_in, k_out, sc, kk = inp
        kb = jnp.full((x.shape[0],), k_in, jnp.int32)
        mean_old = batched_mean(mean_net, kb, x)
        noise_shape = (1,) + x.shape[1:] if shared_noise else x.shape
        x_new = mean_old + sc * jax.random.normal(kk, noise_shape, x.dtype)
        mean_new = batched_mean(mean_net, kb, x_new)
        return x_new, (x_new, mean_old - mean_new)

    _, (xs, targets) = jax.lax.scan(step, x_init, (ks_in, ks_out, noise_scale, keys))
    return Cache(xs=xs, targets=targets, ks=ks_out)


def generate(b_net: eqx.Module, x_prior: jax.Array, gammas: jax.Array, key, *, last_step_noise=False, shared_noise=False):
    """Sample X_0 from the generative model: X_N ~ p_prior, then the backward chain."""
    cache = simulate(b_net, x_prior, gammas, key, forward=False, last_step_noise=last_step_noise, shared_noise=shared_noise)
    return cache.xs[-1]


def reference_pairs(ref: OUReference, key, x0):
    """Technique 2: closed-form sampling of (X_k, X_{k+1}) under the reference dynamics.

    Returns (k+1, x_{k+1}, target) with target = F^0(k, X_k) - F^0(k, X_{k+1}); the
    backward network is trained so that b_net(k+1, X_{k+1}) ~ target.
    """
    kk, kx, kz = jax.random.split(key, 3)
    b = x0.shape[0]
    n = ref.num_steps
    gammas = jnp.asarray(ref.gammas)
    k = jax.random.randint(kk, (b,), 0, n)
    xk = ref.sample_marginal(kx, k, x0)
    mean_old = ref.forward_mean(k, xk)
    xk1 = mean_old + _bcast(jnp.sqrt(2.0 * gammas[k]), xk) * jax.random.normal(kz, xk.shape, xk.dtype)
    mean_new = ref.forward_mean(k, xk1)
    return k + 1, xk1, mean_old - mean_new


def sample_cache(cache: Cache, key, batch_size: int):
    """Uniformly sample (k, j) pairs from the cache (Algorithm 3, line 5). Returns (ks, xs, targets)."""
    n, m = cache.xs.shape[:2]
    ki, kj = jax.random.split(key)
    i = jax.random.randint(ki, (batch_size,), 0, n)
    j = jax.random.randint(kj, (batch_size,), 0, m)
    return cache.ks[i], cache.xs[i, j], cache.targets[i, j]


# --------------------------------------------------------------------------- training
def mean_matching_loss(net: eqx.Module, ks, xs, targets, weights=None, key=None):
    """Empirical version of (12)/(13): mean squared error between the predicted increment
    and the target increment.  Unweighted by default (Appendix I); ``weights`` (B,) allows
    per-step weighting, e.g. 1/(2 gamma_k) for an epsilon-style loss."""
    if key is None:
        pred = jax.vmap(net)(ks, xs)
    else:
        keys = jax.random.split(key, xs.shape[0])
        pred = jax.vmap(lambda k, x, kk: net(k, x, key=kk, inference=False))(ks, xs, keys)
    err = jnp.mean(jnp.square(pred - targets), axis=tuple(range(1, xs.ndim)))
    if weights is not None:
        err = err * weights
    return jnp.mean(err)


class TrainState(NamedTuple):
    params: eqx.Module      # trainable arrays of the network (eqx.partition)
    ema_params: eqx.Module  # exponential moving average (Technique 6)
    opt_state: optax.OptState
    step: jax.Array


def make_optimizer(lr: float, grad_clip: float | None = 1.0, warmup: int = 0):
    sched = lr if warmup == 0 else optax.linear_schedule(0.0, lr, warmup)
    chain = []
    if grad_clip is not None and grad_clip > 0:
        chain.append(optax.clip_by_global_norm(grad_clip))
    chain.append(optax.adam(sched, b1=0.9, b2=0.999))
    return optax.chain(*chain)


def init_train_state(net: eqx.Module, optimizer) -> TrainState:
    params, _ = eqx.partition(net, eqx.is_array)
    return TrainState(params=params, ema_params=params, opt_state=optimizer.init(params), step=jnp.zeros((), jnp.int32))


def make_train_step(optimizer, static, ema_decay: float, use_dropout: bool):
    """Build a jitted gradient step for the mean-matching loss on a batch."""

    def loss_fn(params, ks, xs, targets, weights, key):
        net = eqx.combine(params, static)
        return mean_matching_loss(net, ks, xs, targets, weights, key if use_dropout else None)

    @eqx.filter_jit
    def train_step(state: TrainState, ks, xs, targets, weights, key):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, ks, xs, targets, weights, key)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = eqx.apply_updates(state.params, updates)
        # EMA with a warm-up so that early steps do not carry the random init for long
        decay = jnp.minimum(ema_decay, (1.0 + state.step) / (10.0 + state.step))
        ema = jax.tree_util.tree_map(lambda e, p: decay * e + (1.0 - decay) * p, state.ema_params, params)
        return TrainState(params, ema, opt_state, state.step + 1), loss

    return train_step
