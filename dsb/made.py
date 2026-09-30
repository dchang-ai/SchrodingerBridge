"""Smooth autoregressive densities used as DSB increment parameterizations.

The score mode is a time-scaled gradient field, hence conservative in x. It is
trained with the existing mean-matching objective: this is not an assertion
that a DSB mean-map increment equals an exact marginal-density score.
"""
from __future__ import annotations

import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .networks import sinusoidal_embedding


class _MaskedLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    input_ranks: tuple[int, ...] = eqx.field(static=True)
    output_ranks: tuple[int, ...] = eqx.field(static=True)
    strict: bool = eqx.field(static=True)

    def __init__(self, input_ranks, output_ranks, *, strict=False, key):
        self.input_ranks = tuple(input_ranks)
        self.output_ranks = tuple(output_ranks)
        self.strict = strict
        mask = self.mask()
        fan_in = jnp.maximum(mask.sum(axis=1, keepdims=True), 1)
        bound = jnp.sqrt(3.0 / fan_in)
        self.weight = jax.random.uniform(key, mask.shape, minval=-1.0, maxval=1.0) * bound * mask
        self.bias = jnp.zeros((len(output_ranks),))

    def mask(self):
        inputs = jnp.asarray(self.input_ranks)[None, :]
        outputs = jnp.asarray(self.output_ranks)[:, None]
        return inputs < outputs if self.strict else inputs <= outputs

    def __call__(self, x):
        return (self.weight * self.mask()) @ x + self.bias


class MADE(eqx.Module):
    """Normalized Gaussian-mixture MADE in natural, flattened coordinate order.

    Hidden edges have nondecreasing ranks; output edges require strictly lower
    ranks. Rank-zero time features and hidden units make even the first
    conditional a trainable, time-dependent mixture. Only the *conditional
    parameters* are strictly autoregressive: the full joint-density score can
    depend on a coordinate itself and on later coordinates.

    ``mode='score'`` returns ``gain[k] * grad_x log_prob(k, x)``. The learned
    scalar gains start at ``output_scale``; k must be an integer in
    [0, num_steps]. ``mode='strict'`` instead returns the mixture conditional
    mean as an increment, a strictly autoregressive negative control (it does
    not subtract x). Neither mode changes the trainer's regression targets.
    Batch examples with ``jax.vmap``; key and inference are accepted but unused.
    """

    _log_scale_bound: ClassVar[float] = 3.0
    layers: tuple[_MaskedLinear, ...]
    gain: jax.Array
    dim: int = eqx.field(static=True)
    hidden: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    components: int = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    output_scale: float = eqx.field(static=True)

    def __init__(self, dim, hidden=128, depth=2, components=8, mode="score",
                 num_steps=20, output_scale=0.01, *, key):
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in (dim, hidden, depth, components, num_steps)):
            raise ValueError("dim, hidden, depth, components and num_steps must be positive integers")
        if mode not in ("score", "strict"):
            raise ValueError("mode must be 'score' or 'strict'")
        if not math.isfinite(output_scale):
            raise ValueError("output_scale must be finite")
        self.dim, self.hidden, self.depth = dim, hidden, depth
        self.components, self.mode = components, mode
        self.num_steps, self.output_scale = num_steps, float(output_scale)
        self.gain = jnp.full((num_steps + 1,), output_scale, dtype=jnp.float32)
        hidden_ranks = [i * dim // hidden for i in range(hidden)]
        hidden_ranks[-1] = dim - 1
        input_ranks = tuple(range(1, dim + 1)) + (0,) * 16
        keys = jax.random.split(key, depth + 1)
        layers = []
        for layer_key in keys[:-1]:
            layers.append(_MaskedLinear(input_ranks, hidden_ranks, key=layer_key))
            input_ranks = hidden_ranks
        output_ranks = tuple(rank for rank in range(1, dim + 1) for _ in range(3 * components))
        layers.append(_MaskedLinear(hidden_ranks, output_ranks, strict=True, key=keys[-1]))
        self.layers = tuple(layers)

    def _coordinates(self, x):
        return jnp.asarray(x).reshape((self.dim,))

    def conditional_parameters(self, k, x):
        """Return (logits, locations, log_scales), each shaped (dim, components)."""
        x = self._coordinates(x)
        h = jnp.concatenate((x, sinusoidal_embedding(k, 16)))
        for layer in self.layers[:-1]:
            h = jax.nn.silu(layer(h))
        raw = self.layers[-1](h).reshape((self.dim, 3, self.components))
        return raw[:, 0], raw[:, 1], self._log_scale_bound * jnp.tanh(raw[:, 2] / self._log_scale_bound)

    def log_prob(self, k, x):
        """Scalar log density, normalized over all of R**dim for each fixed k."""
        logits, locations, log_scales = self.conditional_parameters(k, x)
        residual = (self._coordinates(x)[:, None] - locations) * jnp.exp(-log_scales)
        log_components = -0.5 * residual**2 - log_scales - 0.5 * math.log(2.0 * math.pi)
        return jnp.sum(logsumexp(jax.nn.log_softmax(logits, axis=-1) + log_components, axis=-1))

    def __call__(self, k, x, *, key=None, inference=False):
        if self.mode == "score":
            return (self.gain[k] * jax.grad(self.log_prob, argnums=1)(k, x)).reshape(x.shape)
        logits, locations, _ = self.conditional_parameters(k, x)
        return jnp.sum(jax.nn.softmax(logits, axis=-1) * locations, axis=-1).reshape(x.shape)

    def metadata(self):
        return {
            "architecture": "made",
            "dim": self.dim,
            "hidden": self.hidden,
            "depth": self.depth,
            "components": self.components,
            "mode": self.mode,
            "num_steps": self.num_steps,
            "output_scale": self.output_scale,
            "ordering": "natural_flattened",
            "hidden_rank_assignment": "balanced_including_endpoints",
            "log_scale_bounds": [-self._log_scale_bound, self._log_scale_bound],
        }


class ImageMADE(MADE):
    """CIFAR density in raster RGB order, with conditional scales down to exp(-7)."""

    _log_scale_bound: ClassVar[float] = 7.0

    def __init__(self, dim, hidden=128, depth=2, components=8, mode="score",
                 num_steps=20, output_scale=0.01, *, key):
        if dim != 3072 or mode != "score":
            raise ValueError("ImageMADE requires dim=3072 and mode='score'")
        super().__init__(dim, hidden, depth, components, mode, num_steps, output_scale, key=key)

    def _coordinates(self, x):
        return jnp.asarray(x).reshape(3, 32, 32).transpose(1, 2, 0).reshape(self.dim)

    def metadata(self):
        return {**super().metadata(), "architecture": "image_made", "ordering": "raster_pixels_rgb"}
