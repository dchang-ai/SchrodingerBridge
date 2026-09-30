"""Spatially weight-shared causal MADE density for CIFAR-10."""
from __future__ import annotations

import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from .made import ImageMADE
from .networks import TimeEmbedding, scale_module


class _MaskedConv2d(eqx.Module):
    """A Conv2d whose causal mask is reapplied on every call."""

    conv: eqx.nn.Conv2d
    kind: str = eqx.field(static=True)
    components: int = eqx.field(static=True)
    def __init__(self, in_channels, out_channels, kernel_size, *, kind, components=1, key):
        padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(size // 2 for size in kernel_size)
        self.conv = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, key=key
        )
        self.kind = kind
        self.components = components

    def _mask(self):
        out_channels, in_channels, kh, kw = self.conv.weight.shape
        if self.kind == "vertical_first":
            return (jnp.arange(kh)[:, None] == 0)[None, None, :, :]
        if self.kind == "vertical":
            return (jnp.arange(kh)[:, None] <= 1)[None, None, :, :]
        offsets = jnp.arange(kw)[None, None, None, :]
        output_groups = jnp.arange(out_channels)[:, None, None, None] // (out_channels // 3)
        input_channels = jnp.arange(in_channels)[None, :, None, None]
        if self.kind == "horizontal_first":
            center = input_channels < output_groups
        elif self.kind == "horizontal":
            center = input_channels // (in_channels // 3) <= output_groups
        elif self.kind == "output":
            output_groups = jnp.arange(out_channels)[:, None, None, None] // (3 * self.components)
            center = input_channels // (in_channels // 3) <= output_groups
        else:
            raise ValueError(f"unknown convolution mask kind: {self.kind}")
        center_offset = kw // 2
        return (offsets < center_offset) | ((offsets == center_offset) & center)

    def __call__(self, x):
        masked = eqx.tree_at(
            lambda layer: layer.weight,
            self.conv,
            self.conv.weight * self._mask(),
        )
        return masked(x)


class _CausalStage(eqx.Module):
    vertical: _MaskedConv2d
    horizontal: _MaskedConv2d
    vertical_to_horizontal: eqx.nn.Conv2d
    vertical_time: eqx.nn.Linear
    horizontal_time: eqx.nn.Linear
    residual: bool = eqx.field(static=True)

    def __init__(self, in_channels, hidden, *, first, temb_dim, key):
        kv, kh, kproj, kvt, kht = jax.random.split(key, 5)
        self.vertical = _MaskedConv2d(
            in_channels,
            hidden,
            3,
            kind="vertical_first" if first else "vertical",
            key=kv,
        )
        self.horizontal = _MaskedConv2d(
            in_channels,
            hidden,
            (1, 3),
            kind="horizontal_first" if first else "horizontal",
            key=kh,
        )
        self.vertical_to_horizontal = eqx.nn.Conv2d(hidden, hidden, 1, key=kproj)
        self.vertical_time = eqx.nn.Linear(temb_dim, hidden, key=kvt)
        self.horizontal_time = eqx.nn.Linear(temb_dim, hidden, key=kht)
        self.residual = not first

    def __call__(self, vertical, horizontal, temb):
        v = self.vertical(vertical) + self.vertical_time(temb)[:, None, None]
        v = jax.nn.silu(v)
        if self.residual:
            v = vertical + v
        h = (
            self.horizontal(horizontal)
            + self.vertical_to_horizontal(v)
            + self.horizontal_time(temb)[:, None, None]
        )
        h = jax.nn.silu(h)
        if self.residual:
            h = horizontal + h
        return v, h


class ConvMADE(ImageMADE):
    """CIFAR normalized autoregressive density with convolutional weight sharing.

    ``depth`` is the number of vertical/horizontal causal stages.  The first
    stage consumes RGB input with strict masks, while later stages use same-rank
    residuals.  Hidden channels are split into three RGB-order groups.
    """

    _log_scale_bound: ClassVar[float] = 7.0
    layers: tuple[_CausalStage, ...]
    temb: TimeEmbedding
    head: _MaskedConv2d

    def __init__(self, dim, hidden=96, depth=8, components=4, mode="score",
                 num_steps=20, output_scale=0.01, *, key):
        values = (dim, hidden, depth, components, num_steps)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
            raise ValueError("dim, hidden, depth, components and num_steps must be positive integers")
        if dim != 3072 or mode != "score":
            raise ValueError("ConvMADE requires dim=3072 and mode='score'")
        if hidden % 3:
            raise ValueError("hidden must be divisible by 3")
        if not math.isfinite(output_scale):
            raise ValueError("output_scale must be finite")

        self.dim, self.hidden, self.depth = dim, hidden, depth
        self.components, self.mode = components, mode
        self.num_steps, self.output_scale = num_steps, float(output_scale)
        self.gain = jnp.full((num_steps + 1,), output_scale, dtype=jnp.float32)

        keys = jax.random.split(key, depth + 2)
        self.temb = TimeEmbedding(16, hidden, key=keys[0])
        self.layers = tuple(
            _CausalStage(
                3 if index == 0 else hidden,
                hidden,
                first=index == 0,
                temb_dim=hidden,
                key=keys[index + 1],
            )
            for index in range(depth)
        )
        self.head = scale_module(
            _MaskedConv2d(hidden, 9 * components, 1, kind="output", components=components, key=keys[-1]),
            1e-2,
        )

    def conditional_parameters(self, k, x):
        """Return raster-ordered ``(logits, locations, log_scales)``."""
        image = jnp.asarray(x).reshape(3, 32, 32)
        temb = self.temb(k)
        vertical = horizontal = image
        for layer in self.layers:
            vertical, horizontal = layer(vertical, horizontal, temb)
        raw = self.head(horizontal)
        raw = raw.reshape(3, 3, self.components, 32, 32).transpose(3, 4, 0, 1, 2)
        raw = raw.reshape(self.dim, 3, self.components)
        return raw[:, 0], raw[:, 1], self._log_scale_bound * jnp.tanh(raw[:, 2] / self._log_scale_bound)

    def metadata(self):
        return {
            **super().metadata(),
            "architecture": "conv_made",
            "ordering": "raster_pixels_rgb",
            "weight_sharing": "spatial_convolutional",
            "causal_streams": "vertical_and_horizontal",
            "depth_semantics": "number_of_causal_vertical_horizontal_stages",
            "hidden_group_count": 3,
            "hidden_rank_assignment": "vertical_row_ranks_and_horizontal_rgb_groups",
        }
