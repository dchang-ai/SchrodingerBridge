"""Networks approximating the DSB mean maps, written in Equinox.

Every network maps ``(k, x) -> increment`` for a *single* example ``x`` (no batch
axis) and an integer step index ``k``; the mean map is ``x + increment``.  The
residual parametrisation matches Appendix E.2.3 (mean-matching works best with
networks of the form x -> x + Phi(x)).  Batch with ``jax.vmap``.

* :class:`ToyMLP` – the fully connected network of Figure 9 (2-D experiments).
* :class:`UNet` – a reduced version of the U-Net of Nichol & Dhariwal (2021)
  used for the image experiments (Appendix J.2), 64 base channels.
"""
from __future__ import annotations

import math
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------- utils
def sinusoidal_embedding(k, dim: int, max_period: float = 10_000.0):
    """Transformer positional encoding of an integer (scalar) step index."""
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = jnp.asarray(k, jnp.float32) * freqs
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)])
    if dim % 2:
        emb = jnp.concatenate([emb, jnp.zeros((1,), emb.dtype)])
    return emb


def zero_module(module: eqx.Module) -> eqx.Module:
    """Return a copy of ``module`` with all float arrays set to zero (DDPM-style zero init)."""
    is_float = lambda x: eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if is_float(x) else x, module)


def scale_module(module: eqx.Module, scale: float) -> eqx.Module:
    is_float = lambda x: eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
    return jax.tree_util.tree_map(lambda x: x * scale if is_float(x) else x, module)


# --------------------------------------------------------------------------- toy MLP
class MLP(eqx.Module):
    layers: list

    def __init__(self, sizes: Sequence[int], *, key):
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [eqx.nn.Linear(a, b, key=k) for a, b, k in zip(sizes[:-1], sizes[1:], keys)]

    def __call__(self, x, final_act: bool = True):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or final_act:
                x = jax.nn.silu(x)
        return x


class ToyMLP(eqx.Module):
    """Figure 9: x -> MLP(1a); k -> PosEnc -> MLP(1b); concat -> MLP(2) -> increment."""

    x_block: MLP
    k_block: MLP
    out_block: MLP
    pe_dim: int

    def __init__(self, dim: int = 2, hidden: int = 128, pe_dim: int = 16, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.pe_dim = pe_dim
        self.x_block = MLP((dim, hidden // 4, hidden // 2), key=k1)
        self.k_block = MLP((pe_dim, hidden // 4, hidden // 2), key=k2)
        out = MLP((hidden, hidden, hidden, dim), key=k3)
        # small output at init so that the mean map starts close to the identity
        out = eqx.tree_at(lambda m: m.layers[-1], out, scale_module(out.layers[-1], 1e-2))
        self.out_block = out

    def __call__(self, k, x):
        hx = self.x_block(x)
        hk = self.k_block(sinusoidal_embedding(k, self.pe_dim))
        return self.out_block(jnp.concatenate([hx, hk]), final_act=False)


# --------------------------------------------------------------------------- U-Net
class TimeEmbedding(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    dim: int

    def __init__(self, dim: int, out_dim: int, *, key):
        k1, k2 = jax.random.split(key)
        self.dim = dim
        self.lin1 = eqx.nn.Linear(dim, out_dim, key=k1)
        self.lin2 = eqx.nn.Linear(out_dim, out_dim, key=k2)

    def __call__(self, k):
        h = sinusoidal_embedding(k, self.dim)
        return self.lin2(jax.nn.silu(self.lin1(h)))


class ResBlock(eqx.Module):
    norm1: eqx.nn.GroupNorm
    conv1: eqx.nn.Conv2d
    temb_proj: eqx.nn.Linear
    norm2: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d
    skip: eqx.nn.Conv2d | None
    dropout: eqx.nn.Dropout

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, dropout: float, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.norm1 = eqx.nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = eqx.nn.Conv2d(in_ch, out_ch, 3, padding=1, key=k1)
        self.temb_proj = eqx.nn.Linear(temb_dim, out_ch, key=k2)
        self.norm2 = eqx.nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = zero_module(eqx.nn.Conv2d(out_ch, out_ch, 3, padding=1, key=k3))
        self.skip = None if in_ch == out_ch else eqx.nn.Conv2d(in_ch, out_ch, 1, key=k4)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(self, x, temb, *, key=None, inference: bool = True):
        h = self.conv1(jax.nn.silu(self.norm1(x)))
        h = h + self.temb_proj(jax.nn.silu(temb))[:, None, None]
        h = jax.nn.silu(self.norm2(h))
        h = self.dropout(h, key=key, inference=inference)
        h = self.conv2(h)
        return h + (x if self.skip is None else self.skip(x))


class AttnBlock(eqx.Module):
    norm: eqx.nn.GroupNorm
    qkv: eqx.nn.Conv2d
    proj: eqx.nn.Conv2d
    heads: int

    def __init__(self, ch: int, heads: int, *, key):
        k1, k2 = jax.random.split(key)
        self.heads = heads
        self.norm = eqx.nn.GroupNorm(min(32, ch), ch)
        self.qkv = eqx.nn.Conv2d(ch, 3 * ch, 1, key=k1)
        self.proj = zero_module(eqx.nn.Conv2d(ch, ch, 1, key=k2))

    def __call__(self, x):
        c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(3, self.heads, c // self.heads, h * w)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (heads, d, hw)
        scale = 1.0 / math.sqrt(c // self.heads)
        attn = jax.nn.softmax(jnp.einsum("hdi,hdj->hij", q, k) * scale, axis=-1)
        out = jnp.einsum("hij,hdj->hdi", attn, v).reshape(c, h, w)
        return x + self.proj(out)


class Downsample(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(self, ch: int, *, key):
        self.conv = eqx.nn.Conv2d(ch, ch, 3, stride=2, padding=1, key=key)

    def __call__(self, x):
        return self.conv(x)


class Upsample(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(self, ch: int, *, key):
        self.conv = eqx.nn.Conv2d(ch, ch, 3, padding=1, key=key)

    def __call__(self, x):
        c, h, w = x.shape
        x = jax.image.resize(x, (c, 2 * h, 2 * w), method="nearest")
        return self.conv(x)


class UNet(eqx.Module):
    """Reduced Nichol & Dhariwal U-Net (Appendix J.2): ``ch`` base channels."""

    temb: TimeEmbedding
    conv_in: eqx.nn.Conv2d
    down: list  # list of levels, each a list of blocks (ResBlock or (ResBlock, AttnBlock))
    downsamples: list
    mid: tuple
    up: list
    upsamples: list
    norm_out: eqx.nn.GroupNorm
    conv_out: eqx.nn.Conv2d
    skip_channels: list = eqx.field(static=True)
    attn_levels: tuple = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int = 3,
        ch: int = 64,
        ch_mult: Sequence[int] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,),
        image_size: int = 32,
        dropout: float = 0.0,
        heads: int = 4,
        *,
        key,
    ):
        keys = iter(jax.random.split(key, 256))
        temb_dim = 4 * ch
        self.temb = TimeEmbedding(ch, temb_dim, key=next(keys))
        self.conv_in = eqx.nn.Conv2d(in_channels, ch, 3, padding=1, key=next(keys))

        levels = len(ch_mult)
        res = image_size
        attn_levels = []
        chans = [ch]
        cur = ch
        down, downsamples = [], []
        for i, m in enumerate(ch_mult):
            use_attn = res in attn_resolutions
            attn_levels.append(use_attn)
            blocks = []
            for _ in range(num_res_blocks):
                rb = ResBlock(cur, ch * m, temb_dim, dropout, key=next(keys))
                cur = ch * m
                ab = AttnBlock(cur, heads, key=next(keys)) if use_attn else None
                blocks.append((rb, ab))
                chans.append(cur)
            down.append(blocks)
            if i != levels - 1:
                downsamples.append(Downsample(cur, key=next(keys)))
                chans.append(cur)
                res //= 2
        self.down, self.downsamples = down, downsamples
        self.attn_levels = tuple(attn_levels)

        self.mid = (
            ResBlock(cur, cur, temb_dim, dropout, key=next(keys)),
            AttnBlock(cur, heads, key=next(keys)),
            ResBlock(cur, cur, temb_dim, dropout, key=next(keys)),
        )

        skip_channels = list(chans)
        up, upsamples = [], []
        for i in reversed(range(levels)):
            m = ch_mult[i]
            use_attn = attn_levels[i]
            blocks = []
            for _ in range(num_res_blocks + 1):
                skip = skip_channels.pop()
                rb = ResBlock(cur + skip, ch * m, temb_dim, dropout, key=next(keys))
                cur = ch * m
                ab = AttnBlock(cur, heads, key=next(keys)) if use_attn else None
                blocks.append((rb, ab))
            up.append(blocks)
            if i != 0:
                upsamples.append(Upsample(cur, key=next(keys)))
        self.up, self.upsamples = up, upsamples
        self.skip_channels = list(chans)

        self.norm_out = eqx.nn.GroupNorm(min(32, cur), cur)
        self.conv_out = zero_module(eqx.nn.Conv2d(cur, in_channels, 3, padding=1, key=next(keys)))

    def __call__(self, k, x, *, key=None, inference: bool = True):
        temb = self.temb(k)
        n_drop = sum(len(b) for b in self.down) + 2 + sum(len(b) for b in self.up)
        dkeys = iter([None] * n_drop if key is None else jax.random.split(key, n_drop))

        h = self.conv_in(x)
        hs = [h]
        for i, blocks in enumerate(self.down):
            for rb, ab in blocks:
                h = rb(h, temb, key=next(dkeys), inference=inference)
                if ab is not None:
                    h = ab(h)
                hs.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)
                hs.append(h)

        rb1, ab, rb2 = self.mid
        h = rb1(h, temb, key=next(dkeys), inference=inference)
        h = ab(h)
        h = rb2(h, temb, key=next(dkeys), inference=inference)

        for i, blocks in enumerate(self.up):
            for rb, ab in blocks:
                h = rb(jnp.concatenate([h, hs.pop()], axis=0), temb, key=next(dkeys), inference=inference)
                if ab is not None:
                    h = ab(h)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        return self.conv_out(jax.nn.silu(self.norm_out(h)))


def count_params(model: eqx.Module) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
