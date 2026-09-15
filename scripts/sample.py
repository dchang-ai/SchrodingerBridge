"""Generate CIFAR-10 samples from a DSB checkpoint (EMA backward network).

    python scripts/sample.py runs/cifar10_N50 [--ckpt iter011_backward.eqx] [--n 64] [--interpolate]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from dsb import ipf
from dsb.data import cifar_to_uint8
from dsb.networks import ReferenceResidual, ScaledIncrement, UNet
from dsb.train import DSBConfig, Nets, output_scales
from dsb.utils import image_grid, save_png


def load(rundir: str, ckpt: str | None, ch: int = 64):
    cfg = DSBConfig(**json.load(open(os.path.join(rundir, "config.json"))))
    ckdir = os.path.join(rundir, "ckpt")
    if ckpt is None:
        ckpt = json.load(open(os.path.join(ckdir, "latest.json")))["file"]
    f_net = b_net = UNet(in_channels=3, ch=ch, ch_mult=(1, 2, 2, 2), num_res_blocks=2, attn_resolutions=(16,),
                         image_size=32, key=jax.random.PRNGKey(0))
    ref = cfg.reference()
    if cfg.scaled_output:
        sf, sb = output_scales(ref)
        f_net, b_net = ScaledIncrement(f_net, sf), ScaledIncrement(b_net, sb)
    if cfg.ref_residual:
        f_net = ReferenceResidual(f_net, ref.gammas, ref.alpha, ref.mean)
    f_params, f_static = eqx.partition(f_net, eqx.is_array)
    b_params, b_static = eqx.partition(b_net, eqx.is_array)
    nets = eqx.tree_deserialise_leaves(os.path.join(ckdir, ckpt), Nets(f_params, f_params, b_params, b_params))
    return cfg, eqx.combine(nets.f_ema, f_static), eqx.combine(nets.b_ema, b_static), ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rundir")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--interpolate", action="store_true", help="latent interpolation between two prior samples (Fig. 13)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg, f_net, b_net, ckpt = load(args.rundir, args.ckpt, args.ch)
    gammas = jnp.asarray(cfg.reference().gammas)
    key = jax.random.PRNGKey(args.seed)
    k1, k2 = jax.random.split(key)
    out = args.out or os.path.join(args.rundir, f"samples_{ckpt.replace('.eqx', '')}_seed{args.seed}.png")
    if args.interpolate:
        za, zb = jax.random.normal(k1, (2, 3, 32, 32))
        lam = jnp.linspace(0, 1, 8)
        xN = (1 - lam)[:, None, None, None] * za + lam[:, None, None, None] * zb
        # the same Gaussian noise along the chain for every lambda -> deterministic embedding
        x0 = ipf.generate(b_net, xN, gammas, k2, last_step_noise=False, shared_noise=True)
        save_png(out, image_grid(np.asarray(cifar_to_uint8(x0)), nrow=8))
    else:
        xN = jax.random.normal(k1, (args.n, 3, 32, 32))
        x0 = ipf.generate(b_net, xN, gammas, k2, last_step_noise=False)
        save_png(out, image_grid(np.asarray(cifar_to_uint8(x0)), nrow=int(np.sqrt(args.n))))
    print("wrote", out)


if __name__ == "__main__":
    main()
