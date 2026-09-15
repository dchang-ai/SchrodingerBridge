"""Diffusion Schrödinger Bridge on CIFAR-10 (Appendix J.2 configuration adapted to CIFAR-10).

p_data: CIFAR-10 training images scaled to [-1, 1] (d = 3072), p_prior = N(0, I).
Reference dynamics: OU process towards the prior, symmetric gamma schedule.
Networks: reduced U-Net (64 channels) for the forward and backward mean maps.
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
from dsb.data import cifar_to_uint8, load_cifar10, sample_cifar_batch
from dsb.networks import UNet, count_params
from dsb.train import DSBConfig, run_dsb, run_dsb_parallel
from dsb.utils import Logger, image_grid, save_png


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--outdir", default="runs/cifar10")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--gamma-max", type=float, default=0.1)
    p.add_argument("--gamma-power", type=float, default=1.0, help="ramp exponent of the symmetric gamma schedule (1 = paper)")
    p.add_argument("--gamma-schedule", default="symmetric", choices=["symmetric", "geometric"])
    p.add_argument("--iters", type=int, default=10, help="number of DSB (IPF) iterations L")
    p.add_argument("--steps", type=int, default=5000, help="gradient steps per half-step")
    p.add_argument("--first-steps", type=int, default=None, help="steps for the first backward half-step")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--cache-size", type=int, default=512)
    p.add_argument("--cache-refresh", type=int, default=400)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=0, help="linear lr warm-up steps (restarts with the optimizer at every half-step)")
    p.add_argument("--loss-weighting", default="none", choices=["none", "gamma"])
    p.add_argument("--scaled-output", action="store_true", help="increment = sqrt(2 gamma_k) * net(k, x)")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--fid-samples", type=int, default=0, help="if > 0, dump this many generated samples per DSB iteration as .npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--single", action="store_true", help="use the simple single-device driver instead of the data-parallel one")
    p.add_argument("--bf16", action="store_true", help="bfloat16 network evaluation (master weights, loss and chain stay float32)")
    args = p.parse_args()

    cfg = DSBConfig(
        num_steps=args.num_steps, gamma_schedule=args.gamma_schedule, gamma_min=1e-5, gamma_max=args.gamma_max, gamma_power=args.gamma_power,
        prior_mean=0.0, prior_std=1.0,
        num_dsb_iters=args.iters, train_steps=args.steps, first_train_steps=args.first_steps,
        batch_size=args.batch, cache_size=args.cache_size, cache_refresh=args.cache_refresh,
        lr=args.lr, warmup=args.warmup, grad_clip=1.0, ema_decay=0.999, loss_weighting=args.loss_weighting, scaled_output=args.scaled_output,
        use_dropout=args.dropout > 0, ref_residual=True, seed=args.seed, log_every=100,
    )
    ref = cfg.reference()
    gammas = jnp.asarray(ref.gammas)
    A, S = ref.marginal_coefficients()
    os.makedirs(args.outdir, exist_ok=True)
    logger = Logger(os.path.join(args.outdir, "log.jsonl"))
    print(f"N={ref.num_steps} T={ref.horizon:.3f}; reference endpoint: X_N ~ {A[-1]:.3f} X_0 + N(0, {S[-1]:.3f}^2)", flush=True)
    print("devices:", jax.devices(), flush=True)

    images, _ = load_cifar10(args.data, train=True, download=False)
    images = jnp.asarray(images)  # uint8 on device (150 MB)
    print("cifar10 train:", images.shape, flush=True)

    def sample_data(key, n, data):
        return sample_cifar_batch(key, data, n, flip=True)

    def sample_prior(key, n):
        return jax.random.normal(key, (n, 3, 32, 32))

    def make_net(key):
        return UNet(in_channels=3, ch=args.ch, ch_mult=(1, 2, 2, 2), num_res_blocks=2,
                    attn_resolutions=(16,), image_size=32, dropout=args.dropout, key=key)

    print("params per network:", count_params(make_net(jax.random.PRNGKey(0))), flush=True)
    with open(os.path.join(args.outdir, "model.json"), "w") as fh:
        json.dump(dict(ch=args.ch, dropout=args.dropout, bf16=args.bf16, single=args.single, argv=sys.argv[1:]), fh, indent=2)
    grid = image_grid(np.asarray(images[:64]))
    save_png(os.path.join(args.outdir, "data.png"), grid)

    @eqx.filter_jit
    def _generate(b_net, xN, key):
        return ipf.generate(b_net, xN, gammas, key, last_step_noise=False)

    def eval_fn(n, b_net, f_net, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        xN = sample_prior(k1, args.eval_samples)
        traj = ipf.simulate(b_net, xN, gammas, k2, forward=False, last_step_noise=False)
        x0 = traj.xs[-1]
        save_png(os.path.join(args.outdir, f"samples_iter{n:03d}.png"), image_grid(np.asarray(cifar_to_uint8(x0))))
        # trajectory strip for the first 8 samples: k = N, 3N/4, N/2, N/4, 0
        N = ref.num_steps
        ks = [N, 3 * N // 4, N // 2, N // 4, 0]
        rows = [np.asarray(cifar_to_uint8(xN[:8] if k == N else traj.xs[N - 1 - k][:8])) for k in ks]
        save_png(os.path.join(args.outdir, f"trajectory_iter{n:03d}.png"), image_grid(np.concatenate(rows), nrow=8))
        # forward chain from data (does F^n reach the prior?)
        x0d = sample_data(k3, 512, images)
        fwd = ipf.simulate(f_net, x0d, gammas, k4, forward=True)
        xf = np.asarray(fwd.xs[-1])
        rows = [np.asarray(cifar_to_uint8(x0d[:8] if k == 0 else fwd.xs[k - 1][:8])) for k in [0, N // 4, N // 2, 3 * N // 4, N]]
        save_png(os.path.join(args.outdir, f"forward_iter{n:03d}.png"), image_grid(np.concatenate(rows), nrow=8))
        m = dict(gen_mean=float(np.asarray(x0).mean()), gen_std=float(np.asarray(x0).std()),
                 fwd_end_mean=float(xf.mean()), fwd_end_std=float(xf.std()),
                 fwd_end_kurtosis=float(((xf - xf.mean()) ** 4).mean() / xf.var() ** 2))
        if args.fid_samples > 0:
            outs = []
            for i in range(0, args.fid_samples, 500):
                kk = jax.random.fold_in(key, i)
                ka, kb = jax.random.split(kk)
                xg = _generate(b_net, sample_prior(ka, min(500, args.fid_samples - i)), kb)
                outs.append(np.asarray(cifar_to_uint8(xg)))
            np.savez_compressed(os.path.join(args.outdir, f"gen_iter{n:03d}.npz"), images=np.concatenate(outs))
        with open(os.path.join(args.outdir, "metrics.jsonl"), "a") as fh:
            fh.write(json.dumps(dict(n=n, **m)) + "\n")
        return m

    common = dict(make_net=make_net, sample_data=sample_data, sample_prior=sample_prior, data=images,
                  outdir=args.outdir, eval_fn=eval_fn, logger=logger)
    if args.single:
        run_dsb(cfg, **common)
    else:
        run_dsb_parallel(cfg, compute_dtype=jnp.bfloat16 if args.bf16 else None, **common)


if __name__ == "__main__":
    main()
