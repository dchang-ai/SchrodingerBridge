"""Gaussian sanity check (Section 4 / Appendix G.2): the Schrödinger bridge between
p_data = N(a, I) and p_prior = N(-a, I) w.r.t. a Brownian reference over T = 1/2 is
Gaussian with Cov(X_0, X_N) = beta I, beta = (sqrt(5) - 1)/2.  We check that DSB recovers
the mean, variance and this covariance."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from dsb import ipf
from dsb.networks import ToyMLP
from dsb.made import MADE
from dsb.train import DSBConfig, run_dsb
from dsb.utils import Logger, save_model_config

BETA = (np.sqrt(5.0) - 1.0) / 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=5)
    p.add_argument("--a", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--architecture", choices=["mlp", "made", "strict"], default="mlp")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--output-scale", type=float, default=0.025)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--outdir", default="runs/gaussian")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    save_model_config(args.outdir, vars(args), p)

    d, a = args.dim, args.a
    cfg = DSBConfig(
        num_steps=20, gamma_schedule="constant", gamma=1.0 / 40.0,  # T = 1/2 -> unit transition variance
        prior_mean=-a, prior_std=1.0, ref_alpha=0.0,                # Brownian reference
        num_dsb_iters=args.iters, train_steps=args.steps, batch_size=128,
        cache_size=2000, cache_refresh=500, lr=1e-4, ema_decay=0.999, ref_residual=True, seed=args.seed, log_every=500,
        checkpoint_every=args.checkpoint_every,
    )
    ref = cfg.reference()
    gammas = jnp.asarray(ref.gammas)
    logger = Logger(os.path.join(args.outdir, "log.jsonl"))

    def sample_data(key, n, data):
        return a + jax.random.normal(key, (n, d))

    def sample_prior(key, n):
        return -a + jax.random.normal(key, (n, d))

    def make_forward_net(key):
        return ToyMLP(dim=d, hidden=args.hidden, key=key)

    def make_backward_net(key):
        if args.architecture != "mlp":
            return MADE(dim=d, hidden=args.hidden, depth=args.depth, components=args.components,
                        mode="score" if args.architecture == "made" else "strict",
                        num_steps=cfg.num_steps, output_scale=args.output_scale, key=key)
        return ToyMLP(dim=d, hidden=args.hidden, key=key)

    def eval_fn(n, b_net, f_net, key):
        k1, k2 = jax.random.split(key)
        xN = sample_prior(k1, 20000)
        x0 = ipf.generate(b_net, xN, gammas, k2, last_step_noise=True)
        x0, xN = np.asarray(x0), np.asarray(xN)
        mean = x0.mean(0)
        var = x0.var(0)
        cov = np.mean((x0 - x0.mean(0)) * (xN - xN.mean(0)), axis=0)
        m = dict(mean0=float(mean[0]), var0=float(var[0]), cov0=float(cov[0]),
                 mean_err=float(np.abs(mean - a).mean()), var_err=float(np.abs(var - 1).mean()),
                 cov_err=float(np.abs(cov - BETA).mean()))
        with open(os.path.join(args.outdir, "metrics.jsonl"), "a") as fh:
            fh.write(json.dumps(dict(n=n, **m)) + "\n")
        return m

    print(f"target: mean={a} var=1 cov={BETA:.4f}; N={ref.num_steps} T={ref.horizon:.3f}")
    run_dsb(cfg, make_forward_net=make_forward_net, make_backward_net=make_backward_net,
            sample_data=sample_data, sample_prior=sample_prior, data=jnp.zeros(()),
            outdir=args.outdir, eval_fn=eval_fn, logger=logger)


if __name__ == "__main__":
    main()
