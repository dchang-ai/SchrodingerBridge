"""2-D toy experiments (Section 4, Appendix J.1): DSB between a 2-D dataset and a Gaussian
with a *short* horizon (T = N gamma = 0.2), where the first DSB iteration (= classical SGM)
fails and subsequent IPF iterations fix the marginals (Figures 1 and 3 of the paper)."""
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
from dsb.data import sample_pool, toy_dataset
from dsb.networks import ToyMLP
from dsb.made import MADE
from dsb.local_made import LocalMADE
from dsb.train import DSBConfig, run_dsb
from dsb.utils import Logger, save_model_config


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Energy distance between two samples (a proper metric on distributions)."""
    def mean_pdist(a, b):
        d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1) + 1e-12)
        return d.mean()
    return float(2 * mean_pdist(x, y) - mean_pdist(x, x) - mean_pdist(y, y))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="moons")
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--gamma", type=float, default=1e-2)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--forward-hidden", type=int, default=None, help="hold F's MLP width fixed when sweeping B")
    p.add_argument("--architecture", choices=["mlp", "made", "strict", "local"], default="mlp")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--output-scale", type=float, default=0.01)
    p.add_argument("--no-ref-residual", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--max-degree", type=int, default=None)
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--planar", action="store_true")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--prior-std", type=float, default=None, help="default: std of the data (Technique 4)")
    p.add_argument("--prior-scale", type=float, default=1.0, help="multiply the (data-derived) prior std by this factor")
    p.add_argument("--cache-size", type=int, default=10_000)
    p.add_argument("--cache-refresh", type=int, default=1000)
    p.add_argument("--outdir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.architecture != "local" and (args.max_degree is not None or args.radius is not None or args.planar):
        p.error("--max-degree, --radius and --planar require --architecture local")
    outdir = args.outdir or f"runs/toy_{args.dataset}"
    os.makedirs(outdir, exist_ok=True)
    save_model_config(outdir, vars(args), p)

    rng = np.random.default_rng(args.seed)
    pool = jnp.asarray(toy_dataset(args.dataset, 100_000, rng))
    data_mean = float(np.asarray(pool).mean())
    data_std = float(np.asarray(pool).std())
    prior_std = args.prior_std or data_std * args.prior_scale
    cfg = DSBConfig(
        num_steps=args.num_steps, gamma_schedule="constant", gamma=args.gamma,
        prior_mean=data_mean, prior_std=prior_std,        # OU reference with alpha = 1/sigma^2
        num_dsb_iters=args.iters, train_steps=args.steps, batch_size=args.batch,
        cache_size=args.cache_size, cache_refresh=args.cache_refresh, lr=args.lr, ema_decay=0.999,
        ref_residual=not args.no_ref_residual, seed=args.seed, log_every=1000,
        checkpoint_every=args.checkpoint_every,
    )
    ref = cfg.reference()
    gammas = jnp.asarray(ref.gammas)
    logger = Logger(os.path.join(outdir, "log.jsonl"))
    print(f"dataset={args.dataset} N={ref.num_steps} T={ref.horizon:.3f} prior=N({data_mean:.3f},{prior_std:.3f}^2)")

    def sample_data(key, n, data):
        return sample_pool(key, data, n)

    def sample_prior(key, n):
        return cfg.prior_mean + cfg.prior_std * jax.random.normal(key, (n, 2))

    def make_forward_net(key):
        return ToyMLP(dim=2, hidden=args.forward_hidden or args.hidden, key=key)

    def make_backward_net(key):
        if args.architecture != "mlp":
            options = dict(dim=2, hidden=args.hidden, depth=args.depth, components=args.components,
                           mode="strict" if args.architecture == "strict" else "score",
                           num_steps=args.num_steps, output_scale=args.output_scale, key=key)
            if args.architecture == "local":
                return LocalMADE(**options, max_degree=args.max_degree, radius=args.radius, planar=args.planar)
            return MADE(**options)
        return ToyMLP(dim=2, hidden=args.hidden, key=key)

    if args.architecture != "mlp":
        template = make_backward_net(jax.random.PRNGKey(args.seed))
        metadata = template.metadata()
        metadata["allocated_parameters"] = sum(
            int(leaf.size) for leaf in jax.tree_util.tree_leaves(template))
        with open(os.path.join(outdir, "architecture.json"), "w") as fh:
            json.dump(metadata, fh, indent=2)
        if args.architecture == "local":
            with open(os.path.join(outdir, "graph.json"), "w") as fh:
                json.dump(template.graph(), fh)

    def eval_fn(n, b_net, f_net, key):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k1, k2, k3, k4 = jax.random.split(key, 4)
        n_eval = 5000
        xN = sample_prior(k1, n_eval)
        bwd = ipf.simulate(b_net, xN, gammas, k2, forward=False, last_step_noise=False)  # xs[i] = X_{N-1-i}
        x0_data = sample_data(k3, n_eval, pool)
        fwd = ipf.simulate(f_net, x0_data, gammas, k4, forward=True)                    # xs[i] = X_{i+1}
        gen = np.asarray(bwd.xs[-1])
        np.savez_compressed(os.path.join(outdir, f"samples_iter{n:03d}.npz"),
                            generated=gen, data=np.asarray(x0_data),
                            forward=np.asarray(fwd.xs[-1]), prior=np.asarray(xN))
        ed = energy_distance(gen[:2000], np.asarray(x0_data[:2000]))
        # backward endpoint vs data; forward endpoint vs prior
        fwd_end = np.asarray(fwd.xs[-1])
        ed_prior = energy_distance(fwd_end[:2000], np.asarray(xN[:2000]))

        N = ref.num_steps
        idx = [0, N // 4, N // 2, 3 * N // 4, N]
        fig, axes = plt.subplots(2, len(idx), figsize=(3 * len(idx), 6))
        lim = float(np.abs(np.asarray(pool)).max()) * 1.3 + 0.5
        # colour each particle by the angle of its data-space endpoint so transport is visible
        col_f = np.arctan2(np.asarray(x0_data)[:, 1], np.asarray(x0_data)[:, 0])
        col_b = np.arctan2(gen[:, 1], gen[:, 0])
        for j, k in enumerate(idx):
            xf = np.asarray(x0_data) if k == 0 else np.asarray(fwd.xs[k - 1])
            xb = np.asarray(xN) if k == N else np.asarray(bwd.xs[N - 1 - k])
            for row, x, col, name in ((0, xf, col_f, "forward (F)"), (1, xb, col_b, "backward (B)")):
                ax = axes[row, j]
                ax.scatter(x[:, 0], x[:, 1], s=1, c=col, cmap="viridis", alpha=0.6)
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{name}: k={k} (t={float(ref.gammas[:k].sum()):.2f})", fontsize=9)
        fig.suptitle(f"{args.dataset}: DSB iteration {n + 1}   energy dist(gen, data)={ed:.4f}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"iter{n:03d}.png"), dpi=110)
        plt.close(fig)
        m = dict(energy_dist_data=ed, energy_dist_prior=ed_prior)
        with open(os.path.join(outdir, "metrics.jsonl"), "a") as fh:
            fh.write(json.dumps(dict(n=n, **m)) + "\n")
        return m

    run_dsb(cfg, make_forward_net=make_forward_net, make_backward_net=make_backward_net,
            sample_data=sample_data, sample_prior=sample_prior, data=pool,
            outdir=outdir, eval_fn=eval_fn, logger=logger)


if __name__ == "__main__":
    main()
