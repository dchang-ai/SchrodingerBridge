"""Collect run outputs into summary figures under figures/ (used by the README).

    .venv/bin/python scripts/plot_results.py
"""
from __future__ import annotations

import glob
import json
import os
import shutil

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)
BETA = (np.sqrt(5.0) - 1.0) / 2.0


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def gaussian_figure():
    runs = {d: read_jsonl(os.path.join(RUNS, d, "metrics.jsonl")) for d in ("gaussian_d5", "gaussian_d50")}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for name, rows in runs.items():
        it = [r["n"] + 1 for r in rows]
        d = name.split("_d")[1]
        axes[0].plot(it, [r["mean0"] for r in rows], "o-", label=f"d={d}")
        axes[1].plot(it, [r["var0"] for r in rows], "o-", label=f"d={d}")
        axes[2].plot(it, [r["cov0"] for r in rows], "o-", label=f"d={d}")
    for ax, truth, title in zip(axes, (0.1, 1.0, BETA), ("mean of X_0", "variance of X_0", "Cov(X_0, X_N)")):
        ax.axhline(truth, ls="--", c="k", label="analytic SB")
        ax.set_title(title + " (first coordinate)")
        ax.set_xlabel("DSB iteration")
    axes[0].legend()
    fig.suptitle("Gaussian check: DSB between N(a,I) and N(-a,I), Brownian reference, T=1/2 (Prop. 46)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "gaussian_convergence.png"), dpi=120)
    plt.close(fig)


def toy_figure():
    dirs = sorted(glob.glob(os.path.join(RUNS, "toy_*")))
    rows = {os.path.basename(d)[4:]: read_jsonl(os.path.join(d, "metrics.jsonl")) for d in dirs}
    rows = {k: v for k, v in rows.items() if v}
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for name, r in rows.items():
        it = [x["n"] + 1 for x in r]
        axes[0].plot(it, [x["energy_dist_data"] for x in r], "o-", label=name)
        axes[1].plot(it, [x["energy_dist_prior"] for x in r], "o-", label=name)
    axes[0].set_title("energy distance: generated X_0 vs data")
    axes[1].set_title("energy distance: forward X_N vs prior")
    for ax in axes:
        ax.set_yscale("log"); ax.set_xlabel("DSB iteration"); ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "toy_energy_distance.png"), dpi=120)
    plt.close(fig)
    # first vs last iteration panels for each toy
    for name in rows:
        pngs = sorted(glob.glob(os.path.join(RUNS, f"toy_{name}", "iter*.png")))
        if pngs:
            shutil.copy(pngs[0], os.path.join(FIGS, f"toy_{name}_iter01.png"))
            shutil.copy(pngs[-1], os.path.join(FIGS, f"toy_{name}_iter{len(pngs):02d}.png"))


def cifar_figures():
    dirs = sorted(glob.glob(os.path.join(RUNS, "cifar10_*")))
    fid = {os.path.basename(d): read_jsonl(os.path.join(d, "fid.jsonl")) for d in dirs}
    metrics = {os.path.basename(d): read_jsonl(os.path.join(d, "metrics.jsonl")) for d in dirs}
    if any(fid.values()):
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for name, rows in fid.items():
            if rows:
                rows = sorted(rows, key=lambda r: r["n"])
                ax.plot([r["n"] + 1 for r in rows], [r["fid"] for r in rows], "o-", label=name.replace("cifar10_", ""))
        ax.set_xlabel("DSB iteration"); ax.set_ylabel("FID (5k samples vs train)"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title("CIFAR-10: FID vs DSB iteration")
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, "cifar_fid.png"), dpi=120); plt.close(fig)
    if any(metrics.values()):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for name, rows in metrics.items():
            if rows:
                it = [r["n"] + 1 for r in rows]
                axes[0].plot(it, [r["fwd_end_std"] for r in rows], "o-", label=name.replace("cifar10_", ""))
                axes[1].plot(it, [r["fwd_end_kurtosis"] for r in rows], "o-", label=name.replace("cifar10_", ""))
        axes[0].axhline(1.0, ls="--", c="k"); axes[0].set_title("std of forward endpoint X_N (prior: 1)")
        axes[1].axhline(3.0, ls="--", c="k"); axes[1].set_title("kurtosis of forward endpoint X_N (Gaussian: 3)")
        for ax in axes:
            ax.set_xlabel("DSB iteration"); ax.grid(alpha=0.3)
        axes[0].legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, "cifar_forward_endpoint.png"), dpi=120); plt.close(fig)
    for d in dirs:
        name = os.path.basename(d)
        for kind in ("samples", "trajectory", "forward"):
            pngs = sorted(glob.glob(os.path.join(d, f"{kind}_iter*.png")))
            if pngs:
                shutil.copy(pngs[0], os.path.join(FIGS, f"{name}_{kind}_iter01.png"))
                shutil.copy(pngs[-1], os.path.join(FIGS, f"{name}_{kind}_iter{len(pngs):02d}.png"))
        if os.path.exists(os.path.join(d, "data.png")):
            shutil.copy(os.path.join(d, "data.png"), os.path.join(FIGS, "cifar10_data.png"))


if __name__ == "__main__":
    gaussian_figure()
    toy_figure()
    cifar_figures()
    print("figures:", sorted(os.listdir(FIGS)))
