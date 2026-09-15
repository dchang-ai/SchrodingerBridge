"""FID of the samples dumped by ``train_cifar.py --fid-samples`` (gen_iterXXX.npz), one value
per DSB iteration (cf. Figure 6 of the paper).  Runs with the CPU torch venv (.venv-fid):

    .venv-fid/bin/python scripts/fid.py runs/cifar10 [--ref data/cifar10_train_fid_stats.npz]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import time

import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from scipy import linalg


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^{1/2}) (Heusel et al. 2017).
    (pytorch_fid's version passes ``disp=`` to scipy.linalg.sqrtm, removed in SciPy 1.18.)"""
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


@torch.no_grad()
def activations(model, images_uint8: np.ndarray, batch_size: int = 100) -> np.ndarray:
    out = []
    t0 = time.time()
    for i in range(0, len(images_uint8), batch_size):
        x = torch.from_numpy(images_uint8[i : i + batch_size]).float() / 255.0  # (b,3,32,32) in [0,1]
        out.append(model(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"  {i + len(x)}/{len(images_uint8)} images, {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(out)


def stats(act: np.ndarray):
    return act.mean(0), np.cov(act, rowvar=False)


def load_cifar_train(root: str) -> np.ndarray:
    xs = []
    for i in range(1, 6):
        with open(os.path.join(root, "cifar-10-batches-py", f"data_batch_{i}"), "rb") as fh:
            xs.append(pickle.load(fh, encoding="bytes")[b"data"].reshape(-1, 3, 32, 32))
    return np.concatenate(xs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rundir")
    p.add_argument("--data", default="data")
    p.add_argument("--ref", default="data/cifar10_train_fid_stats.npz")
    p.add_argument("--max-ref", type=int, default=50000)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    model = InceptionV3([3], resize_input=True, normalize_input=True).eval()
    if os.path.exists(args.ref):
        d = np.load(args.ref)
        mu_r, sig_r = d["mu"], d["sigma"]
    else:
        print("computing reference statistics on CIFAR-10 train ...", flush=True)
        imgs = load_cifar_train(args.data)[: args.max_ref]
        mu_r, sig_r = stats(activations(model, imgs))
        np.savez(args.ref, mu=mu_r, sigma=sig_r)

    out_path = os.path.join(args.rundir, "fid.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["n"] for l in open(out_path)}
    for f in sorted(glob.glob(os.path.join(args.rundir, "gen_iter*.npz"))):
        n = int(re.search(r"gen_iter(\d+)", f).group(1))
        if n in done:
            continue
        stats_file = f.replace(".npz", "_fidstats.npz")
        if os.path.exists(stats_file):
            d = np.load(stats_file)
            mu, sig, n_img = d["mu"], d["sigma"], int(d["n"])
        else:
            imgs = np.load(f)["images"]
            mu, sig = stats(activations(model, imgs))
            n_img = len(imgs)
            np.savez(stats_file, mu=mu, sigma=sig, n=n_img)
        fid = frechet_distance(mu, sig, mu_r, sig_r)
        print(f"iter {n}: FID = {fid:.2f}  ({n_img} samples)", flush=True)
        with open(out_path, "a") as fh:
            fh.write(json.dumps({"n": n, "fid": fid, "num_samples": n_img}) + "\n")


if __name__ == "__main__":
    main()
