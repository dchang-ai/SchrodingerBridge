"""FID of the samples dumped by ``train_cifar.py --fid-samples`` (gen_iterXXX.npz), one value
per DSB iteration (cf. Figure 6 of the paper).  Runs with the CPU torch venv (.venv-fid):

    .venv-fid/bin/python scripts/fid.py runs/cifar10 [--ref data/cifar10_train_fid_stats.npz]

    .venv-fid/bin/python scripts/fid.py runs/cifar10 --watch --expected-iters 10

Watch mode requires checkpoint-committed metrics rows for iterations 0 through 9.
Statistics and FID caches are checked against sample and reference checksums.
Run only one evaluator per run directory.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import re
import tempfile
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


def _atomic_write(path, write):
    """Use the checkpoint writer's publication protocol without importing JAX."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _committed_rows(path, limit=-1):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as fh:
        # A writer may still be appending the final line, including a UTF-8 character.
        return [json.loads(line) for line in fh.read(limit).splitlines(keepends=True)
                if line.endswith(b"\n")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rundir")
    p.add_argument("--data", default="data")
    p.add_argument("--ref", default="data/cifar10_train_fid_stats.npz")
    p.add_argument("--max-ref", type=int, default=50000)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--watch", action="store_true", help="wait for checkpoint-committed sample files")
    p.add_argument("--expected-iters", type=int, help="with --watch, require every index 0..N-1")
    p.add_argument("--poll-seconds", type=float, default=60, help="watch polling interval (default: 60)")
    args = p.parse_args()
    if args.watch != (args.expected_iters is not None):
        p.error("--watch and --expected-iters must be supplied together")
    if args.expected_iters is not None and args.expected_iters <= 0:
        p.error("--expected-iters must be positive")
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        p.error("--poll-seconds must be finite and positive")
    torch.set_num_threads(args.threads)

    model = None
    if os.path.exists(args.ref):
        with np.load(args.ref) as d:
            mu_r, sig_r = d["mu"], d["sigma"]
    else:
        print("computing reference statistics on CIFAR-10 train ...", flush=True)
        model = InceptionV3([3], resize_input=True, normalize_input=True).eval()
        imgs = load_cifar_train(args.data)[: args.max_ref]
        mu_r, sig_r = stats(activations(model, imgs))
        _atomic_write(args.ref, lambda fh: np.savez(fh, mu=mu_r, sigma=sig_r))

    with open(args.ref, "rb") as fh:
        reference_sha256 = hashlib.file_digest(fh, "sha256").hexdigest()
    out_path = os.path.join(args.rundir, "fid.jsonl")
    rows = {row["n"]: row for row in _committed_rows(out_path)}
    done = set()
    required = set(range(args.expected_iters)) if args.watch else set()
    while True:
        committed = None
        if args.watch:
            resume_path = os.path.join(args.rundir, "ckpt", "resume.json")
            limit = 0
            if os.path.exists(resume_path):
                with open(resume_path) as fh:
                    limit = json.load(fh)["logs"].get("metrics.jsonl", 0)
            committed = {row["n"] for row in _committed_rows(
                os.path.join(args.rundir, "metrics.jsonl"), limit)}
        for f in sorted(glob.glob(os.path.join(args.rundir, "gen_iter*.npz"))):
            match = re.fullmatch(r"gen_iter([0-9]+)\.npz", os.path.basename(f))
            if match is None:
                continue
            n = int(match.group(1))
            if n in done or (args.watch and (n not in required or n not in committed)):
                continue
            with open(f, "rb") as fh:
                sample_sha256 = hashlib.file_digest(fh, "sha256").hexdigest()
            previous = rows.get(n, {})
            if (previous.get("sample_sha256") == sample_sha256
                    and previous.get("reference_sha256") == reference_sha256):
                done.add(n)
                continue
            stats_file = f.removesuffix(".npz") + "_fidstats.npz"
            mu = None
            if os.path.exists(stats_file):
                with np.load(stats_file) as d:
                    if str(d.get("sample_sha256", "")) == sample_sha256:
                        mu, sig, n_img = d["mu"], d["sigma"], int(d["n"])
            if mu is None:
                with np.load(f) as d:
                    imgs = d["images"]
                if model is None:
                    model = InceptionV3([3], resize_input=True, normalize_input=True).eval()
                mu, sig = stats(activations(model, imgs))
                n_img = len(imgs)
                _atomic_write(stats_file, lambda fh: np.savez(
                    fh, mu=mu, sigma=sig, n=n_img, sample_sha256=sample_sha256))
            fid = frechet_distance(mu, sig, mu_r, sig_r)
            rows[n] = {"n": n, "fid": fid, "num_samples": n_img,
                       "sample_sha256": sample_sha256, "reference_sha256": reference_sha256}
            journal = "".join(json.dumps(rows[i]) + "\n" for i in sorted(rows)).encode()
            _atomic_write(out_path, lambda fh: fh.write(journal))
            done.add(n)
            print(f"iter {n}: FID = {fid:.2f}  ({n_img} samples)", flush=True)
        if not args.watch or required.issubset(done & committed):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
