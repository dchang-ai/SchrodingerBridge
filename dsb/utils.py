"""Checkpointing, image grids, logging."""
from __future__ import annotations

import json
import os
import time

import equinox as eqx
import jax
import numpy as np


def save_tree(path: str, tree):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    eqx.tree_serialise_leaves(path, tree)


def load_tree(path: str, like):
    return eqx.tree_deserialise_leaves(path, like)


def save_model_config(outdir: str, config: dict, parser):
    """Compare launch settings after filling defaults absent from older metadata."""
    path = os.path.join(outdir, "model.json")
    comparable = lambda d: {k: v for k, v in d.items() if k not in ("argv", "outdir", "checkpoint_every")}
    if os.path.exists(path):
        with open(path) as fh:
            previous = json.load(fh)
        previous = {**vars(parser.parse_args(previous.get("argv", []))), **previous}
        if comparable(previous) != comparable(config):
            raise ValueError(f"Model configuration differs from {path}. Use a new output directory.")
        return
    with open(path + ".tmp", "w") as fh:
        json.dump(config, fh, indent=2)
    os.replace(path + ".tmp", path)


def image_grid(images_uint8: np.ndarray, nrow: int = 8, pad: int = 1) -> np.ndarray:
    """(n, 3, h, w) uint8 -> (H, W, 3) grid."""
    x = np.asarray(images_uint8)
    n, c, h, w = x.shape
    ncol = int(np.ceil(n / nrow))
    grid = np.full((ncol * (h + pad) + pad, nrow * (w + pad) + pad, c), 255, dtype=np.uint8)
    for i in range(n):
        r, cc = divmod(i, nrow)
        y0, x0 = pad + r * (h + pad), pad + cc * (w + pad)
        grid[y0 : y0 + h, x0 : x0 + w] = x[i].transpose(1, 2, 0)
    return grid


def save_png(path: str, array_hwc: np.ndarray):
    from PIL import Image

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(np.asarray(array_hwc)).save(path)


class Logger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.t0 = time.time()

    def log(self, **kw):
        kw = {k: (float(v) if hasattr(v, "__float__") and not isinstance(v, (int, float)) else v)
              for k, v in kw.items()}
        kw["time"] = round(time.time() - self.t0, 1)
        with open(self.path, "a") as f:
            f.write(json.dumps(kw) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)
