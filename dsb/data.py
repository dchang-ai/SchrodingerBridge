"""Datasets: CIFAR-10 (python pickles), 2-D toys and the Gaussian test case."""
from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


# --------------------------------------------------------------------------- CIFAR-10
def load_cifar10(root: str, train: bool = True, download: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return images as uint8 (n, 3, 32, 32) and labels (n,)."""
    folder = os.path.join(root, "cifar-10-batches-py")
    if not os.path.isdir(folder):
        if not download:
            raise FileNotFoundError(f"{folder} not found (download=False)")
        os.makedirs(root, exist_ok=True)
        tgz = os.path.join(root, "cifar-10-python.tar.gz")
        if not os.path.exists(tgz):
            urllib.request.urlretrieve(CIFAR_URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(root)
    files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
    xs, ys = [], []
    for f in files:
        with open(os.path.join(folder, f), "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        xs.append(d[b"data"].reshape(-1, 3, 32, 32))
        ys.append(np.asarray(d[b"labels"]))
    return np.concatenate(xs).astype(np.uint8), np.concatenate(ys)


def cifar_to_float(x_uint8):
    """uint8 (…,3,32,32) -> float32 in [-1, 1]."""
    return x_uint8.astype(jnp.float32) / 127.5 - 1.0


def cifar_to_uint8(x):
    return jnp.clip(jnp.round((x + 1.0) * 127.5), 0, 255).astype(jnp.uint8)


def sample_cifar_batch(key, images_uint8, batch_size: int, flip: bool = True):
    """Sample a random batch (device-side), optionally with random horizontal flips."""
    ki, kf = jax.random.split(key)
    idx = jax.random.randint(ki, (batch_size,), 0, images_uint8.shape[0])
    x = cifar_to_float(images_uint8[idx])
    if flip:
        do_flip = jax.random.bernoulli(kf, 0.5, (batch_size,))
        x = jnp.where(do_flip[:, None, None, None], x[..., ::-1], x)
    return x


# --------------------------------------------------------------------------- 2-D toys
def toy_dataset(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """2-D toy distributions (all roughly unit scale)."""
    if name == "moons":
        n1 = n // 2
        t1 = rng.uniform(0, np.pi, n1)
        t2 = rng.uniform(0, np.pi, n - n1)
        a = np.stack([np.cos(t1), np.sin(t1)], 1)
        b = np.stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5], 1)
        x = np.concatenate([a, b]) + 0.08 * rng.standard_normal((n, 2))
        x = (x - [0.5, 0.25]) * 1.5
    elif name == "circles":
        n1 = n // 2
        t = rng.uniform(0, 2 * np.pi, n)
        r = np.concatenate([np.ones(n1), 0.5 * np.ones(n - n1)])
        x = np.stack([r * np.cos(t), r * np.sin(t)], 1) * 2 + 0.05 * rng.standard_normal((n, 2))
    elif name == "scurve":
        t = 3 * np.pi * (rng.uniform(0, 1, n) - 0.5)
        x = np.stack([np.sin(t), np.sign(t) * (np.cos(t) - 1)], 1)
        x = (x + [0, 1]) * [1.5, 1.0] + 0.05 * rng.standard_normal((n, 2))
    elif name == "swissroll":
        t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))
        x = np.stack([t * np.cos(t), t * np.sin(t)], 1) / 7.0 + 0.05 * rng.standard_normal((n, 2))
    elif name == "checkerboard":
        x1 = rng.uniform(-2, 2, n)
        x2 = rng.uniform(0, 1, n) - rng.integers(0, 2, n) * 2 + (np.floor(x1) % 2)
        x = np.stack([x1, x2], 1)
    elif name == "8gaussians":
        centers = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 9)[:-1]]) * 2.5
        idx = rng.integers(0, 8, n)
        x = centers[idx] + 0.2 * rng.standard_normal((n, 2))
    elif name == "pinwheel":
        radial_std, tangential_std, num_classes, rate = 0.3, 0.1, 5, 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
        labels = rng.integers(0, num_classes, n)
        feats = rng.standard_normal((n, 2)) * np.array([radial_std, tangential_std]) + np.array([1.0, 0.0])
        angles = rads[labels] + rate * np.exp(feats[:, 0])
        rot = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)], 1).reshape(n, 2, 2)
        x = np.einsum("ni,nij->nj", feats, rot) * 2.0
    else:
        raise ValueError(name)
    return x.astype(np.float32)


def sample_pool(key, pool, batch_size: int):
    idx = jax.random.randint(key, (batch_size,), 0, pool.shape[0])
    return pool[idx]
