"""Single-GPU efficiency audit of the DSB training loop (task 1 of the performance review).

Times, on one device, the pieces of ``dsb.train.run_dsb`` exactly as they are dispatched
there (``eqx.filter_jit`` of ``ipf.make_train_step`` / ``ipf.simulate`` / ``ipf.sample_cache``):

* forward pass ``jax.vmap(unet)`` at batch 128 (+ achieved TFLOP/s from XLA's cost analysis);
* train step at batch 128 / 256 / 512, default (TF32) precision, ``highest`` (true fp32)
  precision and bf16 mixed precision (:func:`dsb.parallel.mixed_precision`);
* the cache refresh ``simulate`` (N steps x 2 network evaluations at batch M), fp32 / bf16;
* per-step Python/dispatch overhead: the ``run_dsb`` loop (split + sample_cache + train_step)
  versus a fused sample-and-step, and the pure dispatch cost of the small kernels;
* the evaluation sample dump (``generate`` at batch 500 + ``np.savez_compressed``).

Usage (inside a Slurm job, see slurm/run.sbatch):
    python scripts/bench_single.py [--ch 64 --num-steps 50 --steps 30 --cache-size 512]
    python scripts/bench_single.py --quick          # only train step @128 and simulate (for XLA-flag A/B)
    JAX_PLATFORMS=cpu python scripts/bench_single.py --tiny   # smoke test on CPU
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from dsb import ipf, parallel
from dsb.data import cifar_to_float, cifar_to_uint8
from dsb.networks import UNet, count_params
from dsb.schedule import OUReference, symmetric_gammas

RESULTS = []


def block(x):
    return jax.block_until_ready(x)


def record(name, ms, **extra):
    row = dict(name=name, ms=round(ms, 3), **extra)
    RESULTS.append(row)
    extras = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in extra.items())
    print(f"[bench] {name:48s} {ms:9.2f} ms  {extras}", flush=True)


def timeit(fn, n, *, chain=None):
    """Average wall time per call of ``fn()`` over ``n`` calls after 1 compile + 2 warm-up calls.
    ``chain`` (state-in/state-out) makes successive calls data dependent, as in training."""
    out = block(fn())
    for _ in range(2):
        out = block(fn())
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    block(out)
    return 1e3 * (time.perf_counter() - t0) / n


def flops_of(jitted, *args):
    try:
        cost = jitted.lower(*args).compile().cost_analysis()
        cost = cost[0] if isinstance(cost, (list, tuple)) else cost
        return float(cost.get("flops", float("nan")))
    except Exception as e:  # cost analysis is best effort
        print("[bench] cost_analysis unavailable:", type(e).__name__, e)
        return float("nan")


def peak_gb():
    try:
        return round(jax.local_devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30, 2)
    except Exception:
        return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--steps", type=int, default=30, help="timed iterations per measurement")
    p.add_argument("--cache-size", type=int, default=512)
    p.add_argument("--batches", default="128,256,512")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--tiny", action="store_true", help="CPU smoke test (ch=8, N=4, tiny batches)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.tiny:
        args.ch, args.num_steps, args.steps, args.cache_size, args.batches = 8, 4, 2, 16, "4,8"

    print("devices:", jax.devices(), "| jax", jax.__version__, "| XLA_FLAGS=", os.environ.get("XLA_FLAGS", ""), flush=True)
    ref = OUReference(symmetric_gammas(args.num_steps, 1e-5, 0.1, 2.0), alpha=1.0)
    gammas = jnp.asarray(ref.gammas)
    N = ref.num_steps
    key = jax.random.PRNGKey(0)
    net = UNet(ch=args.ch, ch_mult=(1, 2, 2, 2), num_res_blocks=2, attn_resolutions=(16,), image_size=32, key=key)
    n_params = count_params(net)
    print(f"UNet ch={args.ch}: {n_params/1e6:.2f}M params, N={N}", flush=True)
    params, static = eqx.partition(net, eqx.is_array)
    opt = ipf.make_optimizer(1e-4, 1.0, 0)
    state = ipf.init_train_state(net, opt)
    # random "CIFAR" images on device (same cost as the real ones; avoids I/O in the benchmark)
    n_img = 50000 if not args.tiny else 256
    images = jnp.asarray(np.random.default_rng(0).integers(0, 256, (n_img, 3, 32, 32), dtype=np.uint8))

    def sample_data(k, n):
        idx = jax.random.randint(k, (n,), 0, images.shape[0])
        return cifar_to_float(images[idx])

    def sample_prior(k, n):
        return jax.random.normal(k, (n, 3, 32, 32))

    def batch(k, b):
        k1, k2, k3 = jax.random.split(k, 3)
        return jax.random.randint(k1, (b,), 1, N + 1), sample_prior(k2, b), 0.1 * sample_prior(k3, b)

    batches = [int(b) for b in args.batches.split(",")]
    b0 = batches[0]
    train_step = ipf.make_train_step(opt, static, 0.999, False)

    # ---------------------------------------------------------------- forward pass
    if not args.quick:
        ks, xs, _ = batch(key, b0)
        fwd = jax.jit(lambda prm, k, x: jax.vmap(eqx.combine(prm, static))(k, x))
        ms = timeit(lambda: fwd(params, ks, xs), args.steps)
        fl = flops_of(fwd, params, ks, xs)
        record(f"forward vmap(unet) B={b0}", ms, samples_per_s=b0 / ms * 1e3, tflops=fl / ms / 1e9, gflop_per_call=fl / 1e9)

    # ---------------------------------------------------------------- train step, default precision (TF32 on Ampere)
    step_flops = {}
    for b in batches if not args.quick else [b0]:
        ks, xs, tg = batch(key, b)
        ms = timeit(lambda: train_step(state, ks, xs, tg, None, key), args.steps)
        fl = flops_of(jax.jit(lambda s, k, x, t, kk: train_step(s, k, x, t, None, kk)), state, ks, xs, tg, key)
        step_flops[b] = fl
        record(f"train_step f32(default/TF32) B={b}", ms, it_per_s=1e3 / ms, samples_per_s=b / ms * 1e3, tflops=fl / ms / 1e9, peak_gb=peak_gb())

    # ---------------------------------------------------------------- train step, highest precision (true fp32 matmuls/convs)
    if not args.quick:
        ks, xs, tg = batch(key, b0)
        with jax.default_matmul_precision("highest"):
            step_hi = ipf.make_train_step(opt, static, 0.999, False)
            ms = timeit(lambda: step_hi(state, ks, xs, tg, None, key), args.steps)
        record(f"train_step f32(highest) B={b0}", ms, it_per_s=1e3 / ms, samples_per_s=b0 / ms * 1e3)

    # ---------------------------------------------------------------- train step, bf16 mixed precision
    dp1 = parallel.DataParallel(jax.devices()[:1])
    state1 = dp1.replicate(state)
    for b in batches if not args.quick else [b0]:
        ks, xs, tg = batch(key, b)
        step_bf = parallel.make_train_step(dp1, opt, static, 0.999, False, compute_dtype=jnp.bfloat16)
        ms = timeit(lambda: step_bf(state1, ks, xs, tg, key), args.steps)
        record(f"train_step bf16-mixed B={b}", ms, it_per_s=1e3 / ms, samples_per_s=b / ms * 1e3, tflops=step_flops.get(b, float("nan")) / ms / 1e9, peak_gb=peak_gb())
    if not args.quick:
        # shard_map with a single device vs. plain eqx.filter_jit (should be identical)
        ks, xs, tg = batch(key, b0)
        step_sm = parallel.make_train_step(dp1, opt, static, 0.999, False)
        ms = timeit(lambda: step_sm(state1, ks, xs, tg, key), args.steps)
        record(f"train_step f32 via shard_map(1 dev) B={b0}", ms, it_per_s=1e3 / ms)

    # ---------------------------------------------------------------- cache refresh: simulate (N steps x 2 evals at batch M)
    @eqx.filter_jit
    def backward_cache(ema, k, m):
        k1, k2 = jax.random.split(k)
        return ipf.simulate(eqx.combine(ema, static), sample_prior(k1, m), gammas, k2, forward=False)

    @eqx.filter_jit
    def forward_cache(ema, k, m):
        k1, k2 = jax.random.split(k)
        return ipf.simulate(eqx.combine(ema, static), sample_data(k1, m), gammas, k2, forward=True)

    @eqx.filter_jit
    def backward_cache_bf16(ema, k, m):
        k1, k2 = jax.random.split(k)
        return ipf.simulate(parallel.mixed_precision(eqx.combine(ema, static)), sample_prior(k1, m), gammas, k2, forward=False)

    M = args.cache_size
    n_sim = max(2, args.steps // 10)
    ms = timeit(lambda: backward_cache(state.ema_params, key, M), n_sim)
    evals = 2 * N * M
    record(f"simulate backward M={M} N={N} f32", ms, sample_evals_per_s=evals / ms * 1e3, per_400_steps_overhead_s=ms / 1e3, peak_gb=peak_gb())
    if not args.quick:
        ms = timeit(lambda: forward_cache(state.ema_params, key, M), n_sim)
        record(f"simulate forward(data) M={M} N={N} f32", ms, sample_evals_per_s=evals / ms * 1e3)
        ms = timeit(lambda: backward_cache_bf16(state.ema_params, key, M), n_sim)
        record(f"simulate backward M={M} N={N} bf16", ms, sample_evals_per_s=evals / ms * 1e3)
        for m2 in (2 * M, 4 * M):
            ms = timeit(lambda: backward_cache(state.ema_params, key, m2), n_sim)
            record(f"simulate backward M={m2} N={N} f32", ms, sample_evals_per_s=2 * N * m2 / ms * 1e3, peak_gb=peak_gb())

    # ---------------------------------------------------------------- per-step overhead: the run_dsb loop vs fused
    if not args.quick:
        cache = block(backward_cache(state.ema_params, key, M))
        sample_cache = eqx.filter_jit(ipf.sample_cache)
        n_loop = max(4, 3 * args.steps)

        def loop_unfused(st, k):
            for _ in range(n_loop):
                k, k1, k2, k3 = jax.random.split(k, 4)
                ks, xs, tg = sample_cache(cache, k2, b0)
                st, loss = train_step(st, ks, xs, tg, None, k3)
            return st, loss

        block(loop_unfused(state, key))
        t0 = time.perf_counter()
        block(loop_unfused(state, key))
        ms = 1e3 * (time.perf_counter() - t0) / n_loop
        record(f"run_dsb loop (split+sample_cache+train_step) B={b0}", ms, it_per_s=1e3 / ms)

        fused = parallel.make_train_step(dp1, opt, static, 0.999, False, fused_cache_sampling=b0)
        cache1 = dp1.replicate(cache)

        def loop_fused(st, k):
            for _ in range(n_loop):
                k, k1 = jax.random.split(k)
                st, loss = fused(st, cache1, k1)
            return st, loss

        block(loop_fused(state1, key))
        t0 = time.perf_counter()
        block(loop_fused(state1, key))
        ms = 1e3 * (time.perf_counter() - t0) / n_loop
        record(f"fused sample+train_step loop B={b0}", ms, it_per_s=1e3 / ms)

        # host-side split done with numpy-free fold_in instead of split(4) each step
        # dispatch cost of the small kernels (async, no sync): sample_cache and key split
        ks_, xs_, tg_ = sample_cache(cache, key, b0)
        block((ks_, xs_, tg_))
        t0 = time.perf_counter()
        for i in range(200):
            out = sample_cache(cache, jax.random.fold_in(key, i), b0)
        t1 = time.perf_counter()
        block(out)
        t2 = time.perf_counter()
        record("dispatch: fold_in+sample_cache (per call, async)", 1e3 * (t1 - t0) / 200, exec_incl_ms=1e3 * (t2 - t0) / 200)
        t0 = time.perf_counter()
        for i in range(200):
            k4 = jax.random.split(jax.random.fold_in(key, i), 4)
        block(k4)
        record("dispatch: jax.random.split(key, 4) (per call)", 1e3 * (time.perf_counter() - t0) / 200)

    # ---------------------------------------------------------------- eval sample dump
    if not args.quick:
        nb = 500 if not args.tiny else 8

        @eqx.filter_jit
        def gen(ema, xN, k):
            return ipf.generate(eqx.combine(ema, static), xN, gammas, k, last_step_noise=False)

        xN = sample_prior(key, nb)
        ms = timeit(lambda: gen(state.ema_params, xN, key), n_sim)
        record(f"generate {nb} samples (N={N})", ms, per_5000_samples_s=ms / 1e3 * 5000 / nb)
        imgs = np.asarray(cifar_to_uint8(gen(state.ema_params, xN, key)))
        imgs = np.concatenate([imgs] * (5000 // nb))
        t0 = time.perf_counter()
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"bench_dump_{os.getpid()}.npz")
        np.savez_compressed(path, images=imgs)
        record("np.savez_compressed 5000 uint8 samples", 1e3 * (time.perf_counter() - t0), mb=os.path.getsize(path) / 2**20)
        os.remove(path)

    print(f"[bench] peak device memory {peak_gb()} GB")
    out = args.out or os.path.join("runs", f"bench_single_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    if not args.tiny:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(dict(ch=args.ch, N=N, params=n_params, xla_flags=os.environ.get("XLA_FLAGS", ""), results=RESULTS)) + "\n")
        print("[bench] wrote", out)


if __name__ == "__main__":
    main()
