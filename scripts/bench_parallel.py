"""Multi-GPU (one node) benchmark of the data-parallel DSB pieces in ``dsb/parallel.py``.

For every configuration (number of devices, network width, global batch, precision) it
builds the replicated train state, times the sharded cache refresh (``simulate`` of
M = per_device_cache x ndev trajectories, N steps x 2 network evaluations) and the fused
sample-from-cache + train step, and reports it/s, samples/s and the effective it/s with a
refresh every ``--refresh-every`` steps (the ``run_dsb`` schedule).

Configurations (``--configs``):
  weak    per-device batch fixed (128): global batch 128 x ndev, cache 512 x ndev
  strong  global batch fixed (512): per-device batch 512 / ndev, cache 512
  bf16    weak configuration at max ndev with bf16 mixed precision
  big     UNet ch=128 (~36M params) at per-device batch 128 on max ndev (fp32 and bf16)
  unfused separate sample_cache + train_step dispatches (as run_dsb does) at 1 and max ndev

Usage (Slurm, see slurm/run.sbatch):
    python scripts/bench_parallel.py --ndev-list 1,2,4,8 --configs weak,strong,bf16,big
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 python scripts/bench_parallel.py --tiny
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from dsb import ipf, parallel
from dsb.networks import UNet, count_params
from dsb.schedule import OUReference, symmetric_gammas

T_START = time.time()
RESULTS = []


def block(x):
    return jax.block_until_ready(x)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"({type(e).__name__}: {e})"


def node_info():
    print("== host", os.uname().nodename, "| devices:", [str(d) for d in jax.devices()], flush=True)
    print("== nvidia-smi topo -m\n" + sh("nvidia-smi topo -m 2>&1 | head -20"), flush=True)
    print("== /sys/class/infiniband:", sh("ls /sys/class/infiniband 2>&1"), "| ibstat:", sh("ibstat 2>&1 | head -5"), flush=True)
    print("== ip -br link:\n" + sh("ip -br link 2>&1 | head -12"), flush=True)
    print("== ethtool speeds:", sh("for i in $(ls /sys/class/net | grep -v -E 'lo|docker|veth|tailscale|br-'); do printf '%s=%s ' $i $(cat /sys/class/net/$i/speed 2>/dev/null); done"), flush=True)
    print("== nccl:", sh("python -c \"import importlib.metadata as m; print(m.version('nvidia-nccl-cu12'))\" 2>&1"), flush=True)


def peak_gb(dev):
    try:
        return round(dev.memory_stats()["peak_bytes_in_use"] / 2**30, 2)
    except Exception:
        return float("nan")


def bench_config(label, devices, ch, global_batch, cache_size, N, steps, refresh_every, dtype=None, unfused=False, tiny=False):
    ndev = len(devices)
    dp = parallel.DataParallel(devices)
    ref = OUReference(symmetric_gammas(N, 1e-5, 0.1, 2.0), alpha=1.0)
    gammas = jnp.asarray(ref.gammas)
    key = jax.random.PRNGKey(0)
    net = UNet(ch=ch, ch_mult=(1, 2, 2, 2), num_res_blocks=2, attn_resolutions=(16,), image_size=32, key=key)
    n_params = count_params(net)
    _, static = eqx.partition(net, eqx.is_array)
    opt = ipf.make_optimizer(1e-4, 1.0, 0)
    state = parallel.init_train_state(dp, net, opt)
    sample_prior = lambda k, n: jax.random.normal(k, (n, 3, 32, 32))
    refresh = parallel.make_cache_fn(dp, static, gammas, sample_prior, cache_size, forward=False, compute_dtype=dtype)
    step = parallel.make_train_step(dp, opt, static, 0.999, False, compute_dtype=dtype, fused_cache_sampling=global_batch)

    # cache refresh: compile + time
    t0 = time.perf_counter()
    cache = block(refresh(state.ema_params, key))
    t_compile_cache = time.perf_counter() - t0
    n_rep = 1 if tiny else 3
    t0 = time.perf_counter()
    for i in range(n_rep):
        cache = refresh(state.ema_params, jax.random.fold_in(key, i))
    block(cache)
    t_refresh = (time.perf_counter() - t0) / n_rep

    # train steps: compile + warm-up + time
    t0 = time.perf_counter()
    state, loss = block(step(state, cache, key))
    t_compile_step = time.perf_counter() - t0
    for i in range(2):
        state, loss = step(state, cache, jax.random.fold_in(key, i))
    block(loss)
    t0 = time.perf_counter()
    for i in range(steps):
        state, loss = step(state, cache, jax.random.fold_in(key, 100 + i))
    block(loss)
    t_step = (time.perf_counter() - t0) / steps

    if unfused:
        sample = parallel.make_sample_cache(dp, global_batch)
        step_u = parallel.make_train_step(dp, opt, static, 0.999, False, compute_dtype=dtype)
        k = key
        for i in range(3):
            k, k1, k2, k3 = jax.random.split(k, 4)
            ks, xs, tg = sample(cache, k2)
            state, loss = step_u(state, ks, xs, tg, k3)
        block(loss)
        t0 = time.perf_counter()
        for i in range(steps):
            k, k1, k2, k3 = jax.random.split(k, 4)
            ks, xs, tg = sample(cache, k2)
            state, loss = step_u(state, ks, xs, tg, k3)
        block(loss)
        t_step_u = (time.perf_counter() - t0) / steps
    else:
        t_step_u = float("nan")

    it_s = 1.0 / t_step
    eff_it_s = refresh_every / (refresh_every * t_step + t_refresh)
    row = dict(
        label=label, ndev=ndev, ch=ch, params_M=round(n_params / 1e6, 2), global_batch=global_batch, per_device_batch=global_batch // ndev,
        cache_size=cache_size, dtype="bf16" if dtype is not None else "f32",
        step_ms=round(1e3 * t_step, 2), it_per_s=round(it_s, 2), samples_per_s=round(it_s * global_batch), refresh_s=round(t_refresh, 3),
        eff_it_per_s=round(eff_it_s, 2), eff_samples_per_s=round(eff_it_s * global_batch),
        unfused_step_ms=round(1e3 * t_step_u, 2) if unfused else None,
        compile_s=round(t_compile_cache + t_compile_step, 1), peak_gb_dev0=peak_gb(devices[0]), loss=float(loss),
    )
    RESULTS.append(row)
    print("[bench] " + json.dumps(row), flush=True)
    # free device memory before the next configuration
    del state, cache, step, refresh
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ndev-list", default="1,2,4,8")
    p.add_argument("--configs", default="weak,strong,bf16,big,unfused")
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--big-ch", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--per-device-batch", type=int, default=128)
    p.add_argument("--global-batch", type=int, default=512)
    p.add_argument("--per-device-cache", type=int, default=512)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--refresh-every", type=int, default=400)
    p.add_argument("--time-budget", type=float, default=1500.0, help="seconds; skip remaining configs beyond this")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.tiny:
        args.ch, args.big_ch, args.num_steps, args.per_device_batch, args.global_batch, args.per_device_cache, args.steps = 8, 16, 4, 4, 16, 8, 2

    node_info()
    devices = jax.devices()
    ndevs = [n for n in (int(x) for x in args.ndev_list.split(",")) if n <= len(devices)]
    if not ndevs:
        ndevs = [len(devices)]
    nmax = max(ndevs)
    configs = args.configs.split(",")
    print(f"== ndev list {ndevs} (available {len(devices)}), configs {configs}", flush=True)
    N, steps, R, tiny = args.num_steps, args.steps, args.refresh_every, args.tiny

    plan = []
    if "weak" in configs:
        for n in ndevs:
            plan.append(dict(label="weak", devices=devices[:n], ch=args.ch, global_batch=args.per_device_batch * n, cache_size=args.per_device_cache * n))
    if "strong" in configs:
        for n in ndevs:
            if args.global_batch % n == 0 and args.per_device_cache % n == 0:
                plan.append(dict(label="strong", devices=devices[:n], ch=args.ch, global_batch=args.global_batch, cache_size=args.per_device_cache))
    if "bf16" in configs:
        for n in sorted({1, nmax}):
            plan.append(dict(label="bf16", devices=devices[:n], ch=args.ch, global_batch=args.per_device_batch * n, cache_size=args.per_device_cache * n, dtype=jnp.bfloat16))
    if "big" in configs:
        plan.append(dict(label="big", devices=devices[:nmax], ch=args.big_ch, global_batch=args.per_device_batch * nmax, cache_size=args.per_device_cache * nmax))
        plan.append(dict(label="big-bf16", devices=devices[:nmax], ch=args.big_ch, global_batch=args.per_device_batch * nmax, cache_size=args.per_device_cache * nmax, dtype=jnp.bfloat16))
        if nmax > 1:
            plan.append(dict(label="big", devices=devices[:1], ch=args.big_ch, global_batch=args.per_device_batch, cache_size=args.per_device_cache))
    if "unfused" in configs:
        for n in sorted({1, nmax}):
            plan.append(dict(label="unfused", devices=devices[:n], ch=args.ch, global_batch=args.per_device_batch * n, cache_size=args.per_device_cache * n, unfused=True))

    for cfg in plan:
        elapsed = time.time() - T_START
        if elapsed > args.time_budget:
            print(f"[bench] time budget exhausted ({elapsed:.0f}s), skipping {cfg['label']} ndev={len(cfg['devices'])}", flush=True)
            continue
        print(f"== config {cfg['label']} ndev={len(cfg['devices'])} ch={cfg['ch']} B={cfg['global_batch']} M={cfg['cache_size']} ({elapsed:.0f}s elapsed)", flush=True)
        try:
            bench_config(N=N, steps=steps, refresh_every=R, tiny=tiny, **cfg)
        except Exception as e:  # keep going (e.g. OOM of the big model)
            print(f"[bench] FAILED {cfg['label']} ndev={len(cfg['devices'])}: {type(e).__name__}: {str(e)[:400]}", flush=True)

    print("\n== summary (it/s = fused sample+step; eff = incl. one cache refresh per %d steps)" % R)
    print(f"{'config':9s} {'ndev':>4s} {'ch':>4s} {'dtype':>5s} {'B':>5s} {'M':>5s} {'step ms':>8s} {'it/s':>7s} {'samples/s':>10s} {'refresh s':>9s} {'eff it/s':>8s} {'eff smp/s':>9s} {'peakGB':>6s}")
    for r in RESULTS:
        print(f"{r['label']:9s} {r['ndev']:4d} {r['ch']:4d} {r['dtype']:>5s} {r['global_batch']:5d} {r['cache_size']:5d} {r['step_ms']:8.2f} {r['it_per_s']:7.2f} {r['samples_per_s']:10d} {r['refresh_s']:9.2f} {r['eff_it_per_s']:8.2f} {r['eff_samples_per_s']:9d} {r['peak_gb_dev0']:6.2f}")
    out = args.out or os.path.join("runs", f"bench_parallel_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    if not tiny:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(dict(host=os.uname().nodename, xla_flags=os.environ.get("XLA_FLAGS", ""), results=RESULTS)) + "\n")
        print("[bench] wrote", out)
    print(f"[bench] total {time.time() - T_START:.0f}s")


if __name__ == "__main__":
    main()
