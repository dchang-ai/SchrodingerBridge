"""Diagnostics for the multi-GPU step overhead: all-reduce micro-benchmark (NCCL over NVLink),
train step with a single raveled all-reduce vs. per-leaf all-reduces, at several device counts.

    NCCL_DEBUG=INFO python scripts/bench_collective.py --ndev-list 2,4,6
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from dsb import ipf, parallel
from dsb.networks import UNet


def timeit(fn, n=20):
    out = jax.block_until_ready(fn())
    for _ in range(3):
        out = fn()
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    jax.block_until_ready(out)
    return 1e3 * (time.perf_counter() - t0) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ndev-list", default="1,2,4,6")
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--batch", type=int, default=128, help="per-device batch")
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()
    devices = jax.devices()
    print("devices:", devices, "| XLA_FLAGS=", os.environ.get("XLA_FLAGS", ""), "| NCCL env:", {k: v for k, v in os.environ.items() if k.startswith("NCCL")}, flush=True)
    ndevs = [n for n in (int(x) for x in args.ndev_list.split(",")) if n <= len(devices)]

    # --- pure all-reduce of a gradient-sized vector -------------------------------------
    for n_el in (8_952_067, 35_746_307):
        for ndev in ndevs:
            if ndev == 1:
                continue
            dp = parallel.DataParallel(devices[:ndev])
            f = jax.jit(dp.shard_map(lambda x: jax.lax.pmean(x, dp.axis), in_specs=P(), out_specs=P(), check_vma=False))
            x = dp.replicate(jnp.ones((n_el,), jnp.float32))
            ms = timeit(lambda: f(x))
            mb = 4 * n_el / 2**20
            print(f"[coll] all-reduce {mb:6.1f} MB f32 over {ndev} devices: {ms:7.3f} ms  algbw={mb / 1024 / (ms / 1e3):6.1f} GB/s", flush=True)

    # --- train step: raveled single all-reduce vs per-leaf all-reduces ------------------
    key = jax.random.PRNGKey(0)
    net = UNet(ch=args.ch, ch_mult=(1, 2, 2, 2), num_res_blocks=2, attn_resolutions=(16,), image_size=32, key=key)
    _, static = eqx.partition(net, eqx.is_array)
    opt = ipf.make_optimizer(1e-4, 1.0, 0)

    # --- single device: ipf.make_train_step (eqx.filter_jit) vs jax.jit of the same body vs shard_map(1)
    B = args.batch
    state0 = ipf.init_train_state(net, opt)
    state0 = state0._replace(ema_params=jax.tree_util.tree_map(lambda a: a + 0.0, state0.params))  # distinct buffers (donation)
    ks0 = jax.random.randint(key, (B,), 1, 51)
    xs0 = jax.random.normal(key, (B, 3, 32, 32))
    tg0 = 0.1 * jax.random.normal(key, (B, 3, 32, 32))
    # host-side dispatch cost of a trivial jitted function on the TrainState pytree (no compute)
    n_leaves = len(jax.tree_util.tree_leaves(state0))
    def trivial(state, ks, xs, targets, weights, key):
        return state._replace(step=state.step + 1), jnp.float32(0.0)
    for name, f in (("jax.jit", jax.jit(trivial)), ("eqx.filter_jit", eqx.filter_jit(trivial))):
        out = jax.block_until_ready(f(state0, ks0, xs0, tg0, None, key))
        t0 = time.perf_counter()
        for _ in range(50):
            out = f(state0, ks0, xs0, tg0, None, key)
        t1 = time.perf_counter()
        jax.block_until_ready(out)
        print(f"[coll] host dispatch of trivial {name:14s} on TrainState ({n_leaves} leaves): {1e3 * (t1 - t0) / 50:6.2f} ms/call (incl. exec {1e3 * (time.perf_counter() - t0) / 50:6.2f})", flush=True)
    step_eqx = ipf.make_train_step(opt, static, 0.999, False)
    ms = timeit(lambda: step_eqx(state0, ks0, xs0, tg0, None, key), args.steps)
    print(f"[coll] 1 dev ipf.make_train_step (eqx.filter_jit)  B={B}: {ms:7.2f} ms", flush=True)
    ms = timeit(lambda: step_eqx(state0, ks0, xs0, tg0, jnp.ones((B,)), key), args.steps)
    print(f"[coll] 1 dev ipf.make_train_step with weights=ones B={B}: {ms:7.2f} ms", flush=True)

    def body(state, ks, xs, targets, key):
        def loss_fn(params):
            return ipf.mean_matching_loss(eqx.combine(params, static), ks, xs, targets)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        updates, opt_state = opt.update(grads, state.opt_state, state.params)
        params = eqx.apply_updates(state.params, updates)
        decay = jnp.minimum(0.999, (1.0 + state.step) / (10.0 + state.step))
        ema = jax.tree_util.tree_map(lambda e, p: decay * e + (1.0 - decay) * p, state.ema_params, params)
        return ipf.TrainState(params, ema, opt_state, state.step + 1), loss

    step_jit = jax.jit(body)
    ms = timeit(lambda: step_jit(state0, ks0, xs0, tg0, key), args.steps)
    print(f"[coll] 1 dev jax.jit(same body)                    B={B}: {ms:7.2f} ms", flush=True)
    st = jax.tree_util.tree_map(lambda a: a + 0.0, state0)
    step_jit_d = jax.jit(body, donate_argnums=(0,))
    def run_donated():
        nonlocal st
        st, loss = step_jit_d(st, ks0, xs0, tg0, key)
        return loss
    ms = timeit(run_donated, args.steps)
    print(f"[coll] 1 dev jax.jit(same body, donate state)      B={B}: {ms:7.2f} ms", flush=True)

    # --- per device count: fused cache-sampling step, K steps per call, f32 / bf16 -----------
    N = 50
    gammas = jnp.asarray(__import__("dsb.schedule", fromlist=["symmetric_gammas"]).symmetric_gammas(N, 1e-5, 0.1, 2.0))
    nmax = max(ndevs)
    for ndev in ndevs:
        dp = parallel.DataParallel(devices[:ndev])
        B = args.batch * ndev
        state = parallel.init_train_state(dp, net, opt)
        refresh = parallel.make_cache_fn(dp, static, gammas, lambda k, n: jax.random.normal(k, (n, 3, 32, 32)), 512 * ndev, forward=False)
        cache = jax.block_until_ready(refresh(state.ema_params, key))
        variants = [(None, 1), (None, 8)] + ([(jnp.bfloat16, 1), (jnp.bfloat16, 8)] if ndev in (1, nmax) else [])
        for dtype, K in variants:
            step = parallel.make_train_step(dp, opt, static, 0.999, False, compute_dtype=dtype, fused_cache_sampling=B, steps_per_call=K)
            n_calls = max(3, args.steps // K)
            ms = timeit(lambda: step(state, cache, key), n_calls) / K
            print(f"[coll] fused step ndev={ndev} B/dev={args.batch} {'bf16' if dtype else 'f32 '} steps_per_call={K}: {ms:7.2f} ms/step  {B / ms * 1e3:8.0f} samples/s  {1e3 / ms:6.2f} it/s", flush=True)
        del cache, refresh, state

    # --- ravel vs per-leaf all-reduce, compute-only reference (1 and max ndev) -----------------
    for ndev in sorted({1, nmax}):
        dp = parallel.DataParallel(devices[:ndev])
        B = args.batch * ndev
        state = parallel.init_train_state(dp, net, opt)
        ks = dp.shard(jax.random.randint(key, (B,), 1, 51))
        xs = dp.shard(jax.random.normal(key, (B, 3, 32, 32)))
        tg = dp.shard(0.1 * jax.random.normal(key, (B, 3, 32, 32)))
        for ravel in (True, False):
            step = parallel.make_train_step(dp, opt, static, 0.999, False, ravel_grads=ravel)
            ms = timeit(lambda: step(state, ks, xs, tg, key), args.steps)
            print(f"[coll] train_step ch={args.ch} ndev={ndev} B/dev={args.batch} ravel_grads={ravel}: {ms:7.2f} ms  {B / ms * 1e3:8.0f} samples/s", flush=True)
        # forward+backward only (no all-reduce, no optimizer): pure compute per device
        def loss_fn(params, ks, xs, tg):
            return ipf.mean_matching_loss(eqx.combine(params, static), ks, xs, tg)
        g = jax.jit(dp.shard_map(lambda prm, k, x, t: jax.value_and_grad(loss_fn)(prm, k, x, t), in_specs=(P(), P(dp.axis), P(dp.axis), P(dp.axis)), out_specs=(P(), P()), check_vma=False))
        ms = timeit(lambda: g(state.params, ks, xs, tg), args.steps)
        print(f"[coll] value_and_grad only (no collective) ndev={ndev} B/dev={args.batch}: {ms:7.2f} ms", flush=True)


if __name__ == "__main__":
    main()
