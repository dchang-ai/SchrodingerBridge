#!/bin/bash
# Overnight 2-D toy sweep on CPU (12 runs in parallel, 4 threads each; ~1 h).
#   A "paper":  N=20, gamma=0.01 (T=0.2), hidden 128, 10k steps/half-step, 20 iterations  (already in runs/toy_*)
#   B "long":   N=20, gamma=0.01, hidden 256, 20k steps, 30 iterations
#   C "fine":   N=50, gamma=0.004 (T=0.2), hidden 256, 10k steps, 20 iterations
#   D "wide":   N=20, gamma=0.01, hidden 256, 10k steps, 20 iterations, prior std = 1.3 x data std (Technique 4)
set -uo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
PY=.venv/bin/python
pids=()
for ds in moons scurve circles checkerboard; do
  $PY scripts/train_toy.py --dataset $ds --hidden 256 --steps 20000 --iters 30 --outdir runs/toyB_long_$ds > logs/toyB_$ds.log 2>&1 & pids+=($!)
  $PY scripts/train_toy.py --dataset $ds --hidden 256 --num-steps 50 --gamma 0.004 --steps 10000 --iters 20 --cache-refresh 500 --outdir runs/toyC_fine_$ds > logs/toyC_$ds.log 2>&1 & pids+=($!)
  $PY scripts/train_toy.py --dataset $ds --hidden 256 --steps 10000 --iters 20 --prior-scale 1.3 --outdir runs/toyD_wide_$ds > logs/toyD_$ds.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait $p || echo "process $p failed"; done
echo "toy sweep finished $(date)"
