#!/bin/bash
# Overnight plan as executed (2026-09-15, 04:00-04:15 UTC).  GPU budget: 20 A100s; the cluster had no node
# with 8 free A100s any more, so the plan was re-cut to what was free.  Job ids are in logs/jobids.txt.
#
#   cifar10_big_N50q   6 GPUs  lambda-hyperplane0  job 524926 (running since 02:31 UTC; 128-ch U-Net, N=50 quadratic)
#   cifar10_big_N100q  6 GPUs  hyperplane-3        job 525233  N=100 quadratic ramp, gamma_max 0.05 (T=1.68), 128-ch U-Net,
#                                                      batch 768, 30k-step first half-step then 10k per half-step
#   cifar10_big_N50geo 2 GPUs  hyperplane-4        job 525234  resumed from iter001_backward.eqx at reduced speed
#                                                      (its 6-GPU job 524929 was stopped to make room for N100q)
#   cifar_N20          1 GPU   (finishing its 12th iteration)
#   fidloop / fidloop2 / fidloop3   CPU jobs computing FID for every new sample dump
#   toy_sweep          48 CPUs planck              job 525216  slurm/toy_sweep.sh (12 toy runs, ~1-2 h)
# Stopped early (checkpoints kept, results in the README): cifar_N50 (10 iterations), cifar_N50q (8), cifar_N20q (8).
set -euo pipefail
cd /mnt/data0/shared/dominic/SchrodingerBridge
sbatch --job-name=cifar_big_N100q --time=12:00:00 --gres=gpu:a100:6 --cpus-per-task=24 --mem=96G slurm/run.sbatch \
  python scripts/train_cifar.py --outdir runs/cifar10_big_N100q --num-steps 100 --gamma-power 2 --gamma-max 0.05 \
  --ch 128 --iters 12 --steps 10000 --first-steps 30000 --batch 768 --cache-size 3072 --cache-refresh 800 \
  --lr 2e-4 --warmup 1000 --fid-samples 5000
sbatch --job-name=cifar_big_geo --time=12:00:00 --gres=gpu:a100:2 --cpus-per-task=8 --mem=48G slurm/run.sbatch \
  python scripts/train_cifar.py --outdir runs/cifar10_big_N50geo --gamma-schedule geometric --ch 128 --num-steps 50 \
  --iters 12 --steps 8000 --first-steps 20000 --batch 768 --cache-size 3072 --cache-refresh 400 --lr 2e-4 --warmup 1000 --fid-samples 5000
sbatch --job-name=fidloop3 --time=13:00:00 --cpus-per-task=16 --mem=32G slurm/cpu.sbatch bash -c \
  'sleep 3000; for i in $(seq 1 90); do .venv-fid/bin/python scripts/fid.py runs/cifar10_big_N100q 2>&1 | grep FID; if ! squeue -h -n cifar_big_N100q -o %j | grep -q cifar; then break; fi; sleep 600; done'
sbatch --job-name=toy_sweep --time=06:00:00 --cpus-per-task=48 --mem=64G slurm/cpu.sbatch bash slurm/toy_sweep.sh
