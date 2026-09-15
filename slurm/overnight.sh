#!/bin/bash
# Overnight plan (2026-09-15, 04:00 UTC).  GPU budget: 20 A100s.
#   * cifar10_big_N50q  / cifar10_big_N50geo : 6 GPUs each, already running (jobs 524926 / 524929).
#   * cifar10_big_N100q : 8 GPUs, N=100 quadratic ramp (gamma_max 0.05 -> T=1.7), 128-ch U-Net, batch 1024,
#                         40k-step first half-step then 10k per half-step; starts when the last small run ends.
#   * 2-D toy sweep on CPU (slurm/toy_sweep.sh).
set -euo pipefail
cd /mnt/data0/shared/dominic/SchrodingerBridge
DEP=${DEP:-524205}   # cifar_N20 (finishing) - keeps the total at 12 + 8 = 20 GPUs
J=$(sbatch --job-name=cifar_big_N100q --dependency=afterany:$DEP --time=12:00:00 --gres=gpu:a100:8 --cpus-per-task=32 --mem=128G \
    slurm/run.sbatch python scripts/train_cifar.py --outdir runs/cifar10_big_N100q --num-steps 100 --gamma-power 2 --gamma-max 0.05 \
    --ch 128 --iters 12 --steps 10000 --first-steps 40000 --batch 1024 --cache-size 4096 --cache-refresh 800 \
    --lr 2e-4 --warmup 1000 --fid-samples 5000 | awk '{print $NF}')
echo "cifar_big_N100q $J" | tee -a logs/jobids.txt
F=$(sbatch --job-name=fidloop3 --dependency=after:$J --time=13:00:00 --cpus-per-task=16 --mem=32G slurm/cpu.sbatch bash -c \
    'sleep 3600; for i in $(seq 1 90); do .venv-fid/bin/python scripts/fid.py runs/cifar10_big_N100q 2>&1 | grep FID; if ! squeue -h -n cifar_big_N100q -o %j | grep -q cifar; then echo "training job finished"; break; fi; sleep 600; done' | awk '{print $NF}')
echo "fidloop3 $F" | tee -a logs/jobids.txt
T=$(sbatch --job-name=toy_sweep --time=06:00:00 --cpus-per-task=48 --mem=64G slurm/cpu.sbatch bash slurm/toy_sweep.sh | awk '{print $NF}')
echo "toy_sweep $T" | tee -a logs/jobids.txt
