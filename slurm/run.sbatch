#!/bin/bash
# Generic single-GPU job runner.
#   sbatch [--job-name=X --time=HH:MM:SS --gres=gpu:a100:1] slurm/run.sbatch <command...>
# e.g. sbatch --job-name=cifar slurm/run.sbatch python scripts/train_cifar.py --outdir runs/cifar10
#SBATCH --job-name=dsb
#SBATCH --partition=express            # 1-day max, higher priority tier than 'main'
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --exclude=192-222-51-248,192-222-56-117,192-222-56-240,129-146-8-103,129-146-99-85,192-222-54-123   # GH200 nodes are aarch64; the 129-146-*/192-222-* VMs mount NFS over a VPN and hang importing jax
#SBATCH --output=/mnt/data0/shared/dominic/SchrodingerBridge/logs/%x_%j.out   # logs/ must exist before sbatch
#SBATCH --error=/mnt/data0/shared/dominic/SchrodingerBridge/logs/%x_%j.err
set -euo pipefail

PROJ=/mnt/data0/shared/dominic/SchrodingerBridge
VENV=${VENV:-$PROJ/.venv}              # uv venv: python 3.12 + jax[cuda12] + equinox + optax (+ numpy, pillow, matplotlib)

# /home is not mounted on compute nodes; only /mnt/data0/shared is.
export HOME=/mnt/data0/shared/dominic
export XDG_CACHE_HOME=$HOME/.cache
export TMPDIR=$HOME/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$PROJ/logs"

# Good-neighbour settings on shared nodes: no GPU memory preallocation, bounded CPU threads.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

source "$VENV/bin/activate"
cd "$PROJ"
echo "== job $SLURM_JOB_ID on $(hostname) | gpus=$CUDA_VISIBLE_DEVICES | cpus=$SLURM_CPUS_PER_TASK | $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader | head -1
echo "== running: $*"
"$@"
echo "== finished $(date)"
