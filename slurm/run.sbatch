#!/bin/bash
# Generic single-GPU job runner.
#   sbatch [--job-name=X --time=HH:MM:SS --gres=gpu:a100:1] slurm/run.sbatch <command...>
# e.g. sbatch --job-name=cifar slurm/run.sbatch python scripts/train_cifar.py --outdir runs/cifar10
#SBATCH --job-name=dsb
#SBATCH --partition=express            # 1-day max, higher priority tier than 'main'
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

PROJ=${PROJ:-$SLURM_SUBMIT_DIR}
VENV=${VENV:-$PROJ/.venv}

export XDG_CACHE_HOME="$PROJ/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$PROJ/.scratch/${SLURM_JOB_ID}.${SLURM_STEP_ID:-batch}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR" "$PROJ/logs"
trap 'rm -rf "$TMPDIR"' EXIT

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
python -c 'import jax; print("JAX devices:", jax.devices(), flush=True); assert jax.default_backend() == "gpu", "GPU required"'
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader --id=0 || true   # (no pipe to head: SIGPIPE + pipefail killed a job)
echo "== running: $*"
"$@"
echo "== finished $(date)"
