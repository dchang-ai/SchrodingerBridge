#!/bin/bash
# Usage:  sbatch cluster_probe/template.sbatch path/to/script.py [args...]
#         (override defaults, e.g.:  sbatch --gres=gpu:a100:1 --time=04:00:00 cluster_probe/template.sbatch train.py --epochs 50)
#SBATCH --job-name=sb_train
#SBATCH --partition=express            # express: 1-day max, PriorityTier=100 (schedules ahead of 'main'); use main for >24h (8d8h max)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                   # or gpu:a100:1 / gpu:h100:1 / gpu:rtx6000:1 / gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=192-222-51-248,192-222-56-117,192-222-56-240,129-146-8-103,129-146-99-85,192-222-54-123   # GH200 nodes are aarch64; the IP-named nodes mount /mnt/data0/shared over Tailscale VPN -> importing jax from the NFS venv took >10 min (jobs 524084/524118 hung). LAN nodes: hyperplane-3/4, lambda-hyperplane(0), mklgpusrv1-3, bacon, darwin, langevin, planck
#SBATCH --output=/mnt/data0/shared/dominic/SchrodingerBridge/logs/%x_%j.out   # logs/ MUST exist before sbatch (Slurm opens it before the script runs; missing dir => job fails silently)
#SBATCH --error=/mnt/data0/shared/dominic/SchrodingerBridge/logs/%x_%j.err
set -euo pipefail

PROJ=/mnt/data0/shared/dominic/SchrodingerBridge
VENV=${VENV:-/mnt/data0/shared/dominic/SchrodingerBridge/.venv}   # uv-managed Python 3.12.13 on NFS (/mnt/data0/shared/dominic/.uv-python) + jax[cuda12], equinox, optax, numpy, pillow, matplotlib
# NOTE: do NOT use /mnt/data0/shared/dominic/onboarding on compute nodes: it wraps the login node's /usr/bin/python3.10, but nodes run Ubuntu 24.04 (python3.12) so its packages are invisible there.

# /home/dom does NOT exist on compute nodes (only /mnt/data0/shared is NFS-exported) -> relocate HOME/caches/tmp
export HOME=/mnt/data0/shared/dominic
export XDG_CACHE_HOME=$HOME/.cache
export TMPDIR=$HOME/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$PROJ/logs"

# Be a good neighbour on shared GPUs/CPUs
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # allocate GPU memory on demand instead of grabbing ~75% up front
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6       # hard cap if preallocation is ever re-enabled
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

source "$VENV/bin/activate"
cd "$PROJ"

echo "== job $SLURM_JOB_ID on $(hostname) ($(uname -m)) | gpus=$CUDA_VISIBLE_DEVICES | cpus=$SLURM_CPUS_PER_TASK | $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
echo "== python: $(which python) $(python --version 2>&1)"
echo "== running: python $*"
python "$@"
echo "== finished $(date)"
