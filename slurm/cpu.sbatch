#!/bin/bash
# CPU-only job runner (e.g. FID with the torch venv):  sbatch slurm/cpu.sbatch .venv-fid/bin/python scripts/fid.py runs/cifar10
#SBATCH --job-name=cpu
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --exclude=192-222-51-248,192-222-56-117,192-222-56-240
#SBATCH --output=/mnt/data0/shared/dominic/SchrodingerBridge/logs/%x_%j.out
#SBATCH --error=/mnt/data0/shared/dominic/SchrodingerBridge/logs/%x_%j.err
set -euo pipefail
PROJ=/mnt/data0/shared/dominic/SchrodingerBridge
export HOME=/mnt/data0/shared/dominic
export XDG_CACHE_HOME=$HOME/.cache
export TORCH_HOME=$HOME/.cache/torch
export TMPDIR=$HOME/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
cd "$PROJ"
echo "== job $SLURM_JOB_ID on $(hostname) | cpus=$SLURM_CPUS_PER_TASK | $(date)"
echo "== running: $*"
"$@"
echo "== finished $(date)"
