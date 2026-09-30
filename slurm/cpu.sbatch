#!/bin/bash
# CPU-only job runner (e.g. FID with the torch venv):  sbatch slurm/cpu.sbatch .venv-fid/bin/python scripts/fid.py runs/cifar10
#SBATCH --job-name=cpu
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
PROJ=${PROJ:-$SLURM_SUBMIT_DIR}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$PROJ/.cache}
export TORCH_HOME=${TORCH_HOME:-$XDG_CACHE_HOME/torch}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}
export TMPDIR="$PROJ/.scratch/$SLURM_JOB_ID.${SLURM_STEP_ID:-cpu}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
cd "$PROJ"
echo "== job $SLURM_JOB_ID on $(hostname) | cpus=$SLURM_CPUS_PER_TASK | $(date)"
echo "== running: $*"
"$@"
echo "== finished $(date)"
