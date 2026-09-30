#!/bin/bash
# Run on one allocated GPU through slurm/run.sbatch. Historical A has no F residual.
set -euo pipefail
dataset=${1:?dataset required}
architecture=${2:-made}
root=${3:-runs/made}
python scripts/train_toy.py --dataset "$dataset" --architecture "$architecture" \
  --hidden 128 --iters 20 --steps 10000 --no-ref-residual \
  --outdir "$root/toy_${dataset}"
python scripts/train_toy.py --dataset "$dataset" --architecture "$architecture" \
  --hidden 256 --num-steps 50 --gamma 0.004 --iters 20 --steps 10000 --cache-refresh 500 \
  --outdir "$root/toyC_fine_${dataset}"
python scripts/train_toy.py --dataset "$dataset" --architecture "$architecture" \
  --hidden 256 --iters 30 --steps 20000 \
  --outdir "$root/toyB_long_${dataset}"
python scripts/train_toy.py --dataset "$dataset" --architecture "$architecture" \
  --hidden 256 --iters 20 --steps 10000 --prior-scale 1.3 \
  --outdir "$root/toyD_wide_${dataset}"
