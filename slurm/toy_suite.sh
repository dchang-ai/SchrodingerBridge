#!/bin/bash
# Cheap experiments run sequentially on one GPU: Gaussian analytic check + 2-D toys.
set -euo pipefail
python scripts/train_gaussian.py --dim 5 --hidden 128 --iters 10 --steps 5000 --outdir runs/gaussian_d5
python scripts/train_gaussian.py --dim 50 --hidden 256 --iters 10 --steps 5000 --outdir runs/gaussian_d50
for ds in moons scurve circles checkerboard; do
  python scripts/train_toy.py --dataset $ds --iters 20 --steps 10000 --outdir runs/toy_$ds
done
