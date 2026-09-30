#!/bin/bash
# Run through slurm/run.sbatch with the desired GPU count and DATA_DIR set.
set -euo pipefail
experiment=${1:?historical run name required}
architecture=${2:-made}
root=${3:-runs/made}
args=(--architecture "$architecture" --hidden 512 --depth 3 --components 8
      --data "${DATA_DIR:-data}" --outdir "$root/$experiment" --fid-samples 5000 --checkpoint-every 10000)
case "$experiment" in
  cifar10_N20|cifar10_N50|cifar10_N20q|cifar10_N50q)
    args+=(--single --legacy-ramp --no-ref-residual --ch 64 --steps 10000
           --first-steps 20000 --batch 128 --cache-size 512 --cache-refresh 400)
    case "$experiment" in
      cifar10_N20) args+=(--num-steps 20 --iters 12) ;;
      cifar10_N50) args+=(--num-steps 50 --iters 10) ;;
      cifar10_N20q) args+=(--num-steps 20 --gamma-power 2 --iters 9) ;;
      cifar10_N50q) args+=(--num-steps 50 --gamma-power 2 --iters 8) ;;
    esac
    ;;
  cifar10_big_N50q|cifar10_big_N100q|cifar10_big_N50geo)
    args+=(--ch 128 --iters 12 --batch 768 --cache-size 3072 --lr 2e-4 --warmup 1000)
    case "$experiment" in
      cifar10_big_N100q)
        args+=(--num-steps 100 --gamma-power 2 --gamma-max 0.05 --steps 10000
               --first-steps 30000 --cache-refresh 800) ;;
      cifar10_big_N50q)
        args+=(--num-steps 50 --gamma-power 2 --steps 8000 --first-steps 20000 --cache-refresh 400) ;;
      cifar10_big_N50geo)
        args+=(--num-steps 50 --gamma-schedule geometric --steps 8000 --first-steps 20000 --cache-refresh 400) ;;
    esac
    ;;
  *) echo "Unknown experiment: $experiment" >&2; exit 2 ;;
esac
python scripts/train_cifar.py "${args[@]}"
