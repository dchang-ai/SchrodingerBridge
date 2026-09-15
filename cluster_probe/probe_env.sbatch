#!/bin/bash
#SBATCH --job-name=probe_env
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --exclude=192-222-51-248,192-222-56-117,192-222-56-240
#SBATCH --output=/mnt/data0/shared/dominic/SchrodingerBridge/cluster_probe/probe_env_%j.out
#SBATCH --error=/mnt/data0/shared/dominic/SchrodingerBridge/cluster_probe/probe_env_%j.err
export HOME=/mnt/data0/shared/dominic
export XDG_CACHE_HOME=/mnt/data0/shared/dominic/.cache
echo "=== HOST $(hostname) arch=$(uname -m) job=$SLURM_JOB_ID gres=$SLURM_JOB_GPUS/$CUDA_VISIBLE_DEVICES"; cat /etc/os-release | grep PRETTY; date
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv
echo "=== system pythons"; ls /usr/bin/python3* 2>&1 | tr '\n' ' '; echo; python3 --version
echo "=== /tmp space"; df -h /tmp | tail -1
echo "=== curl pypi from node"; curl -sI -m 10 https://pypi.org | head -1
for py in /mnt/data0/shared/dominic/onboarding/bin/python /mnt/data0/shared/dominic/fanclub_venv/bin/python; do
  echo "=== candidate $py -> $(readlink -f $py)"
  $py --version 2>&1 || { echo "BROKEN interpreter"; continue; }
  timeout 300 $py - <<'PY' 2>&1
import time, os
t=time.time()
import jax, jax.numpy as jnp
print("jax", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices(), "import+init %.1fs"%(time.time()-t))
try:
    import equinox, optax; print("equinox", equinox.__version__, "optax", optax.__version__)
except Exception as e: print("equinox/optax import failed:", e)
for m in ["numpy","PIL","matplotlib","tensorflow_datasets","torchvision","torch","einops","tqdm"]:
    try:
        mod=__import__(m); print("  ok", m, getattr(mod,"__version__","?"))
    except Exception as e: print("  missing", m, type(e).__name__)
x=jnp.ones((4096,4096)); y=(x@x).block_until_ready(); print("matmul ok, sum=", float(y.sum()), "on", y.devices())
PY
done
echo "=== DONE $(date)"
