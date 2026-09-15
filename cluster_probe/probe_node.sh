echo "=== HOST"; hostname; date; echo "SLURM_JOB_ID=$SLURM_JOB_ID SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
echo "=== project dir visible?"; ls /mnt/data0/shared/dominic/SchrodingerBridge 2>&1
echo "=== mounts"; mount | grep -E "data0|home|scratch|nfs|/mnt" 2>&1
echo "=== df"; df -h /mnt/data0 "$HOME" /tmp /scratch /local /local_scratch /raid /ssd /data 2>&1
echo "=== HOME=$HOME"; ls "$HOME" 2>&1 | head
echo "=== scratch dirs"; ls -ld /scratch /tmp /local /local_scratch /raid /ssd /data /mnt/* 2>&1
echo "=== nvidia-smi"; nvidia-smi 2>&1
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv 2>&1
echo "=== env"; env | grep -i -E 'slurm|cuda|conda|module|^PATH|LD_LIBRARY' | sort
echo "=== cuda/tools"; ls -d /usr/local/cuda* 2>&1 | head; which python3 python conda module nvcc pip3 2>&1; python3 --version 2>&1
echo "=== internet"; for u in https://pypi.org https://storage.googleapis.com https://github.com; do printf "%-40s " $u; curl -sI -m 10 $u 2>&1 | head -1; done
printf "%-40s " toronto; curl -sIL -m 15 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 2>&1 | grep -iE "^HTTP|content-length" | tr '\n' ' '; echo
echo "=== python candidates"
for py in /usr/bin/python3 "$HOME"/.pyenv/versions/*/bin/python /mnt/data0/shared/dominic/.venv/bin/python /mnt/data0/shared/dominic/*/.venv/bin/python /mnt/data0/shared/dominic/*/bin/python /mnt/data0/shared/dominic/SchrodingerBridge/.venv/bin/python; do
  [ -x "$py" ] || continue
  echo "--- $py : $($py -c 'import sys; print(sys.version.split()[0])' 2>&1)"
  timeout 120 "$py" -c "import jax, equinox; print('jax', jax.__version__, 'equinox', equinox.__version__); print(jax.devices())" 2>&1 | tail -4
done
echo "=== DONE"
