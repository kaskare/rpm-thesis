#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=3-0:0:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com

#SBATCH --job-name=pynl384
#SBATCH --output=log/slurm_pynl_384_%j.out
#SBATCH --error=log/slurm_pynl_384_%j.err

# Hybrid: PyNeuraLogic message passing + a PyTorch head, 8 candidates scored in
# one pass. CPU only -- both the JVM engine and the CPU torch build.

echo_green() {
    echo -e "\033[0;32m [$(date)] $1\033[0m"
}

# `ml load Java/21` silently no-ops on some compute nodes; pin JAVA_HOME and verify.
ml load Java/21 2>/dev/null || true
if [ -z "${JAVA_HOME}" ] || [ ! -f "${JAVA_HOME}/lib/server/libjvm.so" ]; then
    for cand in /mnt/appl/software/Java/21.0.8 /mnt/appl/software/Java/21.0.2 \
                /mnt/appl/software/Java/17.0.4; do
        if [ -f "$cand/lib/server/libjvm.so" ]; then export JAVA_HOME="$cand"; break; fi
    done
fi
if [ ! -f "${JAVA_HOME}/lib/server/libjvm.so" ]; then
    echo "FATAL: no libjvm.so found; JAVA_HOME='${JAVA_HOME}'" >&2; exit 1
fi
export PATH="${JAVA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH}"

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1

export ROOT_PATH="/home/kassyask/neuro-vector-symbolic-architectures-raven"
export PYTHONPATH="${ROOT_PATH}:$PYTHONPATH"
PYTHON_SCRIPT="${ROOT_PATH}/train_pynl_torch.py"

mkdir -p "${ROOT_PATH}/log"
cd "${ROOT_PATH}"

echo_green "node:        $(hostname)"
echo_green "python:      $(which python3) ($(python3 -V 2>&1))"
echo_green "neuralogic:  $(python3 -c 'import neuralogic; print(neuralogic.__version__)' 2>/dev/null || echo MISSING)"
echo_green "torch:       $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo MISSING)"
echo_green "JAVA_HOME:   ${JAVA_HOME}"

srun python3 -u "${PYTHON_SCRIPT}" \
    --dataset-root "${ROOT_PATH}/data/I-RAVEN-10000" \
    --constellation center_single \
    --n-train 6000 \
    --n-val 2000 \
    --embed-dim 256 \
    --hidden-dim 384 \
    --layers 3 \
    --lr 1e-3 \
    --nl-lr 1e-4 \
    --batch-size 8 \
    --epochs 130 \
    --jvm-gb 80 \
    --save-dir "${ROOT_PATH}/experiments/pynl_hybrid_384_center_single"

echo_green "Job finished."
