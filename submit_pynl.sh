#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=3-0:0:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com

#SBATCH --job-name=pynl
#SBATCH --output=log/slurm_pynl_%j.out
#SBATCH --error=log/slurm_pynl_%j.err

# NOTE: no --gres=gpu. PyNeuraLogic runs its engine on the JVM (CPU only) via
# JPype; a GPU would sit idle. Memory and cores are what this workload needs —
# grounding every candidate graph up front is the expensive part.

echo_green() {
    echo -e "\033[0;32m [$(date)] $1\033[0m"
}

# Java is mandatory (PyNeuraLogic's engine is a JVM program driven through JPype).
# `ml load Java/21` silently does nothing on some compute nodes, so pin JAVA_HOME
# explicitly and verify libjvm.so before doing any work.
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

# ~/.local/lib/python3.11/site-packages shadows the conda env — keep it out.
export PYTHONNOUSERSITE=1

export ROOT_PATH="/home/kassyask/neuro-vector-symbolic-architectures-raven"
export PYTHONPATH="${ROOT_PATH}:$PYTHONPATH"

PYTHON_SCRIPT="${ROOT_PATH}/train_pynl.py"

mkdir -p "${ROOT_PATH}/log"
cd "${ROOT_PATH}"

echo_green "node:        $(hostname)"
echo_green "python:      $(which python3) ($(python3 -V 2>&1))"
echo_green "neuralogic:  $(python3 -c 'import neuralogic; print(neuralogic.__version__)' 2>/dev/null || echo MISSING)"
echo_green "JAVA_HOME:   ${JAVA_HOME}"
echo_green "Running: ${PYTHON_SCRIPT} $*"

srun python3 -u "${PYTHON_SCRIPT}" \
    --dataset-root "${ROOT_PATH}/data/I-RAVEN-10000" \
    --constellations center_single \
    --n-train 2000 \
    --n-val 500 \
    --dim 64 \
    --layers 3 \
    --lr 0.001 \
    --epochs 300 \
    --activation leaky_relu \
    --jvm-gb 100 \
    --save-dir "${ROOT_PATH}/experiments/pynl_full_center_single"

echo_green "Job finished."
