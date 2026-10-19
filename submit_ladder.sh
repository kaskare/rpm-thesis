#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --time=2-0:0:0
#SBATCH --array=1-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com

#SBATCH --job-name=ladder
#SBATCH --output=log/slurm_ladder_L%a_%A.out
#SBATCH --error=log/slurm_ladder_L%a_%A.err

# One array task per ladder level. Every level gets byte-identical
# hyperparameters -- only the torch/PyNeuraLogic boundary moves.

echo_green() { echo -e "\033[0;32m [$(date)] $1\033[0m"; }

ml load Java/21 2>/dev/null || true
if [ -z "${JAVA_HOME}" ] || [ ! -f "${JAVA_HOME}/lib/server/libjvm.so" ]; then
    for cand in /mnt/appl/software/Java/21.0.8 /mnt/appl/software/Java/21.0.2; do
        if [ -f "$cand/lib/server/libjvm.so" ]; then export JAVA_HOME="$cand"; break; fi
    done
fi
if [ ! -f "${JAVA_HOME}/lib/server/libjvm.so" ]; then
    echo "FATAL: no libjvm.so; JAVA_HOME='${JAVA_HOME}'" >&2; exit 1
fi
export PATH="${JAVA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH}"

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1

export ROOT_PATH="/home/kassyask/neuro-vector-symbolic-architectures-raven"
export PYTHONPATH="${ROOT_PATH}:$PYTHONPATH"
mkdir -p "${ROOT_PATH}/log"
cd "${ROOT_PATH}"

LEVEL=${SLURM_ARRAY_TASK_ID}
echo_green "node $(hostname) — LADDER LEVEL ${LEVEL}"

srun python3 -u "${ROOT_PATH}/train_ladder.py" \
    --level "${LEVEL}" \
    --dataset-root "${ROOT_PATH}/data/I-RAVEN-10000" \
    --constellation center_single \
    --n-train 2000 \
    --n-val 500 \
    --embed-dim 64 \
    --hidden-dim 64 \
    --layers 2 \
    --lr 1e-3 \
    --nl-lr 1e-4 \
    --batch-size 8 \
    --epochs 80 \
    --jvm-gb 70 \
    --save-dir "${ROOT_PATH}/experiments/ladder"

echo_green "Level ${LEVEL} finished."
