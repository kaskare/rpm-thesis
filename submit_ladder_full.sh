#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=3-0:0:0
#SBATCH --array=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com

#SBATCH --job-name=ladderF
#SBATCH --output=log/slurm_ladderfull_L%a_%A.out
#SBATCH --error=log/slurm_ladderfull_L%a_%A.err

# Full-scale run of ladder level 2 (best of the ladder at 39.0%):
#   L1 = torch head   (32.6% at small scale, 52.9% at full scale previously)
#   L2 = pynl head    (39.0% at small scale -- best of the ladder)
# Identical hyperparameters so the comparison stays clean.

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
echo_green "node $(hostname) — FULL-SCALE LADDER LEVEL ${LEVEL}"

srun python3 -u "${ROOT_PATH}/train_ladder.py" \
    --level "${LEVEL}" \
    --dataset-root "${ROOT_PATH}/data/I-RAVEN-10000" \
    --constellation center_single \
    --n-train 6000 \
    --n-val 2000 \
    --embed-dim 128 \
    --hidden-dim 128 \
    --layers 3 \
    --lr 1e-3 \
    --nl-lr 1e-4 \
    --batch-size 8 \
    --epochs 150 \
    --jvm-gb 100 \
    --save-dir "${ROOT_PATH}/experiments/ladder_full"

echo_green "Level ${LEVEL} finished."
