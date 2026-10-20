#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=110G
#SBATCH --nodes=1
#SBATCH --time=1-12:0:0
#SBATCH --array=1-7
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com

#SBATCH --job-name=sweep
#SBATCH --output=log/slurm_sweep_%a_%A.out
#SBATCH --error=log/slurm_sweep_%a_%A.err

# Why the current all-constellation runs may be stalling: padding every
# constellation to 9 slots means 62-89% of nodes are pad tokens, and AVG
# aggregation over a FIXED edge set averages real objects together with them.
# For center_single, 100% of within-panel edges connect two pads.
#
# Each arm changes one thing against arm 1. All share level 2, 200 train /
# 100 val per constellation, dim 64, 40 epochs.

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

case ${SLURM_ARRAY_TASK_ID} in
  1) NAME=baseline    ; OPTS="--agg avg --layers 3 --lr 1e-3 --n-train 200" ;;
  2) NAME=max         ; OPTS="--agg max --layers 3 --lr 1e-3 --n-train 200" ;;
  3) NAME=nosamepanel ; OPTS="--agg avg --layers 3 --lr 1e-3 --n-train 200 --no-same-panel" ;;
  4) NAME=max_nosp    ; OPTS="--agg max --layers 3 --lr 1e-3 --n-train 200 --no-same-panel" ;;
  5) NAME=max_lr3     ; OPTS="--agg max --layers 3 --lr 3e-3 --n-train 200" ;;
  6) NAME=max_2layer  ; OPTS="--agg max --layers 2 --lr 1e-3 --n-train 200" ;;
  7) NAME=max_nosp_x5 ; OPTS="--agg max --layers 3 --lr 1e-3 --n-train 1000 --no-same-panel" ;;
esac

echo_green "node $(hostname) — SWEEP ARM ${SLURM_ARRAY_TASK_ID}: ${NAME}"
echo_green "opts: ${OPTS}"

srun python3 -u "${ROOT_PATH}/train_all_constellations.py" \
    --level 2 \
    --dataset-root "${ROOT_PATH}/data/I-RAVEN-10000" \
    --n-val 100 \
    --embed-dim 64 \
    --hidden-dim 64 \
    --nl-lr 1e-4 \
    --batch-size 8 \
    --epochs 40 \
    --jvm-gb 90 \
    --save-dir "${ROOT_PATH}/experiments/sweep_${NAME}" \
    ${OPTS}

echo_green "Arm ${NAME} finished."
