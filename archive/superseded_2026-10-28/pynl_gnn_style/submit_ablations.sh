#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=130G
#SBATCH --nodes=1
#SBATCH --time=3-0:0:0
#SBATCH --array=1-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com

#SBATCH --job-name=ablate
#SBATCH --output=log/slurm_ablate_%a_%A.out
#SBATCH --error=log/slurm_ablate_%a_%A.err

# Ablations aimed at the model's weak spot: rules over the SET of occupied slots
# (Number, Position) in distribute_four / distribute_nine / in_distribute_four.
# Same budget as the laygrid baseline (laygrid_nosp_off: 45.1% val after 12 epochs):
# level 2, 2000/constellation, dim 128, 3 layers, 12 epochs.
#
#   1 control     the best config again (fresh template init -> seed noise vs laygrid_nosp_off)
#   2 rowcol      cross-panel edges tagged same-row / same-column (same 1656 edges)
#   3 sp_pertype  within-panel edges ON, one message matrix per edge type, each type averaged
#                 separately (so within-panel messages cannot dilute the same-slot ones)
#   4 absentpos   empty slots carry their position + panel embedding
#   5 combined    2 + 3 + 4

echo_green() { echo -e "\033[0;32m [$(date)] $1\033[0m"; }

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
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
  1) NAME=control    ; OPTS="--no-same-panel" ;;
  2) NAME=rowcol     ; OPTS="--no-same-panel --edge-labels rowcol" ;;
  3) NAME=sp_pertype ; OPTS="--per-type-weights" ;;
  4) NAME=absentpos  ; OPTS="--no-same-panel --absent-pos" ;;
  5) NAME=combined   ; OPTS="--edge-labels rowcol --per-type-weights --absent-pos" ;;
esac

echo_green "node $(hostname) — ABLATION ARM ${SLURM_ARRAY_TASK_ID}: ${NAME}  (${OPTS})"

srun python3 -u "${ROOT_PATH}/pynl/train_all_constellations.py" \
    --level 2 \
    --dataset-root "${ROOT_PATH}/data/I-RAVEN-10000" \
    --n-train 2000 \
    --n-val 300 \
    --embed-dim 128 \
    --hidden-dim 128 \
    --layers 3 \
    --lr 1e-3 \
    --nl-lr 1e-4 \
    --batch-size 8 \
    --agg avg \
    --epochs 12 \
    --jvm-gb 110 \
    --save-dir "${ROOT_PATH}/experiments/ablate_${NAME}" \
    ${OPTS}

echo_green "Arm ${NAME} finished."
