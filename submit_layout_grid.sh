#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=130G
#SBATCH --nodes=1
#SBATCH --time=3-0:0:0
#SBATCH --array=1-6
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com

#SBATCH --job-name=laygrid
#SBATCH --output=log/slurm_laygrid_%a_%A.out
#SBATCH --error=log/slurm_laygrid_%a_%A.err

# Grid search over the two fixes for LAYOUT reasoning.
#
# Diagnosis (analyze_constellation.py on allcons_v2_L2): when Number/Position is
# Constant the model scores ~50% on distribute_four/nine; when either follows a
# rule it drops to 19-30%, Position-Progression near chance. The model reasons
# about object ATTRIBUTES but not object LAYOUT -- which is why distribute_four
# isn't easier than distribute_nine despite having fewer slots.
#
#   within-panel edges : none | all          (--no-same-panel removed them)
#   layout features    : off | count | count_occ
#                        count     -> per-panel object count  (targets Number rules)
#                        count_occ -> + 9-bit slot occupancy  (targets Position rules)
#
# Arm 1 reproduces allcons_v2's config exactly and is the control.
# All: level 2, 2000/constellation, dim 128, 3 layers, 12 epochs.

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
  1) NAME=nosp_off       ; OPTS="--no-same-panel --layout off" ;;
  2) NAME=nosp_count     ; OPTS="--no-same-panel --layout count" ;;
  3) NAME=nosp_countocc  ; OPTS="--no-same-panel --layout count_occ" ;;
  4) NAME=sp_off         ; OPTS="--layout off" ;;
  5) NAME=sp_count       ; OPTS="--layout count" ;;
  6) NAME=sp_countocc    ; OPTS="--layout count_occ" ;;
esac

echo_green "node $(hostname) — LAYOUT GRID ARM ${SLURM_ARRAY_TASK_ID}: ${NAME}  (${OPTS})"

srun python3 -u "${ROOT_PATH}/train_all_constellations.py" \
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
    --save-dir "${ROOT_PATH}/experiments/laygrid_${NAME}" \
    ${OPTS}

echo_green "Arm ${NAME} finished."
