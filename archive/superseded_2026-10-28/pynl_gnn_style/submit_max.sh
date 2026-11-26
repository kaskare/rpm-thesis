#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=140G
#SBATCH --nodes=1
#SBATCH --time=3-0:0:0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com

#SBATCH --job-name=pynlmax
#SBATCH --output=log/slurm_max_%x_%j.out
#SBATCH --error=log/slurm_max_%x_%j.err

# Train the BEST PyNeuraLogic model (allcons_v2_L2, 51.78% test) to its maximum.
#
# allcons_v2 stopped at epoch 40 only because it hit the 3-day wall: val was
# still climbing (53.8% @35, 53.9% @38, 53.5% @40) and train loss was still
# falling (0.87). An epoch costs ~1h44m at 2000/constellation, so reaching
# convergence needs several chained allocations.
#
# Both arms WARM-START from experiments/allcons_v2_L2 (copied into their own
# save-dir so the published 51.78% checkpoint stays untouched). --resume picks
# up model_best.pt, so each arm continues from epoch 39 rather than from
# scratch. That checkpoint predates per-epoch saves, so AdamW moments restart
# once here; every later link resumes from model_last.pt with optimiser state.
#
#   ARM=conv  2000/constellation, 150 epochs  -- the known-good recipe, run to
#             convergence. ~1h44m/epoch => ~8 days => 4 chained links.
#   ARM=data  6000/constellation (the FULL train split), 80 epochs -- tests
#             whether data was the binding constraint at a dim that actually
#             learns. ~5h12m/epoch => ~9 days => 4 chained links.
#             (The earlier 42k run failed at dim 256, which learns ~3.7x slower
#             per epoch AND far slower per update; this one keeps dim 128.)
#             Its 2000/constellation warm start is a SUBSET of these 6000, so
#             no test data is involved.
#
# Everything else is allcons_v2's exact config: level 2, dim 128/128, 3 layers,
# lr 1e-3 / nl-lr 1e-4, batch 8, --agg avg, --no-same-panel, plain training
# (no balancing, no adaptive weights).
#
# Submit as a chain -- each link --resume's from the previous one's last epoch
# (the data arm needs more RAM than the 140G default directive):
#   ml=$(sbatch --parsable --export=ALL,ARM=conv pynl/submit_max.sh)
#   for i in 2 3 4; do ml=$(sbatch --parsable --dependency=afterany:$ml \
#                             --export=ALL,ARM=conv pynl/submit_max.sh); done
#   md=$(sbatch --parsable --mem=200G --export=ALL,ARM=data pynl/submit_max.sh)
#   for i in 2 3 4; do md=$(sbatch --parsable --mem=200G --dependency=afterany:$md \
#                             --export=ALL,ARM=data pynl/submit_max.sh); done
# A link that starts after the run is already finished loads the data, sees
# "Already completed all N epochs" and exits in ~10 minutes.

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

# EPOCHS may be raised at submit time (--export=ALL,ARM=data,EPOCHS=120) to
# extend a chain that is still improving when it reaches the arm's default.
case "${ARM}" in
  conv) N_TRAIN=2000; EPOCHS=${EPOCHS:-150}; JVM=110; SAVE=max_conv ;;
  data) N_TRAIN=6000; EPOCHS=${EPOCHS:-80};  JVM=170; SAVE=max_data ;;
  *) echo "FATAL: set ARM=conv or ARM=data (got '${ARM}')" >&2; exit 1 ;;
esac

echo_green "node $(hostname) — MAX TRAINING, arm ${ARM}: ${N_TRAIN}/constellation, up to ${EPOCHS} epochs"

srun python3 -u "${ROOT_PATH}/pynl/train_all_constellations.py" \
    --level 2 \
    --dataset-root "${ROOT_PATH}/data/I-RAVEN-10000" \
    --n-train ${N_TRAIN} \
    --n-val 300 \
    --embed-dim 128 \
    --hidden-dim 128 \
    --layers 3 \
    --lr 1e-3 \
    --nl-lr 1e-4 \
    --batch-size 8 \
    --agg avg \
    --no-same-panel \
    --resume \
    --epochs ${EPOCHS} \
    --jvm-gb ${JVM} \
    --save-dir "${ROOT_PATH}/experiments/${SAVE}"

echo_green "Arm ${ARM} link finished (continues in the chained follow-up if epochs remain)."
