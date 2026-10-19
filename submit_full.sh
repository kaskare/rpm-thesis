#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=220G
#SBATCH --nodes=1
#SBATCH --time=3-0:0:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com

#SBATCH --job-name=pynlfull
#SBATCH --output=log/slurm_full_%j.out
#SBATCH --error=log/slurm_full_%j.err

# FULL RUN — best architecture (ladder level 2), the ENTIRE I-RAVEN train split
# (6000 puzzles x 7 constellations = 42,000), embed/hidden 256, 3 layers.
#
# Measured 863 ms/puzzle at dim 256 => 10.24 h/epoch. Runs on amdlong (3-day
# cap): amdextralong SUSPENDS QOS nonrci jobs -- the previous attempt sat
# suspended for 15 h after 1h50 of work. 6 epochs = ~62 h + 12 min loading;
# 7 would be 72 h and get killed mid-epoch.
#
# Why 20 epochs is enough: 42,000 puzzles at batch 8 is 5,250 optimiser steps per
# epoch. Sweep arm max_nosp_x5 (7,000 puzzles) reached 27.6% and was still
# climbing after 12 epochs = 10,500 steps. 20 epochs here is ~105,000 steps, an
# order of magnitude more.
#
# RESUMABLE: --resume continues from model_last.pt (written every epoch, with
# optimiser state), falling back to model_best.pt. The first attempt (11562097)
# timed out after 3 of 6 epochs -- real epochs took ~19.4 h, not the benchmarked
# 10.24 h -- and had only best-checkpoints, so this relaunch continues from its
# epoch-3 best (== its last epoch). Resubmitting a finished run is a no-op.
#
# Plain training: uniform shuffle over all 42,000 puzzles, unweighted loss. The
# constellations are already equal in count (6000 each), so random sampling is
# balanced in expectation. --balanced-batches / --adaptive-weights exist in the
# script if per-constellation evenness is ever wanted, but are NOT used here.

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

echo_green "node $(hostname) — FULL I-RAVEN, all 7 constellations, dim 256"

srun python3 -u "${ROOT_PATH}/train_all_constellations.py" \
    --level 2 \
    --dataset-root "${ROOT_PATH}/data/I-RAVEN-10000" \
    --n-train 6000 \
    --n-val 300 \
    --embed-dim 256 \
    --hidden-dim 256 \
    --layers 3 \
    --lr 1e-3 \
    --nl-lr 1e-4 \
    --batch-size 8 \
    --agg avg \
    --no-same-panel \
    --resume \
    --epochs 6 \
    --jvm-gb 190 \
    --save-dir "${ROOT_PATH}/experiments/full_iraven"

echo_green "Full run finished."
