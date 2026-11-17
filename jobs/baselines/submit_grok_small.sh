#!/bin/bash -l
#SBATCH --partition amdgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=grok_small
#SBATCH --output=log/grok_small_%j.out

# Grokking needs a training set small enough to memorise: on 100,000 Arithmetic puzzles train and unseen never parted
# (submit_arith_big.sh). Attribute GNN B at 16x on the first N of them, with that run's recipe (it reached 99.3): AdamW,
# learning rate 1e-3 held constant, weight decay WD (default 0.1), no dropout. 300,000 steps of 64 puzzles whatever N is, an
# evaluation every 2000 steps, a warm-up of about 1000; about 15 hours. The grokking papers' setting, decay 1 and beta2 0.98
# (WD=1 B2=0.98), does not fit: the training loss rises after about 6000 steps at every N tried (docs/ARITHMETIC.md).
# amdgpu, not amdgpulong: a higher tier, so it starts sooner. Resubmit with the same settings to continue from last.pt.
#   N=1000 WD=0 sbatch jobs/baselines/submit_grok_small.sh
# SCALE sets the width (default 16x, 25.6M weights; 1x is 1.6M, the facet model's size on 3x10), STEPS the updates.
#   N=1000 WD=0 SCALE=1 STEPS=40000 sbatch jobs/baselines/submit_grok_small.sh
# DATA picks the dataset (default iravenx_arith, every attribute Arithmetic: a diagnostic, the rule is known), SELECT
# what picks the epoch (default arithmetic), SEED the seed. The same recipe on the mixed-rule puzzles, the benchmark
# setting, separates the mixing of rules from the regularisation of the earlier mixed run (dropout 0.3, decay 0.5):
#   DATA=iravenx_n10_m50 N=6000 WD=0 SCALE=1 STEPS=40000 SELECT=matched SEED=0 sbatch jobs/baselines/submit_grok_small.sh

N=${N:?set N, the number of training puzzles}
WD=${WD:-0.1}; B2=${B2:-0.999}; SCALE=${SCALE:-16}; STEPS=${STEPS:-300000}
DATA=${DATA:-iravenx_arith}; SELECT=${SELECT:-arithmetic}; SEED=${SEED:-0}
PER=$(( (N + 63) / 64 ))
EPOCHS=$(( STEPS / PER )); EVAL=$(( (2000 + PER - 1) / PER )); WARM=$(( (1000 + PER - 1) / PER ))

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
echo "${DATA}, N ${N}, width ${SCALE}x, seed ${SEED}, select ${SELECT}, decay ${WD}, beta2 ${B2}: ${PER} steps per epoch, ${EPOCHS} epochs, evaluation every ${EVAL}, warm-up ${WARM}"
python core/baselines/train_baseline.py --data ${DATA} --arch attr-rel --seed ${SEED} --scale ${SCALE} --n-train ${N} --epochs ${EPOCHS} \
    --eval-every ${EVAL} --batch 64 --lr 1e-3 --warmup ${WARM} --clip 1 --wd ${WD} --beta2 ${B2} --dropout 0 \
    --train-eval $(( N < 1000 ? N : 1000 )) --select ${SELECT} --resume --tag grok_n${N}_wd${WD}$([ "${B2}" = 0.999 ] || echo _b2${B2})
