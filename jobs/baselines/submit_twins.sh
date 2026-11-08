#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-23
#SBATCH --job-name=bl_twins
#SBATCH --output=log/baseline_twins_%A_%a.out

# The regularized MLP on the test rules that replace Arithmetic (data_prep/rule_twins.py, docs/ARITHMETIC.md), on CPU:
# rows of 3, values 0..50. Settings of the 3x3 run in submit_final.sh; smaller training sets take proportionally more
# epochs, so every run takes the same number of updates. Task i: rule i % 4, size (i / 4) % 3, seed i / 12.

RULES=(sum midpoint mirror lookup)
SIZES=(100 600 6000)
i=${SLURM_ARRAY_TASK_ID}
RULE=${RULES[$((i % 4))]}; N=${SIZES[$(((i / 4) % 3))]}; SEED=$((i / 12)); F=$((6000 / N))

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
export CUDA_VISIBLE_DEVICES=
python core/baselines/train_baseline.py --data twin_${RULE}_m50 --arch mlp --seed ${SEED} --n-train ${N} --epochs $((300 * F)) \
    --warmup $((5 * F)) --eval-every $((3 * F)) --norm --schedule cosine --clip 1 --dropout 0.3 --wd 0.5 --tag twin_n${N}
