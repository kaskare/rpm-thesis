#!/bin/bash -l
#SBATCH --partition amdgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-23
#SBATCH --job-name=bl_final
#SBATCH --output=log/baseline_final_%A_%a.out

# The regularized references completed for the baseline table (notes §23, thesis/BASELINES.md). The setting
# is the one picked on selection puzzles in submit_regularized.sh and is NOT re-tuned per dataset:
# I-RAVEN weight decay 0.5, no dropout; I-RAVEN-X weight decay 0.5, dropout 0.3; all with LayerNorm (MLP),
# cosine learning rate after 5 warm-up epochs and gradients clipped at 1.
#   tasks 0-11:  I-RAVEN-X 3x3 values 0..10, 3x10 values 0..100 and 0..1000; MLP and GNN; seeds 0 and 1
#   tasks 12-15: seed 1 of the I-RAVEN and 3x10 values 0..50 runs (seed 0 is in submit_regularized.sh)
#   tasks 16-23: 4x and 16x weights, regularized, seed 0, I-RAVEN and 3x10 values 0..50
# Task i: see the table below. Runs are grouped with the grid's runs by their tag.

i=${SLURM_ARRAY_TASK_ID}
TASKS=(
  "iravenx_n3_m10 mlp 0 1" "iravenx_n3_m10 mlp 1 1" "iravenx_n3_m10 gnn 0 1" "iravenx_n3_m10 gnn 1 1"
  "iravenx_n10_m100 mlp 0 1" "iravenx_n10_m100 mlp 1 1" "iravenx_n10_m100 gnn 0 1" "iravenx_n10_m100 gnn 1 1"
  "iravenx_n10_m1000 mlp 0 1" "iravenx_n10_m1000 mlp 1 1" "iravenx_n10_m1000 gnn 0 1" "iravenx_n10_m1000 gnn 1 1"
  "iraven mlp 1 1" "iraven gnn 1 1" "iravenx_n10_m50 mlp 1 1" "iravenx_n10_m50 gnn 1 1"
  "iraven mlp 0 4" "iraven mlp 0 16" "iraven gnn 0 4" "iraven gnn 0 16"
  "iravenx_n10_m50 mlp 0 4" "iravenx_n10_m50 mlp 0 16" "iravenx_n10_m50 gnn 0 4" "iravenx_n10_m50 gnn 0 16"
)
read -r DATA ARCH SEED SCALE <<< "${TASKS[$i]}"
if [ "$DATA" = iraven ]; then DROP=0; else DROP=0.3; fi
WD=0.5

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
python baselines/train_baseline.py --data "$DATA" --arch "$ARCH" --seed "$SEED" --scale "$SCALE" --epochs "${EPOCHS:-300}" \
    --norm --schedule cosine --warmup 5 --clip 1 --dropout "$DROP" --wd "$WD" --tag "reg_do${DROP}_wd${WD}"
