#!/bin/bash -l
#SBATCH --partition amdgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --job-name=bl_rx_vit
#SBATCH --output=log/baseline_rx_vit_%A_%a.out

# The ViT on I-RAVEN-X, drawn by data_prep/render_iravenx.py, seed 0, with the recipe of submit_vit.sh. It is the same
# task as the other baselines': 6000 puzzles, the answer's facet code generated, the candidates matched by their labels.
#   tasks 0-2  3x3, values 0-10:  ViT at widths 64 and 224, row-symmetric ViT at 224; 80 px panels, 16 px patches
#   tasks 3-5  3x10, values 0-50: the same three; 160 px panels (a value step is 1 px), 32 px patches, 725-750 tokens
#              (width 224 runs out of GPU memory at batch 128, so its batches are accumulated in 4 parts)
TASKS=("iravenx_n3_m10 vit 64 80 16" "iravenx_n3_m10 vit 224 80 16" "iravenx_n3_m10 vit-sym 224 80 16"
       "iravenx_n10_m50 vit 64 160 32" "iravenx_n10_m50 vit 224 160 32 4" "iravenx_n10_m50 vit-sym 224 160 32 4")
read -r DATA ARCH WIDTH SIZE PATCH ACCUM <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

# SCHEDULE: cosine (default) or alternating, the old cosine's bug of 2026-11-02 to 11-23 (train_baseline.py docstring).
# Tasks 4 and 5, rerun on 2026-11-02 with accumulated batches, ran the bug; tasks 0-3 predate it. TAGSUFFIX is added to
# the run's tag (_cosfix for the reruns with the fix, 2026-11-23).

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
python core/baselines/train_baseline.py --data "$DATA" --arch "$ARCH" --seed "${SEED:-0}" --width "$WIDTH" --epochs 300 --eval-every 3 \
    --image-size "$SIZE" --patch "$PATCH" --lr 5e-4 --wd 0.05 --schedule ${SCHEDULE:-cosine} --warmup 10 --clip 1 --dropout 0.1 --accum "${ACCUM:-1}" --tag "w${WIDTH}${TAGSUFFIX}"
