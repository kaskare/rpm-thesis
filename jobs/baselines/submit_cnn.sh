#!/bin/bash -l
#SBATCH --partition amdgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --job-name=bl_cnn
#SBATCH --output=log/baseline_cnn_%A_%a.out

# The small ResNet over the I-RAVEN matrix, its panels stacked as channels (train_baseline.py --arch cnn), with the ViT's
# recipe of submit_vit.sh except dropout: AdamW, learning rate 5e-4, weight decay 0.05, cosine after 10 warm-up epochs,
# gradients clipped at 1, no dropout, 80 px panels. The first runs (job 11932164, RESULTS §1) came from a script kept
# outside the repo; this one has their arguments, read back from their best.json.
#   task 0: the facet model's weight count (286k)    task 1: 16x (4.27M)
# SCHEDULE: cosine (default) or alternating, the old cosine's bug of 2026-11-02 to 11-23 (train_baseline.py docstring).
# Job 11932164 predates the bug. TAGSUFFIX is added to the run's tag.

SCALES=(1 16)
SCALE=${SCALES[${SLURM_ARRAY_TASK_ID}]}

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
python core/baselines/train_baseline.py --data iraven --arch cnn --seed "${SEED:-0}" --scale "$SCALE" --epochs 300 --eval-every 3 \
    --lr 5e-4 --wd 0.05 --schedule ${SCHEDULE:-cosine} --warmup 10 --clip 1 --dropout 0 --tag "recipe${TAGSUFFIX}"
