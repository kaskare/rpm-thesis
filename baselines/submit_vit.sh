#!/bin/bash -l
#SBATCH --partition amdgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --job-name=bl_vit
#SBATCH --output=log/baseline_vit_%A_%a.out

# The black-box vision reference (train_baseline.py --arch vit): a plain vision transformer that reads the I-RAVEN
# matrix as one image, end to end, on the other baselines' task, splits and metrics (it generates the answer's
# facet code and never sees the candidates). The usual recipe for a ViT from scratch, not tuned here: AdamW,
# learning rate 5e-4, weight decay 0.05, cosine after 10 warm-up epochs, gradients clipped at 1, dropout 0.1.
#   task 0: the width closest to the facet model's weight count (width 64, 335k weights, 1.26x)
#   task 1: the width closest to 16x (width 224, 3.75M weights, 14x)
# The images come from baselines/prepare_baseline_images.py; build them first, then this array:
#   PREP=$(sbatch --parsable -p amd -c 8 --mem=24G -t 1:00:00 -J bl_images -o log/baseline_images_%j.out \
#     --wrap "source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh && conda activate nvsa && \
#             cd /home/kassyask/neuro-vector-symbolic-architectures-raven && python baselines/prepare_baseline_images.py")
#   sbatch --dependency=afterok:${PREP} baselines/submit_vit.sh

SCALES=(1 16)
SCALE=${SCALES[${SLURM_ARRAY_TASK_ID}]}

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
python baselines/train_baseline.py --data iraven --arch vit --seed "${SEED:-0}" --scale "$SCALE" --epochs "${EPOCHS:-300}" \
    --lr 5e-4 --wd 0.05 --schedule cosine --warmup 10 --clip 1 --dropout 0.1 --tag wd0.05_do0.1
