#!/bin/bash -l
#SBATCH --partition amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --array=0-39
#SBATCH --job-name=bl_reg
#SBATCH --output=log/baseline_reg_%A_%a.out

# Regularized torch baselines at the facet model's number of weights, seed 0, on I-RAVEN and I-RAVEN-X 3x10
# values 0-50 (baselines/train_baseline.py). Every run: LayerNorm in the MLP, cosine learning rate after 5
# warm-up epochs, gradient norm clipped at 1. The configuration is picked on the selection split.
#   tasks 0-35: dropout 0 / 0.1 / 0.3 x weight decay 0 / 0.05 / 0.5;
#               i -> dataset i / 18, architecture (i / 9) % 2, dropout (i / 3) % 3, weight decay i % 3
#   36, 37: GNN with 20% of messages dropped (dropout 0.1, weight decay 0.05), I-RAVEN and I-RAVEN-X
#   38, 39: I-RAVEN MLP and GNN with rows 1 and 2 swapped in half the training samples (same setting);
#           the swap keeps every I-RAVEN rule but breaks Distribute on I-RAVEN-X 3x10, so it runs on I-RAVEN only

DATASETS=(iraven iravenx_n10_m50)
ARCHS=(mlp gnn)
DROPOUTS=(0 0.1 0.3)
WDS=(0 0.05 0.5)
i=${SLURM_ARRAY_TASK_ID}
EXTRA=()
if [ "$i" -lt 36 ]; then
    DATA=${DATASETS[$((i / 18))]}; ARCH=${ARCHS[$(((i / 9) % 2))]}
    DROP=${DROPOUTS[$(((i / 3) % 3))]}; WD=${WDS[$((i % 3))]}; TAG=reg_do${DROP}_wd${WD}
else
    DROP=0.1; WD=0.05
    case $i in
        36) DATA=iraven; ARCH=gnn; EXTRA=(--drop-edge 0.2); TAG=reg_do0.1_wd0.05_de0.2 ;;
        37) DATA=iravenx_n10_m50; ARCH=gnn; EXTRA=(--drop-edge 0.2); TAG=reg_do0.1_wd0.05_de0.2 ;;
        38) DATA=iraven; ARCH=mlp; EXTRA=(--augment rows); TAG=reg_do0.1_wd0.05_rows ;;
        39) DATA=iraven; ARCH=gnn; EXTRA=(--augment rows); TAG=reg_do0.1_wd0.05_rows ;;
    esac
fi

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
python baselines/train_baseline.py --data "$DATA" --arch "$ARCH" --seed 0 --epochs "${EPOCHS:-300}" \
    --norm --schedule cosine --warmup 5 --clip 1 --dropout "$DROP" --wd "$WD" "${EXTRA[@]}" --tag "$TAG"
