#!/bin/bash -l
#SBATCH --partition amdgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --job-name=attr_gnn
#SBATCH --output=log/baseline_attr_%A_%a.out

# Plan models A and B (thesis/THESIS_PLAN_symmetry.md §2; baselines/train_baseline.py): the facet model's
# attribute-level nodes on an all-to-all graph, with absolute labels (attr-abs, A: no puzzle symmetry) or
# relative labels (attr-rel, B: rows 1 <-> 2, components, size <-> colour and the middle columns built in).
# At the facet model's weight count, seed 0, with the object GNN's best regularized setting (notes §23.3):
# weight decay 0.5, cosine learning rate after 5 warm-up epochs, gradient norm clipped at 1; dropout 0 on
# I-RAVEN, 0.3 on I-RAVEN-X. Batch 32 on I-RAVEN (up to 292 attribute nodes), 64 on I-RAVEN-X 3x10 (120).
# Task i: dataset i / 2, architecture i % 2. EPOCHS overrides the default 300.

DATASETS=(iraven iravenx_n10_m50)
ARCHS=(attr-abs attr-rel)
i=${SLURM_ARRAY_TASK_ID}
DATA=${DATASETS[$((i / 2))]}
ARCH=${ARCHS[$((i % 2))]}
if [ "$DATA" = iraven ]; then BATCH=32; DROP=0; else BATCH=64; DROP=0.3; fi

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
python baselines/train_baseline.py --data "$DATA" --arch "$ARCH" --seed 0 --epochs "${EPOCHS:-300}" --batch "$BATCH" \
    --schedule cosine --warmup 5 --clip 1 --wd 0.5 --dropout "$DROP" --tag reg_do${DROP}_wd0.5
