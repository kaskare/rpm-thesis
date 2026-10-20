#!/bin/bash -l
#SBATCH --partition amdgpulong
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --time=3-0:0:0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com

#SBATCH --job-name=gnnfix
#SBATCH --output=log/slurm_gnnfix_%j.out
#SBATCH --error=log/slurm_gnnfix_%j.err

# Retrain the best torch GNN with CORRECTED positions (gnn/data.py fix).
#
# The published I-RAVEN numbers were trained AND evaluated on the buggy
# parse_ground_truth, where ~1/3 of puzzles in distribute_four, distribute_nine
# and in_distribute_four_out_center_single had indistinguishable candidates.
# Re-evaluating that old model on fixed positions is a distribution shift, not a
# baseline -- only a retrain gives an honest number for those three.
#
# Config reproduced EXACTLY from the best checkpoint's stored args
# (object_vsa_gnn_frontend_gnn_granular_object_only/..._ep406/model_best.pt).
#
# The original needed ~85 h for 406 epochs, over amdgpulong's 3-day cap. The
# trainer auto-resumes from checkpoint.pth.tar in --save-dir, so this script is
# submitted TWICE with an afterany dependency; the second job picks up where the
# first stopped. amdgpulong rather than an extralong partition because QOS
# nonrci jobs get SUSPENDED there.

echo_green() { echo -e "\033[0;32m [$(date)] $1\033[0m"; }

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa

export ROOT_PATH="/home/kassyask/neuro-vector-symbolic-architectures-raven"
cd "${ROOT_PATH}"
mkdir -p log

echo_green "node $(hostname) — torch GNN retrain, FIXED positions"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

srun python3 -u torch_gnn/train_object_vsa_gnn_frontend.py \
    --dataset-root data/I-RAVEN-10000 \
    --use-gt \
    --node-repr embedding \
    --gnn-type gnn \
    --granular \
    --object-only \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --embed-dim 256 \
    --hidden-dim 384 \
    --num-layers 3 \
    --dropout 0.1 \
    --batch-size 16 \
    --num-workers 2 \
    --epochs 406 \
    --seed 42 \
    --save-dir "${ROOT_PATH}/experiments/gnn_FIXEDPOS_granular_object_only"

echo_green "Job finished (may resume in the chained follow-up)."
