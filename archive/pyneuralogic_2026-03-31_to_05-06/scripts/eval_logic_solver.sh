#!/bin/bash
#SBATCH --job-name=logic_solver
#SBATCH --partition=cpufast
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=log/logic_solver_%j.out
#SBATCH --error=log/logic_solver_%j.err

cd /home/kassyask/neuro-vector-symbolic-architectures-raven
mkdir -p log

module load Java/21
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa

MODE=${1:-eval_gt}
DATASET=${2:-data/I-RAVEN-10000}
SPLIT=${3:-test}

echo "=== Logic Rule Solver Evaluation ==="
echo "Mode:    $MODE"
echo "Dataset: $DATASET"
echo "Split:   $SPLIT"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo ""

PYTHONUNBUFFERED=1 python -m neuralogic_solver.train \
    --mode "$MODE" \
    --dataset-root "$DATASET" \
    --split "$SPLIT" \
    --batch-size 128
