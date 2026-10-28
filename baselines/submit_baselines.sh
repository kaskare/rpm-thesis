#!/bin/bash -l
#SBATCH --partition amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --array=0-19
#SBATCH --job-name=baseline
#SBATCH --output=log/baseline_%A_%a.out

# Torch baselines for the facet model: MLP and all-to-all GNN, each with the facet model's number of
# weights, on I-RAVEN and four I-RAVEN-X sets, two seeds each (baselines/train_baseline.py).
# Needs the data from baselines/prepare_baseline_data.py (pynl environment) first.
#   sbatch baselines/submit_baselines.sh                  # all 20 runs
#   sbatch --array=0-3 baselines/submit_baselines.sh      # I-RAVEN only
# Task i: dataset i / 4, architecture (i / 2) % 2, seed i % 2. EPOCHS overrides the default 300.
# (Run 11910172 used three seeds, array 0-29; its seed-2 tasks were cancelled.)

DATASETS=(iraven iravenx_n3_m10 iravenx_n10_m50 iravenx_n10_m100 iravenx_n10_m1000)
ARCHS=(mlp gnn)
i=${SLURM_ARRAY_TASK_ID}
DATA=${DATASETS[$((i / 4))]}
ARCH=${ARCHS[$(((i / 2) % 2))]}
SEED=$((i % 2))

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
python baselines/train_baseline.py --data "$DATA" --arch "$ARCH" --seed "$SEED" --epochs "${EPOCHS:-300}"
