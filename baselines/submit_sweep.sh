#!/bin/bash -l
#SBATCH --partition amdgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --job-name=bl_sweep
#SBATCH --output=log/baseline_sweep_%A_%a.out

# Capacity sweep of the torch baselines: MLP and all-to-all GNN at 4x, 16x and 64x the facet model's
# weights, on I-RAVEN and I-RAVEN-X 3x10 values 0-49, seed 0 (the 1x runs are submit_baselines.sh).
# If accuracy levels off below the facet model's as the weights grow, the gap is the architecture, not
# its size. Task i: dataset i / 6, architecture (i / 3) % 2, scale 4 / 16 / 64 by i % 3. The I-RAVEN GNN
# at 64x uses batch 64: its all-pairs message tensors (73 nodes, width ~1190) would not fit a 40 GB A100
# at 128. amdgpu (1-day limit) because the largest runs exceed amdgpufast's 4 hours; best.json is written
# every epoch, so a run stopped by the limit still reports its best epoch so far. LR overrides lr 1e-3:
# the I-RAVEN-X MLP at 64x (task 8) never left its starting point at 1e-3 and was rerun at LR=1e-4.

DATASETS=(iraven iravenx_n10_m50)
ARCHS=(mlp gnn)
SCALES=(4 16 64)
i=${SLURM_ARRAY_TASK_ID}
DATA=${DATASETS[$((i / 6))]}
ARCH=${ARCHS[$(((i / 3) % 2))]}
SCALE=${SCALES[$((i % 3))]}
BATCH=128
if [ "$DATA" = iraven ] && [ "$ARCH" = gnn ] && [ "$SCALE" = 64 ]; then BATCH=64; fi

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
python baselines/train_baseline.py --data "$DATA" --arch "$ARCH" --seed 0 --scale "$SCALE" --batch "$BATCH" --lr "${LR:-1e-3}" --epochs "${EPOCHS:-300}"
