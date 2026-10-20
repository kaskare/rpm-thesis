#!/bin/bash -l
#SBATCH --partition amdgpulong
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --time=3-0:0:0
#SBATCH --exclude=g07
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com

#SBATCH --job-name=gnnrel
#SBATCH --output=log/slurm_gnnrel_%x_%j.out
#SBATCH --error=log/slurm_gnnrel_%x_%j.err

# Retrain the best torch GNN with CORRECT positions AND correct spatial relations,
# over three seeds.
#
# Two bugs, both now fixed:
#   positions  gnn/data.py::parse_ground_truth read slots from I-RAVEN's shrunken
#              Position attribute (fixed 2026-09-10; retrain -> 55.09% test).
#   relations  infer_relations looked centres up by UNIVERSAL position id, but
#              build_position_centers keyed them by sorted-bbox RANK, so 33-100% of
#              within-panel relations were wrong in five of six multi-object
#              constellations. train_object_vsa_gnn_frontend.py now uses
#              universal_position_centers() unless --legacy-centers is given.
#
# Why three seeds: two March runs with identical recorded arguments and the same
# seed (ep405, ep406) finished at 27.35% and 68.83%. A single torch number is not a
# trustworthy baseline; the seed spread is part of the result.
#
# Config otherwise EXACTLY as the fixed-position retrain (submit_gnn_fixedpos.sh).
# 406 epochs take ~85 h, over amdgpulong's 3-day cap: the trainer auto-resumes from
# checkpoint.pth.tar in --save-dir, so each seed is submitted as a chain of links
# (afterany). A link that starts after the run is complete prints
# "Already completed all epochs." and exits.
#
# Submit:  for s in 42 43 44; do j=$(sbatch --parsable --job-name=gnnrel_s$s \
#            --export=ALL,SEED=$s torch_gnn/submit_gnn_fixrel.sh); for i in 2 3; do \
#            j=$(sbatch --parsable --job-name=gnnrel_s$s --dependency=afterany:$j \
#            --export=ALL,SEED=$s torch_gnn/submit_gnn_fixrel.sh); done; done

echo_green() { echo -e "\033[0;32m [$(date)] $1\033[0m"; }

if [ -z "${SEED}" ]; then echo "FATAL: set SEED (e.g. --export=ALL,SEED=42)" >&2; exit 1; fi

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
export PYTHONNOUSERSITE=1

export ROOT_PATH="/home/kassyask/neuro-vector-symbolic-architectures-raven"
cd "${ROOT_PATH}"
mkdir -p log

echo_green "node $(hostname) — torch GNN retrain, fixed positions + fixed relations, seed ${SEED}"
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
    --seed "${SEED}" \
    --save-dir "${ROOT_PATH}/experiments/gnn_FIXREL_s${SEED}"

echo_green "Seed ${SEED} link finished (continues in the chained follow-up if epochs remain)."
