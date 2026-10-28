#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com
#SBATCH --job-name=pixels
#SBATCH --output=log/slurm_pixels_%x_%j.out
#SBATCH --error=log/slurm_pixels_%x_%j.err

# The ResNet-18 frontends over the I-RAVEN val and test images (28,000 puzzles, 448,000 panels),
# cached in experiments/frontend_preds/ for the end-to-end runs (pynl_symbolic_e2e.py). GPU, env nvsa:
#   sbatch --partition=amdgpufast --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=3:00:00 \
#          --export=ALL,STAGE=perceive pynl/submit_pixels.sh
# (The solve stage, for the GNN-style PyNeuraLogic model, is archived with that model.)

echo_green() { echo -e "\033[0;32m [$(date)] $1\033[0m"; }

export ROOT_PATH="/home/kassyask/neuro-vector-symbolic-architectures-raven"
export PYTHONPATH="${ROOT_PATH}:$PYTHONPATH"
export PYTHONNOUSERSITE=1
mkdir -p "${ROOT_PATH}/log"
cd "${ROOT_PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

case "${STAGE}" in
  perceive)
    conda activate nvsa
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo_green "node $(hostname) — PERCEIVE: ResNet-18 frontends over I-RAVEN val+test"
    srun python3 -u pynl/pixel_pipeline.py perceive --splits val,test --workers 14 --batch 16
    ;;
  *)
    echo "FATAL: set STAGE=perceive (got '${STAGE}')" >&2; exit 1 ;;
esac

echo_green "Stage ${STAGE} finished."
