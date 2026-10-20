#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=3-0:0:0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com
#SBATCH --job-name=pynlirx
#SBATCH --output=log/slurm_iravenx_%x_%j.out
#SBATCH --error=log/slurm_iravenx_%x_%j.err

# Best graph model (max_data_L2 config) on the 3x3 I-RAVEN-X.
# Data: python src/datasets/generation/iravenx_task.py --n 3 --maxval 10 --nconf 0 --save_dir data/iravenx
# Chain with --dependency=afterany like submit_max.sh; --resume continues from model_last.pt.
# Score the test split once, after the last link: sbatch --export=ALL,TEST=1 pynl/submit_iravenx.sh

echo_green() { echo -e "\033[0;32m [$(date)] $1\033[0m"; }

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1

export ROOT_PATH="/home/kassyask/neuro-vector-symbolic-architectures-raven"
mkdir -p "${ROOT_PATH}/log"
cd "${ROOT_PATH}"
DATA=${DATA:-data/iravenx/center_single_shuffle_n_3_maxval_10.json}
ARGS="--dataset-root ${DATA} --level 2 --n-val 300 --embed-dim 128 --hidden-dim 128 --layers 3 \
      --lr 1e-3 --nl-lr 1e-4 --batch-size 8 --agg avg --no-same-panel --jvm-gb 48 \
      --save-dir ${ROOT_PATH}/experiments/iravenx"

if [ -n "${TEST}" ]; then srun python3 -u pynl/train_iravenx.py ${ARGS} --test; exit; fi
echo_green "node $(hostname) — I-RAVEN-X: ${DATA}, 6000 train, up to ${EPOCHS:-60} epochs"
srun python3 -u pynl/train_iravenx.py ${ARGS} --n-train 6000 --epochs ${EPOCHS:-60} --resume
