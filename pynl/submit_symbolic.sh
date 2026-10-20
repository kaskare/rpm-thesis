#!/bin/bash -l
#SBATCH --partition amdfast
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com
#SBATCH --job-name=symb
#SBATCH --output=log/slurm_symb_%x_%A_%a.out
#SBATCH --error=log/slurm_symb_%x_%A_%a.err

# Symbolic PyNeuraLogic I-RAVEN models (pynl_symbolic.py), trained from answers on
# ground-truth attributes. Array arms sweep training size and learning rate.
#   MODEL=prims   generic primitives + analogy principle, learned weight per (attribute, primitive)
# Validation = unseen val puzzles 300.. (never used for training or for anything else).

echo_green() { echo -e "\033[0;32m [$(date)] $1\033[0m"; }
export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
mkdir -p log

MODEL=${MODEL:-prims}
case ${SLURM_ARRAY_TASK_ID:-1} in
  1) NTR=100; LR=0.01 ;;
  2) NTR=100; LR=0.003 ;;
  3) NTR=300; LR=0.01 ;;
  4) NTR=300; LR=0.003 ;;
esac
echo_green "node $(hostname) — symbolic ${MODEL}, ${NTR}/constellation, lr ${LR}"
srun python3 -u pynl/pynl_symbolic.py train --model ${MODEL} --n-train ${NTR} --n-val 300 --epochs 30 \
    --lr ${LR} --jvm-gb 40 --save-dir experiments/symb_${MODEL}_n${NTR}_lr${LR}
echo_green "done"
