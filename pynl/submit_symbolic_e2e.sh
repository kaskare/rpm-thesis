#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --array=3-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com
#SBATCH --job-name=symbe2e
#SBATCH --output=log/slurm_symbe2e_%a_%A.out
#SBATCH --error=log/slurm_symbe2e_%a_%A.err

# Symbolic PyNeuraLogic models on the I-RAVEN TEST split, end to end (ResNet-18 perception -> symbolic
# facts -> trained template) and with XML labels, plus the non-learned reference solvers.
# Checkpoints = the best-by-validation runs of submit_symbolic.sh (arm 3: 300/constellation, lr 0.01).
# Each checkpoint's test split is scored once: arms 1-2 were scored on 2026-10-19 (job 11825915).

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

case ${SLURM_ARRAY_TASK_ID} in
  1) MODEL=prims ;;
  2) MODEL=rules ;;
  3) MODEL=rules_cols ;;
  4) MODEL=rules_cols_ms ;;
esac
srun python3 -u pynl/pynl_symbolic_e2e.py --model ${MODEL} --run-dir experiments/symb_${MODEL}_n300_lr0.01 \
    --split test --jvm-gb 40
