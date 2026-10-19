#!/bin/bash -l
#SBATCH --partition amdfast
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --array=1-12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com
#SBATCH --job-name=symbx
#SBATCH --output=log/slurm_symbx_%a_%A.out
#SBATCH --error=log/slurm_symbx_%a_%A.err

# Extra symbolic runs (pynl_symbolic.py), lr 0.01, 30 epochs, validation = unseen val puzzles 300+:
#   learning curve  : 10, 30, 1000 puzzles/constellation (100 and 300 exist from submit_symbolic.sh)
#   random init     : 300/constellation, rule weights start uniform in [0, 0.2], seeds 1-3
# for both vocabularies (prims, rules).

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

T=${SLURM_ARRAY_TASK_ID}
MODEL=$([ $T -le 6 ] && echo prims || echo rules)
i=$(( (T - 1) % 6 ))
case $i in
  0) NTR=10;   EXTRA="";              SAVE=symb_${MODEL}_n10_lr0.01 ;;
  1) NTR=30;   EXTRA="";              SAVE=symb_${MODEL}_n30_lr0.01 ;;
  2) NTR=1000; EXTRA="";              SAVE=symb_${MODEL}_n1000_lr0.01 ;;
  3) NTR=300;  EXTRA="--init-seed 1"; SAVE=symb_${MODEL}_n300_lr0.01_init1 ;;
  4) NTR=300;  EXTRA="--init-seed 2"; SAVE=symb_${MODEL}_n300_lr0.01_init2 ;;
  5) NTR=300;  EXTRA="--init-seed 3"; SAVE=symb_${MODEL}_n300_lr0.01_init3 ;;
esac
JVM=$([ $NTR -ge 1000 ] && echo 100 || echo 40)
echo "node $(hostname) - ${SAVE}"
srun python3 -u pynl_symbolic.py train --model ${MODEL} --n-train ${NTR} --n-val 300 --epochs 30 \
    --lr 0.01 --jvm-gb ${JVM} ${EXTRA} --save-dir experiments/${SAVE}
