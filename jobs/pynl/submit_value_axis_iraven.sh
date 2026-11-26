#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0:0:0
#SBATCH --job-name=va_iraven
#SBATCH --output=log/valaxis_iraven_%x_%j.out

# The value-axis model on I-RAVEN, all seven layouts (core/pynl/value_axis_iraven.py). One run per job, with its memory:
#   sbatch --mem=220G -J va_ir --export=ALL,RUN=full jobs/pynl/submit_value_axis_iraven.sh    # 6000 puzzles per layout
#   sbatch --mem=64G  -J va_ir1k --export=ALL,RUN=n1000 jobs/pynl/submit_value_axis_iraven.sh # 1000 puzzles per layout
#   sbatch --mem=220G -J va_irnp --export=ALL,RUN=noplus jobs/pynl/submit_value_axis_iraven.sh     # no given arithmetic
#   sbatch --mem=64G  -J va_ir1knp --export=ALL,RUN=noplus1k jobs/pynl/submit_value_axis_iraven.sh

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

case "${RUN}" in
  full)  ARGS="--n 6000 --train-epochs 60 --jvm-gb 180"; SAVE=experiments/valaxis_iraven ;;
  n1000) ARGS="--n 1000 --train-epochs 60 --jvm-gb 48"; SAVE=experiments/valaxis_iraven_n1000 ;;
  noplus)   ARGS="--n 6000 --train-epochs 60 --no-plus --jvm-gb 180"; SAVE=experiments/valaxis_iraven_noplus ;;
  noplus1k) ARGS="--n 1000 --train-epochs 60 --no-plus --jvm-gb 48"; SAVE=experiments/valaxis_iraven_n1000_noplus ;;
  *) echo "set RUN to full, n1000, noplus or noplus1k"; exit 1 ;;
esac

echo "node $(hostname), run ${RUN}, saving to ${SAVE}"
python3 -u core/pynl/value_axis_iraven.py ${ARGS} --save-dir ${SAVE} 2>&1 | grep -v "^WARNING\|SLF4J"
