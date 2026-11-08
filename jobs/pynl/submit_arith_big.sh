#!/bin/bash -l
#SBATCH --partition amdextralong
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=42-00:00:00
#SBATCH --job-name=grok_facet
#SBATCH --output=log/grok_facet_%j.out

# Will the facet model grok Arithmetic? Column roles at the wide widths (1,390k weights) on 100,000 Arithmetic puzzles with
# decoupled weight decay 0.1 (core/pynl/facet_stream.py, data data_prep/arith_big_data.py). An epoch takes 5-7 hours. Resubmit to continue.
#   sbatch jobs/pynl/submit_arith_big.sh

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

SAVE=experiments/facet_arith100k_wide
echo "node $(hostname), saving to ${SAVE}"
python3 -u core/pynl/facet_stream.py --save-dir ${SAVE} --epochs ${EPOCHS:-140} --wd ${WD:-0.1} --resume 2>&1 | grep --line-buffered -v "^WARNING\|SLF4J"
python3 -u core/pynl/facet_rule_eval.py ${SAVE} --iravenx data/iravenx/center_single_shuffle_n_10_maxval_50.json --jvm-gb 64 2>&1 | grep -v "^WARNING\|SLF4J"
