#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=facet_roles
#SBATCH --output=log/facet_roles_%x_%j.out

# The facet model with column roles (--columns roles, notes §24) on I-RAVEN-X 3x10, values 0..49: the middle
# columns, the operands of Arithmetic, become interchangeable. Same settings as the 60-epoch runs it is
# compared with (facet_iravenx_n10_m50, facet_iravenx_n10_m50_copy_content). One run per job, chosen by RUN:
#   sbatch -J roles         pynl/submit_column_roles.sh                             # RUN=plain, ~5 h
#   sbatch -J roles_content --export=ALL,RUN=content pynl/submit_column_roles.sh    # content copy gates
#   sbatch -J roles_wide    -p amdlong -t 3-0 --mem=160G --export=ALL,RUN=wide pynl/submit_column_roles.sh
#   sbatch -J roles_wide_c  -p amdlong -t 3-0 --mem=160G --export=ALL,RUN=wide_content pynl/submit_column_roles.sh
# Roles share weights across the middle columns: at widths 32,64,128 the model has 288k weights (named columns:
# 1,411k); the wide runs, widths 64,144,288, have 1,390k (1,448k with content gates). EPOCHS overrides the 60.

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1

cd /home/kassyask/neuro-vector-symbolic-architectures-raven
RUN=${RUN:-plain}
COMMON="--template facet --lr 3e-4 --n 6000 --n-sel 300 --n-val 1000 --n-train-eval 300 --train-epochs ${EPOCHS:-60}
        --eval-every 5 --iravenx data/iravenx/center_single_shuffle_n_10_maxval_50.json --columns roles"
case "${RUN}" in
  plain)        ARGS="--widths 32,64,128 --jvm-gb 64"; EVAL_GB=64 ;;
  content)      ARGS="--widths 32,64,128 --jvm-gb 64 --copy-links --copy-gates content"; EVAL_GB=64 ;;
  wide)         ARGS="--widths 64,144,288 --jvm-gb 120"; EVAL_GB=120 ;;
  wide_content) ARGS="--widths 64,144,288 --jvm-gb 120 --copy-links --copy-gates content"; EVAL_GB=120 ;;
  *) echo "set RUN to plain, content, wide or wide_content"; exit 1 ;;
esac
SAVE=experiments/facet_iravenx_n10_m50_roles$( [ "${RUN}" = plain ] || echo "_${RUN}" )

echo "node $(hostname), run ${RUN}, saving to ${SAVE}"
python3 -u pynl/bench_hub_grounding.py ${COMMON} ${ARGS} --save-dir ${SAVE}
python3 -u pynl/facet_rule_eval.py ${SAVE} --transform none cols23 rows12 --jvm-gb ${EVAL_GB} 2>&1 | grep -v "^WARNING\|SLF4J"
