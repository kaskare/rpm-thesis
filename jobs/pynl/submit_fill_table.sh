#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0:0:0
#SBATCH --job-name=fill
#SBATCH --output=log/fill_%x_%j.out

# The facet variants on the datasets they had not run on, with the settings of their runs elsewhere (widths 32,64,128,
# 6000 puzzles), then scored per rule. One run per job, with its memory:
#   sbatch --mem=48G  -J fold_n3 --export=ALL,RUN=fold_n3 jobs/pynl/submit_fill_table.sh    # row fold, I-RAVEN-X 3x3
#   sbatch --mem=48G  -J copy_n3 --export=ALL,RUN=copy_n3 jobs/pynl/submit_fill_table.sh    # copy links, I-RAVEN-X 3x3
#   sbatch --mem=420G -J copy_all --export=ALL,RUN=copy_all jobs/pynl/submit_fill_table.sh  # copy links, I-RAVEN
#   sbatch --mem=48G  -J facet_n3 --export=ALL,RUN=facet_n3 jobs/pynl/submit_fill_table.sh  # the 3x3 facet run, by rule only
#   sbatch --mem=48G  -J content_n3 --export=ALL,RUN=content_n3 jobs/pynl/submit_fill_table.sh    # content gates, 3x3
#   sbatch --mem=420G -J content_all --export=ALL,RUN=content_all jobs/pynl/submit_fill_table.sh  # content gates, I-RAVEN

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

COMMON="--template facet --widths 32,64,128 --lr 3e-4 --n 6000 --n-sel 300 --n-val 1000 --n-train-eval 300 --eval-every 5"
X3=data/iravenx/center_single_shuffle_n_3_maxval_10.json
case "${RUN}" in
  fold_n3)  ARGS="--columns fold --train-epochs 60 --iravenx ${X3}"; JVM=36; SAVE=experiments/facet_iravenx_n3_m10_fold ;;
  copy_n3)  ARGS="--copy-links --train-epochs 60 --iravenx ${X3}"; JVM=36; SAVE=experiments/facet_iravenx_n3_m10_copy ;;
  copy_all) ARGS="--copy-links --train-epochs 80 --constellation all"; JVM=360; SAVE=experiments/facet_all_copy ;;
  facet_n3) JVM=36; SAVE=experiments/facet_iravenx_n3_m10 ;;
  content_n3)  ARGS="--copy-links --copy-gates content --train-epochs 60 --iravenx ${X3}"; JVM=36
               SAVE=experiments/facet_iravenx_n3_m10_copy_content ;;
  content_all) ARGS="--copy-links --copy-gates content --train-epochs 80 --constellation all"; JVM=360
               SAVE=experiments/facet_all_copy_content ;;
  *) echo "set RUN to fold_n3, copy_n3, copy_all, facet_n3, content_n3 or content_all"; exit 1 ;;
esac

echo "node $(hostname), run ${RUN}, saving to ${SAVE}"
[ "${RUN}" = facet_n3 ] || python3 -u core/pynl/bench_hub_grounding.py ${COMMON} ${ARGS} --jvm-gb ${JVM} --save-dir ${SAVE}
python3 -u core/pynl/facet_rule_eval.py ${SAVE} --transform none --jvm-gb ${JVM} 2>&1 | grep -v "^WARNING\|SLF4J"
