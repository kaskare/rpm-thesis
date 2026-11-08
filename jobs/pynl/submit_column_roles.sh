#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=facet_roles
#SBATCH --output=log/facet_roles_%x_%j.out

# The facet model with column roles (--columns roles, notes §24) on I-RAVEN-X 3x10, values 0..50: the middle
# columns, the operands of Arithmetic, become interchangeable. Same settings as the 60-epoch runs it is
# compared with (facet_iravenx_n10_m50, facet_iravenx_n10_m50_copy_content). One run per job, chosen by RUN:
#   sbatch -J roles         jobs/pynl/submit_column_roles.sh                             # RUN=plain, ~5 h
#   sbatch -J roles_content --export=ALL,RUN=content jobs/pynl/submit_column_roles.sh    # content copy gates
#   sbatch -J roles_wide    -p amdlong -t 3-0 --mem=160G --export=ALL,RUN=wide jobs/pynl/submit_column_roles.sh
#   sbatch -J roles_wide_c  -p amdlong -t 3-0 --mem=160G --export=ALL,RUN=wide_content jobs/pynl/submit_column_roles.sh
# Roles share weights across the middle columns: at widths 32,64,128 the model has 288k weights (named columns:
# 1,411k); the wide runs, widths 64,144,288, have 1,390k (1,448k with content gates). EPOCHS overrides the 60.
# RUN=fold / fold_content: the recursive row fold instead (--columns fold, notes §25), 305k weights.
# Roles and fold runs are also scored on rows of 5 and 20 (same values 0..50; generated with IBM's
# iravenx_task.py --n 5 / 20 --maxval 50 --nconf 0, seed 1234), to test row-length transfer.
# RUN=fold_mixed / roles_mixed: trained on rows of 5 and 10 together (6000 puzzles of each, selection 300 of
# each), then scored on rows of 5, 10 and 20: does a second training length make the fold transfer?
#   sbatch -J fold_mixed  --mem=160G --export=ALL,RUN=fold_mixed  jobs/pynl/submit_column_roles.sh
#   sbatch -J roles_mixed --mem=160G --export=ALL,RUN=roles_mixed jobs/pynl/submit_column_roles.sh

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1

cd /home/kassyask/neuro-vector-symbolic-architectures-raven
RUN=${RUN:-plain}
D=data/iravenx/center_single_shuffle_n
IRX=${D}_10_maxval_50.json
COMMON="--template facet --lr 3e-4 --n 6000 --n-sel 300 --n-val 1000 --n-train-eval 300 --train-epochs ${EPOCHS:-60}
        --eval-every 5"
case "${RUN}" in
  plain)        ARGS="--columns roles --widths 32,64,128 --jvm-gb 64"; EVAL_GB=64 ;;
  content)      ARGS="--columns roles --widths 32,64,128 --jvm-gb 64 --copy-links --copy-gates content"; EVAL_GB=64 ;;
  wide)         ARGS="--columns roles --widths 64,144,288 --jvm-gb 120"; EVAL_GB=120 ;;
  wide_content) ARGS="--columns roles --widths 64,144,288 --jvm-gb 120 --copy-links --copy-gates content"; EVAL_GB=120 ;;
  fold)         ARGS="--columns fold --widths 32,64,128 --jvm-gb 64"; EVAL_GB=120 ;;
  fold_content) ARGS="--columns fold --widths 32,64,128 --jvm-gb 64 --copy-links --copy-gates content"; EVAL_GB=120 ;;
  fold_mixed)   ARGS="--columns fold --widths 32,64,128 --jvm-gb 96"; EVAL_GB=120; IRX=${D}_5_maxval_50.json,${D}_10_maxval_50.json ;;
  roles_mixed)  ARGS="--columns roles --widths 32,64,128 --jvm-gb 96"; EVAL_GB=96; IRX=${D}_5_maxval_50.json,${D}_10_maxval_50.json ;;
  *) echo "set RUN to plain, content, wide, wide_content, fold, fold_content, fold_mixed or roles_mixed"; exit 1 ;;
esac
case "${RUN}" in
  *_mixed) SAVE=experiments/facet_iravenx_n5-10_m50_${RUN%_mixed} ;;
  fold*) SAVE=experiments/facet_iravenx_n10_m50_${RUN} ;;
  *)     SAVE=experiments/facet_iravenx_n10_m50_roles$( [ "${RUN}" = plain ] || echo "_${RUN}" ) ;;
esac

echo "node $(hostname), run ${RUN}, saving to ${SAVE}"
python3 -u core/pynl/bench_hub_grounding.py ${COMMON} --iravenx ${IRX} ${ARGS} --save-dir ${SAVE}
python3 -u core/pynl/facet_rule_eval.py ${SAVE} --iravenx ${D}_10_maxval_50.json --transform none cols23 rows12 \
    --jvm-gb ${EVAL_GB} 2>&1 | grep -v "^WARNING\|SLF4J"
for n in 5 20; do
    python3 -u core/pynl/facet_rule_eval.py ${SAVE} --iravenx data/iravenx/center_single_shuffle_n_${n}_maxval_50.json \
        --jvm-gb ${EVAL_GB} 2>&1 | grep -v "^WARNING\|SLF4J"
done
