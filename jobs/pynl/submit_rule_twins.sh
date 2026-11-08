#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=twins
#SBATCH --output=log/twins_%x_%j.out

# Is Arithmetic hard because it has no value symmetry? Rows of 3, values 0..50, the fourth rule swapped (data_prep/rule_twins.py,
# docs/ARITHMETIC.md). One run per job, with RULE = sum, midpoint, mirror or lookup and N = puzzles:
#   shared    value axis, one weight per offset on a ring (exactly shift-equivariant), reading both operands
#   pervalue  the same with weights per value: the value symmetry removed
#   facet     the plain facet model (thermometer codes, no value symmetry)
#   lplus     shared, plus the plus facts used by learned rules (--learned-plus): sum's own symmetry, the moves of one
#             operand with the answer, given as background knowledge; the kernel stays a ring
# No run but lplus is given plus. The value axis takes 200 epochs at 100 and 600 puzzles and 60 at 6000, as before; the facet
# model takes the same number of updates at every size.
#   sbatch -J sh_midpoint_600 --export=ALL,RUN=shared,RULE=midpoint,N=600 jobs/pynl/submit_rule_twins.sh
# SELECT=exact picks the facet model's epoch by exact on the selection puzzles (it saves one checkpoint); matched sits at
# 97-99% here, so picking by it is noise.

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

X=data/iravenx_twins/${RULE}/center_single_shuffle_n_3_maxval_50.json
EPOCHS=${EPOCHS:-$([ "${N}" -ge 6000 ] && echo 60 || echo 200)}
EVERY=${EVERY:-$([ "${N}" -ge 6000 ] && echo 5 || echo 10)}
SAVE=experiments/twins/${RUN}_${RULE}_n${N}${SELECT:+_sel${SELECT}}
echo "node $(hostname), ${RUN} on ${RULE}, ${N} puzzles, saving to ${SAVE}"
case "${RUN}" in
  shared)   python3 -u core/pynl/value_axis.py --iravenx ${X} --n ${N} --n-val ${NVAL:-1000} --train-epochs ${EPOCHS} --eval-every ${EVERY} \
                --no-plus --pair --kernel 50 --ring --jvm-gb ${JVM:-64} --save-dir ${SAVE} ;;
  lplus)    python3 -u core/pynl/value_axis.py --iravenx ${X} --n ${N} --n-val ${NVAL:-1000} --train-epochs ${EPOCHS} --eval-every ${EVERY} \
                --learned-plus --pair --kernel 50 --ring --jvm-gb ${JVM:-64} --save-dir ${SAVE} ;;
  pervalue) python3 -u core/pynl/value_axis.py --iravenx ${X} --n ${N} --n-val ${NVAL:-1000} --train-epochs ${EPOCHS} --eval-every ${EVERY} \
                --no-plus --pair --kernel 50 --per-value --jvm-gb ${JVM:-64} --save-dir ${SAVE} ;;
  facet)    FE=$((60 * 6000 / N)); FV=$((FE / 12))
            python3 -u core/pynl/bench_hub_grounding.py --template facet --widths 32,64,128 --lr 3e-4 --n ${N} --n-sel 300 --select ${SELECT:-matched} \
                --n-val ${NVAL:-1000} --n-train-eval 300 --train-epochs ${EPOCHS_F:-${FE}} --eval-every ${EVERY_F:-${FV}} --iravenx ${X} --jvm-gb ${JVM:-64} --save-dir ${SAVE}
            python3 -u core/pynl/facet_rule_eval.py ${SAVE} --transform none --jvm-gb ${JVM:-64} 2>&1 | grep -v "^WARNING\|SLF4J" ;;
  *) echo "set RUN to shared, lplus, pervalue or facet"; exit 1 ;;
esac
