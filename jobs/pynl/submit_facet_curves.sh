#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0:0:0
#SBATCH --job-name=facet_cur
#SBATCH --output=log/facet_curves_%x_%j.out

# The facet model on I-RAVEN for the comparisons with the baselines (jobs/baselines/submit_curves.sh): the learning curve
# at 300 / 1000 / 3000 puzzles per layout (6000 is facet_all_w32-64-128_hidden), the transfer test (trained without
# up-down, scored on it) and the held-out pair (no puzzle with Progression on Size in training or selection). Small
# training sets get more epochs, as in submit_symmetry_ablation.sh; the epoch is picked on selection puzzles. One run
# per job, with its memory:
#   sbatch --mem=120G -J n300 --export=ALL,RUN=n300 jobs/pynl/submit_facet_curves.sh
#   sbatch --mem=160G -J n1000 --export=ALL,RUN=n1000 jobs/pynl/submit_facet_curves.sh
#   sbatch --mem=250G -J n3000 --export=ALL,RUN=n3000 jobs/pynl/submit_facet_curves.sh
#   sbatch --mem=340G -J noUD --export=ALL,RUN=noUD jobs/pynl/submit_facet_curves.sh
#   sbatch --mem=340G -J holdPS --export=ALL,RUN=holdPS jobs/pynl/submit_facet_curves.sh

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

COMMON="--template facet --widths 32,64,128 --lr 3e-4 --n-sel 300 --n-val 1000 --n-train-eval 300"
UD=up_center_single_down_center_single
NOUD=center_single,distribute_four,distribute_nine,left_center_single_right_center_single,in_center_single_out_center_single,in_distribute_four_out_center_single
EVAL="--transform none"
case "${RUN}" in
  n300)   ARGS="--n 300 --train-epochs 200 --eval-every 10 --constellation all"; JVM=96 ;;
  n1000)  ARGS="--n 1000 --train-epochs 120 --eval-every 5 --constellation all"; JVM=128 ;;
  n3000)  ARGS="--n 3000 --train-epochs 80 --eval-every 5 --constellation all"; JVM=200 ;;
  noUD)   ARGS="--n 6000 --train-epochs 60 --eval-every 5 --constellation ${NOUD}"; JVM=280; EVAL="--transform none --layouts ${UD}" ;;
  holdPS) ARGS="--n 6000 --train-epochs 60 --eval-every 5 --constellation all --hold-pair Progression:Size"; JVM=280 ;;
  *) echo "set RUN to n300, n1000, n3000, noUD or holdPS"; exit 1 ;;
esac
SAVE=experiments/facet_cmp_${RUN}

echo "node $(hostname), run ${RUN}, saving to ${SAVE}"
python3 -u core/pynl/bench_hub_grounding.py ${COMMON} ${ARGS} --jvm-gb ${JVM} --save-dir ${SAVE}
python3 -u core/pynl/facet_rule_eval.py ${SAVE} ${EVAL} --jvm-gb ${JVM} 2>&1 | grep -v "^WARNING\|SLF4J"
