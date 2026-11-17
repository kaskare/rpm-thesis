#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=wrong_sym
#SBATCH --output=log/wrong_sym_%x_%j.out

# Wrong-symmetry controls (plan E4): the facet model on I-RAVEN-X 3x3, values 0..10, with a symmetry built in that the
# data does not have, against the same model without it (experiments/facet_iravenx_n3_m10: 99.2 matched, 6000 puzzles).
#   tie12      columns 1 and 2 tied (--columns tie12). Swapping them takes Progression rows out of the data's support, so
#              the prediction is no ceiling: the answer is still determined (Progression from the multiset and the last).
#   fold       v and 10 - v get the same input code (--fold-values). Reflection keeps Constant, Progression and
#              Distribute valid with a different answer, so they are capped at 54-56% exact; Arithmetic has no forced cap.
#   *_600      the same at 600 puzzles, with as many updates (600 epochs), against named_600: does the tie cost data?
# Both symmetries are exact with random weights (max output difference 0 under columns 1 <-> 2 and under reflection).
#   for r in tie12 fold tie12_600 named_600 fold_600; do sbatch -J ws_$r --export=ALL,RUN=$r jobs/pynl/submit_wrong_symmetry.sh; done

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

X=data/iravenx/center_single_shuffle_n_3_maxval_10.json
case "${RUN}" in
  tie12)      N=6000; EP=60;  EV=5;  ARGS="--columns tie12" ;;
  fold)       N=6000; EP=60;  EV=5;  ARGS="--fold-values" ;;
  tie12_600)  N=600;  EP=600; EV=50; ARGS="--columns tie12" ;;
  named_600)  N=600;  EP=600; EV=50; ARGS="" ;;
  fold_600)   N=600;  EP=600; EV=50; ARGS="--fold-values" ;;
  *) echo "set RUN to tie12, fold, tie12_600, named_600 or fold_600"; exit 1 ;;
esac
SAVE=experiments/wrongsym_n3_m10_${RUN}
echo "node $(hostname), run ${RUN}, saving to ${SAVE}"
python3 -u core/pynl/bench_hub_grounding.py --template facet --iravenx ${X} --n ${N} --n-sel 300 --n-val 1000 --widths 32,64,128 \
    --lr 3e-4 --train-epochs ${EP} --eval-every ${EV} --jvm-gb 32 --save-dir ${SAVE} ${ARGS} 2>&1 | grep --line-buffered -v "^WARNING\|SLF4J"
python3 -u core/pynl/facet_rule_eval.py ${SAVE} --transform none --jvm-gb 32 2>&1 | grep -v "^WARNING\|SLF4J"
