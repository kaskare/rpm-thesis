#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=3-0:0:0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com
#SBATCH --job-name=facet150
#SBATCH --output=log/facet150_%x_%j.out

# The facet model (notes §18) for 150 epochs, same settings as the 60/80-epoch runs.
# One run per job, chosen by RUN; pass the memory to sbatch:
#   sbatch --mem=360G --export=ALL,RUN=all   -J f150_all   pynl/submit_facet_long.sh   # ~27 h
#   sbatch --mem=96G  --export=ALL,RUN=m50   -J f150_m50   pynl/submit_facet_long.sh   # ~13 h
#   sbatch --mem=96G  --export=ALL,RUN=m100  -J f150_m100  pynl/submit_facet_long.sh   # ~13 h
#   sbatch --mem=180G --export=ALL,RUN=m1000 -J f150_m1000 pynl/submit_facet_long.sh   # ~25 h
#   sbatch --mem=360G --export=ALL,RUN=fold_all,EPOCHS=80 -J fold_all pynl/submit_facet_long.sh  # ~16 h
# EPOCHS overrides the 150. RUN=fold_all: the recursive row fold (--columns fold, notes §25) on I-RAVEN, 80 epochs
# like facet_all_w32-64-128_hidden, then scored once per rule on the unseen puzzles.

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1

cd /home/kassyask/neuro-vector-symbolic-architectures-raven
EPOCHS=${EPOCHS:-150}
COMMON="--template facet --widths 32,64,128 --lr 3e-4 --n 6000 --n-sel 300 --n-val 1000 --n-train-eval 300 \
        --train-epochs ${EPOCHS} --eval-every 5"

case "${RUN}" in
  all)   ARGS="--constellation all --jvm-gb 300" ;;
  fold_all) ARGS="--constellation all --columns fold --jvm-gb 300" ;;
  m50)   ARGS="--iravenx data/iravenx/center_single_shuffle_n_10_maxval_50.json --jvm-gb 64" ;;
  m100)  ARGS="--iravenx data/iravenx/center_single_shuffle_n_10_maxval_100.json --jvm-gb 64" ;;
  m1000) ARGS="--iravenx data/iravenx/center_single_shuffle_n_10_maxval_1000.json --jvm-gb 140" ;;
  *) echo "set RUN to all, fold_all, m50, m100 or m1000"; exit 1 ;;
esac
SAVE=experiments/facet_ep${EPOCHS}_${RUN}

echo "node $(hostname), run ${RUN}, ${EPOCHS} epochs, saving to ${SAVE}"
python3 -u pynl/bench_hub_grounding.py ${COMMON} ${ARGS} --save-dir ${SAVE}
if [ "${RUN}" = fold_all ]; then
    python3 -u pynl/facet_rule_eval.py ${SAVE} --transform none --jvm-gb 300 2>&1 | grep -v "^WARNING\|SLF4J"
fi
