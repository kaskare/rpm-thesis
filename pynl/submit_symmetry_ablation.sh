#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=sym_abl
#SBATCH --output=log/facet_symabl_%x_%j.out

# Symmetry ablations (notes §26): the facet model with one built-in symmetry removed. Arms:
#   base  the default template (symmetric)
#   lab   one-hot facts name what the symmetry makes interchangeable (--label-components / --label-attributes);
#         the weights stay shared, nearly equal capacity
#   sep   separate weights per component / per attribute (--per-comp-weights / --per-attr-weights), same widths:
#         no weight sharing either, 2x / 3.2x the weights (§26.1)
#   sepm  as sep, narrower, at the default's weight count (lrud 22,45,90: 266k; irx 18,35,69: 288k)
# Trained at a small and a full training set, then scored on the unseen puzzles as they are and transformed.
#   components: I-RAVEN left-right + up-down (their two components are a data symmetry, §22), the default
#               template vs --label-components, at 100 / 1000 / 6000 puzzles per layout; scored also mirrored.
#   attributes: I-RAVEN-X 3x10 values 0..50 with column roles (the base at 6000 is facet_iravenx_n10_m50_roles),
#               vs --label-attributes, at 600 / 6000 puzzles; scored also with size <-> colour swapped.
# Small training sets get more epochs; the epoch is picked on the selection puzzles as always. One run per job:
#   for r in lrud_base_100 lrud_lab_100 lrud_base_1000 lrud_lab_1000 lrud_base_6000 lrud_lab_6000 \
#            irx_base_600 irx_lab_600 irx_lab_6000; do sbatch -J $r --export=ALL,RUN=$r pynl/submit_symmetry_ablation.sh; done
#   for a in sep sepm; do for r in lrud_${a}_100 lrud_${a}_1000 lrud_${a}_6000 irx_${a}_600 irx_${a}_6000; do
#       sbatch -J $r --export=ALL,RUN=$r pynl/submit_symmetry_ablation.sh; done; done

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

IFS=_ read -r SET ARM N <<< "${RUN}"
case "${N}" in
  100) EP=300; EV=10 ;;
  600) EP=200; EV=10 ;;
  1000) EP=120; EV=5 ;;
  *) EP=60; EV=5 ;;
esac
COMMON="--template facet --lr 3e-4 --n ${N} --n-sel 300 --n-val 1000 --n-train-eval 300
        --train-epochs ${EP} --eval-every ${EV}"
case "${SET}" in
  lrud) DATA="--constellation left_center_single_right_center_single,up_center_single_down_center_single --jvm-gb 64"
        LABEL="--label-components"; SEP="--per-comp-weights"; NARROW=22,45,90; TRANSFORMS="none mirror" ;;
  irx)  DATA="--iravenx data/iravenx/center_single_shuffle_n_10_maxval_50.json --columns roles --jvm-gb 64"
        LABEL="--label-attributes"; SEP="--per-attr-weights"; NARROW=18,35,69; TRANSFORMS="none sizecolor" ;;
  *) echo "set RUN to <lrud|irx>_<base|lab>_<n>"; exit 1 ;;
esac
case "${ARM}" in
  base) ARGS="--widths 32,64,128" ;;
  lab)  ARGS="--widths 32,64,128 ${LABEL}" ;;
  sep)  ARGS="--widths 32,64,128 ${SEP}" ;;
  sepm) ARGS="--widths ${NARROW} ${SEP}" ;;
  *) echo "arm must be base, lab, sep or sepm"; exit 1 ;;
esac
SAVE=experiments/facet_symabl_${RUN}

echo "node $(hostname), run ${RUN}, saving to ${SAVE}"
python3 -u pynl/bench_hub_grounding.py ${COMMON} ${DATA} ${ARGS} --save-dir ${SAVE}
python3 -u pynl/facet_rule_eval.py ${SAVE} --transform ${TRANSFORMS} --jvm-gb 64 2>&1 | grep -v "^WARNING\|SLF4J"
