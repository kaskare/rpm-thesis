#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=valaxis
#SBATCH --output=log/valaxis_%x_%j.out

# The value-axis model (pynl/value_axis.py, notes §27) on I-RAVEN-X 3x10, values 0..49. One run per job:
#   smoke    500 puzzles, 20 epochs, 300 unseen
#   plus     6000 puzzles, 60 epochs (the full model)
#   noplus   6000 puzzles, without the plus relation (exactly shift-equivariant)
#   plus600  600 puzzles, 200 epochs (training data; the column-roles facet model: 66.3% at 600)
#   plus100  100 puzzles, 200 epochs
#   pervalue, pervalue600, pervalue100   the same with weights per value (--per-value): the value symmetry
#            removed, 335k weights (the facet model with column roles: 288k); not scored on values 0..99
#   findzero, findzero600, findzero100   no plus table: sums counted from every candidate zero, the zero found
#            per puzzle as the one both complete rows check out under (--find-zero); 6000 puzzles need
#            JVM=150 and --mem=180G
# After training: scored on values 0..99 (a range it never saw; the template has no weight per value) and on
# rows of 5 and 20.
#   for r in plus noplus plus600 plus100 pervalue pervalue600 pervalue100 findzero600 findzero100; do sbatch -J va_$r --export=ALL,RUN=$r pynl/submit_value_axis.sh; done

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

X=data/iravenx/center_single_shuffle_n_10_maxval_50.json
case "${RUN}" in
  smoke)   ARGS="--n 500 --train-epochs 20 --n-val 300" ;;
  plus)    ARGS="--n 6000 --train-epochs 60" ;;
  noplus)  ARGS="--n 6000 --train-epochs 60 --no-plus" ;;
  plus600) ARGS="--n 600 --train-epochs 200 --eval-every 10" ;;
  plus100) ARGS="--n 100 --train-epochs 200 --eval-every 10" ;;
  pervalue)    ARGS="--n 6000 --train-epochs 60 --per-value" ;;
  pervalue600) ARGS="--n 600 --train-epochs 200 --eval-every 10 --per-value" ;;
  pervalue100) ARGS="--n 100 --train-epochs 200 --eval-every 10 --per-value" ;;
  findzero)    ARGS="--n 6000 --train-epochs 60 --find-zero" ;;
  findzero600) ARGS="--n 600 --train-epochs 200 --eval-every 10 --find-zero" ;;
  findzero100) ARGS="--n 100 --train-epochs 200 --eval-every 10 --find-zero" ;;
  *) echo "RUN must be smoke, plus, noplus, plus600, plus100, pervalue[600|100] or findzero[600|100]"; exit 1 ;;
esac
SAVE=experiments/valaxis_n10_m50_${RUN}

echo "node $(hostname), run ${RUN}, saving to ${SAVE}"
python3 -u pynl/value_axis.py --iravenx ${X} --dim 32 --lr 1e-3 ${ARGS} --save-dir ${SAVE} --jvm-gb ${JVM:-64} 2>&1 \
    | grep -v "^WARNING\|SLF4J"
[ "${RUN}" = smoke ] && exit 0
RANGES=center_single_shuffle_n_10_maxval_100.json
[[ "${RUN}" == pervalue* ]] && RANGES=""     # weights tied to values 0..50
for f in center_single_shuffle_n_10_maxval_50.json ${RANGES} \
         center_single_shuffle_n_5_maxval_50.json center_single_shuffle_n_20_maxval_50.json; do
    python3 -u pynl/value_axis.py --iravenx data/iravenx/${f} --eval ${SAVE} --jvm-gb 64 2>&1 | grep -v "^WARNING\|SLF4J"
done
