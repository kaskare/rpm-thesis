#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=arith_op
#SBATCH --output=log/arith_op_%x_%j.out

# Does taking away the choice of operator lift Arithmetic on I-RAVEN-X 3x10? Column roles, the model with 40.5 exact on it,
# with Arithmetic only plus or only minus (data_prep/arith_op_data.py). Settings as facet_iravenx_n10_m50_roles and the same number
# of updates, 78 epochs on about 4600 puzzles against 60 on 6000. One run per job:
#   sbatch -J arith_plus  --export=ALL,OP=plus  jobs/pynl/submit_arith_op.sh
#   sbatch -J arith_minus --export=ALL,OP=minus jobs/pynl/submit_arith_op.sh
# The run and the two-operator checkpoint are then scored on the same unseen puzzles, so the reference is that operator's score.

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

case "${OP}" in
  plus)  N=4622; SEL=229; VAL=751 ;;
  minus) N=4588; SEL=220; VAL=769 ;;
  *) echo "set OP to plus or minus"; exit 1 ;;
esac
X=data/iravenx_ops/${OP}/center_single_shuffle_n_10_maxval_50.json
SAVE=experiments/facet_iravenx_n10_m50_roles_${OP}
echo "node $(hostname), column roles, ${OP} only, saving to ${SAVE}"
python3 -u core/pynl/bench_hub_grounding.py --template facet --lr 3e-4 --n ${N} --n-sel ${SEL} --n-val ${VAL} --n-train-eval 300 \
    --train-epochs ${EPOCHS:-78} --eval-every 6 --iravenx ${X} --columns roles --widths 32,64,128 --jvm-gb 64 --save-dir ${SAVE}
python3 -u core/pynl/facet_rule_eval.py ${SAVE} --n ${VAL} --jvm-gb 64 2>&1 | grep -v "^WARNING\|SLF4J"
python3 -u core/pynl/facet_rule_eval.py experiments/facet_iravenx_n10_m50_roles --iravenx ${X} --n ${VAL} --jvm-gb 64 2>&1 | grep -v "^WARNING\|SLF4J"
