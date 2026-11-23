#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --job-name=test_pass
#SBATCH --output=log/test_pass_%x_%j.out

# The test pass of Chapters 6 and 7 (2026-11-23, after an external review): every final checkpoint behind a number in
# those chapters, scored once on puzzles no run had touched. I-RAVEN-X and the rule twins: puzzles 8000..9999 of the
# training file; I-RAVEN (the components ablation on left-right and up-down): the test split 0..1999 per layout.
# Each checkpoint writes <ckpt>/test.json, and a checkpoint that has one is skipped, so the pass cannot be repeated.
# LIST: a file of checkpoint directories (experiments/test_pass/*.txt); KIND: facet (core/pynl/facet_rule_eval.py
# --test) or valaxis (core/pynl/value_axis.py --eval --eval-start 8000). One checkpoint at a time, so one failure
# skips only itself.
#   sbatch --export=ALL,LIST=experiments/test_pass/irx_facet.txt,KIND=facet jobs/pynl/submit_test_pass.sh

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

N=${N:-2000}
JVM=${JVM:-96}
echo "node $(hostname), list ${LIST}, kind ${KIND}, ${N} puzzles (per layout on I-RAVEN)"
while read -r d; do
  [ -z "${d}" ] && continue
  if [ -f "${d}/test.json" ]; then echo "${d}: already scored, skipped"; continue; fi
  case "${KIND}" in
    facet)   python3 -u core/pynl/facet_rule_eval.py "${d}" --test --n ${N} --jvm-gb ${JVM} 2>&1 | grep --line-buffered -v "^WARNING\|SLF4J" ;;
    valaxis) X=$(python3 -c "import json; print(json.load(open('${d}/best.json'))['args']['iravenx'])")
             python3 -u core/pynl/value_axis.py --iravenx "${X}" --eval "${d}" --eval-start 8000 --n-val ${N} --jvm-gb ${JVM} \
                 2>&1 | grep --line-buffered -v "^WARNING\|SLF4J" ;;
    *) echo "set KIND to facet or valaxis"; exit 1 ;;
  esac
  [ -f "${d}/test.json" ] || echo "${d}: FAILED, no test.json"
done < "${LIST}"
