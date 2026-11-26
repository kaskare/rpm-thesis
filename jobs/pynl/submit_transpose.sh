#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=transpose
#SBATCH --output=log/transpose_%x_%j.out

# Pricing "rules run along rows" (thesis Section 8.7, item 2), with no training. Every model here reads rules along
# rows only, a prior taken from the generator. On I-RAVEN, the context is transposed (panel (r, c) <- (c, r)); the
# answer keeps its place, so the candidates are unchanged and the rules now run down the columns.
#   1. The reference solvers (rpm_reference_solvers.py, ground-truth attributes; all read along rows) as they are and
#      transposed: how much of RAVEN's rules, and of generic row relations, still works when the rules run down the
#      columns. Not a ceiling for rows-only models (the smoke test of 2026-11-22: rules 49.0, primitives 74.9 on 700).
#   2. The I-RAVEN facet checkpoints, as they are (must reproduce RESULTS) and transposed, val puzzles 300..1299.
#   sbatch jobs/pynl/submit_transpose.sh
# CKPTS overrides the checkpoints.

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

CKPTS=${CKPTS:-"experiments/facet_all_w32-64-128_hidden experiments/facet_all_copy_content experiments/facet_ep80_fold_all"}
echo "node $(hostname)"
python3 core/pynl/rpm_reference_solvers.py --n 2000
python3 core/pynl/rpm_reference_solvers.py --n 2000 --transpose
for c in ${CKPTS}; do                                  # one process per checkpoint, so the JVM starts clean each time
    python3 -u core/pynl/facet_rule_eval.py "$c" --n 1000 --transform none transpose --jvm-gb 140 2>&1 | grep -v "^WARNING\|SLF4J"
done
