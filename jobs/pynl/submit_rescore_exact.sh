#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=rescore
#SBATCH --output=log/rescore_%x_%j.out

# Rescores saved I-RAVEN-X facet checkpoints after the exact-scorer fix of 2026-11-08 (bench_hub_grounding.evaluate
# decoded the answer through the one-hot description, which cannot hold the value maxval). Matched must come out
# as in best.json; exact changes. One process per checkpoint, so the JVM starts clean each time:
#   sbatch -J rescore_n3 --export=ALL,CKPTS="experiments/facet_iravenx_n3_m10 experiments/facet_iravenx_n3_m10_copy" \
#       jobs/pynl/submit_rescore_exact.sh

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

[ -n "${CKPTS}" ] || { echo "set CKPTS to the run directories"; exit 1; }
echo "node $(hostname)"
for c in ${CKPTS}; do
    python3 -c "import json; b = json.load(open('$c/best.json')); u = b['unseen']; print('$c: before the fix, epoch', b['epoch'], 'matched', u['matched'], 'exact', u['exact'], 'of 1000')"
    python3 -u core/pynl/facet_rule_eval.py "$c" --transform none --jvm-gb 48 2>&1 | grep -v "^WARNING\|SLF4J"
done
