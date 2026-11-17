#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=facet_e2e
#SBATCH --output=log/facet_e2e_%x_%j.out

# The facet model end to end on I-RAVEN (core/pynl/facet_e2e_eval.py): context panels and candidates from the ViT slot
# recognizer (experiments/recognizer/vit64, jobs/frontend/submit_recognizer.sh) instead of the XML labels. Three passes:
# labels (must reproduce the checkpoint's record on the unseen split), hard, soft. Writes <CKPT>/e2e_<SPLIT>.json.
#   sbatch jobs/pynl/submit_facet_e2e.sh                               # unseen = val 300..1299, the plain facet model
#   CKPT=experiments/facet_all_copy_content sbatch jobs/pynl/submit_facet_e2e.sh
# The test split is scored once per final checkpoint, with the mapping picked on the unseen split:
#   SPLIT=test MODES=labels,hard sbatch jobs/pynl/submit_facet_e2e.sh

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

CKPT=${CKPT:-experiments/facet_all_w32-64-128_hidden}
SPLIT=${SPLIT:-unseen}
MODES=${MODES:-labels,hard,soft}
echo "node $(hostname), checkpoint ${CKPT}, split ${SPLIT}, passes ${MODES}"
python3 -u core/pynl/facet_e2e_eval.py ${CKPT} --split ${SPLIT} --modes ${MODES} --jvm-gb 64 2>&1 | grep --line-buffered -v "^WARNING\|SLF4J"
exit ${PIPESTATUS[0]}
