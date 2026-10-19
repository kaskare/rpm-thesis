#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com
#SBATCH --job-name=pixels
#SBATCH --output=log/slurm_pixels_%x_%j.out
#SBATCH --error=log/slurm_pixels_%x_%j.err

# End-to-end I-RAVEN from pixels: ResNet-18 perception -> PyNeuraLogic reasoning.
# See pixel_pipeline.py for what each stage does. The two stages need different
# environments and hardware, so they run as two chained jobs:
#
#   STAGE=perceive  GPU, conda env nvsa. Both frontend checkpoints over the I-RAVEN val
#                   and test images (28,000 puzzles, 448,000 panels, images loaded once).
#   STAGE=solve     CPU, conda env pynl. Picks the frontend with the better VALIDATION
#                   perception, then scores the test split from pixels and from XML with
#                   the same PyNeuraLogic checkpoint.
#
# Submit (solve only starts if perceive succeeded):
#   p=$(sbatch --parsable --partition=amdgpufast --gres=gpu:1 --cpus-per-task=16 --mem=64G \
#              --time=3:00:00 --export=ALL,STAGE=perceive submit_pixels.sh)
#   sbatch --dependency=afterok:$p --partition=amdfast --cpus-per-task=8 --mem=64G \
#          --time=4:00:00 --export=ALL,STAGE=solve submit_pixels.sh

echo_green() { echo -e "\033[0;32m [$(date)] $1\033[0m"; }

export ROOT_PATH="/home/kassyask/neuro-vector-symbolic-architectures-raven"
export PYTHONPATH="${ROOT_PATH}:$PYTHONPATH"
export PYTHONNOUSERSITE=1
mkdir -p "${ROOT_PATH}/log"
cd "${ROOT_PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

case "${STAGE}" in
  perceive)
    conda activate nvsa
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo_green "node $(hostname) — PERCEIVE: ResNet-18 frontends over I-RAVEN val+test"
    srun python3 -u pixel_pipeline.py perceive --splits val,test --workers 14 --batch 16
    ;;
  solve)
    ml load Java/21 2>/dev/null || true
    if [ -z "${JAVA_HOME}" ] || [ ! -f "${JAVA_HOME}/lib/server/libjvm.so" ]; then
        for cand in /mnt/appl/software/Java/21.0.8 /mnt/appl/software/Java/21.0.2; do
            if [ -f "$cand/lib/server/libjvm.so" ]; then export JAVA_HOME="$cand"; break; fi
        done
    fi
    if [ ! -f "${JAVA_HOME}/lib/server/libjvm.so" ]; then
        echo "FATAL: no libjvm.so; JAVA_HOME='${JAVA_HOME}'" >&2; exit 1
    fi
    export PATH="${JAVA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH}"
    conda activate pynl
    echo_green "node $(hostname) — SOLVE: PyNeuraLogic on perceived objects, I-RAVEN test"
    srun python3 -u pixel_pipeline.py solve --frontend auto --split test --jvm-gb 40
    ;;
  *)
    echo "FATAL: set STAGE=perceive or STAGE=solve (got '${STAGE}')" >&2; exit 1 ;;
esac

echo_green "Stage ${STAGE} finished."
