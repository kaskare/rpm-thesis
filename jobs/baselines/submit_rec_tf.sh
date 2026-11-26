#!/bin/bash -l
#SBATCH --partition amdgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-10%11
#SBATCH --job-name=rec_tf
#SBATCH --output=log/rec_tf_%A_%a.out

# The control for supervision (RESULTS §8, 2026-11-23): the facet model's perception with a black-box reasoner
# (core/baselines/train_baseline.py --arch rec-tf). The ViT slot recognizer trained on the labelled panels of the same
# first N puzzles per layout (experiments/recognizer/vit64_nN; at 6000 the one trained on 2000, as end to end) reads
# the eight context panels; a plain transformer over its slot states answers. Settings are the ViT learning curve's
# (jobs/baselines/submit_curves.sh): epochs 300 * 6000 / N, the same learning rate, decay, warm-up and dropout, so the
# reasoner is trained as the ViT it is compared with: the fixed cosine schedule, as the ViT reruns tagged _cosfix
# (submit_curves.sh, submit_vit.sh with TAGSUFFIX=_cosfix, 2026-11-23).
#   tasks 0-3   frozen recognizer, reasoner width 64 (about the facet model's weights), 300 / 1000 / 3000 / 6000
#   tasks 4-7   frozen recognizer, reasoner width 224 (the larger ViT's width), 300 / 1000 / 3000 / 6000
#   tasks 8-10  recognizer fine-tuned with its label loss kept on the context panels, width 64, 300 / 1000 / 3000
# SMOKE=1: two epochs on 300 puzzles per layout, into experiments/baselines/smoke.

read -r -d '' TABLE <<EOF
frozen 64 300
frozen 64 1000
frozen 64 3000
frozen 64 6000
frozen 224 300
frozen 224 1000
frozen 224 3000
frozen 224 6000
finetune 64 300
finetune 64 1000
finetune 64 3000
EOF
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" <<< "$TABLE")
read -r MODE WIDTH N <<< "$LINE"
F=$((6000 / N))
REC=experiments/recognizer/vit64_n${N}
[ "$N" = 6000 ] && REC=experiments/recognizer/vit64
FT=""; [ "$MODE" = finetune ] && FT="--finetune"
EPOCHS=$((300 * F)); WARM=$((10 * F)); EVERY=$((3 * F)); OUT=experiments/baselines/runs
if [ -n "${SMOKE}" ]; then EPOCHS=2; WARM=1; EVERY=1; OUT=experiments/baselines/smoke; fi

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
echo "task ${SLURM_ARRAY_TASK_ID}: ${MODE} recognizer ${REC}, reasoner width ${WIDTH}, ${N} puzzles per layout, ${EPOCHS} epochs"
python core/baselines/train_baseline.py --data iraven --arch rec-tf --recognizer "$REC" $FT --width "$WIDTH" \
    --seed "${SEED:-0}" --n-train "$N" --epochs "$EPOCHS" --warmup "$WARM" --eval-every "$EVERY" \
    --lr 5e-4 --wd 0.05 --schedule cosine --clip 1 --dropout 0.1 --tag "${MODE}_w${WIDTH}_n${N}" --out "$OUT"
