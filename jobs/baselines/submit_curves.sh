#!/bin/bash -l
#SBATCH --partition amdgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-42%12
#SBATCH --job-name=bl_curves
#SBATCH --output=log/baseline_curves_%A_%a.out

# The baselines on I-RAVEN against the symmetric models, seed 0 (jobs/pynl/submit_facet_curves.sh has the facet model's).
# Settings are the ones the full runs used. Smaller training sets get proportionally more epochs, so every run takes
# the same number of updates and no model loses by being under-trained.
#   tasks 0-17   learning curves at 300 / 1000 / 3000 puzzles per layout: regularized MLP, object GNN, attribute GNNs
#                A and B, ViT at widths 64 and 224
#   tasks 18-25  the ViT with the row symmetry built in, widths 64 and 224, at 300 / 1000 / 3000 / 6000
#   tasks 26-31  transfer: trained and selected without up-down, scored on it
#   tasks 32-34  grids only, width 224: 80 px with 16 px patches (the control), 80 px with 8 px patches, 160 px with 16 px
#   tasks 35-36  every patch with its own filter, width 64, at 1000 and 6000 puzzles
#   tasks 37-42  held-out pair: no puzzle with Progression on Size in training or selection
# Task i: line i of the table. Each line: arch, scale, puzzles per layout, tag, extra flags.
# SEED=k (default 0) for further seeds. The seed runs of 2026-11-22, the ViT at width 64 with and without rows built
# in, 3000 puzzles: for S in 1 2; do sbatch --array=16,22 -J bl_vit3000_s$S --export=ALL,SEED=$S jobs/baselines/submit_curves.sh; done

UD=center_single,distribute_four,distribute_nine,left_center_single_right_center_single,in_center_single_out_center_single,in_distribute_four_out_center_single
GR=distribute_four,distribute_nine,in_distribute_four_out_center_single
read -r -d '' TABLE <<EOF
mlp 1 300 n300
gnn 1 300 n300
attr-abs 1 300 n300
attr-rel 1 300 n300
vit 1 300 n300
vit 16 300 n300
mlp 1 1000 n1000
gnn 1 1000 n1000
attr-abs 1 1000 n1000
attr-rel 1 1000 n1000
vit 1 1000 n1000
vit 16 1000 n1000
mlp 1 3000 n3000
gnn 1 3000 n3000
attr-abs 1 3000 n3000
attr-rel 1 3000 n3000
vit 1 3000 n3000
vit 16 3000 n3000
vit-sym 1 300 n300
vit-sym 16 300 n300
vit-sym 1 1000 n1000
vit-sym 16 1000 n1000
vit-sym 1 3000 n3000
vit-sym 16 3000 n3000
vit-sym 1 6000 n6000
vit-sym 16 6000 n6000
mlp 1 6000 noUD --layouts $UD
gnn 1 6000 noUD --layouts $UD
attr-abs 1 6000 noUD --layouts $UD
attr-rel 1 6000 noUD --layouts $UD
vit 16 6000 noUD --layouts $UD
vit-sym 16 6000 noUD --layouts $UD
vit 16 6000 grids_p16_s80 --width 224 --layouts $GR
vit 16 6000 grids_p8_s80 --width 224 --layouts $GR --patch 8 --accum 4
vit 16 6000 grids_p16_s160 --width 224 --layouts $GR --image-size 160 --accum 4
vit 1 1000 unshared_n1000 --width 64 --unshared-patches
vit 1 6000 unshared_n6000 --width 64 --unshared-patches
mlp 1 6000 holdPS --hold-pair Progression:Size
gnn 1 6000 holdPS --hold-pair Progression:Size
attr-abs 1 6000 holdPS --hold-pair Progression:Size
attr-rel 1 6000 holdPS --hold-pair Progression:Size
vit 16 6000 holdPS --hold-pair Progression:Size
vit-sym 16 6000 holdPS --hold-pair Progression:Size
EOF
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" <<< "$TABLE")
read -r ARCH SCALE N TAG FLAGS <<< "$LINE"
F=$((6000 / N))
case "$ARCH" in
  mlp|gnn)  SET="--norm --wd 0.5 --schedule cosine --clip 1 --dropout 0"; WARM=5 ;;
  attr-*)   SET="--batch 32 --wd 0.5 --schedule cosine --clip 1 --dropout 0"; WARM=5 ;;
  *)        SET="--lr 5e-4 --wd 0.05 --schedule cosine --clip 1 --dropout 0.1"; WARM=10 ;;
esac

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
python core/baselines/train_baseline.py --data iraven --arch "$ARCH" --seed "${SEED:-0}" --scale "$SCALE" --n-train "$N" \
    --epochs $((300 * F)) --warmup $((WARM * F)) --eval-every $((3 * F)) $SET --tag "$TAG" $FLAGS
python core/baselines/rule_eval.py experiments/baselines/runs/iraven_${ARCH}$([ "$SCALE" = 1 ] || echo _x$SCALE)_${TAG}_s${SEED:-0}
