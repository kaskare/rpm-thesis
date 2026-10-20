#!/bin/bash -l
#SBATCH --partition amdgpulong
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --nodes=1

#SBATCH --job-name=nvsa

ml load Anaconda3
conda activate nvsa

EXPNR=000
SAVEPATH="/home/kassyask/neuro-vector-symbolic-architectures-raven/experiments"
DATAPATH="/home/kassyask/neuro-vector-symbolic-architectures-raven/data/RAVEN-10000"
DATAPATHIRAVEN="/home/kassyask/data/I-RAVEN/"
EXPNAME="nvsa_universal_c"
EXPPATH="${SAVEPATH}/${EXPNAME}/"
SEED=1234
RUN=0
batch_size=16
num_workers=0

# NVSA specific backend
# source /opt/share/anaconda3-2019.03/x86_64/bin/activate py37_torch111

# # Center
# #jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_6h \
# python raven/main_nvsa_marg.py --mode train --config center_single --epochs 50 --s 7 --trainable-s \
#  --exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN --batch-size $batch_size 

# # 2x2
# #jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_24h \
# python raven/main_nvsa_marg.py --mode train --config distribute_four --epochs 150 --s 6 --trainable-s \
# --exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN --batch-size $batch_size 

# # 3x3
# #jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_24h \
# python raven/main_nvsa_marg.py --mode train --config distribute_nine --epochs 150 --s 2 --trainable-s \
# --exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN --batch-size $batch_size

# # Left-right
# #jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_12h \
# python raven/main_nvsa_marg.py --mode train --config left_center_single_right_center_single --epochs 100 --s 5 --trainable-s \
# --exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN --batch-size $batch_size

# # Up-down
# #jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_12h \
# python raven/main_nvsa_marg.py --mode train --config up_center_single_down_center_single --epochs 100 --s 5 --trainable-s \
# --exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN --batch-size $batch_size

# Out-in center
# jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_12h \
# python raven/main_nvsa_marg.py --mode train --config in_center_single_out_center_single --epochs 100 --s 5 --trainable-s \
# --exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN --batch-size $batch_size

# Out-in grid
# jbsub -cores 8+1 -mem 32g -require a100 -name "j${EXPNR}_${RUN}" -queue x86_12h \
# python raven/main_nvsa_marg.py --mode train --config in_distribute_four_out_center_single --epochs 100 --s 5 --trainable-s \
# --exp_dir $EXPPATH --dataset $DATAPATH --dataset-i-raven $DATAPATHIRAVEN  --seed $SEED --run $RUN --batch-size $batch_size

# python3 frontend/train_frontend.py  --dataset data/RAVEN-10000\
#    --load-from log/universal_frontend_correct_proportional_mean3 \
#    --save-dir log/universal_frontend_correct_proportional_mean3  --epochs 200 \
#     --batch-size 16 --lr 0.001 --num-workers 2 --per-object-loss --loss-reduction mean

# Step 1: Precompute frontend features (run once, then comment out)
# python3 torch_gnn/gnn_solver.py \
#     --frontend-checkpoint log/universal_frontend_correct_proportional_mean1 \
#     --dataset data/RAVEN-10000 \
#     --save-dir log/gnn_solver \
#     --mode precompute --device cuda:0

# Step 2: Train GNN (config: configs/gnn_solver.yaml)
# python3 torch_gnn/train_object_vsa_gnn_frontend.py \
#   --dataset-root data/RAVEN-10000 \
#   --frontend-checkpoint log/universal_frontend_correct_proportional_mean1 \
#   --device cuda:0 \
#   --frontend-device cuda:0


# Run mode: "single" = one config (set granular/object_only/unified_graph below); "all" = all 8 graph representation combos
# Usage: ./run.sh (single)  |  RUN_MODE=all ./run.sh  |  RUN_MODE=all sbatch run.sh
# Or: ./scripts/train_all_gnn.sh  (standalone script with same 8 commands)
RUN_MODE="${RUN_MODE:-single}"

# Experiment hyperparameters (used in save dir and command)
lr=0.001
node_repr=embedding
gnn_type=gnn
granular=false
object_only=false
unified_graph=false
embed_dim=256
hidden_dim=384
epochs=402

# Save dir: object_vsa_gnn_frontend_{gnn_type}[_granular]/lr{lr}_{node_repr}[_emb{dim}]_h{hidden}_ep{epochs}
SAVE_SUBFOLDER="lr${lr}_${node_repr}"
[ "$node_repr" = "embedding" ] && SAVE_SUBFOLDER="${SAVE_SUBFOLDER}_emb${embed_dim}"
SAVE_SUBFOLDER="${SAVE_SUBFOLDER}_h${hidden_dim}_ep${epochs}"
SAVE_BASE="object_vsa_gnn_frontend_${gnn_type}"
[ "$granular" = "true" ] && SAVE_BASE="${SAVE_BASE}_granular"
[ "$object_only" = "true" ] && SAVE_BASE="${SAVE_BASE}_object_only"
[ "$unified_graph" = "true" ] && SAVE_BASE="${SAVE_BASE}_unified"
SAVE_DIR="${SAVEPATH}/${SAVE_BASE}/${SAVE_SUBFOLDER}"

GRANULAR_FLAG=""
[ "$granular" = "true" ] && GRANULAR_FLAG="--granular"
OBJECT_ONLY_FLAG=""
[ "$object_only" = "true" ] && OBJECT_ONLY_FLAG="--object-only"
UNIFIED_FLAG=""
[ "$unified_graph" = "true" ] && UNIFIED_FLAG="--unified-graph"

# ========== Single config (set granular/object_only/unified_graph above) ==========
run_single() {
  python3 torch_gnn/train_object_vsa_gnn_frontend.py --dataset-root data/RAVEN-10000 --use-gt --node-repr "$node_repr" \
    --gnn-type "$gnn_type" $GRANULAR_FLAG $OBJECT_ONLY_FLAG $UNIFIED_FLAG --lr $lr --embed-dim $embed_dim --hidden-dim $hidden_dim \
    --save-dir "$SAVE_DIR" --epochs $epochs --batch-size 16 --num-workers 2
}

# ========== Train ALL graph representation configs (8 combinations) ==========
# granular x object_only x unified_graph: baseline, granular, object_only, unified, granular+obj, granular+unified, obj+unified, all
train_all_configs() {
  local lr=0.001 node_repr=embedding gnn_type=gnn embed_dim=256 hidden_dim=384 epochs=402
  local SAVE_SUBFOLDER SAVE_BASE SAVE_DIR GRANULAR_FLAG OBJECT_ONLY_FLAG UNIFIED_FLAG
  for granular in false true; do
    for object_only in false true; do
      for unified_graph in false true; do
        SAVE_SUBFOLDER="lr${lr}_${node_repr}_emb${embed_dim}_h${hidden_dim}_ep${epochs}"
        SAVE_BASE="object_vsa_gnn_frontend_${gnn_type}"
        [ "$granular" = "true" ] && SAVE_BASE="${SAVE_BASE}_granular"
        [ "$object_only" = "true" ] && SAVE_BASE="${SAVE_BASE}_object_only"
        [ "$unified_graph" = "true" ] && SAVE_BASE="${SAVE_BASE}_unified"
        SAVE_DIR="${SAVEPATH}/${SAVE_BASE}/${SAVE_SUBFOLDER}"
        GRANULAR_FLAG=""; [ "$granular" = "true" ] && GRANULAR_FLAG="--granular"
        OBJECT_ONLY_FLAG=""; [ "$object_only" = "true" ] && OBJECT_ONLY_FLAG="--object-only"
        UNIFIED_FLAG=""; [ "$unified_graph" = "true" ] && UNIFIED_FLAG="--unified-graph"
        echo "=== Training: granular=$granular object_only=$object_only unified_graph=$unified_graph -> $SAVE_DIR ==="
        python3 torch_gnn/train_object_vsa_gnn_frontend.py --dataset-root data/RAVEN-10000 --use-gt --node-repr "$node_repr" \
          --gnn-type "$gnn_type" $GRANULAR_FLAG $OBJECT_ONLY_FLAG $UNIFIED_FLAG --lr $lr --embed-dim $embed_dim --hidden-dim $hidden_dim \
          --save-dir "$SAVE_DIR" --epochs $epochs --batch-size 16 --num-workers 2
      done
    done
  done
}

if [ "$RUN_MODE" = "all" ]; then
  train_all_configs
else
  run_single
fi

## Frontend
# python3 torch_gnn/train_object_vsa_gnn_frontend.py --dataset-root data/RAVEN-10000 \
#   --frontend-checkpoint log/universal_frontend_correct_proportional_mean1 \
#   --node-repr "$node_repr" \
#   --gnn-type "$gnn_type" $GRANULAR_FLAG --lr $lr --embed-dim $embed_dim --hidden-dim $hidden_dim \
#   --save-dir "$SAVE_DIR" --epochs $epochs --batch-size 16 --num-workers 2

# VSA example:
# node_repr=vsa; lr=0.001; hidden_dim=256
# SAVE_SUBFOLDER="lr${lr}_${node_repr}_h${hidden_dim}"
# python3 torch_gnn/train_object_vsa_gnn_frontend.py --dataset-root data/RAVEN-10000 --use-gt --node-repr vsa \
#   --frontend-checkpoint log/universal_frontend_correct_proportional_mean1 --lr $lr --hidden-dim $hidden_dim \
#   --save-dir "${SAVEPATH}/object_vsa_gnn_frontend/${SAVE_SUBFOLDER}" --epochs 200 --batch-size 16 --num-workers 2