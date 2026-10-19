#!/bin/bash
# Train all 8 GNN graph representation configs (granular x object_only x unified_graph).
# Usage:
#   ./scripts/train_all_gnn.sh [--gat|--gcn|--gnn]
#   ./scripts/submit_all_gnn.sh [--gat|--gcn|--gnn]
#   RUN_MODE=all sbatch run.sh

set -e
cd "$(dirname "$0")/.."

gnn_type=gat
for arg in "$@"; do
  case "$arg" in
    --gat) gnn_type=gat ;;
    --gcn) gnn_type=gcn ;;
    --gnn) gnn_type=gnn ;;
    *) echo "Unknown flag: $arg (use --gat, --gcn, or --gnn)"; exit 1 ;;
  esac
done

SAVEPATH="${SAVEPATH:-/home/kassyask/neuro-vector-symbolic-architectures-raven/experiments}"
lr=0.001
node_repr=embedding
embed_dim=256
hidden_dim=384
epochs=402

echo "Training all 8 ${gnn_type^^} graph configs -> $SAVEPATH"

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
      echo ""
      echo "=== granular=$granular object_only=$object_only unified_graph=$unified_graph ==="
      echo "    -> $SAVE_DIR"
      python3 train_object_vsa_gnn_frontend.py --dataset-root data/RAVEN-10000 --use-gt --node-repr "$node_repr" \
        --gnn-type "$gnn_type" $GRANULAR_FLAG $OBJECT_ONLY_FLAG $UNIFIED_FLAG --lr $lr --embed-dim $embed_dim --hidden-dim $hidden_dim \
        --save-dir "$SAVE_DIR" --epochs $epochs --batch-size 16 --num-workers 2
    done
  done
done

echo ""
echo "Done. All 8 configs trained."
