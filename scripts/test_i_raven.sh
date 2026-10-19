#!/bin/bash
# Run test evaluation on I-RAVEN dataset with the best-performing model.
# Best I-RAVEN model (ep406): object_vsa_gnn_frontend_gnn_granular_object_only @ 68.83% val
#
# Usage: ./scripts/test_i_raven.sh [--checkpoint PATH] [--dataset-root PATH]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-$ROOT/experiments/object_vsa_gnn_frontend_gnn_granular_object_only/lr0.001_embedding_emb256_h384_ep406/model_best.pt}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/data/I-RAVEN-10000}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Error: Checkpoint not found: $CHECKPOINT"
  echo "Set CHECKPOINT env var or pass --checkpoint PATH"
  exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
  echo "Error: Dataset not found: $DATASET_ROOT"
  echo "Set DATASET_ROOT env var or pass --dataset-root PATH"
  exit 1
fi

echo "=== I-RAVEN Test Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATASET_ROOT"
echo ""

python3 eval_gnn_test.py \
  --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" \
  "$@"
