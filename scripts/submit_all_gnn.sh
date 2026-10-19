#!/bin/bash
# Submit 8 separate SLURM jobs (one per graph config) instead of running sequentially.
# Usage: ./scripts/submit_all_gnn.sh [--gat|--gcn|--gnn] [--nvsa-frontend]
#
# Architecture flags:
#   --gat   Use GAT (attention + transformer) — default
#   --gcn   Use GCN (degree-normalized convolution)
#   --gnn   Use simple message-passing GNN (RelGN)
#
# Object source:
#   (default)      train_object_vsa_gnn_frontend.py with --use-gt (XML ground truth)
#   --nvsa-frontend  omit --use-gt; extract objects from panel images via pretrained NVSA
#                    frontend (same as run.sh commented "Step 2" example). Override checkpoint:
#                    FRONTEND_CHECKPOINT=/path/to/dir  FRONTEND_DEVICE=cuda:0  ./scripts/submit_all_gnn.sh --nvsa-frontend
#
# Resuming: If a job was stopped (e.g. time limit), resubmit with the same SAVEPATH.
# The trainer automatically resumes from checkpoint.pth.tar in each save-dir.

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
mkdir -p log

gnn_type=gat
use_nvsa_frontend=false
for arg in "$@"; do
  case "$arg" in
    --gat) gnn_type=gat ;;
    --gcn) gnn_type=gcn ;;
    --gnn) gnn_type=gnn ;;
    --nvsa-frontend) use_nvsa_frontend=true ;;
    *) echo "Unknown flag: $arg (use --gat, --gcn, --gnn, --nvsa-frontend)"; exit 1 ;;
  esac
done

SAVEPATH="${SAVEPATH:-/home/kassyask/neuro-vector-symbolic-architectures-raven/experiments}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$ROOT/log/universal_frontend_correct_proportional_mean1}"
FRONTEND_DEVICE="${FRONTEND_DEVICE:-cuda:0}"

if [ "$use_nvsa_frontend" = true ]; then
  if [ ! -d "$FRONTEND_CHECKPOINT" ]; then
    echo "ERROR: --nvsa-frontend requires a frontend save dir (config + weights). Not found:" >&2
    echo "  FRONTEND_CHECKPOINT=$FRONTEND_CHECKPOINT" >&2
    echo "Set FRONTEND_CHECKPOINT to your trained NVSA frontend directory." >&2
    exit 1
  fi
fi
lr=0.001
node_repr=embedding
embed_dim=256
hidden_dim=256
epochs=100

# SLURM options (match run.sh)
PARTITION="amdgpulong"
JOB_PREFIX=""

obj_mode="ground-truth (XML)"
[ "$use_nvsa_frontend" = true ] && obj_mode="NVSA frontend ($FRONTEND_CHECKPOINT)"
echo "Submitting 8 ${gnn_type^^} jobs to partition=$PARTITION -> $SAVEPATH"
echo "Object source: $obj_mode"
echo ""

job_ids=()
for granular in false true; do
  for object_only in false true; do
    for unified_graph in false true; do
      SAVE_SUBFOLDER="lr${lr}_${node_repr}_emb${embed_dim}_h${hidden_dim}_ep${epochs}"
      SAVE_BASE="object_vsa_gnn_frontend_${gnn_type}"
      [ "$use_nvsa_frontend" = true ] && SAVE_BASE="${SAVE_BASE}_nvsa_det"
      [ "$granular" = "true" ] && SAVE_BASE="${SAVE_BASE}_granular"
      [ "$object_only" = "true" ] && SAVE_BASE="${SAVE_BASE}_object_only"
      [ "$unified_graph" = "true" ] && SAVE_BASE="${SAVE_BASE}_unified"
      SAVE_DIR="${SAVEPATH}/${SAVE_BASE}/${SAVE_SUBFOLDER}"
      GRANULAR_FLAG=""; [ "$granular" = "true" ] && GRANULAR_FLAG="--granular"
      OBJECT_ONLY_FLAG=""; [ "$object_only" = "true" ] && OBJECT_ONLY_FLAG="--object-only"
      UNIFIED_FLAG=""; [ "$unified_graph" = "true" ] && UNIFIED_FLAG="--unified-graph"

      # Job name <= 8 chars for SLURM: g0o0u0 = granular, object_only, unified (0/1)
      g=$([ "$granular" = "true" ] && echo 1 || echo 0)
      o=$([ "$object_only" = "true" ] && echo 1 || echo 0)
      u=$([ "$unified_graph" = "true" ] && echo 1 || echo 0)
      JOB_NAME="g${g}o${o}u${u}"
      [ "$use_nvsa_frontend" = true ] && JOB_NAME="${JOB_NAME}n"

      # Write job script (shebang must be first char - no leading space/newline)
      JOB_SCRIPT="log/job_${JOB_NAME}.sh"
      {
        echo '#!/bin/bash -l'
        echo "#SBATCH --partition $PARTITION"
        echo "#SBATCH --gres=gpu:1"
        echo "#SBATCH --cpus-per-task=16"
        echo "#SBATCH --mem=100G"
        echo "#SBATCH --job-name=$JOB_NAME"
        echo "#SBATCH --output=log/slurm_${JOB_NAME}_%j.out"
        echo "#SBATCH --error=log/slurm_${JOB_NAME}_%j.err"
        echo ""
        echo "ml load Anaconda3 2>/dev/null || true"
        echo "conda activate nvsa 2>/dev/null || true"
        echo ""
        echo "cd \"$ROOT\""
        if [ "$use_nvsa_frontend" = true ]; then
          echo "python3 train_object_vsa_gnn_frontend.py --dataset-root data/I-RAVEN-10000 --node-repr $node_repr \\"
          echo "  --frontend-checkpoint \"$FRONTEND_CHECKPOINT\" --frontend-device $FRONTEND_DEVICE \\"
        else
          echo "python3 train_object_vsa_gnn_frontend.py --dataset-root data/I-RAVEN-10000 --use-gt --node-repr $node_repr \\"
        fi
        echo "  --gnn-type $gnn_type $GRANULAR_FLAG $OBJECT_ONLY_FLAG $UNIFIED_FLAG --lr $lr --embed-dim $embed_dim --hidden-dim $hidden_dim \\"
        echo "  --save-dir \"$SAVE_DIR\" --epochs $epochs --batch-size 16 --num-workers 2"
      } > "$JOB_SCRIPT"

      JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $NF}')
      job_ids+=("$JOB_ID")
      echo "  Submitted $JOB_NAME -> job $JOB_ID -> $SAVE_DIR"
    done
  done
done

echo ""
echo "Submitted ${#job_ids[@]} jobs: ${job_ids[*]}"
echo "Monitor: squeue -u \$USER"
