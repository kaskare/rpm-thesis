#!/bin/bash
# Submit a SLURM job for the PyNeuraLogic GNN solver on I-RAVEN.
#
# Trains a SINGLE unified model on data from ALL 7 constellations
# using per-candidate sub-graphs, then evaluates per-constellation.
#
# Usage:
#   ./scripts/submit_pynl_gnn.sh
#   ./scripts/submit_pynl_gnn.sh --embed-dim 64 --epochs 200 --oversample-pos 5
#
# Environment overrides:
#   SAVEPATH=/my/experiments  ./scripts/submit_pynl_gnn.sh
#   PARTITION=amdgpushort     ./scripts/submit_pynl_gnn.sh

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
mkdir -p log

# ── Defaults ────────────────────────────────────────────────────────
embed_dim=512
epochs=411
lr=0.001
n_train=6000
n_val=2000
n_test=2000
oversample_pos=3
seed=42

# ── Parse CLI flags ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --embed-dim)       embed_dim="$2";      shift 2 ;;
    --epochs)          epochs="$2";         shift 2 ;;
    --lr)              lr="$2";             shift 2 ;;
    --n-train)         n_train="$2";        shift 2 ;;
    --n-val)           n_val="$2";          shift 2 ;;
    --n-test)          n_test="$2";         shift 2 ;;
    --oversample-pos)  oversample_pos="$2"; shift 2 ;;
    --seed)            seed="$2";           shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

SAVEPATH="${SAVEPATH:-$ROOT/experiments}"
PARTITION="${PARTITION:-amdgpulong}"
DATASET_ROOT="$ROOT/data/I-RAVEN-10000"

SAVE_SUBFOLDER="emb${embed_dim}_ep${epochs}_lr${lr}_n${n_train}_os${oversample_pos}"
SAVE_DIR="${SAVEPATH}/pynl_gnn/${SAVE_SUBFOLDER}"
JOB_NAME="pynl_gnn"

echo "PyNeuraLogic GNN solver — SLURM submission"
echo "  partition:      $PARTITION"
echo "  save dir:       $SAVE_DIR"
echo "  embed_dim:      $embed_dim"
echo "  epochs:         $epochs"
echo "  lr:             $lr"
echo "  oversample_pos: ${oversample_pos}x"
echo "  n_train:        $n_train/config  n_val: $n_val/config  n_test: $n_test/config"
echo "  seed:           $seed"
echo ""

# ── Write and submit a single job ──────────────────────────────────
JOB_SCRIPT="log/job_pynl_gnn.sh"
cat > "$JOB_SCRIPT" <<SLURM_EOF
#!/bin/bash -l
#SBATCH --partition $PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=log/slurm_pynl_gnn_%j.out
#SBATCH --error=log/slurm_pynl_gnn_%j.err

ml load Anaconda3 2>/dev/null || true
ml load Java 2>/dev/null || true
conda activate nvsa 2>/dev/null || true

# PyNeuraLogic needs Java (JPype). Set JAVA_HOME if not already provided.
if [ -z "\$JAVA_HOME" ]; then
  for jdir in /usr/lib/jvm/java-*-openjdk-*/jre /usr/lib/jvm/java-*-openjdk-* /usr/lib/jvm/default-java; do
    if [ -d "\$jdir" ]; then export JAVA_HOME="\$jdir"; break; fi
  done
fi

cd "$ROOT"

python3 -u scripts/run_pynl_gnn.py \\
  --dataset-root "$DATASET_ROOT" \\
  --embed-dim $embed_dim \\
  --epochs $epochs \\
  --lr $lr \\
  --n-train $n_train \\
  --n-val $n_val \\
  --n-test $n_test \\
  --oversample-pos $oversample_pos \\
  --seed $seed \\
  --save-dir "$SAVE_DIR"
SLURM_EOF

JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $NF}')
echo "Submitted job $JOB_ID -> $SAVE_DIR"
echo "Monitor: squeue -u \$USER"
echo "Log:     tail -f log/slurm_pynl_gnn_${JOB_ID}.out"
