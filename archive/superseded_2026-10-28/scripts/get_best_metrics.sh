#!/usr/bin/env bash
# Retrieve best model metrics for each checkpoint found in experiments/

EXPERIMENTS_DIR="${1:-experiments}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

echo "Scanning ${EXPERIMENTS_DIR}/ for model checkpoints..."
echo

printf "%-72s  %10s  %8s  %8s  %s\n" "PATH" "BEST_ACC" "BEST_AT" "STOPPED" "CONFIG"
printf "%-72s  %10s  %8s  %8s  %s\n" "----" "--------" "-------" "-------" "------"

find "$EXPERIMENTS_DIR" -type f \( -name "model_best.pt" -o -name "model_best.pth.tar" -o -name "checkpoint.pth.tar" \) 2>/dev/null | sort | while read -r ckpt; do
    python3 - "$ckpt" <<'PY'
import sys
import os
import torch

path = sys.argv[1]
try:
    ckpt = torch.load(path, map_location="cpu")
except Exception as e:
    print(f"{path}\tERROR: {e}")
    sys.exit(0)

acc = ckpt.get("best_val_acc") or ckpt.get("best_acc")
acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
# BEST_AT: epoch when best model was achieved (best_epoch) or fallback to epoch
best_at_epoch = ckpt.get("best_epoch")
if best_at_epoch is not None:
    best_at_str = str(best_at_epoch)
else:
    epoch = ckpt.get("epoch", "")
    best_at_str = str(epoch) if epoch != "" else "N/A"

# STOPPED: max epoch reached during training (from checkpoint.pth.tar)
stopped_at = "N/A"
ckpt_dir = os.path.dirname(path)
base = os.path.basename(path)
if "model_best" in base:
    final_ckpt = os.path.join(ckpt_dir, "checkpoint.pth.tar")
    if os.path.isfile(final_ckpt):
        try:
            final = torch.load(final_ckpt, map_location="cpu")
            stopped_at = str(final.get("epoch", "N/A"))
        except Exception:
            pass
else:
    # Current file is checkpoint.pth.tar - its epoch is max reached
    stopped_at = str(ckpt.get("epoch", "N/A"))

args = ckpt.get("args", {})
if isinstance(args, dict):
    extra = []
    for k in ("lr", "embed_dim", "hidden_dim", "node_repr", "gnn_type"):
        if k in args and args[k] is not None:
            extra.append(f"{k}={args[k]}")
    extra_str = ", ".join(extra) if extra else ""
else:
    extra_str = ""

print(f"{path}\t{acc_str}\t{best_at_str}\t{stopped_at}\t{extra_str}")
PY
done | while IFS=$'\t' read -r path acc best_at stopped extra; do
    printf "%-72s  %10s  %8s  %8s  %s\n" "$path" "$acc" "$best_at" "$stopped" "$extra"
done

echo
echo "Done."
