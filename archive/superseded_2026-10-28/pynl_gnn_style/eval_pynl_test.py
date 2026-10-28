#!/usr/bin/env python3
"""Evaluate a trained all-constellations model on the I-RAVEN TEST split.

Reports per-constellation and overall accuracy, alongside the torch GNN's
published test numbers for comparison.

A checkpoint needs BOTH files, written together by train_all_constellations.py:
    model_best.pt      torch side  (embeddings, projection, pad token, level-1 head)
    template_best.bin  PyNeuraLogic side (message passing, and for level 2 the
                       scoring head as well)
`nl.internal_weights` inside model_best.pt is the framework's dummy empty
Parameter, NOT the template weights -- a run whose save predates the
template_best.bin fix cannot be evaluated, and this script says so rather than
silently scoring a half-random model.

Usage:
    python pynl/eval_pynl_test.py experiments/full_iraven_L2
    python pynl/eval_pynl_test.py experiments/*_L2 --n-test 2000
"""
import argparse, glob, json, os, sys, time
from collections import defaultdict

import numpy as np
import torch

import neuralogic
from neuralogic.core import Transformation, Aggregation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_all_constellations import (AllConstellations, CONSTELLATIONS, load, log)

# torch GNN per-constellation test accuracy: MEAN of three seeds retrained with both fixes
# (correct slot positions + correct spatial-relation centres), experiments/gnn_FIXREL_s{42,43,44},
# 6000/constellation, 406 epochs. Seeds scored 41.74 / 35.01 / 59.02 overall (sd 12.4), so treat
# the per-constellation means as a rough reference. Superseded references: 68.81% (published,
# buggy positions, not reproducible) and 55.09% (fixed positions, wrong relations, one seed).
TORCH_GNN = {
    "center_single": 50.22, "distribute_four": 36.63, "distribute_nine": 32.55,
    "left_center_single_right_center_single": 47.53,
    "up_center_single_down_center_single": 46.85,
    "in_center_single_out_center_single": 55.53,
    "in_distribute_four_out_center_single": 47.47,
}
TORCH_OVERALL = 45.25
AGGS = {"avg": Aggregation.AVG, "max": Aggregation.MAX, "sum": Aggregation.SUM}


def load_model(run_dir):
    ckpt_p = os.path.join(run_dir, "model_best.pt")
    tmpl_p = os.path.join(run_dir, "template_best.bin")
    if not os.path.exists(ckpt_p):
        raise FileNotFoundError(f"{run_dir}: no model_best.pt")
    if not os.path.exists(tmpl_p):
        raise FileNotFoundError(
            f"{run_dir}: no template_best.bin -- this run saved only the torch side, "
            f"so the PyNeuraLogic weights are unrecoverable. Re-train with the "
            f"current train_all_constellations.py.")

    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    a = ck["args"]
    import train_all_constellations as T                  # ablation switches the run was trained with
    T.EDGE_LABELS, T.ABSENT_POS = a.get("edge_labels", "basic"), a.get("absent_pos", False)
    model = AllConstellations(
        a["level"], a["embed_dim"], a["hidden_dim"], a["layers"],
        Transformation.LEAKY_RELU, a["nl_lr"],
        agg=AGGS.get(a.get("agg", "avg"), Aggregation.AVG),
        same_panel=not a.get("no_same_panel", False),
        layout=a.get("layout", "off"),
        per_type=a.get("per_type_weights", False),
        absent_pos=a.get("absent_pos", False),
    )
    model.load_state_dict(ck["torch_state"])
    model.nl.model.load(tmpl_p)
    model.eval()
    return model, ck, a


def evaluate(model, data):
    per = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for cfg, attrs, active, y in data:
            per[cfg][0] += int(model(attrs, active).argmax()) == y
            per[cfg][1] += 1
    return {k: (v[0], v[1]) for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="experiments/<run>_L<level> directories")
    ap.add_argument("--dataset-root", default="data/I-RAVEN-10000")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--n-test", type=int, default=500, help="puzzles per constellation")
    ap.add_argument("--jvm-gb", type=int, default=100)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    neuralogic.set_max_memory_size(args.jvm_gb)
    results = {}

    for run_dir in args.run_dirs:
        run_dir = run_dir.rstrip("/")
        name = os.path.basename(run_dir)
        print()
        try:
            model, ck, a = load_model(run_dir)
        except FileNotFoundError as e:
            log(f"SKIP {name}: {e}")
            continue

        log(f"{name}  (level {a['level']}, dim {a['embed_dim']}/{a['hidden_dim']}, "
            f"{a['layers']} layers, trained on {a['n_train']}/constellation, "
            f"best epoch {ck['epoch']}, val {ck['val_acc']*100:.1f}%)")

        t0 = time.time()
        data = load(args.dataset_root, CONSTELLATIONS, args.split, args.n_test)
        log(f"loaded {len(data)} {args.split} puzzles in {time.time()-t0:.0f}s; scoring…")

        t0 = time.time()
        per = evaluate(model, data)
        hits = sum(c for c, _ in per.values())
        tot = sum(n for _, n in per.values())
        log(f"scored in {time.time()-t0:.0f}s")

        print(f"\n{'constellation':<44} {'correct':>12} {'acc':>8} {'torch GNN':>10} {'Δ':>8}")
        print("-" * 86)
        for cfg in CONSTELLATIONS:
            if cfg not in per:
                continue
            c, n = per[cfg]
            acc = 100.0 * c / n
            ref = TORCH_GNN[cfg]
            print(f"{cfg:<44} {c:>6}/{n:<5} {acc:>7.2f}% {ref:>9.2f}% {acc-ref:>+7.2f}")
        overall = 100.0 * hits / tot
        print("-" * 86)
        print(f"{'OVERALL':<44} {hits:>6}/{tot:<5} {overall:>7.2f}% "
              f"{TORCH_OVERALL:>9.2f}% {overall-TORCH_OVERALL:>+7.2f}")
        print(f"{'':<44} {'':>12} {'':>8} chance {100/8:.2f}%")

        results[name] = {"split": args.split, "overall": overall,
                         "per_constellation": {k: 100.0*c/n for k, (c, n) in per.items()},
                         "best_epoch": ck["epoch"], "val_acc": ck["val_acc"], "args": a}

    if args.json_out and results:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        log(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
