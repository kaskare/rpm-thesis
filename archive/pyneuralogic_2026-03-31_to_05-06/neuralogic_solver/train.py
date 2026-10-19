#!/usr/bin/env python3
"""
Evaluate / train the logic-rule RPM solver.

Modes
-----
eval_gt     Deterministic evaluation with ground-truth attributes from XML.
            Should reach ~100 % accuracy — serves as a sanity check that the
            rule vocabulary is complete.

eval_soft   Evaluate the differentiable soft solver on GT one-hot distributions.
            Tests the soft rule checkers against the hard baseline.

eval_pynl   Evaluate using PyNeuraLogic's inference engine — the RPM rules
            are expressed as declarative logic rules and grounded by the
            NeuraLogic Java backend.

Usage
-----
    # Ground-truth evaluation (no training needed)
    python -m neuralogic_solver.train --mode eval_gt \\
        --dataset-root data/I-RAVEN-10000 --split test

    # PyNeuraLogic inference engine
    python -m neuralogic_solver.train --mode eval_pynl \\
        --dataset-root data/I-RAVEN-10000 --split test
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gnn.data import RavenGTDataset, collate_gt, get_config_from_path

from .config import ALL_CONFIGS, NUM_CANDIDATES
from .solver import HardLogicSolver, SoftLogicSolver, attrs_to_dists
from .pynl_solver import PynlHardSolver


# ─── Evaluation helpers ─────────────────────────────────────────────

def eval_hard_gt(
    dataset_root: str,
    split: str,
    batch_size: int = 64,
) -> dict:
    """Run HardLogicSolver on ground-truth attributes, report per-config accuracy."""
    solver = HardLogicSolver()
    config_correct = defaultdict(int)
    config_total = defaultdict(int)

    for config in ALL_CONFIGS:
        config_dir = os.path.join(dataset_root, config)
        if not os.path.isdir(config_dir):
            continue

        ds = RavenGTDataset(dataset_root, split)
        ds.files = [f for f in ds.files if get_config_from_path(f) == config]
        if not ds.files:
            continue

        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_gt,
        )

        correct, total = 0, 0
        t0 = time.time()
        for extracted, targets in loader:
            preds = solver.predict_batch(extracted)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        elapsed = time.time() - t0

        acc = 100.0 * correct / total if total else 0.0
        config_correct[config] = correct
        config_total[config] = total
        print(f"  {config:45s}  {acc:6.2f}%  ({correct}/{total})  [{elapsed:.1f}s]")

    return dict(config_correct), dict(config_total)


def eval_soft_gt(
    dataset_root: str,
    split: str,
    batch_size: int = 64,
) -> dict:
    """Run SoftLogicSolver on one-hot GT distributions."""
    solver = SoftLogicSolver()
    solver.eval()

    config_correct = defaultdict(int)
    config_total = defaultdict(int)

    for config in ALL_CONFIGS:
        config_dir = os.path.join(dataset_root, config)
        if not os.path.isdir(config_dir):
            continue

        ds = RavenGTDataset(dataset_root, split)
        ds.files = [f for f in ds.files if get_config_from_path(f) == config]
        if not ds.files:
            continue

        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_gt,
        )

        correct, total = 0, 0
        t0 = time.time()
        with torch.no_grad():
            for extracted, targets in loader:
                for i, (obj_type, obj_size, obj_color, obj_exists) in enumerate(extracted):
                    dists = attrs_to_dists(obj_type, obj_size, obj_color)
                    exist_float = obj_exists.float()
                    logits = solver(dists, exist_float)
                    pred = logits.argmax().item()
                    if pred == targets[i].item():
                        correct += 1
                    total += 1
        elapsed = time.time() - t0

        acc = 100.0 * correct / total if total else 0.0
        config_correct[config] = correct
        config_total[config] = total
        print(f"  {config:45s}  {acc:6.2f}%  ({correct}/{total})  [{elapsed:.1f}s]")

    return dict(config_correct), dict(config_total)


def eval_pynl_gt(
    dataset_root: str,
    split: str,
    batch_size: int = 64,
) -> dict:
    """Run PynlHardSolver (PyNeuraLogic inference engine) on GT attributes."""
    solver = PynlHardSolver()
    config_correct = defaultdict(int)
    config_total = defaultdict(int)

    for config in ALL_CONFIGS:
        config_dir = os.path.join(dataset_root, config)
        if not os.path.isdir(config_dir):
            continue

        ds = RavenGTDataset(dataset_root, split)
        ds.files = [f for f in ds.files if get_config_from_path(f) == config]
        if not ds.files:
            continue

        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_gt,
        )

        correct, total = 0, 0
        t0 = time.time()
        for extracted, targets in loader:
            preds = solver.predict_batch(extracted)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        elapsed = time.time() - t0

        acc = 100.0 * correct / total if total else 0.0
        config_correct[config] = correct
        config_total[config] = total
        print(f"  {config:45s}  {acc:6.2f}%  ({correct}/{total})  [{elapsed:.1f}s]")

    return dict(config_correct), dict(config_total)


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Logic-rule RPM solver: evaluation and training",
    )
    parser.add_argument(
        "--mode", type=str, default="eval_gt",
        choices=["eval_gt", "eval_soft", "eval_pynl"],
        help="eval_gt = hard boolean; eval_soft = differentiable; eval_pynl = PyNeuraLogic inference",
    )
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Mode:    {args.mode}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Split:   {args.split}")
    print()

    if args.mode == "eval_gt":
        print("Per-constellation accuracy (hard logic solver, GT attributes):")
        print("-" * 72)
        cc, ct = eval_hard_gt(args.dataset_root, args.split, args.batch_size)
    elif args.mode == "eval_soft":
        print("Per-constellation accuracy (soft logic solver, one-hot GT):")
        print("-" * 72)
        cc, ct = eval_soft_gt(args.dataset_root, args.split, args.batch_size)
    elif args.mode == "eval_pynl":
        print("Per-constellation accuracy (PyNeuraLogic inference, GT attributes):")
        print("-" * 72)
        cc, ct = eval_pynl_gt(args.dataset_root, args.split, args.batch_size)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    total_correct = sum(cc.values())
    total_samples = sum(ct.values())
    overall = 100.0 * total_correct / total_samples if total_samples else 0.0
    print("-" * 72)
    print(f"  {'OVERALL':45s}  {overall:6.2f}%  ({total_correct}/{total_samples})")


if __name__ == "__main__":
    main()
