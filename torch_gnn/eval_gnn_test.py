#!/usr/bin/env python3
"""
Evaluate a trained GNN checkpoint on RAVEN or I-RAVEN test split.

Reports overall accuracy and per-constellation accuracy.

Usage:
    python torch_gnn/eval_gnn_test.py --checkpoint experiments/.../model_best.pt --dataset-root data/RAVEN-10000
    python torch_gnn/eval_gnn_test.py --checkpoint experiments/.../model_best.pt --dataset-root data/I-RAVEN-10000
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path.insert(0, ROOT)

from gnn import (
    build_packed_unified,
    build_packed_candidates_from_batch,
    collate_gt,
    discover_xml_paths,
    fallback_position_centers,
    to_device_packed,
)
from gnn.data import build_position_centers
from gnn.data import RavenGTDataset, get_config_from_path
from gnn.model import CandidateVsaGNN

# All 7 constellations (identical for RAVEN and I-RAVEN)
ALL_CONFIGS = [
    "center_single",
    "distribute_four",
    "distribute_nine",
    "left_center_single_right_center_single",
    "up_center_single_down_center_single",
    "in_center_single_out_center_single",
    "in_distribute_four_out_center_single",
]


def eval_per_config(
    model,
    dataset_root: str,
    device: torch.device,
    granular: bool,
    object_only: bool,
    unified_graph: bool,
    batch_size: int = 32,
    centers: dict = None,
) -> dict:
    """Evaluate model on test split, returning per-config and overall accuracy.

    `centers` should be the checkpoint's own `pos_centers` -- the ones the model was
    trained with -- so old (rank-keyed) and new (universal) runs both reproduce.
    """
    if centers is None:
        train_xml = discover_xml_paths(dataset_root, "train")
        centers = build_position_centers(train_xml) if train_xml else fallback_position_centers()

    config_correct = defaultdict(int)
    config_total = defaultdict(int)

    for config in ALL_CONFIGS:
        config_dir = os.path.join(dataset_root, config)
        if not os.path.isdir(config_dir):
            continue
        npz_pattern = os.path.join(config_dir, f"*_test.npz")
        import glob
        files = sorted(glob.glob(npz_pattern))
        if not files:
            continue

        ds = RavenGTDataset(dataset_root, "test")
        # Filter to this config only by restricting to files in config dir
        ds.files = [f for f in ds.files if get_config_from_path(f) == config]
        if not ds.files:
            continue

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_gt,
        )

        model.eval()
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            for extracted, target in loader:
                target = target.to(device)
                if unified_graph:
                    packed = [
                        to_device_packed(p, device)
                        for p in build_packed_unified(
                            extracted, centers,
                            granular=granular, object_only=object_only
                        )
                    ]
                else:
                    packed = [
                        to_device_packed(p, device)
                        for p in build_packed_candidates_from_batch(
                            extracted, centers,
                            granular=granular, object_only=object_only
                        )
                    ]
                logits = model(packed)
                pred = logits.argmax(dim=1)
                correct = (pred == target).sum().item()
                total_correct += correct
                total_samples += target.size(0)

        config_correct[config] = total_correct
        config_total[config] = total_samples

    return dict(config_correct), dict(config_total)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GNN on RAVEN/I-RAVEN test split")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_best.pt")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path to RAVEN-10000 or I-RAVEN-10000")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    cargs = ckpt.get("args", {})
    if not cargs:
        raise ValueError("Checkpoint has no 'args'; cannot infer model config")

    granular = cargs.get("granular", False)
    object_only = cargs.get("object_only", False)
    unified_graph = cargs.get("unified_graph", False)
    embed_dim = cargs.get("embed_dim", 256)
    hidden_dim = cargs.get("hidden_dim", 384)
    gnn_type = cargs.get("gnn_type", "gnn")
    node_repr = cargs.get("node_repr", "embedding")

    model = CandidateVsaGNN(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=cargs.get("num_layers", 3),
        dropout=cargs.get("dropout", 0.1),
        node_repr=node_repr,
        W_matrix=None,
        gnn_type=gnn_type,
        granular=granular,
        unified_graph=unified_graph,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset:   {args.dataset_root}")
    print(f"Config:    granular={granular}, object_only={object_only}, unified_graph={unified_graph}")
    stored = ckpt.get("pos_centers")
    print(f"Centres:   {'from checkpoint (as trained)' if stored else 'recomputed, legacy rank-keyed'}")
    print()

    config_correct, config_total = eval_per_config(
        model,
        args.dataset_root,
        device,
        granular=granular,
        object_only=object_only,
        unified_graph=unified_graph,
        batch_size=args.batch_size,
        centers=stored,
    )

    total_correct = sum(config_correct.values())
    total_samples = sum(config_total.values())
    overall_acc = 100.0 * total_correct / total_samples if total_samples else 0.0

    print("Per-constellation test accuracy:")
    print("-" * 60)
    for config in ALL_CONFIGS:
        if config in config_total and config_total[config] > 0:
            acc = 100.0 * config_correct[config] / config_total[config]
            print(f"  {config:45s}  {acc:6.2f}%  ({config_correct[config]}/{config_total[config]})")
        else:
            print(f"  {config:45s}  N/A (no test samples)")
    print("-" * 60)
    print(f"  {'OVERALL':45s}  {overall_acc:6.2f}%  ({total_correct}/{total_samples})")


if __name__ == "__main__":
    main()
