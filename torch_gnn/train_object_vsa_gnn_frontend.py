#!/usr/bin/env python3
"""
Object-level VSA-GNN training for RPM using pretrained frontend extraction.

Entry point: delegates to gnn package.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path.insert(0, ROOT)

from gnn import (
    build_position_centers,
    universal_position_centers,
    collate_gt,
    collate_images,
    discover_xml_paths,
    fallback_position_centers,
    load_codebook_only,
    run_epoch,
    run_epoch_gt,
)
from gnn.data import RavenGTDataset, RavenImageDataset
from gnn.model import CandidateVsaGNN
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train object-level VSA-GNN from pretrained frontend extraction")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--frontend-checkpoint", type=str, default="", help="Required for frontend mode or node-repr=vsa (codebook)")
    parser.add_argument("--use-gt", action="store_true", help="Use ground truth from XML (faster, no frontend inference)")
    parser.add_argument("--node-repr", type=str, default="embedding", choices=["embedding", "vsa"],
                        help="Node encoding: embedding (learned) or vsa (codebook lookup from frontend)")
    parser.add_argument("--gnn-type", type=str, default="gat", choices=["gat", "gcn", "gnn"],
                        help="GNN architecture: gat (attention + transformer), gcn (degree-normalized convolution), or gnn (simple message-passing)")
    parser.add_argument("--granular", action="store_true",
                        help="Use granular graph: attribute nodes (color, size, type, pos) each connected to object node, which connects to panel")
    parser.add_argument("--object-only", action="store_true",
                        help="Use object-only graph: no panel super-nodes; objects connect intra-panel (spatial) + inter-panel (same position across panels)")
    parser.add_argument("--unified-graph", action="store_true",
                        help="Use one graph per sample with all 8 candidates; candidates connect only to context panels 5 and 7")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.23)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy-centers", action="store_true",
                        help="use the old rank-keyed position centres (wrong spatial relations); "
                             "only to reproduce runs from before 2026-09-30")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frontend-device", type=str, default="cuda:0")
    parser.add_argument("--save-dir", type=str, default="experiments/object_vsa_gnn_frontend_exp1")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_gt = args.use_gt
    node_repr = args.node_repr

    W_matrix = None
    if node_repr == "vsa":
        if not args.frontend_checkpoint:
            raise ValueError("--frontend-checkpoint is required for --node-repr vsa (codebook)")
        W_matrix = load_codebook_only(args.frontend_checkpoint, device=device)
        print(f"Loaded VSA codebook: W shape={W_matrix.shape}")

    if not use_gt:
        if not args.frontend_checkpoint:
            raise ValueError("--frontend-checkpoint is required when not using --use-gt")
        from gnn_solver import load_frontend
        frontend_device = torch.device(args.frontend_device if torch.cuda.is_available() else "cpu")
        frontend_model, cos_layer, _, _ = load_frontend(args.frontend_checkpoint, device=frontend_device)
        for p in frontend_model.parameters():
            p.requires_grad = False
        for p in cos_layer.parameters():
            p.requires_grad = False
        frontend_model.eval()
        cos_layer.eval()
    else:
        msg = "Using ground truth from XML (no frontend inference)"
        if node_repr == "vsa":
            msg += " + VSA codebook"
        if args.granular:
            msg += " + granular graph (attr nodes -> object -> panel)"
        if args.object_only:
            msg += " + object-only (no panel nodes, same-pos cross-panel edges)"
        if args.unified_graph:
            msg += " + unified (one graph, all 8 candidates → panels 5 and 7)"
        print(msg)
        frontend_model = cos_layer = None
        frontend_device = device

    train_xml = discover_xml_paths(args.dataset_root, "train")
    if args.legacy_centers:
        # rank-keyed centres: wrong within-panel relations; only to reproduce old runs
        centers = build_position_centers(train_xml) if train_xml else fallback_position_centers()
    else:
        centers = universal_position_centers(args.dataset_root)
    print(f"Position centres: {'LEGACY rank-keyed (buggy)' if args.legacy_centers else 'universal, from bbox geometry'}")

    if use_gt:
        train_loader = DataLoader(
            RavenGTDataset(args.dataset_root, "train"),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_gt
        )
        val_loader = DataLoader(
            RavenGTDataset(args.dataset_root, "val"),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_gt
        )
    else:
        train_loader = DataLoader(
            RavenImageDataset(args.dataset_root, "train"),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_images
        )
        val_loader = DataLoader(
            RavenImageDataset(args.dataset_root, "val"),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_images
        )

    model = CandidateVsaGNN(
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout,
        node_repr=node_repr, W_matrix=W_matrix, gnn_type=args.gnn_type, granular=args.granular,
        unified_graph=args.unified_graph,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = 0.0
    best_epoch = 0
    start_epoch = 1
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pth.tar")
    best_path = os.path.join(args.save_dir, "model_best.pt")

    if os.path.isfile(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val_acc", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        print(f"  Resumed at epoch {start_epoch}, best_val_acc={best_val:.2f}% (at epoch {best_epoch})")
    if start_epoch > args.epochs:
        print("Already completed all epochs.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        if use_gt:
            tr_loss, tr_acc = run_epoch_gt(
                model, train_loader, optimizer, device, centers, True, epoch, args.epochs,
                granular=args.granular, object_only=args.object_only, unified_graph=args.unified_graph
            )
            va_loss, va_acc = run_epoch_gt(
                model, val_loader, optimizer, device, centers, False, epoch, args.epochs,
                granular=args.granular, object_only=args.object_only, unified_graph=args.unified_graph
            )
        else:
            tr_loss, tr_acc = run_epoch(
                model, train_loader, optimizer, device, frontend_model, cos_layer, frontend_device,
                centers, args.threshold, True, epoch, args.epochs,
                granular=args.granular, object_only=args.object_only, unified_graph=args.unified_graph
            )
            va_loss, va_acc = run_epoch(
                model, val_loader, optimizer, device, frontend_model, cos_layer, frontend_device,
                centers, args.threshold, False, epoch, args.epochs,
                granular=args.granular, object_only=args.object_only, unified_graph=args.unified_graph
            )
        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"| train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% "
            f"| val_loss={va_loss:.4f} val_acc={va_acc:.2f}%"
        )
        if va_acc > best_val:
            best_val = va_acc
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "best_val_acc": best_val,
                "best_epoch": best_epoch,
                "pos_centers": centers,
                "args": vars(args),
            }, best_path)
            print(f"  Saved best checkpoint: {best_path} (epoch {best_epoch})")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val,
            "best_epoch": best_epoch,
            "pos_centers": centers,
            "args": vars(args),
        }, checkpoint_path)

    print(f"Training done. Best val acc: {best_val:.2f}% (epoch {best_epoch})")


if __name__ == "__main__":
    main()


# With frontend (slower):
#   python3 torch_gnn/train_object_vsa_gnn_frontend.py --dataset-root data/RAVEN-10000 \
#     --frontend-checkpoint log/... --epochs 100 --batch-size 16 --num-workers 2
# With ground truth + VSA codebook (faster, no frontend inference, uses stored codebook):
#   python3 torch_gnn/train_object_vsa_gnn_frontend.py --dataset-root data/RAVEN-10000 --use-gt --node-repr vsa \
#     --frontend-checkpoint log/universal_frontend_... --epochs 100 --batch-size 32 --num-workers 2
