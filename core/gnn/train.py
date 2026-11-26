"""Training loops and batch graph construction for RAVEN GNN."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import NUM_CANDIDATES
from .graph import build_candidate_graph, build_unified_graph, pack_graphs
from .model import CandidateVsaGNN


def to_device_packed(packed: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in packed.items()}


@torch.no_grad()
def extract_batch_objects(
    images: torch.Tensor,
    frontend_model: nn.Module,
    cos_layer: nn.Module,
    frontend_device: torch.device,
    threshold: float,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Extract object attributes from images using frontend (used when not --use-gt)."""
    from gnn_solver import extract_cosine_sims, extract_discrete_objects

    out = []
    for b in range(images.size(0)):
        cosine = extract_cosine_sims(frontend_model, cos_layer, images[b].cpu().numpy(), frontend_device)
        obj_type, obj_size, obj_color, obj_exists, _ = extract_discrete_objects(cosine, threshold=threshold)
        out.append((obj_type, obj_size, obj_color, obj_exists))
    return out


def build_packed_candidates_from_batch(
    extracted_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    centers: Dict[int, Tuple[float, float]],
    granular: bool = False,
    object_only: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    per_candidate: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(NUM_CANDIDATES)]
    for obj_type, obj_size, obj_color, obj_exists in extracted_batch:
        for c in range(NUM_CANDIDATES):
            per_candidate[c].append(
                build_candidate_graph(obj_type, obj_size, obj_color, obj_exists, c, centers,
                                     granular=granular, object_only=object_only)
            )
    return [pack_graphs(gs) for gs in per_candidate]


def build_packed_unified(
    extracted_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    centers: Dict[int, Tuple[float, float]],
    granular: bool = False,
    object_only: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """Build one unified graph per sample; returns list of 1 packed dict."""
    graphs = []
    for obj_type, obj_size, obj_color, obj_exists in extracted_batch:
        g = build_unified_graph(obj_type, obj_size, obj_color, obj_exists, centers, granular=granular, object_only=object_only)
        graphs.append(g)
    packed = pack_graphs(graphs)
    # pack_graphs doesn't include candidate_id; build_unified_graph returns it, so we need to add it
    packed["candidate_id"] = torch.cat([g["candidate_id"] for g in graphs], dim=0)
    return [packed]


def run_epoch(
    model: CandidateVsaGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    frontend_model: nn.Module,
    cos_layer: nn.Module,
    frontend_device: torch.device,
    centers: Dict[int, Tuple[float, float]],
    threshold: float,
    train: bool,
    epoch: int,
    total_epochs: int,
    granular: bool = False,
    object_only: bool = False,
    unified_graph: bool = False,
) -> Tuple[float, float]:
    model.train(mode=train)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    phase = "train" if train else "val"
    pbar = tqdm(loader, desc=f"{phase} {epoch:03d}/{total_epochs:03d}", leave=False)
    for images, target in pbar:
        target = target.to(device)
        extracted = extract_batch_objects(images, frontend_model, cos_layer, frontend_device, threshold)
        if unified_graph:
            packed = [to_device_packed(p, device) for p in build_packed_unified(extracted, centers, granular=granular, object_only=object_only)]
        else:
            packed = [to_device_packed(p, device) for p in build_packed_candidates_from_batch(extracted, centers, granular=granular, object_only=object_only)]
        with torch.set_grad_enabled(train):
            logits = model(packed)
            loss = F.cross_entropy(logits, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        pred = logits.argmax(dim=1)
        total_correct += int((pred == target).sum().item())
        total_samples += target.size(0)
        total_loss += float(loss.item()) * target.size(0)
        pbar.set_postfix(
            loss=f"{(total_loss / max(total_samples, 1)):.4f}",
            acc=f"{(100.0 * total_correct / max(total_samples, 1)):.2f}%",
        )
    return total_loss / max(total_samples, 1), 100.0 * total_correct / max(total_samples, 1)


def run_epoch_gt(
    model: CandidateVsaGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    centers: Dict[int, Tuple[float, float]],
    train: bool,
    epoch: int,
    total_epochs: int,
    granular: bool = False,
    object_only: bool = False,
    unified_graph: bool = False,
) -> Tuple[float, float]:
    """Run epoch using ground truth objects (no frontend). Faster than run_epoch."""
    model.train(mode=train)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    phase = "train" if train else "val"
    pbar = tqdm(loader, desc=f"{phase} {epoch:03d}/{total_epochs:03d}", leave=False)
    for extracted, target in pbar:
        target = target.to(device)
        if unified_graph:
            packed = [to_device_packed(p, device) for p in build_packed_unified(extracted, centers, granular=granular, object_only=object_only)]
        else:
            packed = [to_device_packed(p, device) for p in build_packed_candidates_from_batch(extracted, centers, granular=granular, object_only=object_only)]
        with torch.set_grad_enabled(train):
            logits = model(packed)
            loss = F.cross_entropy(logits, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        pred = logits.argmax(dim=1)
        total_correct += int((pred == target).sum().item())
        total_samples += target.size(0)
        total_loss += float(loss.item()) * target.size(0)
        pbar.set_postfix(
            loss=f"{(total_loss / max(total_samples, 1)):.4f}",
            acc=f"{(100.0 * total_correct / max(total_samples, 1)):.2f}%",
        )
    return total_loss / max(total_samples, 1), 100.0 * total_correct / max(total_samples, 1)
