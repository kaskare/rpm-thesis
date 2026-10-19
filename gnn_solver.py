"""
GNN Solver for RAVEN using Universal Frontend

Uses the trained universal frontend to detect objects and extract discrete
attribute predictions (type, size, color) per position per panel, then
represents each object as a learned embedding vector for a GNN.

Node representation (per position slot per panel):
    Option 1 — "embedding" (default):
        embed(type) + embed(size) + embed(color) + embed(position) + embed(panel)
        All embeddings learned end-to-end with the GNN.
        Empty positions get a special "no-object" token per attribute.

    Option 2 — "vsa":
        W-row from the codebook + learnable projector (VSA was trained for
        perception, not reasoning). Uses: vsa_proj(W_row) + panel_embed +
        pos_embed + exist_embed. Empty positions use zero W-row.
        VSA mode computes raw VSA vectors live during training (frozen frontend
        runs on images); no precomputation.

Graph structure:
    22 positions × 16 panels = 352 nodes per sample (fixed size).
    Edges: intra-panel + same-position across row/column + candidate-to-context.

Usage:
    # Load defaults from configs/gnn_solver.yaml; override via CLI
    python gnn_solver.py --config configs/gnn_solver.yaml \
        --frontend-checkpoint log/universal_frontend_correct_proportional_mean \
        --dataset data/RAVEN-10000 --save-dir log/gnn_solver

    # Precompute (embedding mode only)
    python gnn_solver.py --config configs/gnn_solver.yaml \
        --frontend-checkpoint ... --dataset ... --mode precompute

    # Test
    python gnn_solver.py --config configs/gnn_solver.yaml \
        --frontend-checkpoint ... --dataset ... --save-dir log/gnn_solver --mode test
"""

import argparse
import os
import sys
import glob

try:
    import yaml
except ImportError:
    yaml = None

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'raven'))
sys.path.insert(0, os.path.join(ROOT, 'neuro-vector-symbolic-architectures'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from nvsa.perception.resnet import resnet18
from nvsa.perception.metrics import fixCos
from const import DIM_TYPE, DIM_SIZE, DIM_COLOR


# ── Constants ────────────────────────────────────────────────────────────────

NUM_POS = 22             # universal position slots
NUM_PANELS = 16          # 8 context + 8 candidates
NUM_PER_POS = DIM_COLOR * DIM_SIZE * DIM_TYPE + 1   # 301
N_OUT = NUM_POS * NUM_PER_POS                         # 6622
NUM_ATTR_COMBOS = DIM_COLOR * DIM_SIZE * DIM_TYPE     # 300
NUM_NODES = NUM_POS * NUM_PANELS                      # 352

# "No object" sentinel values (one past the valid range for each attribute)
NO_TYPE  = DIM_TYPE    # 5
NO_SIZE  = DIM_SIZE    # 6
NO_COLOR = DIM_COLOR   # 10

# RAVEN 3×3 grid layout
CANDIDATE_PANELS = list(range(8, 16))
GRID_ROWS = [[0, 1, 2], [3, 4, 5], [6, 7]]
GRID_COLS = [[0, 3, 6], [1, 4, 7], [2, 5]]


# ══════════════════════════════════════════════════════════════════════════════
# 1. FRONTEND & FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def load_frontend(checkpoint_dir, device="cpu", use_best=True):
    """Load the trained universal frontend."""
    device = torch.device(device)

    cb_path = os.path.join(checkpoint_dir, "codebooks.pt")
    cb_data = torch.load(cb_path, map_location="cpu")
    perception_cb = cb_data["perception_cb"]

    n_out = perception_cb.shape[0]
    d = perception_cb.shape[1]

    ckpt_name = "model_best.pth.tar" if use_best else "checkpoint.pth.tar"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_d = ckpt.get("d", d)
    s = ckpt.get("s", 7.0)

    model = resnet18(pretrained=True, progress=False, num_classes=ckpt_d, no_maxpool=True)
    cos_layer = fixCos(num_features=ckpt_d, num_classes=n_out, mat=perception_cb,
                       fixed_weights=True, s=s, trainable_s=True)

    model.load_state_dict(ckpt["state_dict_model"])
    cos_layer.load_state_dict(ckpt["state_dict_cos"])
    model.to(device).eval()
    cos_layer.to(device).eval()

    print(f"Loaded frontend: d={ckpt_d}, W={perception_cb.shape}, "
          f"epoch={ckpt.get('epoch','?')}, acc={ckpt.get('best_acc','?')}%")

    return model, cos_layer, perception_cb, device


@torch.no_grad()
def extract_cosine_sims(model, cos_layer, images, device):
    """Run frontend → raw cosine similarities (16, N_OUT)."""
    imgs = torch.tensor(images, dtype=torch.float).unsqueeze(1).to(device)
    q = model(imgs)
    W_norm = F.normalize(cos_layer.W.data)
    q_norm = F.normalize(q)
    return F.linear(q_norm, W_norm).cpu()  # (16, N_OUT)


def extract_discrete_objects(cosine_sim, threshold=0.23):
    """
    Extract discrete object attributes from cosine similarities.

    For each position in each panel, find the best-matching (type, size, color)
    combination. If the best similarity is below threshold, mark as "no object".

    Parameters
    ----------
    cosine_sim : Tensor (16, N_OUT)
    threshold  : float

    Returns
    -------
    obj_type  : LongTensor (16, NUM_POS)  — type index (0..4) or NO_TYPE if empty
    obj_size  : LongTensor (16, NUM_POS)  — size index (0..5) or NO_SIZE if empty
    obj_color : LongTensor (16, NUM_POS)  — color index (0..9) or NO_COLOR if empty
    obj_exists: BoolTensor (16, NUM_POS)  — whether object exists at this position
    best_sim  : Tensor (16, NUM_POS)      — best cosine similarity
    """
    # Reshape: (16, NUM_POS, NUM_PER_POS)
    sim = cosine_sim.view(NUM_PANELS, NUM_POS, NUM_PER_POS)

    # Only attribute entries (first 300), ignore detection vector
    attr_sim = sim[:, :, :NUM_ATTR_COMBOS]    # (16, 22, 300)

    # Best match per position
    best_sim, best_idx = attr_sim.max(dim=-1)  # (16, 22), (16, 22)

    # Decode flat index → (color, size, type)
    obj_color = best_idx // (DIM_SIZE * DIM_TYPE)
    remainder = best_idx % (DIM_SIZE * DIM_TYPE)
    obj_size  = remainder // DIM_TYPE
    obj_type  = remainder % DIM_TYPE

    # Threshold for existence
    obj_exists = best_sim > threshold

    # Set "no object" sentinel for empty positions
    obj_type  = torch.where(obj_exists, obj_type,  torch.full_like(obj_type,  NO_TYPE))
    obj_size  = torch.where(obj_exists, obj_size,  torch.full_like(obj_size,  NO_SIZE))
    obj_color = torch.where(obj_exists, obj_color, torch.full_like(obj_color, NO_COLOR))

    return obj_type, obj_size, obj_color, obj_exists, best_sim


def extract_w_indices(cosine_sim, threshold=0.23):
    """
    Extract the W-row index for the best-matching codebook entry per position.
    Used when node_repr="vsa" to look up the VSA binding vector.

    Parameters
    ----------
    cosine_sim : (16, N_OUT) or (B, 16, N_OUT)

    Returns
    -------
    w_indices : LongTensor (16, NUM_POS) or (B, 16, NUM_POS)
    obj_exists: BoolTensor (16, NUM_POS) or (B, 16, NUM_POS)
    """
    if cosine_sim.dim() == 2:
        sim = cosine_sim.view(NUM_PANELS, NUM_POS, NUM_PER_POS)
    else:
        B = cosine_sim.shape[0]
        sim = cosine_sim.view(B, NUM_PANELS, NUM_POS, NUM_PER_POS)

    attr_sim = sim[..., :NUM_ATTR_COMBOS]
    best_sim, best_local_idx = attr_sim.max(dim=-1)

    obj_exists = best_sim > threshold

    pos_offsets = (
        torch.arange(NUM_POS, device=cosine_sim.device) * NUM_PER_POS
    )  # (22,)
    if cosine_sim.dim() == 3:
        pos_offsets = pos_offsets.unsqueeze(0).unsqueeze(0)  # (1, 1, 22)
    else:
        pos_offsets = pos_offsets.unsqueeze(0)  # (1, 22)
    w_indices = best_local_idx + pos_offsets
    w_indices = torch.where(obj_exists, w_indices, torch.full_like(w_indices, -1))
    return w_indices, obj_exists


# ══════════════════════════════════════════════════════════════════════════════
# 2. GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def node_id(panel, pos):
    return panel * NUM_POS + pos


def build_edge_index():
    """
    Build static edge index for the RAVEN graph.

    Edges:
        1. Intra-panel: all positions fully connected within each panel
        2. Same-position across context rows
        3. Same-position across context columns
        4. Candidate-to-context: candidates connect to panels 2, 5, 6, 7
    """
    edges = set()

    def add(u, v):
        edges.add((u, v))
        edges.add((v, u))

    # 1. Intra-panel
    for panel in range(NUM_PANELS):
        for p1 in range(NUM_POS):
            for p2 in range(p1 + 1, NUM_POS):
                add(node_id(panel, p1), node_id(panel, p2))

    # 2. Same-position across rows
    for row in GRID_ROWS:
        for pos in range(NUM_POS):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    add(node_id(row[i], pos), node_id(row[j], pos))

    # 3. Same-position across columns
    for col in GRID_COLS:
        for pos in range(NUM_POS):
            for i in range(len(col)):
                for j in range(i + 1, len(col)):
                    add(node_id(col[i], pos), node_id(col[j], pos))

    # 4. Candidate-to-context
    relevant_context = [2, 5, 6, 7]
    for cand in CANDIDATE_PANELS:
        for ctx in relevant_context:
            for pos in range(NUM_POS):
                add(node_id(cand, pos), node_id(ctx, pos))

    src, dst = zip(*edges)
    return torch.tensor([src, dst], dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════════
# 3. GNN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GCNConv(nn.Module):
    """GCN convolution (no PyG dependency)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        src, dst = edge_index
        N = x.shape[0]

        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
        deg_inv_sqrt = (deg + 1).pow(-0.5)

        msg = x[src] * deg_inv_sqrt[src].unsqueeze(-1)
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        agg = agg * deg_inv_sqrt.unsqueeze(-1)
        agg = agg + x  # self-loop

        return self.linear(agg)


class RAVENSolverGNN(nn.Module):
    """
    GNN that solves RAVEN from per-object discrete attribute representations.

    Node representations:
        "embedding": learned embeddings for each attribute, summed per object
        "vsa":       codebook W-row vectors (512-dim)
    """

    def __init__(self, hidden_dim=128, num_layers=4, dropout=0.1,
                 node_repr="embedding", embed_dim=64, vsa_dim=512,
                 W_matrix=None, frontend_model=None, cos_layer=None,
                 device=None, threshold=0.23):
        super().__init__()
        self.node_repr = node_repr
        self.hidden_dim = hidden_dim

        if node_repr == "vsa" and frontend_model is not None:
            self._frontend_model = frontend_model
            self._cos_layer = cos_layer
            self._frontend_device = device
            self._threshold = threshold
            for p in frontend_model.parameters():
                p.requires_grad = False
            for p in cos_layer.parameters():
                p.requires_grad = False

        if node_repr == "embedding":
            # Learned embedding per attribute (+1 for "no object" sentinel)
            self.type_embed  = nn.Embedding(DIM_TYPE + 1,  embed_dim)
            self.size_embed  = nn.Embedding(DIM_SIZE + 1,  embed_dim)
            self.color_embed = nn.Embedding(DIM_COLOR + 1, embed_dim)
            self.pos_embed   = nn.Embedding(NUM_POS,       embed_dim)
            self.panel_embed = nn.Embedding(NUM_PANELS,    embed_dim)
            self.exist_embed = nn.Embedding(2, embed_dim)  # 0=empty, 1=exists

            # Sum of 6 embeddings, each embed_dim → project to hidden
            input_dim = embed_dim

        elif node_repr == "vsa":
            # W matrix rows + learnable projector (VSA was trained for perception, not reasoning)
            assert W_matrix is not None, "Must provide W_matrix for vsa mode"
            zero_row = torch.zeros(1, W_matrix.shape[1])
            self.register_buffer("W_extended",
                                 torch.cat([W_matrix, zero_row], dim=0))
            # Learnable projection: adapt VSA space for reasoning (key fix for "doesn't learn")
            vsa_proj_dim = min(256, vsa_dim)
            self.vsa_proj = nn.Sequential(
                nn.Linear(vsa_dim, vsa_proj_dim),
                nn.LayerNorm(vsa_proj_dim),
                nn.ReLU(),
            )
            self.panel_embed_vsa = nn.Embedding(NUM_PANELS, 64)
            self.pos_embed_vsa   = nn.Embedding(NUM_POS, 64)
            self.exist_embed_vsa = nn.Embedding(2, 32)  # explicit empty vs. object
            input_dim = vsa_proj_dim + 64 + 64 + 32

        else:
            raise ValueError(f"Unknown node_repr: {node_repr}")

        # Input projection (deeper for VSA to adapt raw features)
        if node_repr == "vsa":
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )

        # GCN layers with residual + norm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Context aggregation
        self.context_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Candidate scoring
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _embed_nodes(self, obj_type, obj_size, obj_color, obj_exists, B):
        """
        Build node features from discrete attributes using learned embeddings.

        Parameters
        ----------
        obj_type   : (B, 16, 22) LongTensor
        obj_size   : (B, 16, 22) LongTensor
        obj_color  : (B, 16, 22) LongTensor
        obj_exists : (B, 16, 22) BoolTensor

        Returns
        -------
        x : (B * NUM_NODES, hidden_dim)
        """
        device = obj_type.device

        # Flatten: (B, 16, 22) → (B * 352)
        flat_type  = obj_type.reshape(-1)
        flat_size  = obj_size.reshape(-1)
        flat_color = obj_color.reshape(-1)
        flat_exist = obj_exists.long().reshape(-1)

        # Position and panel IDs (tiled for batch)
        pos_ids   = torch.arange(NUM_POS, device=device).repeat(NUM_PANELS).repeat(B)
        panel_ids = torch.arange(NUM_PANELS, device=device).repeat_interleave(NUM_POS).repeat(B)

        # Sum of embeddings → (B*352, embed_dim)
        x = (self.type_embed(flat_type)
             + self.size_embed(flat_size)
             + self.color_embed(flat_color)
             + self.pos_embed(pos_ids)
             + self.panel_embed(panel_ids)
             + self.exist_embed(flat_exist))

        return x

    def _vsa_nodes(self, w_indices, obj_exists, B):
        """
        Build node features from W-row VSA vectors + learned projection.

        Parameters
        ----------
        w_indices : (B, 16, 22) LongTensor — W-row index, -1 for no object
        obj_exists: (B, 16, 22) BoolTensor — whether object detected

        Returns
        -------
        x : (B * NUM_NODES, input_dim)
        """
        device = w_indices.device

        # Map -1 to the last row (zero vector)
        safe_idx = w_indices.clone()
        safe_idx[safe_idx < 0] = self.W_extended.shape[0] - 1
        flat_idx = safe_idx.reshape(-1)

        vsa_raw = self.W_extended[flat_idx]  # (B*352, vsa_dim)
        vsa_feat = self.vsa_proj(vsa_raw)    # (B*352, vsa_proj_dim)

        # Panel and position embeddings
        pos_ids   = torch.arange(NUM_POS, device=device).repeat(NUM_PANELS).repeat(B)
        panel_ids = torch.arange(NUM_PANELS, device=device).repeat_interleave(NUM_POS).repeat(B)

        flat_exist = obj_exists.long().reshape(-1)

        x = torch.cat([
            vsa_feat,
            self.pos_embed_vsa(pos_ids),
            self.panel_embed_vsa(panel_ids),
            self.exist_embed_vsa(flat_exist),
        ], dim=-1)

        return x

    def forward(self, obj_data, edge_index, batch_size, images=None):
        """
        Parameters
        ----------
        obj_data   : dict (or None when node_repr=vsa and images provided)
        edge_index : (2, E) for a single sample
        batch_size : int
        images     : (B, 16, 80, 80) optional; for VSA live mode, raw panels

        Returns
        -------
        scores : (B, 8)
        """
        B = batch_size
        device = edge_index.device

        # VSA live: compute w_indices and exists from images (frontend frozen)
        if self.node_repr == "vsa" and images is not None:
            with torch.no_grad():
                imgs = images.to(self._frontend_device)
                # (B, 16, H, W) -> (B, 16, 1, H, W); frontend expects 160×160
                imgs = imgs.unsqueeze(2)
                if imgs.shape[-2:] != (160, 160):
                    imgs = F.interpolate(
                        imgs.view(-1, 1, imgs.shape[-2], imgs.shape[-1]),
                        size=(160, 160), mode="bilinear", align_corners=False
                    ).view(imgs.shape[0], imgs.shape[1], 1, 160, 160)
                B2, N, C, H, W = imgs.shape
                imgs_flat = imgs.view(B2 * N, C, H, W)
                q = self._frontend_model(imgs_flat)
                W_norm = F.normalize(self._cos_layer.W.data)
                q_norm = F.normalize(q)
                cosine_sim = F.linear(q_norm, W_norm).view(B2, N, -1)
            w_indices, obj_exists = extract_w_indices(cosine_sim, self._threshold)
            w_indices = w_indices.to(device)
            obj_exists = obj_exists.to(device)
            x = self._vsa_nodes(w_indices, obj_exists, B)
        elif self.node_repr == "embedding":
            x = self._embed_nodes(
                obj_data["type"], obj_data["size"],
                obj_data["color"], obj_data["exists"], B
            )
        elif self.node_repr == "vsa":
            x = self._vsa_nodes(obj_data["w_indices"], obj_data["exists"], B)

        # Expand edge_index for batch
        offsets = torch.arange(B, device=device) * NUM_NODES
        batch_edge_index = torch.cat([
            edge_index + off for off in offsets
        ], dim=1)

        # Input projection
        x = self.input_proj(x)

        # GCN layers with residual
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, batch_edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual

        # Reshape: (B, 16, 22, hidden)
        x = x.view(B, NUM_PANELS, NUM_POS, -1)

        # Pool over positions → (B, 16, hidden)
        panel_repr = x.mean(dim=2)

        # Context: mean of 8 context panels
        context = panel_repr[:, :8].mean(dim=1)
        context = self.context_pool(context)

        # Score each candidate
        cand_repr = panel_repr[:, 8:]                       # (B, 8, hidden)
        ctx_exp = context.unsqueeze(1).expand_as(cand_repr)  # (B, 8, hidden)
        combined = torch.cat([ctx_exp, cand_repr], dim=-1)   # (B, 8, 2*hidden)
        scores = self.score_mlp(combined).squeeze(-1)        # (B, 8)

        return scores


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASET
# ══════════════════════════════════════════════════════════════════════════════

class RAVENGNNDataset(Dataset):
    """
    RAVEN dataset returning per-object discrete attributes for the GNN.

    Each sample yields:
        obj_type   : (16, 22) LongTensor
        obj_size   : (16, 22) LongTensor
        obj_color  : (16, 22) LongTensor
        obj_exists : (16, 22) BoolTensor
        w_indices  : (16, 22) LongTensor  (for VSA mode)
        target     : int (0-7)
    """

    def __init__(self, dataset_path, split, configs=None,
                 frontend_model=None, cos_layer=None, device="cpu",
                 threshold=0.23, precomputed_dir=None, return_images_only=False):
        if configs is None:
            configs = [
                "center_single", "distribute_four", "distribute_nine",
                "left_center_single_right_center_single",
                "up_center_single_down_center_single",
                "in_center_single_out_center_single",
                "in_distribute_four_out_center_single",
            ]

        self.frontend_model = frontend_model
        self.cos_layer = cos_layer
        self.device = device
        self.threshold = threshold
        self.precomputed_dir = precomputed_dir
        self.split = split
        self.return_images_only = return_images_only

        self.samples = []

        print(f"Loading {split} data...")
        for config in configs:
            pattern = os.path.join(dataset_path, config, f"*_{split}.npz")
            npz_files = sorted(glob.glob(pattern))
            for npz_path in npz_files:
                data = np.load(npz_path)
                target = int(data["target"])
                self.samples.append((npz_path, target))

        print(f"  Total: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, target = self.samples[idx]

        # VSA live mode: return raw images (frontend runs in GNN forward)
        if self.return_images_only:
            npz_data = np.load(npz_path)
            images = npz_data["image"]  # (16, 80, 80)
            return torch.tensor(images, dtype=torch.float32), target

        # Check precomputed (precompute saves to features_{split}/)
        if self.precomputed_dir:
            fname = os.path.basename(npz_path).replace(".npz", "_obj.pt")
            feat_path = os.path.join(
                self.precomputed_dir, f"features_{self.split}", fname
            )
            if not os.path.isfile(feat_path):
                feat_path = os.path.join(self.precomputed_dir, fname)  # fallback
            if os.path.isfile(feat_path):
                data = torch.load(feat_path, map_location="cpu")
                return data, target

        # Extract on-the-fly
        npz_data = np.load(npz_path)
        images = npz_data["image"]

        cosine_sim = extract_cosine_sims(
            self.frontend_model, self.cos_layer, images, self.device
        )

        obj_type, obj_size, obj_color, obj_exists, _ = \
            extract_discrete_objects(cosine_sim, self.threshold)

        w_indices, _ = extract_w_indices(cosine_sim, self.threshold)

        data = {
            "type": obj_type,        # (16, 22)
            "size": obj_size,        # (16, 22)
            "color": obj_color,      # (16, 22)
            "exists": obj_exists,    # (16, 22)
            "w_indices": w_indices,  # (16, 22)
        }
        return data, target


def collate_fn(batch):
    """Custom collate: stack dicts of tensors + targets."""
    data_list, targets = zip(*batch)
    batched = {}
    for key in data_list[0]:
        batched[key] = torch.stack([d[key] for d in data_list])  # (B, 16, 22)
    targets = torch.tensor(targets, dtype=torch.long)
    return batched, targets


def collate_images_fn(batch):
    """Collate for VSA live mode: stack images + targets."""
    images_list, targets = zip(*batch)
    images = torch.stack(images_list)  # (B, 16, 80, 80)
    targets = torch.tensor(targets, dtype=torch.long)
    return images, targets


# ══════════════════════════════════════════════════════════════════════════════
# 5. PRECOMPUTE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def precompute_features(frontend_model, cos_layer, dataset_path, output_dir,
                        split, device, threshold=0.23, configs=None):
    """Pre-extract discrete object attributes and save as .pt files."""
    if configs is None:
        configs = [
            "center_single", "distribute_four", "distribute_nine",
            "left_center_single_right_center_single",
            "up_center_single_down_center_single",
            "in_center_single_out_center_single",
            "in_distribute_four_out_center_single",
        ]

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for config in configs:
        pattern = os.path.join(dataset_path, config, f"*_{split}.npz")
        npz_files = sorted(glob.glob(pattern))
        print(f"  {config}: {len(npz_files)} files")

        for npz_path in tqdm(npz_files, desc=f"  {config}"):
            data = np.load(npz_path)
            images = data["image"]

            cosine_sim = extract_cosine_sims(frontend_model, cos_layer, images, device)
            obj_type, obj_size, obj_color, obj_exists, _ = \
                extract_discrete_objects(cosine_sim, threshold)
            w_indices, _ = extract_w_indices(cosine_sim, threshold)

            feat = {
                "type": obj_type,
                "size": obj_size,
                "color": obj_color,
                "exists": obj_exists,
                "w_indices": w_indices,
            }

            feat_path = os.path.join(
                output_dir,
                os.path.basename(npz_path).replace(".npz", "_obj.pt")
            )
            torch.save(feat, feat_path)
            count += 1

    print(f"Precomputed {count} samples → {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(gnn_model, loader, edge_index, optimizer, device, clip=1.0,
                vsa_live=False):
    gnn_model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for batch, targets in tqdm(loader, desc="  train"):
        B = targets.shape[0]
        targets = targets.to(device)
        if vsa_live:
            images = batch.to(device)
            scores = gnn_model(None, edge_index, B, images=images)
        else:
            obj_data = {k: v.to(device) for k, v in batch.items()}
            scores = gnn_model(obj_data, edge_index, B)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), clip)
        optimizer.step()

        pred = scores.argmax(dim=-1)
        total_correct += (pred == targets).sum().item()
        total_loss += loss.item() * B
        total_samples += B

    return total_loss / total_samples, 100.0 * total_correct / total_samples


@torch.no_grad()
def evaluate(gnn_model, loader, edge_index, device, vsa_live=False):
    gnn_model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for batch, targets in tqdm(loader, desc="  eval "):
        B = targets.shape[0]
        targets = targets.to(device)
        if vsa_live:
            images = batch.to(device)
            scores = gnn_model(None, edge_index, B, images=images)
        else:
            obj_data = {k: v.to(device) for k, v in batch.items()}
            scores = gnn_model(obj_data, edge_index, B)
        loss = criterion(scores, targets)

        pred = scores.argmax(dim=-1)
        total_correct += (pred == targets).sum().item()
        total_loss += loss.item() * B
        total_samples += B

    return total_loss / total_samples, 100.0 * total_correct / total_samples


# ══════════════════════════════════════════════════════════════════════════════
# 7. CONFIG & MAIN
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "gnn_solver.yaml")


def load_config(path):
    """Load YAML config; return dict with defaults for missing keys."""
    # In-code defaults (used when YAML missing or PyYAML not installed)
    defaults = {
        "mode": "train",
        "frontend_checkpoint": "log/universal_frontend_correct_proportional_mean1",
        "dataset": "data/RAVEN-10000",
        "save_dir": "log/gnn_solver",
        "precomputed_dir": "",
        "node_repr": "vsa",
        "embed_dim": 128,
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout": 0.1,
        "epochs": 100,
        "batch_size": 32,
        "lr": 0.01,
        "weight_decay": 0.0001,
        "clip": 1.0,
        "threshold": 0.23,
        "num_workers": 4,
        "seed": 1234,
        "device": "cuda:0",
    }
    if yaml is None:
        return defaults
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return defaults
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in cfg.items():
        key = k.replace("-", "_")
        if key in defaults and v is not None:
            if isinstance(v, str):
                v = v.strip()
            defaults[key] = v
    return defaults


def main():
    # Resolve config path before full parse (CLI overrides take precedence)
    config_path = DEFAULT_CONFIG
    if "--config" in sys.argv:
        i = sys.argv.index("--config")
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
            config_path = sys.argv[i + 1]
    # Try ROOT-relative if path does not exist (e.g. run from different cwd)
    if not os.path.isabs(config_path):
        config_path = os.path.join(ROOT, config_path)
    if not os.path.isfile(config_path):
        config_path = DEFAULT_CONFIG  # fallback to default
    config_path = os.path.abspath(config_path)

    cfg = load_config(config_path)

    parser = argparse.ArgumentParser(description="GNN Solver for RAVEN")
    parser.add_argument("--config", type=str, default=config_path,
                        help=f"Config YAML path (default: {DEFAULT_CONFIG})")
    parser.add_argument("--mode", type=str, default=cfg["mode"],
                        choices=["train", "test", "precompute"])
    parser.add_argument("--frontend-checkpoint", type=str, default=cfg["frontend_checkpoint"])
    parser.add_argument("--dataset", type=str, default=cfg["dataset"])
    parser.add_argument("--save-dir", type=str, default=cfg["save_dir"])
    parser.add_argument("--precomputed-dir", type=str, default=cfg["precomputed_dir"])
    parser.add_argument("--node-repr", type=str, default=cfg["node_repr"],
                        choices=["embedding", "vsa"])
    parser.add_argument("--embed-dim", type=int, default=cfg["embed_dim"])
    parser.add_argument("--hidden-dim", type=int, default=cfg["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=cfg["num_layers"])
    parser.add_argument("--dropout", type=float, default=cfg["dropout"])
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--lr", type=float, default=cfg["lr"])
    parser.add_argument("--weight-decay", type=float, default=cfg["weight_decay"])
    parser.add_argument("--clip", type=float, default=cfg["clip"])
    parser.add_argument("--threshold", type=float, default=cfg["threshold"])
    parser.add_argument("--num-workers", type=int, default=cfg["num_workers"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--device", type=str, default=cfg["device"])

    args = parser.parse_args()

    if not args.frontend_checkpoint or not args.dataset:
        parser.error(
            "--frontend-checkpoint and --dataset are required "
            "(set in config YAML or pass via CLI). Config loaded from: %s" % config_path
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load frontend ──
    print("Loading universal frontend...")
    frontend_model, cos_layer, perception_cb, _ = load_frontend(
        args.frontend_checkpoint, device=device
    )

    # ── Precompute mode ──
    if args.mode == "precompute":
        for split in ["train", "val", "test"]:
            out_dir = os.path.join(
                args.precomputed_dir or args.save_dir, f"features_{split}"
            )
            print(f"\nPrecomputing {split} → {out_dir}")
            precompute_features(frontend_model, cos_layer, args.dataset,
                                out_dir, split, device, args.threshold)
        return

    # ── Build graph ──
    print("Building graph structure...")
    edge_index = build_edge_index().to(device)
    print(f"  Nodes/sample: {NUM_NODES}  |  Edges/sample: {edge_index.shape[1]}")

    # ── Create GNN ──
    vsa_live = args.node_repr == "vsa"
    W_matrix = perception_cb if args.node_repr == "vsa" else None
    gnn_model = RAVENSolverGNN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        node_repr=args.node_repr,
        embed_dim=args.embed_dim,
        vsa_dim=perception_cb.shape[1],
        W_matrix=W_matrix,
        frontend_model=frontend_model if vsa_live else None,
        cos_layer=cos_layer if vsa_live else None,
        device=device if vsa_live else None,
        threshold=args.threshold if vsa_live else 0.23,
    ).to(device)

    num_params = sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)
    print(f"  GNN parameters: {num_params:,}  |  node_repr={args.node_repr}"
          f"{'  |  VSA live (frontend frozen)' if vsa_live else ''}")

    # ── Data ──
    precomputed = None if vsa_live else (args.precomputed_dir or None)
    collate = collate_images_fn if vsa_live else collate_fn

    if args.mode == "test":
        ckpt_path = os.path.join(args.save_dir, "gnn_best.pth.tar")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        gnn_model.load_state_dict(ckpt["state_dict_gnn"])
        gnn_model.to(device)
        print(f"Loaded GNN from {ckpt_path} (acc={ckpt.get('best_acc','?')}%)")

        test_ds = RAVENGNNDataset(
            args.dataset, "test", frontend_model=frontend_model,
            cos_layer=cos_layer, device=device, threshold=args.threshold,
            precomputed_dir=precomputed, return_images_only=vsa_live,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 num_workers=args.num_workers, collate_fn=collate)
        loss, acc = evaluate(gnn_model, test_loader, edge_index, device, vsa_live)
        print(f"Test: loss={loss:.4f}  acc={acc:.2f}%")
        return

    # ── Train mode ──
    print("\nLoading data...")
    train_ds = RAVENGNNDataset(
        args.dataset, "train", frontend_model=frontend_model,
        cos_layer=cos_layer, device=device, threshold=args.threshold,
        precomputed_dir=precomputed, return_images_only=vsa_live,
    )
    val_ds = RAVENGNNDataset(
        args.dataset, "val", frontend_model=frontend_model,
        cos_layer=cos_layer, device=device, threshold=args.threshold,
        precomputed_dir=precomputed, return_images_only=vsa_live,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
    )

    optimizer = optim.Adam(gnn_model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    print(f"\nTraining GNN for {args.epochs} epochs")
    print(f"  hidden={args.hidden_dim}, layers={args.num_layers}, "
          f"node_repr={args.node_repr}, embed_dim={args.embed_dim}")
    print(f"  lr={args.lr}, bs={args.batch_size}, threshold={args.threshold}")
    print(f"  Device: {device}\n")

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs-1}  (lr={lr:.6f})")

        train_loss, train_acc = train_epoch(
            gnn_model, train_loader, edge_index, optimizer, device, args.clip,
            vsa_live=vsa_live,
        )
        val_loss, val_acc = evaluate(gnn_model, val_loader, edge_index, device,
                                      vsa_live=vsa_live)

        scheduler.step()

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print(f"  train: loss={train_loss:.4f} acc={train_acc:.2f}%  |  "
              f"val: loss={val_loss:.4f} acc={val_acc:.2f}%"
              f"{'  *** BEST ***' if is_best else ''}")

        torch.save({
            "epoch": epoch + 1,
            "state_dict_gnn": gnn_model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }, os.path.join(args.save_dir, "gnn_checkpoint.pth.tar"))

        if is_best:
            torch.save({
                "epoch": epoch + 1,
                "state_dict_gnn": gnn_model.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
            }, os.path.join(args.save_dir, "gnn_best.pth.tar"))

    print(f"\nDone. Best val acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
