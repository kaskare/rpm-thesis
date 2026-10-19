"""GNN model components: encoders, layers, CandidateVsaGNN."""

import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    NUM_CANDIDATES,
    NUM_COLOR,
    NUM_NODE_KINDS,
    NUM_PANEL_GRID,
    NUM_POSITIONS,
    NUM_SHAPE,
    NUM_SIZE,
    RELATIONS,
)


def load_codebook_only(checkpoint_dir: str, device: torch.device = None) -> torch.Tensor:
    """Load perception codebook W from frontend checkpoint (no ResNet model)."""
    cb_path = os.path.join(checkpoint_dir, "codebooks.pt")
    if not os.path.isfile(cb_path):
        raise FileNotFoundError(f"Codebook not found: {cb_path}")
    cb_data = torch.load(cb_path, map_location="cpu")
    W = cb_data["perception_cb"]  # (N_OUT, d)
    if device is not None:
        W = W.to(device)
    return W


def compute_w_index(pos: torch.Tensor, typ: torch.Tensor, siz: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    """Compute W row index from (pos, type, size, color). Matches gnn_solver/frontend layout."""
    from .config import NUM_PER_POS
    attr_combo = col * (NUM_SIZE * NUM_SHAPE) + siz * NUM_SHAPE + typ
    return pos * NUM_PER_POS + attr_combo


def scatter_mean(x: torch.Tensor, idx: torch.Tensor, n_groups: int) -> torch.Tensor:
    out = torch.zeros((n_groups, x.size(1)), device=x.device, dtype=x.dtype)
    cnt = torch.zeros((n_groups,), device=x.device, dtype=x.dtype)
    out.index_add_(0, idx, x)
    cnt.index_add_(0, idx, torch.ones_like(idx, dtype=x.dtype))
    return out / cnt.clamp_min(1.0).unsqueeze(-1)


def _softmax_edges_per_dst(edge_index: torch.Tensor, e: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax attention over incoming edges per destination. e: (E,), returns alpha (E,)."""
    dst = edge_index[1]
    if hasattr(torch.Tensor, "scatter_reduce_"):
        e_max = torch.full((n,), -1e9, device=e.device, dtype=e.dtype)
        e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=False)
    else:
        e_max_list = [-1e9] * n
        for d in dst.unique():
            d_val = int(d.item())
            mask = dst == d
            e_max_list[d_val] = float(e[mask].max())
        e_max = torch.tensor(e_max_list, device=e.device, dtype=e.dtype)
    e_exp = torch.exp((e - e_max[dst]).clamp(max=20))
    e_sum = torch.zeros(n, device=e.device, dtype=e.dtype)
    e_sum.scatter_add_(0, dst, e_exp)
    return e_exp / (e_sum[dst] + 1e-8)


class NodeEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, granular: bool = False) -> None:
        super().__init__()
        self.granular = granular
        self.pos_emb = nn.Embedding(NUM_POSITIONS, embed_dim)
        self.col_emb = nn.Embedding(NUM_COLOR, embed_dim)
        self.shp_emb = nn.Embedding(NUM_SHAPE, embed_dim)
        self.siz_emb = nn.Embedding(NUM_SIZE, embed_dim)
        self.role_emb = nn.Embedding(2, embed_dim)
        num_kinds = NUM_NODE_KINDS if granular else 2
        self.kind_emb = nn.Embedding(num_kinds, embed_dim)
        self.panel_emb = nn.Embedding(NUM_PANEL_GRID, embed_dim)

    def forward(self, node_attrs: torch.Tensor, node_kinds: torch.Tensor, panel_roles: torch.Tensor) -> torch.Tensor:
        pos = node_attrs[:, 0].clamp(0, NUM_POSITIONS - 1)
        col = node_attrs[:, 1].clamp(0, NUM_COLOR - 1)
        shp = node_attrs[:, 2].clamp(0, NUM_SHAPE - 1)
        siz = node_attrs[:, 3].clamp(0, NUM_SIZE - 1)
        pnl = node_attrs[:, 4].clamp(0, NUM_PANEL_GRID - 1) if node_attrs.size(1) > 4 else torch.zeros(node_attrs.size(0), dtype=torch.long, device=node_attrs.device)
        knd = node_kinds.clamp(0, NUM_NODE_KINDS - 1 if self.granular else 1)
        rol = panel_roles.clamp(0, 1)

        if self.granular:
            is_obj = (knd == 0).squeeze(-1) if node_kinds.dim() > 1 else (knd == 0)
            is_panel = (knd == 1)
            is_attr_pos = (knd == 2)
            is_attr_col = (knd == 3)
            is_attr_shp = (knd == 4)
            is_attr_siz = (knd == 5)
            emb = torch.zeros(node_attrs.size(0), self.pos_emb.embedding_dim, device=node_attrs.device, dtype=torch.float32)
            emb[is_obj | is_panel] = (
                self.pos_emb(pos[is_obj | is_panel]) + self.col_emb(col[is_obj | is_panel])
                + self.shp_emb(shp[is_obj | is_panel]) + self.siz_emb(siz[is_obj | is_panel])
            )
            emb[is_attr_pos] = self.pos_emb(pos[is_attr_pos])
            emb[is_attr_col] = self.col_emb(col[is_attr_col])
            emb[is_attr_shp] = self.shp_emb(shp[is_attr_shp])
            emb[is_attr_siz] = self.siz_emb(siz[is_attr_siz])
            return emb + self.kind_emb(knd) + self.role_emb(rol) + self.panel_emb(pnl)
        else:
            return (self.pos_emb(pos) + self.col_emb(col) + self.shp_emb(shp)
                    + self.siz_emb(siz) + self.kind_emb(knd) + self.role_emb(rol) + self.panel_emb(pnl))


class VSAEncoder(nn.Module):
    """Encode nodes using W codebook lookup (for GT labels). VSA vectors from frontend codebook."""

    def __init__(self, W: torch.Tensor, vsa_proj_dim: int = 128, granular: bool = False) -> None:
        super().__init__()
        self.granular = granular
        vsa_dim = W.shape[1]
        self.register_buffer("W", W)
        self.register_buffer("zero_row", torch.zeros(1, vsa_dim, device=W.device))
        self.vsa_proj = nn.Sequential(
            nn.Linear(vsa_dim, vsa_proj_dim),
            nn.LayerNorm(vsa_proj_dim),
            nn.ReLU(),
        )
        self.panel_emb = nn.Embedding(NUM_PANEL_GRID, 64)
        self.role_emb = nn.Embedding(2, 64)
        num_kinds = NUM_NODE_KINDS if granular else 2
        self.kind_emb = nn.Embedding(num_kinds, 32)
        self.enc_dim = vsa_proj_dim + 64 + 64 + 32

    def forward(self, node_attrs: torch.Tensor, node_kinds: torch.Tensor, panel_roles: torch.Tensor) -> torch.Tensor:
        pos = node_attrs[:, 0].clamp(0, NUM_POSITIONS - 1)
        col = node_attrs[:, 1].clamp(0, NUM_COLOR - 1)
        typ = node_attrs[:, 2].clamp(0, NUM_SHAPE - 1)
        siz = node_attrs[:, 3].clamp(0, NUM_SIZE - 1)
        pnl = node_attrs[:, 4].clamp(0, NUM_PANEL_GRID - 1) if node_attrs.size(1) > 4 else torch.zeros(node_attrs.size(0), dtype=torch.long, device=node_attrs.device)
        knd = node_kinds.clamp(0, NUM_NODE_KINDS - 1 if self.granular else 1)
        rol = panel_roles.clamp(0, 1)

        is_obj = (node_kinds == 0).squeeze(-1) if node_kinds.dim() > 1 else (node_kinds == 0)
        w_idx = compute_w_index(pos, typ, siz, col)
        w_idx = w_idx.clamp(0, self.W.size(0) - 1)
        vsa_raw = torch.where(is_obj.unsqueeze(-1), self.W[w_idx], self.zero_row.expand(w_idx.size(0), -1))
        vsa_feat = self.vsa_proj(vsa_raw)
        aux = torch.cat([self.panel_emb(pnl), self.role_emb(rol), self.kind_emb(knd)], dim=-1)
        return torch.cat([vsa_feat, aux], dim=-1)


class RelGNLayer(nn.Module):
    """Simple message-passing GNN with edge-type conditioning (no attention, mean aggregation)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.rel_emb = nn.Embedding(len(RELATIONS), hidden_dim)
        self.W = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        rel_feat = self.rel_emb(edge_type)
        msg = self.W(torch.cat([x[src] + rel_feat, x[dst], rel_feat], dim=-1))
        agg = scatter_mean(msg, dst, x.size(0))
        return self.norm(x + self.dropout(self.out_proj(agg)))


class GCNLayer(nn.Module):
    """GCN layer with symmetric degree normalization and edge-type conditioning."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.rel_emb = nn.Embedding(len(RELATIONS), hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n = x.size(0)

        # Symmetric normalization: 1 / sqrt(deg_src * deg_dst)
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device, dtype=x.dtype))
        deg_inv_sqrt = (deg + 1.0).pow(-0.5)  # +1 for implicit self-loop
        norm_coeff = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        rel_feat = self.rel_emb(edge_type)
        msg = self.msg_proj(torch.cat([x[src] + rel_feat, rel_feat], dim=-1))
        msg = msg * norm_coeff.unsqueeze(-1)

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        # Self-loop contribution (degree-normalized)
        agg = agg + deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-1) * x

        return self.norm(x + self.dropout(self.out_proj(agg)))


class GATLayer(nn.Module):
    """Graph Attention layer with edge-type conditioning."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim
        self.rel_emb = nn.Embedding(len(RELATIONS), hidden_dim)
        self.W = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky = 0.2

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        rel_feat = self.rel_emb(edge_type)
        msg = self.W(torch.cat([x[src] + rel_feat, x[dst], rel_feat], dim=-1))
        e = self.attn(F.leaky_relu(msg, self.leaky)).mean(dim=-1)
        alpha = _softmax_edges_per_dst(edge_index, e, x.size(0))
        weighted_msg = alpha.unsqueeze(-1) * msg
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, weighted_msg)
        return self.norm(x + self.dropout(self.out_proj(agg)))


class GraphTransformerBlock(nn.Module):
    """Transformer block with multi-head self-attention over nodes, masked by graph."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, graph_idx: torch.Tensor, num_graphs: int
    ) -> torch.Tensor:
        N, D = x.size()
        q = self.q_proj(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        mask = (graph_idx.unsqueeze(0) != graph_idx.unsqueeze(1)).float() * -1e9
        attn = attn + mask.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(0, 1).reshape(N, D)
        x = self.norm1(x + self.out_proj(out))
        return self.norm2(x + self.ffn(x))


class CandidateVsaGNN(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        node_repr: str = "embedding",
        W_matrix: torch.Tensor = None,
        gnn_type: str = "gat",
        granular: bool = False,
        unified_graph: bool = False,
    ):
        super().__init__()
        self.node_repr = node_repr
        self.gnn_type = gnn_type
        self.granular = granular
        self.unified_graph = unified_graph
        if node_repr == "embedding":
            self.enc = NodeEncoder(embed_dim=embed_dim, granular=granular)
            enc_out_dim = embed_dim
        elif node_repr == "vsa":
            assert W_matrix is not None, "W_matrix required for node_repr='vsa'"
            self.enc = VSAEncoder(W=W_matrix, vsa_proj_dim=128, granular=granular)
            enc_out_dim = self.enc.enc_dim
        else:
            raise ValueError(f"node_repr must be 'embedding' or 'vsa', got {node_repr!r}")
        self.inp = nn.Sequential(
            nn.Linear(enc_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        if gnn_type == "gat":
            self.gnn_layers = nn.ModuleList([
                GATLayer(hidden_dim, num_heads=4, dropout=dropout) for _ in range(num_layers)
            ])
            self.transformer_layers = nn.ModuleList([
                GraphTransformerBlock(hidden_dim, num_heads=4, dropout=dropout) for _ in range(num_layers)
            ])
        elif gnn_type == "gcn":
            self.gnn_layers = nn.ModuleList([
                GCNLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)
            ])
            self.transformer_layers = None
        elif gnn_type == "gnn":
            self.gnn_layers = nn.ModuleList([
                RelGNLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)
            ])
            self.transformer_layers = None
        else:
            raise ValueError(f"gnn_type must be 'gat', 'gcn', or 'gnn', got {gnn_type!r}")
        self.drop = nn.Dropout(dropout)
        self.ctx_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cand_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.score_head[-1].bias)
        nn.init.xavier_uniform_(self.score_head[-1].weight, gain=0.1)

    def forward_unified(self, packed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward for unified graph: one graph per sample, all 8 candidates. Returns [batch_size, 8]."""
        x = self.inp(self.enc(packed["node_attrs"], packed["node_kinds"], packed["panel_roles"]))

        graph_idx = packed["graph_idx"]
        num_graphs = int(packed["num_graphs"])
        candidate_id = packed["candidate_id"]  # -1 context, 0-7 for candidates

        for i, gnn in enumerate(self.gnn_layers):
            x = gnn(x, packed["edge_index"], packed["edge_type"]) + x
            x = self.drop(F.relu(x))
            if self.transformer_layers is not None:
                x = self.transformer_layers[i](x, graph_idx, num_graphs) + x
                x = self.drop(F.relu(x))

        roles = packed["panel_roles"]
        ctx_mask = (roles == 0)
        cand_mask = (roles == 1)

        ctx_emb = scatter_mean(x[ctx_mask], graph_idx[ctx_mask], num_graphs)  # [num_graphs, D]

        # Pool per (graph, candidate_id) -> [num_graphs, 8, D]
        cand_emb = torch.zeros(num_graphs, NUM_CANDIDATES, x.size(1), device=x.device, dtype=x.dtype)
        for c in range(NUM_CANDIDATES):
            mask = cand_mask & (candidate_id == c)
            if mask.any():
                # Map graph_idx[mask] to 0..num_graphs-1; need group ids
                gidx = graph_idx[mask]
                cand_emb[:, c] = scatter_mean(x[mask], gidx, num_graphs)

        ctx_emb = self.ctx_pool(ctx_emb)  # [num_graphs, D]
        cand_emb = self.cand_pool(cand_emb)  # [num_graphs, 8, D]

        combined = torch.cat([
            ctx_emb.unsqueeze(1).expand(-1, NUM_CANDIDATES, -1),
            cand_emb,
        ], dim=-1)  # [num_graphs, 8, 2D]
        logits = self.score_head(combined).squeeze(-1)  # [num_graphs, 8]
        return logits

    def forward_one(self, packed: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.inp(self.enc(packed["node_attrs"], packed["node_kinds"], packed["panel_roles"]))

        graph_idx = packed["graph_idx"]
        num_graphs = int(packed["num_graphs"])
        for i, gnn in enumerate(self.gnn_layers):
            x = gnn(x, packed["edge_index"], packed["edge_type"]) + x
            x = self.drop(F.relu(x))
            if self.transformer_layers is not None:
                x = self.transformer_layers[i](x, graph_idx, num_graphs) + x
                x = self.drop(F.relu(x))

        roles = packed["panel_roles"]
        ctx_mask = (roles == 0)
        cand_mask = (roles == 1)

        ctx_emb = scatter_mean(x[ctx_mask], graph_idx[ctx_mask], num_graphs)
        cand_emb = scatter_mean(x[cand_mask], graph_idx[cand_mask], num_graphs)

        ctx_emb = self.ctx_pool(ctx_emb)
        cand_emb = self.cand_pool(cand_emb)

        combined = torch.cat([ctx_emb, cand_emb], dim=-1)
        return self.score_head(combined).squeeze(-1)

    def forward(self, packed_candidates: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if self.unified_graph:
            return self.forward_unified(packed_candidates[0])
        return torch.stack([self.forward_one(p) for p in packed_candidates], dim=1)
