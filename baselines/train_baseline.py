#!/usr/bin/env python3
"""Torch baselines for the facet model: a plain MLP and an all-to-all GNN, each sized to the facet model's
number of weights, trained on the facet model's task. Everything except the architecture is the facet
model's: the puzzles and splits, the input values as thermometer codes, one sample per component of the
missing panel, the generated target (per attribute the mean code of the component's objects, the object
count, the occupied positions), sigmoid outputs with MSE, the epoch picked on the selection split, and the
metrics (matched = the closest candidate is the answer, ties wrong; exact = the decoded panel equals the
answer's). Candidates are never seen in training.

  mlp: the context panels flattened: for every panel and every position slot in use, an occupancy bit and
       the three attribute codes; plus the component asked for. Three hidden ReLU layers of width H.
  gnn: one node per object of the context panels and one query node for the missing panel. Node input:
       row, column, position slot and component one-hots, the three attribute codes, a query flag. Every
       node is connected to every other; a layer sends m_ij = ReLU(W [h_i; h_j]), averages them over j and
       updates h_i = LayerNorm(h_i + ReLU(U [h_i; mean_j m_ij])). Three layers; the query node is decoded.
  attr-abs, attr-rel: the facet model's granularity on an all-to-all graph (plan models A and B,
       thesis/THESIS_PLAN_symmetry.md). One graph per sample, the queried component's objects only (the facet
       model never mixes components). Attribute channel: one node per (object, attribute) for type, size,
       colour and number (the panel component's object count, on every object), input its thermometer code,
       plus one query node per attribute; position channel, own weights: one node per context panel with the
       component's occupancy, plus a query node. A layer sends m_ij = ReLU(W_i h_i + W_j h_j + E e_ij), with
       edge labels e_ij (attributes: same row, panel, object, attribute, slot; positions: same row), and
       updates as the gnn. A shared head decodes the four attribute query nodes, another the position query.
       attr-rel (B) labels nodes only relatively: complete or question row, column role first / middle /
       last, query flag. It names no row, column index, component, attribute or slot, so rows 1 <-> 2, the
       two components of left-right and up-down, size <-> colour and the middle columns are built in, as in
       the facet model with --columns roles. attr-abs (A) adds row, column, component, attribute and slot
       one-hots, which break all of them.
  H is the width whose weight count is closest to the facet model's (meta "budget").

Regularization, all off by default (the runs before 2026-10-28 used none): --norm puts a LayerNorm after every
hidden layer of the MLP (the GNN has one per layer already); --dropout drops hidden units (MLP: after every
hidden layer; GNN: the encoded nodes and every update); --drop-edge drops GNN messages; --wd is decoupled
weight decay (AdamW) on the weight matrices; --schedule cosine decays the learning rate to 0 after --warmup
epochs; --clip bounds the gradient norm; --augment rows swaps context rows 1 and 2 of half the training
samples. That swap keeps every I-RAVEN rule but breaks Distribute on I-RAVEN-X 3x10 (notes section 22.2),
so it is refused there. --tag names the run directory.

After training, the best checkpoint is scored on the unseen puzzles transformed: rows 1 <-> 2, columns 2 <-> 3
(n >= 4), size <-> colour values (equal ranges), the components of left-right and up-down mirrored
(best.json "transforms"). --eval-only RUN_DIR does this for an existing run. --check-symmetry prints, for
random weights, how far each transformation moves the outputs from what an exactly symmetric model gives.

Data: baselines/prepare_baseline_data.py. Environment: nvsa (Python 3.7, torch 1.11), one GPU. From the
repository root:
    python baselines/train_baseline.py --data iraven --arch gnn --seed 0
"""
import argparse, copy, json, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def therm(v, T):
    """Thermometer codes of integer tensor v: the first v of T bits on (v <= 0: all off)."""
    return (torch.arange(T, device=v.device) < v.unsqueeze(-1)).float()


class Data:
    """One split on the GPU, with the facet model's inputs and targets computed from the raw values."""

    def __init__(self, z, split, meta, dev):
        self.m = meta
        self.vals = torch.as_tensor(z[f"{split}_vals"].astype(np.int64), device=dev)   # [N, P, M, 4]
        self.comp = torch.as_tensor(z[f"{split}_comp"].astype(np.int64), device=dev)   # [N, P, M]
        self.y = z[f"{split}_y"].astype(np.int64)
        self.ncomp = z[f"{split}_ncomp"].astype(np.int64)
        self.cfg = z[f"{split}_cfg"].astype(np.int64)
        self.rules = z[f"{split}_rules"] if f"{split}_rules" in z else None
        self.check_target, self.check_spec = z[f"{split}_check_target"], z[f"{split}_check_spec"]
        # samples: (puzzle, component), as the facet model's per-component queries
        self.units = np.array([(i, k) for i in range(len(self.y)) for k in range(self.ncomp[i])], np.int64)
        slot = torch.full((meta["n_pos"],), -1, dtype=torch.long)
        slot[meta["slots"]] = torch.arange(len(meta["slots"]))
        self.slot = slot.to(dev)

    def target(self, panel_vals, panel_comp, k):
        """The facet code of component k of one panel per row: [B, 4T + n_pos]."""
        T = self.m["therm"]
        sel = (panel_vals[..., 0] >= 0) & (panel_comp == k[:, None])                    # [B, M]
        cnt = sel.sum(1)
        w = sel.float() / cnt.clamp(min=1)[:, None].float()
        codes = [(therm(panel_vals[..., a], T) * w[..., None]).sum(1) for a in (1, 2, 3)]
        occ = torch.zeros(len(k), self.m["n_pos"], device=k.device)
        occ.scatter_add_(1, panel_vals[..., 0].clamp(min=0), sel.float())
        return torch.cat(codes + [therm(cnt, T), (occ > 0).float()], 1)

    def batch_target(self, idx, k):
        rows = torch.as_tensor(idx, device=self.vals.device)
        y = torch.as_tensor(self.y[idx], device=self.vals.device)
        p = self.m["n_ctx"] + y
        return self.target(self.vals[rows, p], self.comp[rows, p], k)

    def candidate_targets(self, idx, k):
        rows = torch.as_tensor(idx, device=self.vals.device)
        return torch.stack([self.target(self.vals[rows, self.m["n_ctx"] + c], self.comp[rows, self.m["n_ctx"] + c], k)
                            for c in range(8)], 1)                                       # [B, 8, D]

    def context(self, idx, swap=None):
        """Values and components of the context panels, [B, C, M, 4] and [B, C, M]; rows 1 and 2 swapped where swap."""
        rows = torch.as_tensor(idx, device=self.vals.device)
        v, c = self.vals[rows, :self.m["n_ctx"]], self.comp[rows, :self.m["n_ctx"]]
        if swap is not None:
            n = self.m["n_cols"]
            perm = torch.arange(self.m["n_ctx"], device=v.device)
            perm[:2 * n] = torch.cat([perm[n:2 * n], perm[:n]])
            v = torch.where(swap[:, None, None, None], v[:, perm], v)
            c = torch.where(swap[:, None, None], c[:, perm], c)
        return v, c

    def mlp_input(self, idx, k, swap=None):
        """[B, n_ctx * slots * (1 + 3T) + 2]: per context panel and slot, occupancy and the attribute codes."""
        T, S = self.m["therm"], len(self.m["slots"])
        v = self.context(idx, swap)[0]                                                  # [B, C, M, 4]
        present = (v[..., 0] >= 0).float()
        feat = torch.cat([present[..., None]] + [therm(v[..., a], T) for a in (1, 2, 3)], -1)
        onehot = F.one_hot(self.slot[v[..., 0].clamp(min=0)], S).float() * present[..., None]
        x = torch.einsum("bcms,bcmf->bcsf", onehot, feat).flatten(1)
        return torch.cat([x, F.one_hot(k, 2).float()], 1)

    def gnn_input(self, idx, k, swap=None):
        """Nodes [B, n_ctx * M + 1, d] (the query node last) and their mask."""
        T, n, S = self.m["therm"], self.m["n_cols"], len(self.m["slots"])
        v, c = self.context(idx, swap)                                                  # [B, C, M, 4], [B, C, M]
        B_, C, M = c.shape
        present = v[..., 0] >= 0
        p = torch.arange(C, device=v.device)[None, :, None].expand(B_, C, M)
        feat = torch.cat([F.one_hot(torch.div(p, n, rounding_mode="floor"), 3), F.one_hot(p % n, n), F.one_hot(self.slot[v[..., 0].clamp(min=0)], S),
                          F.one_hot(c.clamp(min=0), 2)], -1).float()
        feat = torch.cat([feat] + [therm(v[..., a], T) for a in (1, 2, 3)] + [torch.zeros_like(feat[..., :1])], -1)
        feat = feat * present[..., None].float()
        q = torch.zeros(B_, 1, feat.shape[-1], device=v.device)
        q[:, 0, 2], q[:, 0, 3 + n - 1], q[:, 0, -1] = 1, 1, 1                            # row 3, last column, query
        q[:, 0, 3 + n + S:3 + n + S + 2] = F.one_hot(k, 2).float()
        x = torch.cat([feat.reshape(B_, C * M, -1), q], 1)
        mask = torch.cat([present.reshape(B_, C * M), torch.ones(B_, 1, dtype=torch.bool, device=v.device)], 1)
        return x, mask

    def attr_input(self, idx, k, swap=None, absolute=False):
        """The attribute-level graph of component k: attribute nodes [B, C*M*4 + 4, da] (query nodes last,
        type, size, colour, number), their ids [B, ., 5] (row, panel, object, attribute, slot; -1 never
        matches) and mask; position nodes [B, C + 1, dp] (query last), ids [B, ., 1] (row) and mask."""
        T, n, S, P = self.m["therm"], self.m["n_cols"], len(self.m["slots"]), self.m["n_pos"]
        v, c = self.context(idx, swap)                                                  # [B, C, M, 4], [B, C, M]
        B_, C, M = c.shape
        dev = v.device
        sel = (v[..., 0] >= 0) & (c == k[:, None, None])                                # the component's objects
        cnt = sel.sum(2)                                                                # [B, C]
        vals = torch.cat([v[..., 1:], cnt[:, :, None, None].expand(B_, C, M, 1)], -1)  # type, size, colour, number
        p = torch.arange(C, device=dev)
        row, col = torch.div(p, n, rounding_mode="floor"), p % n
        role = (col > 0).long() + (col == n - 1).long()                                 # first 0, middle 1, last 2
        rel_p = torch.cat([F.one_hot((row == 2).long(), 2), F.one_hot(role, 3)], -1).float()        # [C, 5]
        rel_q = torch.tensor([0, 1, 0, 0, 1], dtype=torch.float, device=dev)                         # question row, last
        abs_p = torch.cat([F.one_hot(row, 3), F.one_hot(col, n)], -1).float()                       # [C, 3 + n]
        abs_q = torch.cat([F.one_hot(torch.tensor(2), 3), F.one_hot(torch.tensor(n - 1), n)]).float().to(dev)
        kk = F.one_hot(k, 2).float()                                                   # [B, 2]
        A4 = torch.eye(4, device=dev)

        # attribute channel
        shape = (B_, C, M, 4)
        feats = [therm(vals, T), rel_p[None, :, None, None].expand(*shape, 5), torch.zeros(*shape, 1, device=dev)]
        q = [torch.zeros(B_, 4, T, device=dev), rel_q.expand(B_, 4, 5), torch.ones(B_, 4, 1, device=dev)]
        if absolute:
            slot = F.one_hot(self.slot[v[..., 0].clamp(min=0)], S).float()             # [B, C, M, S]
            feats += [abs_p[None, :, None, None].expand(*shape, 3 + n), kk[:, None, None, None].expand(*shape, 2),
                      A4.expand(*shape, 4), slot[:, :, :, None].expand(*shape, S)]
            q += [abs_q.expand(B_, 4, 3 + n), kk[:, None].expand(B_, 4, 2), A4.expand(B_, 4, 4),
                  torch.zeros(B_, 4, S, device=dev)]
        x = torch.cat(feats, -1) * sel[..., None, None].float()
        xa = torch.cat([x.reshape(B_, C * M * 4, -1), torch.cat(q, -1)], 1)
        a_ids = torch.arange(4, device=dev)
        ids = torch.stack([row[None, :, None, None].expand(*shape), p[None, :, None, None].expand(*shape),
                           (p[:, None] * M + torch.arange(M, device=dev))[None, :, :, None].expand(*shape),
                           a_ids.expand(*shape), v[..., 0][..., None].expand(*shape)], -1)
        full = lambda x: torch.full((4,), x, dtype=torch.long, device=dev)
        qids = torch.stack([full(2), full(-1), full(-1), a_ids, full(-1)], -1)
        ia = torch.cat([ids.reshape(B_, C * M * 4, 5), qids.expand(B_, 4, 5)], 1)
        ma = torch.cat([sel[..., None].expand(*shape).reshape(B_, -1), torch.ones(B_, 4, dtype=torch.bool, device=dev)], 1)

        # position channel
        occ = torch.zeros(B_, C, P, device=dev)
        occ.scatter_add_(2, v[..., 0].clamp(min=0), sel.float())
        feats = [(occ > 0).float(), rel_p.expand(B_, C, 5), torch.zeros(B_, C, 1, device=dev)]
        q = [torch.zeros(B_, 1, P, device=dev), rel_q.expand(B_, 1, 5), torch.ones(B_, 1, 1, device=dev)]
        if absolute:
            feats += [abs_p.expand(B_, C, 3 + n), kk[:, None].expand(B_, C, 2)]
            q += [abs_q.expand(B_, 1, 3 + n), kk[:, None]]
        xp = torch.cat([torch.cat(feats, -1), torch.cat(q, -1)], 1)
        ip = torch.cat([row, torch.tensor([2], device=dev)])[None, :, None].expand(B_, C + 1, 1)
        mp = torch.ones(B_, C + 1, dtype=torch.bool, device=dev)
        return xa, ia, ma, xp, ip, mp


class MLP(nn.Module):
    def __init__(self, d_in, h, d_out, norm=False, dropout=0.0):
        super().__init__()
        layers = []
        for d in (d_in, h, h):
            layers += [nn.Linear(d, h)] + ([nn.LayerNorm(h)] if norm else []) + [nn.ReLU()] + \
                      ([nn.Dropout(dropout)] if dropout else [])
        self.net = nn.Sequential(*layers, nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


class AllToAllGNN(nn.Module):
    def __init__(self, d_in, h, d_out, layers=3, dropout=0.0, drop_edge=0.0):
        super().__init__()
        self.drop, self.drop_edge = nn.Dropout(dropout), drop_edge
        self.enc = nn.Linear(d_in, h)
        self.msg_i = nn.ModuleList([nn.Linear(h, h) for _ in range(layers)])            # W [h_i; h_j] =
        self.msg_j = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(layers)])  # W_i h_i + W_j h_j
        self.upd = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)])
        self.head = nn.Linear(h, d_out)

    def forward(self, x, mask):
        h = self.drop(F.relu(self.enc(x)))
        n = x.shape[1]
        w = (mask[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=x.device)[None]).float()   # [B, i, j]
        if self.training and self.drop_edge:
            w = w * (torch.rand_like(w) >= self.drop_edge).float()
        for mi, mj, up, norm in zip(self.msg_i, self.msg_j, self.upd, self.norm):
            m = F.relu(mi(h)[:, :, None] + mj(h)[:, None, :])                          # [B, i, j, H]
            mbar = (m * w[..., None]).sum(2) / w.sum(2).clamp(min=1)[..., None]
            h = norm(h + self.drop(F.relu(up(torch.cat([h, mbar], -1)))))
        return self.head(h[:, -1])


class LabelledGNN(nn.Module):
    """All-to-all message passing with edge labels: e_ij[f] = 1 where nodes i and j share id f (ids < 0
    never match); m_ij = ReLU(W_i h_i + W_j h_j + E e_ij), averaged over the present j != i;
    h_i = LayerNorm(h_i + ReLU(U [h_i; mean_j m_ij])). Returns every node's final state."""

    def __init__(self, d_in, h, n_labels, layers=3, dropout=0.0, drop_edge=0.0):
        super().__init__()
        self.drop, self.drop_edge = nn.Dropout(dropout), drop_edge
        self.enc = nn.Linear(d_in, h)
        self.msg_i = nn.ModuleList([nn.Linear(h, h) for _ in range(layers)])
        self.msg_j = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(layers)])
        self.edge = nn.ModuleList([nn.Linear(n_labels, h, bias=False) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)])

    def forward(self, x, ids, mask):
        h = self.drop(F.relu(self.enc(x)))
        n = x.shape[1]
        e = ((ids[:, :, None] == ids[:, None]) & (ids[:, :, None] >= 0)).to(x.dtype)        # [B, i, j, labels]
        w = (mask[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=x.device)[None]).to(x.dtype)
        if self.training and self.drop_edge:
            w = w * (torch.rand_like(w) >= self.drop_edge).to(x.dtype)
        for mi, mj, me, up, norm in zip(self.msg_i, self.msg_j, self.edge, self.upd, self.norm):
            m = F.relu(mi(h)[:, :, None] + mj(h)[:, None, :] + me(e))                  # [B, i, j, H]
            mbar = (m * w[..., None]).sum(2) / w.sum(2).clamp(min=1)[..., None]
            h = norm(h + self.drop(F.relu(up(torch.cat([h, mbar], -1)))))
        return h


class AttrGNN(nn.Module):
    """Plan models A and B: an attribute channel and a position channel (LabelledGNN each); one head for
    the four attribute query nodes (type, size, colour, number), one for the position query node."""

    def __init__(self, d_attr, d_pos, h, T, n_pos, dropout=0.0, drop_edge=0.0):
        super().__init__()
        self.attr = LabelledGNN(d_attr, h, 5, 3, dropout, drop_edge)
        self.pos = LabelledGNN(d_pos, h, 1, 3, dropout, drop_edge)
        self.head, self.head_pos = nn.Linear(h, T), nn.Linear(h, n_pos)

    def forward(self, xa, ia, ma, xp, ip, mp):
        ha = self.attr(xa, ia, ma)[:, -4:]
        hp = self.pos(xp, ip, mp)[:, -1]
        return torch.cat([self.head(ha).flatten(1), self.head_pos(hp)], 1)


def n_weights(model):
    return sum(p.numel() for p in model.parameters())


ARCHS = ("mlp", "gnn", "attr-abs", "attr-rel")


def dims(arch, meta):
    """(input width, output width); for attr-*: ((attribute node width, position node width), output width)."""
    T, S, n, P = meta["therm"], len(meta["slots"]), meta["n_cols"], meta["n_pos"]
    d_out = 4 * T + P
    if arch.startswith("attr"):
        extra = arch == "attr-abs"
        return (T + 6 + (3 + n + 2 + 4 + S if extra else 0), P + 6 + (3 + n + 2 if extra else 0)), d_out
    return (meta["n_ctx"] * S * (1 + 3 * T) + 2 if arch == "mlp" else 3 + n + S + 2 + 3 * T + 1), d_out


def count(arch, meta, h, norm=False):
    """Weights of build(arch, meta, h), without building it (an MLP on 1000-bit codes at width 8000 would not fit)."""
    d_in, d_out = dims(arch, meta)
    if arch.startswith("attr"):
        stack = lambda d, labels: d * h + h + 3 * (4 * h * h + labels * h + 4 * h)
        T, P = meta["therm"], meta["n_pos"]
        return stack(d_in[0], 5) + stack(d_in[1], 1) + h * T + T + h * P + P
    middle = 2 * (h * h + h) + (6 * h if norm else 0) if arch == "mlp" else 3 * (4 * h * h + 4 * h)
    return d_in * h + h + middle + h * d_out + d_out


def build(arch, meta, h, norm=False, dropout=0.0, drop_edge=0.0):
    d_in, d_out = dims(arch, meta)
    if arch == "mlp":
        model = MLP(d_in, h, d_out, norm, dropout)
    elif arch == "gnn":
        model = AllToAllGNN(d_in, h, d_out, 3, dropout, drop_edge)
    else:
        model = AttrGNN(d_in[0], d_in[1], h, meta["therm"], meta["n_pos"], dropout, drop_edge)
    assert n_weights(model) == count(arch, meta, h, norm)
    return model


def model_input(arch, data, idx, k, swap=None):
    """The model's arguments for the samples (puzzle idx, component k)."""
    if arch == "mlp":
        return (data.mlp_input(idx, k, swap),)
    if arch == "gnn":
        return data.gnn_input(idx, k, swap)
    return data.attr_input(idx, k, swap, absolute=arch == "attr-abs")


def width_for(arch, meta, budget, norm=False):
    """The width whose weight count is closest to the budget."""
    h = 1
    while count(arch, meta, h + 1, norm) <= budget:
        h += 1
    return min((h, h + 1), key=lambda w: abs(count(arch, meta, w, norm) - budget))


def decode(out, meta):
    """bench_hub_grounding.facet_decode, as one integer vector: occupancy bits, then value counts."""
    T = meta["therm"]
    occ = out[4 * T:] > 0.5
    n = int(occ.sum())
    parts = [occ.astype(np.int64)]
    for i, nv in enumerate(meta["nvals"]):
        gt = np.minimum.accumulate(np.clip(np.rint(out[i * T:(i + 1) * T] * n), 0, n))
        parts.append((np.concatenate([[n], gt[:-1]]) - gt)[:nv].astype(np.int64))
    return np.concatenate(parts)


def spec_parts(meta):
    """Slices of a decoded vector: pos (occupancy, so also the count), type, size, colour."""
    s, at = {"pos": slice(0, meta["n_pos"])}, meta["n_pos"]
    for a, nv in zip(("type", "size", "color"), meta["nvals"]):
        s[a], at = slice(at, at + nv), at + nv
    return s


def check(data, meta):
    """The targets and decoded answers must be bench_hub_grounding's own (first puzzles of the split).
    I-RAVEN-X's Distribute draws values up to maxval INCLUSIVE, one more than the codes and counts hold:
    the code of maxval (all bits on) stays distinct, so matching is unaffected, but bench_hub_grounding's
    decode(describe()) writes that value into the next attribute's counts. Exact generation is therefore
    scored here against the decoded true code, and those answers are left out of this comparison."""
    dev = data.vals.device
    for i in range(len(data.check_target)):
        for k in range(data.ncomp[i]):
            t = data.batch_target(np.array([i]), torch.tensor([k], device=dev))[0].cpu().numpy()
            assert np.allclose(t, data.check_target[i, k], atol=1e-6), f"target differs, puzzle {i} component {k}"
            ans = data.vals[i, meta["n_ctx"] + data.y[i]].cpu().numpy()
            ans = ans[(ans[:, 0] >= 0) & (data.comp[i, meta["n_ctx"] + data.y[i]].cpu().numpy() == k)]
            if (ans[:, 1:] < np.array(meta["nvals"])).all():
                assert (decode(t, meta) == data.check_spec[i, k]).all(), f"decoded answer differs, puzzle {i}"


def run(model, arch, data, units, k_dev, bs=None):
    bs = bs or (64 if arch.startswith("attr") else 256)
    outs = []
    for s in range(0, len(units), bs):
        u = units[s:s + bs]
        k = torch.as_tensor(u[:, 1], device=k_dev)
        outs.append(torch.sigmoid(model(*model_input(arch, data, u[:, 0], k))))
    return torch.cat(outs)


@torch.no_grad()
def evaluate(model, arch, data, meta):
    """Matched, exact (all, per part), per constellation and per rule, as bench_hub_grounding.evaluate."""
    model.eval()
    dev = data.vals.device
    out = run(model, arch, data, data.units, dev)
    k = torch.as_tensor(data.units[:, 1], device=dev)
    cand = torch.cat([data.candidate_targets(data.units[s:s + 512, 0], k[s:s + 512]) for s in range(0, len(k), 512)])
    tgt = torch.cat([data.batch_target(data.units[s:s + 512, 0], k[s:s + 512]) for s in range(0, len(k), 512)])
    mse = float(((out - tgt) ** 2).mean())
    d = ((out[:, None] - cand) ** 2).sum(-1).cpu().numpy()                             # [units, 8]
    out, tgt = out.cpu().numpy(), tgt.cpu().numpy()
    parts = spec_parts(meta)
    n = len(data.y)
    dist = np.zeros((n, 8))
    ok = {p: np.ones(n, bool) for p in parts}
    rule_hits = {}
    for j, (i, kk) in enumerate(data.units):
        dist[i] += d[j]
        got, want = decode(out[j], meta), decode(tgt[j], meta)
        for p, s in parts.items():
            hit = (got[s] == want[s]).all()
            ok[p][i] &= hit
            if data.rules is not None and p != "pos":
                r = meta["rules"][data.rules[i, ("type", "size", "color").index(p)]]
                h, t = rule_hits.get(r, (0, 0))
                rule_hits[r] = (h + int(hit), t + 1)
    best = dist <= dist.min(1, keepdims=True) + 1e-9
    matched = best[np.arange(n), data.y] & (best.sum(1) == 1)
    exact = np.logical_and.reduce(list(ok.values()))
    per_cfg = {}
    for c in np.unique(data.cfg):
        sel = data.cfg == c
        per_cfg[meta["constellations"][c]] = [int(matched[sel].sum()), int(sel.sum())]
    model.train()
    return {"mse": mse, "matched": float(matched.mean()), "exact": float(exact.mean()),
            "exact_parts": {p: float(v.mean()) for p, v in ok.items()}, "per_constellation": per_cfg,
            "exact_per_rule": {r: [h, t] for r, (h, t) in sorted(rule_hits.items())}}


TRANSFORMS = ("rows12", "cols23", "size-colour", "mirror")
MIRRORED = ("left_center_single_right_center_single", "up_center_single_down_center_single")


def transformed(data, name, meta):
    """A copy of data with every puzzle transformed, or None where the transformation does not apply.
    rows12: context rows 1 and 2 exchanged. cols23: columns 2 and 3 exchanged in every row (n >= 4, where both
    are middle columns; the missing cell stays). size-colour: the values of size and colour exchanged in every
    panel (equal value ranges only), and their rules. objects: the objects of every panel in reverse order.
    mirror: left-right and up-down puzzles mirrored, slots 14 <-> 15 and 16 <-> 17, components 0 <-> 1."""
    n = meta["n_cols"]
    d = copy.copy(data)
    perm = torch.arange(data.vals.shape[1], device=data.vals.device)
    if name == "rows12":
        perm[:2 * n] = torch.cat([perm[n:2 * n], perm[:n]])
    elif name == "cols23":
        if n < 4:
            return None
        for r in range(3):
            perm[r * n + 1], perm[r * n + 2] = r * n + 2, r * n + 1
    elif name == "size-colour":
        if meta["nvals"][1] != meta["nvals"][2]:
            return None
        d.vals = data.vals[..., [0, 1, 3, 2]]
        d.rules = None if data.rules is None else data.rules[:, [0, 2, 1]]
    elif name == "objects":
        d.vals, d.comp = data.vals.flip(2), data.comp.flip(2)
    elif name == "mirror":
        lr = [meta["constellations"].index(c) for c in MIRRORED if c in meta["constellations"]]
        if not np.isin(data.cfg, lr).any():
            return None
        on = torch.as_tensor(np.isin(data.cfg, lr), device=data.vals.device)[:, None, None]
        pos = data.vals[..., 0]
        swapped = pos.clone()
        for a, b in ((14, 15), (16, 17)):
            swapped[pos == a], swapped[pos == b] = b, a
        d.vals = data.vals.clone()
        d.vals[..., 0] = torch.where(on, swapped, pos)
        d.comp = torch.where(on & (data.comp >= 0), 1 - data.comp, data.comp)
    if name in ("rows12", "cols23"):
        d.vals, d.comp = data.vals[:, perm], data.comp[:, perm]
    return d


def transform_results(model, arch, data, meta):
    """The unseen puzzles, transformed: matched, exact and the rest of evaluate() per transformation."""
    res = {}
    for t in TRANSFORMS:
        dt = transformed(data, t, meta)
        if dt is not None:
            res[t] = evaluate(model, arch, dt, meta)
            print(f"  {t:12s} unseen matched {100 * res[t]['matched']:5.1f}  exact {100 * res[t]['exact']:5.1f}", flush=True)
    return res


@torch.no_grad()
def symmetry_check(model, arch, data, meta, n_puzzles=32):
    """Max |output difference| between the transformed puzzles and what an exactly symmetric model gives
    (float64, current weights): the same output (rows12, cols23, objects), the size and colour blocks
    exchanged (size-colour), the other component's attribute output (mirror; positions not compared)."""
    model = copy.deepcopy(model).double().eval()
    T = meta["therm"]
    picks = np.concatenate([np.flatnonzero(data.cfg == c)[:max(1, n_puzzles // len(np.unique(data.cfg)))]
                            for c in np.unique(data.cfg)])
    units = data.units[np.isin(data.units[:, 0], picks)]
    where = {(i, k): j for j, (i, k) in enumerate(units)}

    def outs(d):
        res = []
        for s in range(0, len(units), 16):
            u = units[s:s + 16]
            args = model_input(arch, d, u[:, 0], torch.as_tensor(u[:, 1], device=d.vals.device))
            res.append(model(*[a.double() if a.is_floating_point() else a for a in args]))
        return torch.cat(res)

    base = outs(data)
    print(f"symmetry check, {arch}, {len(picks)} selection puzzles, {len(units)} samples, float64")
    for t in ("objects",) + TRANSFORMS:
        dt = transformed(data, t, meta)
        if dt is None:
            print(f"  {t:12s} n/a")
            continue
        o = outs(dt)
        if t == "size-colour":
            want = base.clone()
            want[:, T:2 * T], want[:, 2 * T:3 * T] = base[:, 2 * T:3 * T], base[:, T:2 * T]
            diff = (o - want).abs().max()
        elif t == "mirror":
            on = np.isin(data.cfg[units[:, 0]], [meta["constellations"].index(c) for c in MIRRORED])
            js = [j for j in np.flatnonzero(on)]
            other = [where[(units[j, 0], 1 - units[j, 1])] for j in js]
            if not js:
                print(f"  {t:12s} n/a (no left-right or up-down puzzles)")
                continue
            diff = (o[js, :4 * T] - base[other, :4 * T]).abs().max()
        else:
            diff = (o - base).abs().max()
        print(f"  {t:12s} max |output difference| {float(diff):.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="a dataset name in experiments/baselines/data")
    ap.add_argument("--arch", choices=ARCHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--width", type=int, default=None, help="override the width matched to the facet model")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="weights = scale x the facet model's (the capacity sweep); the run directory gets _x<scale>")
    ap.add_argument("--norm", action="store_true", help="LayerNorm after every hidden layer of the MLP")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--drop-edge", type=float, default=0.0, help="GNN: share of messages dropped in training")
    ap.add_argument("--wd", type=float, default=0.0, help="decoupled weight decay on the weight matrices")
    ap.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    ap.add_argument("--warmup", type=int, default=0, help="epochs of linear warm-up")
    ap.add_argument("--clip", type=float, default=0.0, help="gradient-norm bound (0: none)")
    ap.add_argument("--augment", choices=["none", "rows"], default="none")
    ap.add_argument("--tag", default="", help="added to the run directory's name")
    ap.add_argument("--data-dir", default="experiments/baselines/data")
    ap.add_argument("--out", default="experiments/baselines/runs")
    ap.add_argument("--eval-only", metavar="RUN_DIR", help="score RUN_DIR's model_best.pt on the transformed "
                    "unseen puzzles and add them to its best.json; no training")
    ap.add_argument("--check-symmetry", action="store_true", help="random weights: print the symmetry check, no training")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.eval_only:
        info = json.load(open(os.path.join(args.eval_only, "best.json")))
        a = info["args"]
        z = np.load(os.path.join(args.data_dir, f"{a['data']}.npz"))
        meta = json.loads(str(z["meta"]))
        unseen = Data(z, "unseen", meta, dev)
        model = build(a["arch"], meta, info["width"], a.get("norm", False), a.get("dropout", 0.0)).to(dev)  # dropout layers shift the MLP's indices
        model.load_state_dict(torch.load(os.path.join(args.eval_only, "model_best.pt"), map_location=dev))
        print(f"{args.eval_only}: epoch {info['best']['epoch']}, unseen matched {100 * info['best']['unseen']['matched']:.1f}")
        info["transforms"] = transform_results(model, a["arch"], unseen, meta)
        json.dump(info, open(os.path.join(args.eval_only, "best.json"), "w"), indent=1)
        return
    if not (args.data and args.arch):
        ap.error("--data and --arch are required")
    z = np.load(os.path.join(args.data_dir, f"{args.data}.npz"))
    meta = json.loads(str(z["meta"]))
    sets = {s: Data(z, s, meta, dev) for s in ("train", "sel", "unseen")}
    for d in sets.values():
        check(d, meta)

    if args.augment == "rows" and meta["n_cols"] != 3:
        raise SystemExit("swapping rows 1 and 2 breaks Distribute on I-RAVEN-X 3x10 (notes section 22.2)")
    budget = int(round(args.scale * meta["budget"]))
    h = args.width or width_for(args.arch, meta, budget, args.norm)
    model = build(args.arch, meta, h, args.norm, args.dropout, args.drop_edge).to(dev)
    if args.check_symmetry:
        symmetry_check(model, args.arch, sets["sel"], meta)
        return
    params = list(model.parameters())
    opt = torch.optim.AdamW([{"params": [p for p in params if p.dim() > 1], "weight_decay": args.wd},
                             {"params": [p for p in params if p.dim() <= 1], "weight_decay": 0.0}], lr=args.lr)
    per_epoch = -(-len(sets["train"].units) // args.batch)
    warm, total = args.warmup * per_epoch, args.epochs * per_epoch

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + np.cos(np.pi * (step - warm) / max(1, total - warm))) if args.schedule == "cosine" else 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    name = f"{args.data}_{args.arch}" + (f"_x{args.scale:g}" if args.scale != 1 else "") + \
           (f"_{args.tag}" if args.tag else "") + f"_s{args.seed}"
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    info = {"args": vars(args), "width": h, "weights": n_weights(model), "budget": budget, "facet_weights": meta["budget"],
            "budget_from": meta["budget_from"], "device": str(dev),
            "samples": {s: len(d.units) for s, d in sets.items()}, "puzzles": {s: len(d.y) for s, d in sets.items()}}
    print(json.dumps(info), flush=True)

    train, hist, best = sets["train"], [], None
    for ep in range(1, args.epochs + 1):
        t0, losses = time.time(), []
        units = train.units[np.random.permutation(len(train.units))]
        for s in range(0, len(units), args.batch):
            b = units[s:s + args.batch]
            k = torch.as_tensor(b[:, 1], device=dev)
            swap = torch.rand(len(b), device=dev) < 0.5 if args.augment == "rows" else None
            out = model(*model_input(args.arch, train, b[:, 0], k, swap))
            loss = F.mse_loss(torch.sigmoid(out), train.batch_target(b[:, 0], k))
            opt.zero_grad()
            loss.backward()
            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            losses.append(loss.item())
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "sel": evaluate(model, args.arch, sets["sel"], meta),
               "unseen": evaluate(model, args.arch, sets["unseen"], meta), "seconds": time.time() - t0}
        hist.append(rec)
        print(f"epoch {ep:4d}  loss {rec['train_loss']:.5f}  sel matched {100 * rec['sel']['matched']:5.1f}  "
              f"unseen matched {100 * rec['unseen']['matched']:5.1f} exact {100 * rec['unseen']['exact']:5.1f}  "
              f"({rec['seconds']:.0f}s)", flush=True)
        if best is None or rec["sel"]["matched"] > best["sel"]["matched"]:
            best = rec
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))
        json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"))
        json.dump(dict(info, best=best), open(os.path.join(out_dir, "best.json"), "w"), indent=1)
    print(f"best on selection: epoch {best['epoch']}, unseen matched {100 * best['unseen']['matched']:.1f}%, "
          f"exact {100 * best['unseen']['exact']:.1f}%", flush=True)
    model.load_state_dict(torch.load(os.path.join(out_dir, "model_best.pt"), map_location=dev))
    info["transforms"] = transform_results(model, args.arch, sets["unseen"], meta)
    json.dump(dict(info, best=best), open(os.path.join(out_dir, "best.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
