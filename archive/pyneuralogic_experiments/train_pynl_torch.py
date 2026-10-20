#!/usr/bin/env python3
"""Hybrid: PyNeuraLogic for relational message passing, PyTorch for everything numeric.

Split of responsibilities
  PyNeuraLogic : the logic graph and message passing over typed edges
  PyTorch      : embedding tables, LayerNorm, the scoring head, 8-way softmax
                 cross-entropy, AdamW, gradient accumulation

All 8 context panels and all 8 candidates live in ONE graph and are scored in ONE
forward pass. Candidates are connected to the context but NOT to each other, so
each candidate is still scored independently -- this preserves the property that
made per-candidate graphs safe on I-RAVEN (unified_graph leaked the answer: +15 on
RAVEN, -20 on I-RAVEN).

Constraint from neuralogic.nn.torch_function.NeuraLogic: the sample is built ONCE
at construction and only fact VALUES change per forward. That is fine for
center_single, where every puzzle has exactly 16 objects with identical topology;
constellations with varying object counts would need one module per structure.
"""
import argparse, glob, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import neuralogic
from neuralogic.core import (Model, R, V, Settings, Transformation, Aggregation,
                             Metadata, Combination)
from neuralogic.nn import Adam
from neuralogic.nn.torch_function import NeuraLogic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pynl import parse, POS_MAP, N_POS, N_COL, N_SHP, N_SIZ, N_PNL, log

CTX = [f"ctx{i}" for i in range(8)]
CAND = [f"cand{i}" for i in range(8)]
NODES = CTX + CAND                       # fixed 16-node topology for center_single
EDGE_TYPES = ["ctx_ctx", "ctx_cand", "cand_ctx"]


def static_edges():
    """Context fully connected; every candidate wired to the context both ways.
    Deliberately NO candidate-candidate edges."""
    e = []
    for i in CTX:
        for j in CTX:
            if i != j:
                e.append((i, j, "ctx_ctx"))
    for c in CAND:
        for i in CTX:
            e.append((i, c, "ctx_cand"))
            e.append((c, i, "cand_ctx"))
    return e


def build_template(d, layers):
    m = Model()
    for t in EDGE_TYPES:
        m += R.rel_emb(t)[d,]

    # h0 comes straight from the torch-supplied feature fact
    m += (R.h0(V.O) <= R.feat(V.O))
    m += (R.h0 / 1) | Metadata(transformation=Transformation.IDENTITY)

    prev = "h0"
    for k in range(1, layers + 1):
        cur = f"h{k}"
        m += (R.get(cur)(V.Dd) <= (R.edge(V.Ss, V.Dd, V.T),
                                   R.get(prev)(V.Ss)[d, d],
                                   R.rel_emb(V.T))) | Metadata(aggregation=Aggregation.AVG)
        m += (R.get(cur)(V.Dd) <= (R.edge(V.Ss, V.Dd, V.T),
                                   R.get(prev)(V.Dd)[d, d])) | Metadata(aggregation=Aggregation.AVG)
        m += (R.get(cur)(V.X) <= R.get(prev)(V.X))            # identity residual
        m += (R.get(cur) / 1) | Metadata(transformation=Transformation.NORM)
        prev = cur

    # concatenate the 8 candidate embeddings into one 8*d vector for the torch head
    m += (R.out <= (R.is_candidate(V.O), R.get(prev)(V.O))) | Metadata(aggregation=Aggregation.CONCAT)
    m += (R.out / 0) | Metadata(transformation=Transformation.IDENTITY)
    return m


class RavenHybrid(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64, layers=2, nl_lr=1e-4):
        super().__init__()
        self.d = hidden_dim
        self.emb_pos = nn.Embedding(N_POS, embed_dim)
        self.emb_col = nn.Embedding(N_COL, embed_dim)
        self.emb_shp = nn.Embedding(N_SHP, embed_dim)
        self.emb_siz = nn.Embedding(N_SIZ, embed_dim)
        self.emb_pnl = nn.Embedding(N_PNL, embed_dim)
        self.proj = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                  nn.LayerNorm(hidden_dim), nn.ReLU())

        edges = static_edges()
        input_facts = [R.feat(n)[[0.0] * hidden_dim] for n in NODES]
        input_facts += [R.edge(s, t, r) for s, t, r in edges]
        input_facts += [R.is_candidate(c) for c in CAND]
        input_facts += [R.is_context(c) for c in CTX]

        self.nl = NeuraLogic(
            template=build_template(hidden_dim, layers),
            input_facts=input_facts,
            output_relation=R.out,
            to_logic=self._to_logic,
            settings=Settings(optimizer=Adam(lr=nl_lr)),
        )
        # torch scoring head over each candidate embedding -> one logit
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("perm", self._recover_concat_order())

    def _recover_concat_order(self):
        """Aggregation.CONCAT does NOT emit candidates in cand0..cand7 order -- it
        follows the grounder's own ordering, a fixed but arbitrary permutation.
        Without correcting for it the head scores candidate i against candidate j's
        target and the model cannot learn.

        The real template can't be probed directly (NORM erases the marker values),
        but the order depends only on the constants, not on template depth -- probed
        and confirmed identical for layers=0,1,2. So build a depth-0, identity-only
        companion over the same facts and read the order off that."""
        d = self.d
        p = Model()
        for t in EDGE_TYPES:
            p += R.rel_emb(t)[d,]
        p += (R.h0(V.O) <= R.feat(V.O))
        p += (R.h0 / 1) | Metadata(transformation=Transformation.IDENTITY)
        p += (R.out <= (R.is_candidate(V.O), R.h0(V.O))) | Metadata(aggregation=Aggregation.CONCAT)
        p += (R.out / 0) | Metadata(transformation=Transformation.IDENTITY)

        facts = [R.feat(n)[[0.0] * d] for n in NODES]
        facts += [R.edge(s, t, r) for s, t, r in static_edges()]
        facts += [R.is_candidate(c) for c in CAND] + [R.is_context(c) for c in CTX]
        probe_nl = NeuraLogic(p, facts, R.out,
                              lambda v: [R.feat(n)[v[i]] for i, n in enumerate(NODES)],
                              settings=Settings(optimizer=Adam(lr=1e-4)))

        marker = torch.zeros(len(NODES), d)
        for i, n in enumerate(NODES):
            if n in CAND:
                marker[i] = float(CAND.index(n) + 1)
        with torch.no_grad():
            emitted = probe_nl(marker).reshape(8, d)[:, 0].tolist()
        # is_candidate's scalar value shifts every entry equally; ranks are unaffected
        base = min(emitted)
        order = [int(round(v - base)) for v in emitted]
        slot_of_cand = {c: j for j, c in enumerate(order)}          # candidate k -> row j
        if sorted(slot_of_cand) != list(range(8)):
            raise RuntimeError(f"could not recover CONCAT order from probe: {order}")
        return torch.tensor([slot_of_cand[k] for k in range(8)], dtype=torch.long)

    def _to_logic(self, node_vecs):
        """node_vecs: (16, hidden) -> the facts whose values change each forward."""
        return [R.feat(n)[node_vecs[i]] for i, n in enumerate(NODES)]

    def encode(self, attrs):
        """attrs: LongTensor (16, 5) = pos, color, shape, size, panel."""
        e = (self.emb_pos(attrs[:, 0]) + self.emb_col(attrs[:, 1]) + self.emb_shp(attrs[:, 2])
             + self.emb_siz(attrs[:, 3]) + self.emb_pnl(attrs[:, 4]))
        return self.proj(e)

    def forward(self, attrs):
        node_vecs = self.encode(attrs)                 # (16, d)  torch
        out = self.nl(node_vecs)                       # (8*d,)   PyNeuraLogic
        cand = out.reshape(8, self.d)[self.perm]       # reorder to cand0..cand7
        return self.head(cand).squeeze(-1)             # (8,) logits


# ── data ────────────────────────────────────────────────────────────────────
def puzzle_to_tensor(panels):
    """16 panels -> (16, 5) attrs, ordered ctx0..ctx7 then cand0..cand7."""
    rows = []
    for p in list(range(8)) + list(range(8, 16)):
        o = panels[p][0]                                # center_single: one object per panel
        grid = p if p < 8 else 8
        rows.append([o["pos"], o["color"], o["type"], o["size"], grid])
    return torch.tensor(rows, dtype=torch.long)


def load(root, cfg, split, n):
    X, Y = [], []
    for f in sorted(glob.glob(os.path.join(root, cfg, f"*_{split}.npz")))[:n]:
        panels = parse(f.replace(".npz", ".xml"), cfg)
        if any(len(p) != 1 for p in panels):
            continue                                    # static topology assumption
        X.append(puzzle_to_tensor(panels))
        Y.append(int(np.load(f)["target"]))
    return X, torch.tensor(Y, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/I-RAVEN-10000")
    ap.add_argument("--constellation", default="center_single")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nl-lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--jvm-gb", type=int, default=40)
    ap.add_argument("--save-dir", default="experiments/pynl_hybrid")
    args = ap.parse_args()

    neuralogic.set_max_memory_size(args.jvm_gb)
    torch.manual_seed(0)
    os.makedirs(args.save_dir, exist_ok=True)
    log(f"config: {vars(args)}")

    if args.overfit:
        Xtr, Ytr = load(args.dataset_root, args.constellation, "train", args.overfit)
        Xva, Yva = Xtr, Ytr
        log(f"OVERFIT MODE: {len(Xtr)} puzzles, reporting TRAIN accuracy")
    else:
        Xtr, Ytr = load(args.dataset_root, args.constellation, "train", args.n_train)
        Xva, Yva = load(args.dataset_root, args.constellation, "val", args.n_val)
    log(f"{len(Xtr)} train / {len(Xva)} val puzzles")

    model = RavenHybrid(args.embed_dim, args.hidden_dim, args.layers, args.nl_lr)
    torch_params = [p for n, p in model.named_parameters() if not n.startswith("nl.")]
    opt = torch.optim.AdamW(torch_params, lr=args.lr, weight_decay=1e-4)
    log(f"torch params: {sum(p.numel() for p in torch_params):,} "
        f"(+ PyNeuraLogic template weights, updated inside backward)")

    with torch.no_grad():
        probe = model(Xtr[0])
    log(f"sanity: forward returns {tuple(probe.shape)} (must be (8,))")
    assert probe.shape == (8,), f"expected 8 logits, got {tuple(probe.shape)}"

    log(f"{'epoch':>5} {'loss':>10} {'acc':>8} {'best':>8}   (chance 12.5%)")
    best, hist = 0.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot, opt_steps = 0.0, 0
        opt.zero_grad()
        for i, idx in enumerate(perm):
            logits = model(Xtr[idx])
            loss = F.cross_entropy(logits.unsqueeze(0), Ytr[idx].unsqueeze(0))
            (loss / args.batch_size).backward()
            tot += float(loss)
            if (i + 1) % args.batch_size == 0:
                opt.step(); opt.zero_grad(); opt_steps += 1
        if opt_steps == 0 or (len(perm) % args.batch_size):
            opt.step(); opt.zero_grad()

        model.eval()
        hits = 0
        with torch.no_grad():
            for x, y in zip(Xva, Yva):
                hits += int(model(x).argmax().item()) == int(y)
        acc = hits / len(Xva)
        hist.append({"epoch": ep, "loss": tot / len(Xtr), "val_acc": acc})
        if acc > best:
            best = acc
            torch.save({"epoch": ep, "val_acc": acc, "args": vars(args),
                        "torch_state": model.state_dict()},
                       os.path.join(args.save_dir, "model_best.pt"))
            model.nl.model.save(os.path.join(args.save_dir, "template_best.bin"))
            with open(os.path.join(args.save_dir, "best.json"), "w") as f:
                json.dump({"epoch": ep, "val_acc": acc, "args": vars(args)}, f, indent=2)
        with open(os.path.join(args.save_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)
        log(f"{ep:>5} {tot/len(Xtr):>10.4f} {acc*100:>7.1f}% {best*100:>7.1f}%")
        if args.overfit and acc == 1.0:
            log(f"REACHED 100% TRAIN ACCURACY at epoch {ep}"); break
    log(f"DONE. best {best*100:.2f}%")


if __name__ == "__main__":
    main()
