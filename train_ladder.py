#!/usr/bin/env python3
"""Ablation ladder: shift work from PyTorch to PyNeuraLogic one component at a time.

Every level solves the SAME task on the SAME graph with the SAME hyperparameters.
Only the implementation boundary moves, so a drop between adjacent levels localises
which part of PyNeuraLogic fails to learn.

  LEVEL 1  torch: embeddings, projection, scoring head, loss, optimiser
           pynl : message passing only                        (the 52.9% baseline)

  LEVEL 2  torch: embeddings, projection, loss, optimiser
           pynl : message passing + SCORING HEAD
           -> isolates: can PyNeuraLogic learn the readout/scoring function?

  LEVEL 3  torch: loss, optimiser (no learnable torch params at all)
           pynl : EMBEDDINGS + message passing + head
           -> isolates: can PyNeuraLogic learn the attribute embeddings?
                        (inputs become fixed one-hots; the embedding is a
                         [d, n_values] weight matrix, exactly nn.Embedding)

  LEVEL 4  pynl : everything, including the loss and the optimiser
           -> isolates: is PyNeuraLogic's own training loop the problem?
                        Template is IDENTICAL to level 3.

Levels 1-3 use neuralogic.nn.torch_function.NeuraLogic (one static graph, values
change per forward). Level 4 drops torch entirely and uses model.train().
"""
import argparse, glob, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import neuralogic
from neuralogic.core import (Model, R, V, Settings, Transformation, Aggregation,
                             Metadata, Combination)
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn import Adam, CrossEntropy
from neuralogic.nn.torch_function import NeuraLogic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pynl import parse, N_POS, N_COL, N_SHP, N_SIZ, N_PNL, log

CTX = [f"ctx{i}" for i in range(8)]
CAND = [f"cand{i}" for i in range(8)]
NODES = CTX + CAND
EDGE_TYPES = ["ctx_ctx", "ctx_cand", "cand_ctx"]
ATTRS = [("pos", N_POS), ("col", N_COL), ("shp", N_SHP), ("siz", N_SIZ), ("pnl", N_PNL)]


def static_edges():
    e = [(i, j, "ctx_ctx") for i in CTX for j in CTX if i != j]
    for c in CAND:
        for i in CTX:
            e += [(i, c, "ctx_cand"), (c, i, "cand_ctx")]
    return e


# ── template pieces, shared by every level ──────────────────────────────────
def _message_passing(m, d, layers, act):
    for t in EDGE_TYPES:
        m += R.rel_emb(t)[d,]
    prev = "h0"
    for k in range(1, layers + 1):
        cur = f"h{k}"
        m += (R.get(cur)(V.Dd) <= (R.edge(V.Ss, V.Dd, V.T),
                                   R.get(prev)(V.Ss)[d, d],
                                   R.rel_emb(V.T))) | Metadata(aggregation=Aggregation.AVG)
        m += (R.get(cur)(V.Dd) <= (R.edge(V.Ss, V.Dd, V.T),
                                   R.get(prev)(V.Dd)[d, d])) | Metadata(aggregation=Aggregation.AVG)
        m += (R.get(cur)(V.X) <= R.get(prev)(V.X))                 # identity residual
        m += (R.get(cur) / 1) | Metadata(transformation=Transformation.NORM)
        prev = cur
    return prev


def _pynl_head(m, d, last, act):
    """Scoring head expressed as rules (levels 2-4): per-candidate context pooling,
    a hidden layer, then one scalar. Mirrors the torch head of level 1."""
    m += (R.ctxp <= (R.is_context(V.O), R.get(last)(V.O)[d, d])) | Metadata(aggregation=Aggregation.AVG)
    m += (R.ctxp / 0) | Metadata(transformation=act)
    m += (R.sh(V.C) <= (R.is_candidate(V.C), R.get(last)(V.C)[d, d], R.ctxp[d, d]))
    m += (R.sh / 1) | Metadata(transformation=act)
    m += (R.score(V.C)[1, d] <= R.sh(V.C))
    m += (R.score / 1) | Metadata(transformation=Transformation.IDENTITY)
    return "score"


def build_template(level, d, layers, act):
    m = Model()
    if level <= 2:
        m += (R.h0(V.O) <= R.feat(V.O))                            # torch supplies the vector
        m += (R.h0 / 1) | Metadata(transformation=Transformation.IDENTITY)
    else:
        # embeddings inside pynl: a [d, n_values] matrix over a fixed one-hot
        for name, n in ATTRS:
            m += (R.h0(V.O) <= R.get(f"f_{name}")(V.O)[d, n])
        m += (R.h0 / 1) | Metadata(transformation=act)

    last = _message_passing(m, d, layers, act)

    if level == 1:
        # pynl only concatenates candidate embeddings; torch scores them
        m += (R.out <= (R.is_candidate(V.O), R.get(last)(V.O))) | Metadata(aggregation=Aggregation.CONCAT)
    else:
        score = _pynl_head(m, d, last, act)
        m += (R.out <= R.get(score)(V.C)) | Metadata(aggregation=Aggregation.CONCAT)
    m += (R.out / 0) | Metadata(transformation=Transformation.IDENTITY)
    return m


def onehot_facts(attrs):
    """attrs: (16,5) longs -> fixed one-hot facts, one per attribute per node."""
    facts = []
    for i, n in enumerate(NODES):
        for a, (name, size) in enumerate(ATTRS):
            v = [0.0] * size
            v[int(attrs[i, a])] = 1.0
            facts.append(R.get(f"f_{name}")(n)[v])
    return facts


def structural_facts():
    f = [R.edge(s, t, r) for s, t, r in static_edges()]
    f += [R.is_candidate(c) for c in CAND] + [R.is_context(c) for c in CTX]
    return f


# ── level 1-3 module ────────────────────────────────────────────────────────
class LadderModule(nn.Module):
    def __init__(self, level, embed_dim, hidden_dim, layers, act, nl_lr):
        super().__init__()
        self.level, self.d = level, hidden_dim
        tmpl = build_template(level, hidden_dim, layers, act)

        if level <= 2:
            self.emb = nn.ModuleList([nn.Embedding(n, embed_dim) for _, n in ATTRS])
            self.proj = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                      nn.LayerNorm(hidden_dim), nn.ReLU())
            if level == 1:
                self.head = nn.Sequential(nn.LayerNorm(hidden_dim),
                                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, 1))
            init_facts = [R.feat(n)[[0.0] * hidden_dim] for n in NODES] + structural_facts()
        else:
            zero = torch.zeros(16, 5, dtype=torch.long)
            init_facts = onehot_facts(zero) + structural_facts()

        self.nl = NeuraLogic(tmpl, init_facts, R.out, self._to_logic,
                             settings=Settings(optimizer=Adam(lr=nl_lr)))
        self.register_buffer("perm", self._recover_order(hidden_dim, layers, act))

    def _to_logic(self, x):
        if self.level <= 2:
            return [R.feat(n)[x[i]] for i, n in enumerate(NODES)]
        return [R.get(f"f_{name}")(n)[x[i * len(ATTRS) + a]]
                for i, n in enumerate(NODES) for a, (name, _) in enumerate(ATTRS)]

    def _recover_order(self, d, layers, act):
        """CONCAT emits candidates in the grounder's own order, not cand0..cand7.
        Probe a depth-0 identity companion (order is depth-independent) and invert."""
        p = Model()
        p += (R.h0(V.O) <= R.feat(V.O))
        p += (R.h0 / 1) | Metadata(transformation=Transformation.IDENTITY)
        p += (R.out <= (R.is_candidate(V.O), R.h0(V.O))) | Metadata(aggregation=Aggregation.CONCAT)
        p += (R.out / 0) | Metadata(transformation=Transformation.IDENTITY)
        facts = [R.feat(n)[[0.0] * d] for n in NODES] + structural_facts()
        pn = NeuraLogic(p, facts, R.out, lambda v: [R.feat(n)[v[i]] for i, n in enumerate(NODES)],
                        settings=Settings(optimizer=Adam(lr=1e-4)))
        mark = torch.zeros(len(NODES), d)
        for i, n in enumerate(NODES):
            if n in CAND:
                mark[i] = float(CAND.index(n) + 1)
        with torch.no_grad():
            em = pn(mark).reshape(8, d)[:, 0].tolist()
        base = min(em)
        order = [int(round(v - base)) for v in em]
        slot = {c: j for j, c in enumerate(order)}
        if sorted(slot) != list(range(8)):
            raise RuntimeError(f"cannot recover CONCAT order: {order}")
        return torch.tensor([slot[k] for k in range(8)], dtype=torch.long)

    def forward(self, attrs):
        if self.level <= 2:
            e = sum(self.emb[a](attrs[:, a]) for a in range(len(ATTRS)))
            out = self.nl(self.proj(e))
            if self.level == 1:                       # torch scores the embeddings
                return self.head(out.reshape(8, self.d)[self.perm]).squeeze(-1)
            return out.reshape(8)[self.perm]          # pynl already produced 8 scalars
        # level 3: fixed one-hots. requires_grad=True so autograd records the
        # NeuraLogic node -- the gradient w.r.t. the constant input is discarded,
        # but without it backward() never reaches the template weights.
        oh = []
        for i in range(len(NODES)):
            for a, (_, size) in enumerate(ATTRS):
                v = torch.zeros(size, requires_grad=True)
                with torch.no_grad():
                    v[int(attrs[i, a])] = 1.0
                oh.append(v)
        return self.nl(oh).reshape(8)[self.perm]


# ── data ────────────────────────────────────────────────────────────────────
def puzzle_to_tensor(panels):
    rows = []
    for p in range(16):
        o = panels[p][0]
        rows.append([o["pos"], o["color"], o["type"], o["size"], p if p < 8 else 8])
    return torch.tensor(rows, dtype=torch.long)


def load(root, cfg, split, n):
    X, Y = [], []
    for f in sorted(glob.glob(os.path.join(root, cfg, f"*_{split}.npz")))[:n]:
        panels = parse(f.replace(".npz", ".xml"), cfg)
        if any(len(p) != 1 for p in panels):
            continue
        X.append(puzzle_to_tensor(panels)); Y.append(int(np.load(f)["target"]))
    return X, torch.tensor(Y, dtype=torch.long)


def run_level4(args, Xtr, Ytr, Xva, Yva, act, save_dir):
    """Pure PyNeuraLogic: same template as level 3, but NL's own loss and trainer."""
    tmpl = build_template(3, args.hidden_dim, args.layers, act)
    model = tmpl.build(Settings(optimizer=Adam(lr=args.nl_lr),
                                error_function=CrossEntropy(with_logits=True)))

    def to_sample(attrs, y, gid):
        oh = [0.0] * 8; oh[int(y)] = 1.0
        return Sample(R.out[oh], onehot_facts(attrs) + structural_facts())

    tr_s = [to_sample(x, y, f"tr{i}") for i, (x, y) in enumerate(zip(Xtr, Ytr))]
    va_s = [to_sample(x, y, f"va{i}") for i, (x, y) in enumerate(zip(Xva, Yva))]
    t0 = time.time()
    built = list(model.build_dataset(Dataset(tr_s + va_s), batch_size=args.batch_size))
    tr, va = built[:len(tr_s)], built[len(tr_s):]
    log(f"grounded in {time.time()-t0:.0f}s (batch_size={args.batch_size})")

    probe = np.asarray(model.test([va[0]])[0]).flatten()
    log(f"sanity: query output shape {probe.shape} (must be (8,))")

    best, hist = 0.0, []
    log(f"{'epoch':>5} {'loss':>10} {'acc':>8} {'best':>8}   (chance 12.5%)")
    for ep in range(1, args.epochs + 1):
        res = model.train(tr, epochs=1)
        loss = float(np.mean([float(np.asarray(r[2]).flatten()[0]) for r in res])) if res else float("nan")
        outs = [np.asarray(y).flatten() for y in model.test(va)]
        acc = sum(int(np.argmax(o)) == int(t) for o, t in zip(outs, Yva)) / len(Yva)
        best = max(best, acc); hist.append({"epoch": ep, "loss": loss, "val_acc": acc})
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)
        log(f"{ep:>5} {loss:>10.4f} {acc*100:>7.1f}% {best*100:>7.1f}%")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--dataset-root", default="data/I-RAVEN-10000")
    ap.add_argument("--constellation", default="center_single")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nl-lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--jvm-gb", type=int, default=60)
    ap.add_argument("--save-dir", default="experiments/ladder")
    args = ap.parse_args()

    neuralogic.set_max_memory_size(args.jvm_gb)
    torch.manual_seed(0)
    save_dir = f"{args.save_dir}_L{args.level}"
    os.makedirs(save_dir, exist_ok=True)
    log(f"LEVEL {args.level}")
    log(f"config: {vars(args)}")

    if args.overfit:
        Xtr, Ytr = load(args.dataset_root, args.constellation, "train", args.overfit)
        Xva, Yva = Xtr, Ytr
        log(f"OVERFIT MODE: {len(Xtr)} puzzles, TRAIN accuracy")
    else:
        Xtr, Ytr = load(args.dataset_root, args.constellation, "train", args.n_train)
        Xva, Yva = load(args.dataset_root, args.constellation, "val", args.n_val)
    log(f"{len(Xtr)} train / {len(Xva)} val puzzles")

    act = Transformation.LEAKY_RELU
    if args.level == 4:
        best = run_level4(args, Xtr, Ytr, Xva, Yva, act, save_dir)
        log(f"DONE. level 4 best {best*100:.2f}%")
        return

    model = LadderModule(args.level, args.embed_dim, args.hidden_dim,
                         args.layers, act, args.nl_lr)
    tp = [p for n, p in model.named_parameters() if not n.startswith("nl.")]
    opt = torch.optim.AdamW(tp, lr=args.lr, weight_decay=1e-4) if tp else None
    log(f"torch learnable params: {sum(p.numel() for p in tp):,}"
        f"{' (none - all learning is in PyNeuraLogic)' if not tp else ''}")

    with torch.no_grad():
        probe = model(Xtr[0])
    log(f"sanity: forward returns {tuple(probe.shape)} (must be (8,))")
    assert probe.shape == (8,), f"expected 8 logits, got {tuple(probe.shape)}"

    best, hist = 0.0, []
    log(f"{'epoch':>5} {'loss':>10} {'acc':>8} {'best':>8}   (chance 12.5%)")
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        if opt: opt.zero_grad()
        for i, idx in enumerate(torch.randperm(len(Xtr))):
            logits = model(Xtr[idx])
            loss = F.cross_entropy(logits.unsqueeze(0), Ytr[idx].unsqueeze(0))
            (loss / args.batch_size).backward()
            tot += float(loss.detach())
            if opt and (i + 1) % args.batch_size == 0:
                opt.step(); opt.zero_grad()
        if opt: opt.step(); opt.zero_grad()

        model.eval(); hits = 0
        with torch.no_grad():
            for x, y in zip(Xva, Yva):
                hits += int(model(x).argmax().item()) == int(y)
        acc = hits / len(Xva); best = max(best, acc)
        hist.append({"epoch": ep, "loss": tot / len(Xtr), "val_acc": acc})
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)
        if acc >= best:
            with open(os.path.join(save_dir, "best.json"), "w") as f:
                json.dump({"epoch": ep, "val_acc": acc, "level": args.level,
                           "args": vars(args)}, f, indent=2)
        log(f"{ep:>5} {tot/len(Xtr):>10.4f} {acc*100:>7.1f}% {best*100:>7.1f}%")
    log(f"DONE. level {args.level} best {best*100:.2f}%")


if __name__ == "__main__":
    main()
