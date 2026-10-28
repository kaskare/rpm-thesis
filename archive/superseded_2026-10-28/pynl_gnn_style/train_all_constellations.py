#!/usr/bin/env python3
"""ONE model trained jointly on ALL SEVEN I-RAVEN constellations.

Design notes
------------
The NeuraLogic torch wrapper needs a STATIC computation graph, but object counts
vary across and within constellations (distribute_nine holds 1-9 per panel).

An earlier attempt used one module per constellation sharing a single template.
Forward and weight reads share correctly (`A.model is B.model`), but BACKWARD
throws `Incompatible dimensions ... matrix increment by vector` on the second
topology -- the trainer state is bound to whichever topology was grounded first.
So that approach is unusable.

Instead: pad EVERY constellation to a common 9-slot topology (9 = max objects per
panel, from distribute_nine). 16 panels x 9 slots = 144 nodes, 2808 edges.
Measured 484 ms/puzzle at hidden=128, 3 layers -- 8.8x the center_single-only
cost, which is affordable.

SLOT SEMANTICS: slot k of a panel is the k-th POSITION in that constellation's
sorted universal-position list -- NOT the k-th object present. A puzzle is
entirely one constellation, so slot k denotes the same universal position in all
16 panels, and `ctx_ctx`/`ctx_cand` same-slot edges therefore link objects at the
SAME position across panels (the edge that carries the row/column rule). Absent
objects get an inactive slot and a learned pad embedding.

  --level 1 : torch scoring head  (pynl does message passing only)
  --level 2 : pynl scoring head   (best of the ablation: 39.0% vs 32.6%)
"""
import argparse, glob, json, os, sys, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import neuralogic
from neuralogic.core import (Model, R, V, Settings, Transformation, Aggregation,
                             Metadata)
from neuralogic.nn import Adam
from neuralogic.nn.torch_function import NeuraLogic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pynl import parse, POS_MAP, N_POS, N_COL, N_SHP, N_SIZ, N_PNL, log

CONSTELLATIONS = list(POS_MAP)
POS_LIST = {c: sorted(m.values()) for c, m in POS_MAP.items()}
SLOTS = max(len(v) for v in POS_LIST.values())          # 9, from distribute_nine
ATTRS = [("pos", N_POS), ("col", N_COL), ("shp", N_SHP), ("siz", N_SIZ), ("pnl", N_PNL)]
EDGE_TYPES = ["same_panel", "ctx_ctx", "ctx_cand", "cand_ctx"]
# Ablation switches (set from the command line in main(); defaults reproduce the best run exactly).
EDGE_LABELS = "basic"   # "rowcol": tag cross-panel edges as same-row / same-column (_row / _col)
ABSENT_POS = False      # empty slots carry their position + panel embedding, not only `absent`


def edge_types():
    if EDGE_LABELS != "rowcol":
        return EDGE_TYPES
    return ["same_panel"] + [b + t for b in ("ctx_ctx", "ctx_cand", "cand_ctx") for t in ("", "_row", "_col")]


def _grid(base, i, j):
    """rowcol mode: label an edge between panels i and j; every candidate sits in grid cell 8."""
    if EDGE_LABELS != "rowcol":
        return base
    i, j = min(i, 8), min(j, 8)
    if i // 3 == j // 3:
        return base + "_row"
    if i % 3 == j % 3:
        return base + "_col"
    return base

NODES = [f"p{p}_s{s}" for p in range(16) for s in range(SLOTS)]


def edges(same_panel=True):
    e = []
    if same_panel:                                       # within panel, all pairs
        for p in range(16):
            for a in range(SLOTS):
                for b in range(SLOTS):
                    if a != b:
                        e.append((f"p{p}_s{a}", f"p{p}_s{b}", "same_panel"))
    for s in range(SLOTS):                               # same slot across panels
        for i in range(8):
            for j in range(8):
                if i != j:
                    e.append((f"p{i}_s{s}", f"p{j}_s{s}", _grid("ctx_ctx", i, j)))
            for c in range(8, 16):                       # candidates never see each other
                e.append((f"p{i}_s{s}", f"p{c}_s{s}", _grid("ctx_cand", i, c)))
                e.append((f"p{c}_s{s}", f"p{i}_s{s}", _grid("cand_ctx", c, i)))
    return e


def structural_facts(same_panel=True):
    f = [R.edge(s, t, r) for s, t, r in edges(same_panel)]
    f += [R.is_context(f"p{p}_s{s}") for p in range(8) for s in range(SLOTS)]
    f += [R.in_cand(f"p{8+c}_s{s}", f"c{c}") for c in range(8) for s in range(SLOTS)]
    return f


def build_template(level, d, layers, act, agg=Aggregation.AVG, per_type=False):
    """per_type: one message matrix per edge type, each type averaged separately (then summed),
    instead of one shared matrix with an additive rel_emb and one average over all edges."""
    m = Model()
    for t in edge_types():
        m += R.rel_emb(t)[d,]
    m += (R.h0(V.O) <= R.feat(V.O))
    m += (R.h0 / 1) | Metadata(transformation=Transformation.IDENTITY)

    prev = "h0"
    for k in range(1, layers + 1):
        cur = f"h{k}"
        if per_type:
            for t in edge_types():
                m += (R.get(cur)(V.Dd) <= (R.edge(V.Ss, V.Dd, t),
                                           R.get(prev)(V.Ss)[d, d],
                                           R.rel_emb(t))) | Metadata(aggregation=agg)
        else:
            m += (R.get(cur)(V.Dd) <= (R.edge(V.Ss, V.Dd, V.T),
                                       R.get(prev)(V.Ss)[d, d],
                                       R.rel_emb(V.T))) | Metadata(aggregation=agg)
        m += (R.get(cur)(V.Dd) <= (R.edge(V.Ss, V.Dd, V.T),
                                   R.get(prev)(V.Dd)[d, d])) | Metadata(aggregation=agg)
        m += (R.get(cur)(V.X) <= R.get(prev)(V.X))             # identity residual
        m += (R.get(cur) / 1) | Metadata(transformation=Transformation.NORM)
        prev = cur

    # pool each candidate PANEL over its slots
    m += (R.candp(V.C) <= (R.in_cand(V.O, V.C), R.get(prev)(V.O)[d, d])) \
         | Metadata(aggregation=Aggregation.AVG)
    m += (R.candp / 1) | Metadata(transformation=act)

    if level == 1:
        m += (R.out <= R.candp(V.C)) | Metadata(aggregation=Aggregation.CONCAT)
    else:
        m += (R.ctxp <= (R.is_context(V.O), R.get(prev)(V.O)[d, d])) \
             | Metadata(aggregation=Aggregation.AVG)
        m += (R.ctxp / 0) | Metadata(transformation=act)
        m += (R.sh(V.C) <= (R.candp(V.C)[d, d], R.ctxp[d, d]))
        m += (R.sh / 1) | Metadata(transformation=act)
        m += (R.score(V.C)[1, d] <= R.sh(V.C))
        m += (R.score / 1) | Metadata(transformation=Transformation.IDENTITY)
        m += (R.out <= R.score(V.C)) | Metadata(aggregation=Aggregation.CONCAT)
    m += (R.out / 0) | Metadata(transformation=Transformation.IDENTITY)
    return m


class AllConstellations(nn.Module):
    def __init__(self, level, embed_dim, hidden_dim, layers, act, nl_lr,
                 agg=Aggregation.AVG, same_panel=True, layout="off", per_type=False, absent_pos=False):
        super().__init__()
        self.level, self.d, self.layout, self.absent_pos = level, hidden_dim, layout, absent_pos
        # with absent_pos, position id N_POS marks slots beyond the constellation's layout
        self.emb = nn.ModuleList([nn.Embedding(n + int(absent_pos and k == 0), embed_dim)
                                  for k, (_, n) in enumerate(ATTRS)])
        # Panel-level LAYOUT features. RAVEN's Number and Position rules are
        # properties of a whole panel (how many objects, which slots), which a node
        # cannot perceive from same-slot cross-panel edges alone. Every node in a
        # panel -- real or pad -- receives its panel's layout.
        if layout in ("count", "count_occ"):
            self.emb_count = nn.Embedding(SLOTS + 1, embed_dim)    # 0..9 objects
        if layout == "count_occ":
            self.occ_proj = nn.Linear(SLOTS, embed_dim)            # 9-bit occupancy
        self.absent = nn.Parameter(torch.zeros(embed_dim))          # learned pad token
        self.proj = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                  nn.LayerNorm(hidden_dim), nn.ReLU())
        if level == 1:
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))
        facts = [R.feat(n)[[0.0] * hidden_dim] for n in NODES] + structural_facts(same_panel)
        self.nl = NeuraLogic(build_template(level, hidden_dim, layers, act, agg, per_type),
                             facts, R.out,
                             lambda v: [R.feat(n)[v[i]] for i, n in enumerate(NODES)],
                             settings=Settings(optimizer=Adam(lr=nl_lr)))
        self.register_buffer("perm", self._recover_order(hidden_dim, same_panel))

    def _recover_order(self, d, same_panel=True):
        p = Model()
        p += (R.h0(V.O) <= R.feat(V.O))
        p += (R.h0 / 1) | Metadata(transformation=Transformation.IDENTITY)
        p += (R.cp(V.C) <= (R.in_cand(V.O, V.C), R.h0(V.O))) | Metadata(aggregation=Aggregation.AVG)
        p += (R.cp / 1) | Metadata(transformation=Transformation.IDENTITY)
        p += (R.out <= R.cp(V.C)) | Metadata(aggregation=Aggregation.CONCAT)
        p += (R.out / 0) | Metadata(transformation=Transformation.IDENTITY)
        facts = [R.feat(n)[[0.0] * d] for n in NODES] + structural_facts(same_panel)
        pn = NeuraLogic(p, facts, R.out,
                        lambda v: [R.feat(n)[v[i]] for i, n in enumerate(NODES)],
                        settings=Settings(optimizer=Adam(lr=1e-4)))
        mark = torch.zeros(len(NODES), d)
        for i, n in enumerate(NODES):
            panel = int(n.split("_")[0][1:])
            if panel >= 8:
                mark[i] = float(panel - 8 + 1)
        with torch.no_grad():
            em = pn(mark).reshape(8, d)[:, 0].tolist()
        base = min(em)
        order = [int(round(v - base)) for v in em]
        slot = {c: j for j, c in enumerate(order)}
        if sorted(slot) != list(range(8)):
            raise RuntimeError(f"cannot recover CONCAT order: {order}")
        return torch.tensor([slot[k] for k in range(8)], dtype=torch.long)

    def forward(self, attrs, active):
        e = sum(self.emb[a](attrs[:, a]) for a in range(len(ATTRS)))
        filler = self.absent.expand_as(e)
        if self.absent_pos:                                        # "empty, at this position, in this panel"
            beyond = (attrs[:, 0] == N_POS).unsqueeze(-1)
            filler = torch.where(beyond, filler, self.absent + self.emb[0](attrs[:, 0]) + self.emb[4](attrs[:, 4]))
        e = torch.where(active.unsqueeze(-1), e, filler)
        if self.layout != "off":
            occ = active.view(16, SLOTS)                           # (panels, slots)
            lay = self.emb_count(occ.sum(dim=1))                   # object count
            if self.layout == "count_occ":
                lay = lay + self.occ_proj(occ.float())             # which slots
            e = e + lay.repeat_interleave(SLOTS, dim=0)            # broadcast to nodes
        out = self.nl(self.proj(e))
        if self.level == 1:
            return self.head(out.reshape(8, self.d)[self.perm]).squeeze(-1)
        return out.reshape(8)[self.perm]


# ── data ────────────────────────────────────────────────────────────────────
def encode_puzzle(panels, cfg):
    """Slot k == the k-th universal position of this constellation, in EVERY panel."""
    idx = {pos: k for k, pos in enumerate(POS_LIST[cfg])}
    attrs = torch.zeros(16 * SLOTS, 5, dtype=torch.long)
    active = torch.zeros(16 * SLOTS, dtype=torch.bool)
    if ABSENT_POS:                       # empty slots keep position + panel ids; slots beyond the layout get N_POS
        for p in range(16):
            for k in range(SLOTS):
                attrs[p * SLOTS + k, 0] = POS_LIST[cfg][k] if k < len(POS_LIST[cfg]) else N_POS
                attrs[p * SLOTS + k, 4] = p if p < 8 else 8
    for p in range(16):
        for o in panels[p]:
            k = idx[o["pos"]]
            i = p * SLOTS + k
            attrs[i] = torch.tensor([o["pos"], o["color"], o["type"], o["size"],
                                     p if p < 8 else 8])
            active[i] = True
    return attrs, active


def load(root, cfgs, split, n_per):
    data = []
    for cfg in cfgs:
        for f in sorted(glob.glob(os.path.join(root, cfg, f"*_{split}.npz")))[:n_per]:
            panels = parse(f.replace(".npz", ".xml"), cfg)
            a, m = encode_puzzle(panels, cfg)
            data.append((cfg, a, m, int(np.load(f)["target"])))
    return data


def evaluate(model, data):
    per = defaultdict(lambda: [0, 0])
    model.eval()
    with torch.no_grad():
        for cfg, a, m, y in data:
            per[cfg][0] += int(model(a, m).argmax()) == y
            per[cfg][1] += 1
    tot = sum(v[0] for v in per.values()) / sum(v[1] for v in per.values())
    return tot, {k: v[0] / v[1] for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2, choices=[1, 2])
    ap.add_argument("--dataset-root", default="data/I-RAVEN-10000")
    ap.add_argument("--only", default="", help="comma-separated subset of constellations")
    ap.add_argument("--n-train", type=int, default=500, help="per constellation")
    ap.add_argument("--n-val", type=int, default=200, help="per constellation")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nl-lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--resume", action="store_true",
                    help="continue from model_last.pt (else model_best.pt) in the save dir")
    ap.add_argument("--overfit", type=int, default=0,
                    help="if >0: train on N puzzles PER CONSTELLATION and report TRAIN accuracy")
    ap.add_argument("--balanced-batches", action="store_true",
                    help="every optimiser step sees one puzzle from EACH constellation")
    ap.add_argument("--adaptive-weights", action="store_true",
                    help="reweight each constellation's loss by how far it lags")
    ap.add_argument("--weight-floor", type=float, default=0.5,
                    help="min loss weight, so a solved constellation is not ignored")
    ap.add_argument("--layout", default="off", choices=["off","count","count_occ"],
                    help="panel-level layout features for Number/Position rules")
    ap.add_argument("--agg", default="avg", choices=["avg","max","sum"])
    ap.add_argument("--no-same-panel", action="store_true",
                    help="drop within-panel all-pairs edges (mostly pad-to-pad)")
    ap.add_argument("--edge-labels", default="basic", choices=["basic", "rowcol"],
                    help="rowcol: cross-panel edges tagged same-row / same-column")
    ap.add_argument("--per-type-weights", action="store_true",
                    help="one message matrix per edge type, each type averaged separately")
    ap.add_argument("--absent-pos", action="store_true",
                    help="empty slots carry their position + panel embedding")
    ap.add_argument("--jvm-gb", type=int, default=100)
    ap.add_argument("--save-dir", default="experiments/all_constellations")
    args = ap.parse_args()

    global CONSTELLATIONS, EDGE_LABELS, ABSENT_POS
    EDGE_LABELS, ABSENT_POS = args.edge_labels, args.absent_pos
    if args.only:
        CONSTELLATIONS = args.only.split(",")
    neuralogic.set_max_memory_size(args.jvm_gb)
    torch.manual_seed(0)
    save_dir = f"{args.save_dir}_L{args.level}"
    os.makedirs(save_dir, exist_ok=True)
    log(f"LEVEL {args.level} — ALL {len(CONSTELLATIONS)} CONSTELLATIONS JOINTLY")
    log(f"config: {vars(args)}")
    log(f"common topology: {SLOTS} slots/panel, {len(NODES)} nodes, {len(edges())} edges")

    t0 = time.time()
    if args.overfit:
        tr = load(args.dataset_root, CONSTELLATIONS, "train", args.overfit)
        va = tr                                     # evaluate on the training puzzles
        log(f"OVERFIT MODE: {args.overfit} puzzles per constellation, TRAIN accuracy")
    else:
        tr = load(args.dataset_root, CONSTELLATIONS, "train", args.n_train)
        va = load(args.dataset_root, CONSTELLATIONS, "val", args.n_val)
    log(f"loaded {len(tr)} train / {len(va)} val puzzles in {time.time()-t0:.0f}s")
    cnt = defaultdict(int)
    for cfg, *_ in tr:
        cnt[cfg] += 1
    for k in CONSTELLATIONS:
        log(f"    {k:<44} {cnt[k]} train")

    AGG = {"avg": Aggregation.AVG, "max": Aggregation.MAX, "sum": Aggregation.SUM}[args.agg]
    log(f"layout={args.layout}  aggregation={args.agg}  same_panel_edges={not args.no_same_panel}  "
        f"edges={len(edges(not args.no_same_panel))}  edge_labels={args.edge_labels}  "
        f"per_type_weights={args.per_type_weights}  absent_pos={args.absent_pos}")
    model = AllConstellations(args.level, args.embed_dim, args.hidden_dim,
                              args.layers, Transformation.LEAKY_RELU, args.nl_lr,
                              agg=AGG, same_panel=not args.no_same_panel,
                              layout=args.layout, per_type=args.per_type_weights,
                              absent_pos=args.absent_pos)
    tp = [p for n, p in model.named_parameters() if not n.startswith("nl.")]
    opt = torch.optim.AdamW(tp, lr=args.lr, weight_decay=1e-4)
    log(f"torch params: {sum(p.numel() for p in tp):,}")

    probe = model(tr[0][1], tr[0][2])
    log(f"sanity: forward returns {tuple(probe.shape)} (must be (8,))")
    assert probe.shape == (8,)

    # indices grouped by constellation, for stratified batching
    by_cfg = defaultdict(list)
    for i, (cfg, *_ ) in enumerate(tr):
        by_cfg[cfg].append(i)
    cfg_w = {c: 1.0 for c in CONSTELLATIONS}          # per-constellation loss weights

    def epoch_order():
        """Stratified: each step draws one puzzle from every constellation, so a
        single optimiser update always carries gradient from all seven tasks.
        Without this, a random permutation gives batches that are constellation-
        skewed by chance, and the rarer/harder tasks get noisier updates."""
        if not args.balanced_batches:
            return [int(i) for i in torch.randperm(len(tr))]
        pools = {c: [by_cfg[c][int(j)] for j in torch.randperm(len(by_cfg[c]))]
                 for c in by_cfg}
        order, k = [], max(len(v) for v in pools.values())
        for step in range(k):
            for c in CONSTELLATIONS:
                if c in pools and step < len(pools[c]):
                    order.append(pools[c][step])
        return order

    best, hist, start_ep = 0.0, [], 1
    if args.resume:
        # Preference: model_last.pt (written every epoch, carries optimiser state)
        # over model_best.pt (written only on improvement -- a valid resume point
        # only up to its own epoch; any later history is discarded).
        p = lambda n: os.path.join(save_dir, n)
        if os.path.exists(p("model_last.pt")) and os.path.exists(p("template_last.bin")):
            ck = torch.load(p("model_last.pt"), map_location="cpu", weights_only=False)
            model.load_state_dict(ck["torch_state"])
            model.nl.model.load(p("template_last.bin"))
            if "opt_state" in ck:
                opt.load_state_dict(ck["opt_state"])
            start_ep, best = ck["epoch"] + 1, ck.get("best", 0.0)
            if os.path.exists(p("history.json")):
                hist = [r for r in json.load(open(p("history.json"))) if r["epoch"] <= ck["epoch"]]
            log(f"RESUMED from model_last.pt at epoch {start_ep} (best so far {best*100:.1f}%)")
        elif os.path.exists(p("model_best.pt")) and os.path.exists(p("template_best.bin")):
            ck = torch.load(p("model_best.pt"), map_location="cpu", weights_only=False)
            model.load_state_dict(ck["torch_state"])
            model.nl.model.load(p("template_best.bin"))
            start_ep, best = ck["epoch"] + 1, ck["val_acc"]
            if os.path.exists(p("history.json")):
                full = json.load(open(p("history.json")))
                hist = [r for r in full if r["epoch"] <= ck["epoch"]]
                if len(full) > len(hist):
                    log(f"WARNING: best epoch {ck['epoch']} is older than the last logged epoch; "
                        f"discarding {len(full) - len(hist)} later epoch(s) of history")
            log(f"RESUMED from model_best.pt at epoch {start_ep} (best {best*100:.1f}%); "
                f"no optimiser state saved there, so AdamW moments restart")
        else:
            log("--resume given but no checkpoint found in save dir; starting from epoch 1")
    if start_ep > args.epochs:
        log(f"Already completed all {args.epochs} epochs; nothing to do.")
        return
    log(f"balanced_batches={args.balanced_batches}  adaptive_weights={args.adaptive_weights}")
    log(f"{'epoch':>5} {'loss':>10} {'val acc':>9} {'best':>8} {'worst cons':>11}   (chance 12.5%)")
    for ep in range(start_ep, args.epochs + 1):
        model.train(); tot = 0.0; opt.zero_grad()
        order = epoch_order()
        for i, idx in enumerate(order):
            cfg, a, m, y = tr[idx]
            loss = F.cross_entropy(model(a, m).unsqueeze(0), torch.tensor([y]))
            w = cfg_w[cfg] if args.adaptive_weights else 1.0
            ((loss * w) / args.batch_size).backward()
            tot += float(loss.detach())
            if (i + 1) % args.batch_size == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()

        acc, per = evaluate(model, va)
        hist.append({"epoch": ep, "loss": tot / len(tr), "val_acc": acc,
                     "per_constellation": per})
        if acc > best:
            best = acc
            torch.save({"epoch": ep, "val_acc": acc, "per_constellation": per,
                        "args": vars(args), "torch_state": model.state_dict()},
                       os.path.join(save_dir, "model_best.pt"))
            # The PyNeuraLogic template weights are NOT in state_dict -- `nl` only
            # registers a dummy empty Parameter. For level 2 the template holds
            # the message passing AND the scoring head, i.e. most of the model, so
            # a checkpoint without this is unloadable.
            model.nl.model.save(os.path.join(save_dir, "template_best.bin"))
            with open(os.path.join(save_dir, "best.json"), "w") as f:
                json.dump({"epoch": ep, "val_acc": acc, "per_constellation": per,
                           "args": vars(args)}, f, indent=2)
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)
        # Resume point, written EVERY epoch. Write to a temp name then os.replace, so
        # a wall-clock kill mid-write can never leave a corrupt checkpoint behind.
        tmp = os.path.join(save_dir, "model_last.tmp.pt")
        torch.save({"epoch": ep, "best": best, "torch_state": model.state_dict(),
                    "opt_state": opt.state_dict(), "args": vars(args)}, tmp)
        os.replace(tmp, os.path.join(save_dir, "model_last.pt"))
        tmp = os.path.join(save_dir, "template_last.tmp.bin")
        model.nl.model.save(tmp)
        os.replace(tmp, os.path.join(save_dir, "template_last.bin"))
        worst = min(per, key=per.get) if per else None
        if args.adaptive_weights and per:
            # lagging constellations get proportionally more gradient; the floor
            # keeps an already-solved one from being dropped entirely
            gap = {c: max(0.0, 1.0 - per.get(c, 0.0)) for c in CONSTELLATIONS}
            mean = sum(gap.values()) / max(len(gap), 1) or 1.0
            cfg_w = {c: max(args.weight_floor, g / mean) for c, g in gap.items()}
        wtxt = f"{worst.split('_')[0][:9]} {per[worst]*100:.0f}%" if worst else ""
        log(f"{ep:>5} {tot/len(tr):>10.4f} {acc*100:>8.1f}% {best*100:>7.1f}% {wtxt:>11}")
        if args.overfit and acc == 1.0:
            log(f"REACHED 100% TRAIN ACCURACY at epoch {ep}")
            for k in CONSTELLATIONS:
                log(f"        {k:<44} {per.get(k, 0)*100:5.1f}%")
            break
        if ep % 2 == 0:
            for k in CONSTELLATIONS:
                if k in per:
                    log(f"        {k:<44} {per[k]*100:5.1f}%")
    log(f"DONE. level {args.level} best overall val {best*100:.2f}%")


if __name__ == "__main__":
    main()
