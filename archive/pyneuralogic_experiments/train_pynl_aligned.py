#!/usr/bin/env python3
"""PyNeuraLogic I-RAVEN solver aligned as closely as possible to the torch GNN.

Differences from train_pynl.py, each mirroring a specific detail of
gnn/model.py::RelGNLayer / CandidateVsaGNN:

  1. ONE shared message rule keyed on edge(S,D,Type) + a learnable rel_emb(Type),
     instead of a separate weight matrix per relation. Matches torch's single
     `W` over cat([x_src + rel_emb, x_dst, rel_emb]) -- a linear map of a
     concatenation is the sum of linear maps of the parts, so separate summed
     rules are equivalent, but the WEIGHTS are now shared across relations.
  2. Because all edge types feed one rule, Aggregation.AVG now spans ALL
     incoming edges -- exactly torch's scatter_mean(msg, dst). Previously it
     averaged within each relation and summed across them, inflating magnitude.
  3. TRUE identity residual: `h_k(X) :- h_k-1(X)` with NO weight matrix.
     train_pynl.py put a learned [dim,dim] in the skip path, which is not a
     residual connection at all.
  4. Transformation.NORM as the LayerNorm stand-in (torch normalises every layer;
     without it hidden=384 diverges to loss ~3600).
  5. Two-layer score head with ctx_pool / cand_pool, matching torch's
     Linear+ReLU pools followed by Linear(2D->D)+ReLU+Linear(D->1).
  6. batch_size configurable (torch used 8; PyNeuraLogic defaults to 1).
  7. --granular optionally adds the 4 attribute satellite nodes per object.
"""
import argparse, glob, json, os, sys, time

import numpy as np
import neuralogic
from neuralogic.core import (Model, R, V, Settings, Transformation, Aggregation,
                             Metadata, Combination)
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn import Adam, CrossEntropy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pynl import parse, spatial_rels, POS_MAP, N_POS, N_COL, N_SHP, N_SIZ, N_PNL, log

SPATIAL = ["left_of", "right_of", "above", "below", "same_row", "same_col"]
EDGE_TYPES = SPATIAL + ["same_pos_cross_panel", "attr_to_obj", "obj_to_attr"]
# node kinds, mirroring gnn/config.py: 0=object, 2=attr_pos, 3=attr_color, 4=attr_type, 5=attr_size
KINDS = ["obj", "apos", "acol", "ashp", "asiz"]


# ── data ────────────────────────────────────────────────────────────────────
def sample_for(panels, cand, label, gid, granular):
    keep = list(range(8)) + [8 + cand]
    facts, nodes = [R.graph(gid)], {}

    for p in keep:
        grid = p if p < 8 else 8
        role = R.is_candidate if p == 8 + cand else R.is_context
        for o in panels[p]:
            nid = f"o{p}_{o['pos']}"
            nodes[nid] = (p, o)
            facts += [R.pos(nid, f"p{o['pos']}"), R.color(nid, f"c{o['color']}"),
                      R.shape(nid, f"s{o['type']}"), R.size(nid, f"z{o['size']}"),
                      R.panel(nid, f"g{grid}"), R.kind(nid, "obj"), role(nid)]
            if granular:
                # one satellite node per attribute, each carrying only that attribute
                for kind, pred, val in [("apos", R.pos, f"p{o['pos']}"),
                                        ("acol", R.color, f"c{o['color']}"),
                                        ("ashp", R.shape, f"s{o['type']}"),
                                        ("asiz", R.size, f"z{o['size']}")]:
                    anid = f"{nid}_{kind}"
                    facts += [pred(anid, val), R.panel(anid, f"g{grid}"),
                              R.kind(anid, kind), role(anid),
                              R.edge(anid, nid, "attr_to_obj"),
                              R.edge(nid, anid, "obj_to_attr")]

    for p in keep:                                   # intra-panel spatial edges
        ids = [n for n, (q, _) in nodes.items() if q == p]
        for i in ids:
            for j in ids:
                if i != j:
                    facts += [R.edge(i, j, rel) for rel in spatial_rels(nodes[i][1], nodes[j][1])]

    by_pos = {}                                      # cross-panel, same universal position
    for n, (_, o) in nodes.items():
        by_pos.setdefault(o["pos"], []).append(n)
    for grp in by_pos.values():
        for i in grp:
            for j in grp:
                if i != j:
                    facts.append(R.edge(i, j, "same_pos_cross_panel"))

    return Sample(R.score(gid)[label], facts)


def load(root, constellations, split, n_per, granular):
    samples, groups = [], []
    for cfg in constellations:
        for k, f in enumerate(sorted(glob.glob(os.path.join(root, cfg, f"*_{split}.npz")))[:n_per]):
            panels = parse(f.replace(".npz", ".xml"), cfg)
            target = int(np.load(f)["target"])
            groups.append((list(range(len(samples), len(samples) + 8)), target, cfg))
            samples += [sample_for(panels, c, 1 if c == target else 0, f"{cfg}_{split}{k}_{c}",
                                   granular) for c in range(8)]
    return samples, groups


# ── model ───────────────────────────────────────────────────────────────────
def build_model(embed_dim, hidden_dim, layers, act):
    d = hidden_dim
    m = Model()

    # embedding tables (torch NodeEncoder: pos/col/shp/siz/panel/kind, summed)
    for i in range(N_POS): m += R.emb_pos(f"p{i}")[embed_dim,]
    for i in range(N_COL): m += R.emb_col(f"c{i}")[embed_dim,]
    for i in range(N_SHP): m += R.emb_shp(f"s{i}")[embed_dim,]
    for i in range(N_SIZ): m += R.emb_siz(f"z{i}")[embed_dim,]
    for i in range(N_PNL): m += R.emb_pnl(f"g{i}")[embed_dim,]
    for k in KINDS:        m += R.emb_knd(k)[embed_dim,]
    # relation embeddings: torch's rel_emb, shared across the single message rule
    for t in EDGE_TYPES:   m += R.rel_emb(t)[d,]

    for fact, emb in [("pos", "emb_pos"), ("color", "emb_col"), ("shape", "emb_shp"),
                      ("size", "emb_siz"), ("panel", "emb_pnl"), ("kind", "emb_knd")]:
        m += (R.henc(V.O) <= (R.get(fact)(V.O, V.A), R.get(emb)(V.A)))
    m += (R.henc / 1) | Metadata(transformation=Transformation.IDENTITY)

    m += (R.h0(V.O) <= R.henc(V.O)[d, embed_dim])            # torch `inp`
    m += (R.h0 / 1) | Metadata(transformation=act)

    prev = "h0"
    for k in range(1, layers + 1):
        cur = f"h{k}"
        # (1)+(2): one shared rule over every edge type -> shared W, and AVG spans
        # all incoming edges exactly like scatter_mean
        m += (R.get(cur)(V.Dd) <= (R.edge(V.Ss, V.Dd, V.T),
                                   R.get(prev)(V.Ss)[d, d],
                                   R.rel_emb(V.T))) | Metadata(aggregation=Aggregation.AVG)
        m += (R.get(cur)(V.Dd) <= (R.edge(V.Ss, V.Dd, V.T),
                                   R.get(prev)(V.Dd)[d, d])) | Metadata(aggregation=Aggregation.AVG)
        # (3): true identity skip -- no weight
        m += (R.get(cur)(V.X) <= R.get(prev)(V.X))
        # (4): LayerNorm stand-in
        m += (R.get(cur) / 1) | Metadata(transformation=Transformation.NORM)
        prev = cur

    # (5) readout: mean-pool, then ctx_pool / cand_pool, then a 2-layer score head
    m += (R.ctxraw(V.G)  <= (R.graph(V.G), R.is_context(V.O),   R.get(prev)(V.O))) \
         | Metadata(aggregation=Aggregation.AVG)
    m += (R.candraw(V.G) <= (R.graph(V.G), R.is_candidate(V.O), R.get(prev)(V.O))) \
         | Metadata(aggregation=Aggregation.AVG)
    m += (R.ctxp(V.G)  <= R.ctxraw(V.G)[d, d])
    m += (R.candp(V.G) <= R.candraw(V.G)[d, d])
    m += (R.ctxp / 1)  | Metadata(transformation=act)
    m += (R.candp / 1) | Metadata(transformation=act)
    m += (R.shead(V.G)[d, 2 * d] <= (R.ctxp(V.G), R.candp(V.G))) \
         | Metadata(combination=Combination.CONCAT)
    m += (R.shead / 1) | Metadata(transformation=act)
    m += (R.score(V.G)[1, d] <= R.shead(V.G))
    m += (R.score / 1) | Metadata(transformation=Transformation.IDENTITY)
    return m


def evaluate(model, built, groups):
    out = [float(np.asarray(y).flatten()[0]) for y in model.test(built)]
    per_cfg, hits = {}, 0
    for idx, t, cfg in groups:
        ok = int(np.argmax([out[i] for i in idx])) == t
        hits += ok
        c = per_cfg.setdefault(cfg, [0, 0]); c[0] += ok; c[1] += 1
    return hits / len(groups), float(np.std(out)), {k: v[0] / v[1] for k, v in per_cfg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/I-RAVEN-10000")
    ap.add_argument("--constellations", default="center_single")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)      # torch default
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--granular", action="store_true")
    ap.add_argument("--overfit", type=int, default=0,
                    help="if >0, train on N puzzles and report TRAIN accuracy on them")
    ap.add_argument("--activation", default="leaky_relu",
                    choices=["leaky_relu", "norm", "tanh", "relu"])
    ap.add_argument("--jvm-gb", type=int, default=100)
    ap.add_argument("--save-dir", default="experiments/pynl_aligned")
    args = ap.parse_args()

    cfgs = list(POS_MAP) if args.constellations == "all" else args.constellations.split(",")
    os.makedirs(args.save_dir, exist_ok=True)
    neuralogic.set_max_memory_size(args.jvm_gb)
    log(f"config: {vars(args)}")
    log(f"neuralogic {neuralogic.__version__}, JAVA_HOME={os.environ.get('JAVA_HOME')}")

    act = {"leaky_relu": Transformation.LEAKY_RELU, "norm": Transformation.NORM,
           "tanh": Transformation.TANH, "relu": Transformation.RELU}[args.activation]

    t0 = time.time()
    if args.overfit:
        train_s, train_g = load(args.dataset_root, cfgs, "train", args.overfit, args.granular)
        val_s, val_g = [], train_g          # evaluate on the training puzzles themselves
        log(f"OVERFIT MODE: {len(train_g)} puzzles, reporting TRAIN accuracy")
    else:
        train_s, train_g = load(args.dataset_root, cfgs, "train", args.n_train, args.granular)
        val_s,   val_g   = load(args.dataset_root, cfgs, "val",   args.n_val,   args.granular)
    log(f"loaded {len(train_g)} train / {len(val_g)} val puzzles in {time.time()-t0:.0f}s")

    model = build_model(args.embed_dim, args.hidden_dim, args.layers, act)
    model.build(Settings(optimizer=Adam(lr=args.lr), error_function=CrossEntropy()))

    t0 = time.time()
    built = list(model.build_dataset(Dataset(train_s + val_s), batch_size=args.batch_size))
    tr = built[:len(train_s)]
    va = tr if args.overfit else built[len(train_s):]
    log(f"grounded in {time.time()-t0:.0f}s (batch_size={args.batch_size})")

    best, hist = 0.0, []
    log(f"{'epoch':>5} {'loss':>10} {'acc':>8} {'std':>10} {'best':>8}   (chance 12.5%)")
    for ep in range(1, args.epochs + 1):
        res = model.train(tr, epochs=1)
        loss = float(np.mean([float(np.asarray(r[2]).flatten()[0]) for r in res])) if res else float("nan")
        acc, spread, per_cfg = evaluate(model, va, val_g)
        hist.append({"epoch": ep, "loss": loss, "acc": acc, "std": spread})
        if acc > best:
            best = acc
            model.save(os.path.join(args.save_dir, "model_best.bin"))
            with open(os.path.join(args.save_dir, "best.json"), "w") as f:
                json.dump({"epoch": ep, "acc": acc, "per_constellation": per_cfg,
                           "args": vars(args)}, f, indent=2)
        log(f"{ep:>5} {loss:>10.4f} {acc*100:>7.1f}% {spread:>10.2e} {best*100:>7.1f}%")
        if ep % 25 == 0:
            with open(os.path.join(args.save_dir, "history.json"), "w") as f:
                json.dump(hist, f, indent=2)
        if args.overfit and acc == 1.0:
            log(f"REACHED 100% TRAIN ACCURACY at epoch {ep}"); break

    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    log(f"DONE. best {best*100:.2f}%")


if __name__ == "__main__":
    main()
