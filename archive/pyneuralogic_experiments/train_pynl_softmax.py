#!/usr/bin/env python3
"""Full-scale PyNeuraLogic I-RAVEN solver — 8-way softmax objective.

Faithful port of the torch loss: each candidate is scored on its OWN disjoint
sub-graph (torch `forward_one`), the 8 scalar scores are CONCAT-aggregated into
one 8-vector (torch `torch.stack(..., dim=1)`), and CrossEntropy(with_logits=True)
softmaxes across them (torch `F.cross_entropy`). See gnn/model.py:409, gnn/train.py:102.

Candidates never exchange messages, so this avoids the answer-set leakage that
made `unified_graph` help RAVEN (+15) but hurt I-RAVEN (-20).

Target to beat: center_single 75.4%, I-RAVEN overall 68.8%.
"""
import argparse, glob, json, os, sys, time

import numpy as np
import neuralogic
from neuralogic.core import (Model, R, V, Settings, Transformation, Aggregation,
                             Metadata, Combination)
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn import Adam, CrossEntropy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pynl import (parse, spatial_rels, RELATIONS, POS_MAP,
                        N_POS, N_COL, N_SHP, N_SIZ, N_PNL, log)


# ── data ────────────────────────────────────────────────────────────────────
def puzzle_sample(panels, target, gid):
    """One puzzle -> one sample holding all 8 candidate branches, fully disjoint."""
    facts = []
    for c in range(8):
        branch = f"{gid}c{c}"
        facts.append(R.cand_of(branch, gid))
        keep, nodes = list(range(8)) + [8 + c], {}
        for p in keep:
            grid = p if p < 8 else 8
            for o in panels[p]:
                nid = f"{branch}_o{p}_{o['pos']}"
                nodes[nid] = (p, o)
                facts += [R.pos(nid, f"p{o['pos']}"), R.color(nid, f"c{o['color']}"),
                          R.shape(nid, f"s{o['type']}"), R.size(nid, f"z{o['size']}"),
                          R.panel(nid, f"g{grid}"), R.in_branch(nid, branch),
                          R.is_candidate(nid) if p == 8 + c else R.is_context(nid)]
        for p in keep:
            ids = [n for n, (q, _) in nodes.items() if q == p]
            for i in ids:
                for j in ids:
                    if i != j:
                        facts += [R.get(r)(i, j) for r in spatial_rels(nodes[i][1], nodes[j][1])]
        by_pos = {}
        for n, (_, o) in nodes.items():
            by_pos.setdefault(o["pos"], []).append(n)
        for grp in by_pos.values():
            for i in grp:
                for j in grp:
                    if i != j:
                        facts.append(R.same_pos_cross_panel(i, j))
    onehot = [0.0] * 8
    onehot[target] = 1.0
    return Sample(R.answer(gid)[onehot], facts)


def load(root, constellations, split, n_per):
    samples, meta = [], []
    for cfg in constellations:
        files = sorted(glob.glob(os.path.join(root, cfg, f"*_{split}.npz")))[:n_per]
        for k, f in enumerate(files):
            panels = parse(f.replace(".npz", ".xml"), cfg)
            t = int(np.load(f)["target"])
            samples.append(puzzle_sample(panels, t, f"{cfg}_{split}{k}"))
            meta.append((t, cfg))
    return samples, meta


# ── model ───────────────────────────────────────────────────────────────────
def build_model(dim, layers, act):
    m = Model()
    for i in range(N_POS): m += R.emb_pos(f"p{i}")[dim,]
    for i in range(N_COL): m += R.emb_col(f"c{i}")[dim,]
    for i in range(N_SHP): m += R.emb_shp(f"s{i}")[dim,]
    for i in range(N_SIZ): m += R.emb_siz(f"z{i}")[dim,]
    for i in range(N_PNL): m += R.emb_pnl(f"g{i}")[dim,]

    for fact, emb in [("pos", "emb_pos"), ("color", "emb_col"), ("shape", "emb_shp"),
                      ("size", "emb_siz"), ("panel", "emb_pnl")]:
        m += (R.h0(V.O) <= (R.get(fact)(V.O, V.A), R.get(emb)(V.A)))
    m += (R.h0 / 1) | Metadata(transformation=Transformation.IDENTITY)

    prev = "h0"
    for k in range(1, layers + 1):
        cur = f"h{k}"
        for rel in RELATIONS:
            m += (R.get(cur)(V.Dd) <= (R.get(rel)(V.Ss, V.Dd),
                                       R.get(prev)(V.Ss)[dim, dim],
                                       R.get(prev)(V.Dd)[dim, dim])) \
                 | Metadata(aggregation=Aggregation.AVG)
        m += (R.get(cur)(V.X) <= R.get(prev)(V.X)[dim, dim]) | Metadata(aggregation=Aggregation.AVG)
        m += (R.get(cur) / 1) | Metadata(transformation=act)
        prev = cur

    # pool within each candidate branch
    m += (R.ctx(V.B)  <= (R.in_branch(V.O, V.B), R.is_context(V.O),   R.get(prev)(V.O)[dim, dim])) \
         | Metadata(aggregation=Aggregation.AVG)
    m += (R.cand(V.B) <= (R.in_branch(V.O, V.B), R.is_candidate(V.O), R.get(prev)(V.O)[dim, dim])) \
         | Metadata(aggregation=Aggregation.AVG)
    m += (R.ctx / 1)  | Metadata(transformation=act)
    m += (R.cand / 1) | Metadata(transformation=act)
    m += (R.score(V.B)[1, 2 * dim] <= (R.ctx(V.B)[dim, dim], R.cand(V.B)[dim, dim])) \
         | Metadata(combination=Combination.CONCAT)
    m += (R.score / 1) | Metadata(transformation=Transformation.IDENTITY)

    # the 8 branch scores -> one 8-vector of logits (torch: stack(dim=1))
    m += (R.answer(V.G) <= (R.cand_of(V.B, V.G), R.score(V.B))) \
         | Metadata(aggregation=Aggregation.CONCAT)
    m += (R.answer / 1) | Metadata(transformation=Transformation.IDENTITY)
    return m


def evaluate(model, built, meta):
    outs = [np.asarray(y).flatten() for y in model.test(built)]
    per_cfg, hits = {}, 0
    for o, (t, cfg) in zip(outs, meta):
        ok = len(o) == 8 and int(np.argmax(o)) == t
        hits += ok
        c = per_cfg.setdefault(cfg, [0, 0]); c[0] += ok; c[1] += 1
    spread = float(np.mean([np.std(o) for o in outs])) if outs else 0.0
    return hits / len(meta), spread, {k: v[0] / v[1] for k, v in per_cfg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/I-RAVEN-10000")
    ap.add_argument("--constellations", default="center_single", help="comma-separated, or 'all'")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--activation", default="leaky_relu",
                    choices=["leaky_relu", "norm", "tanh", "relu"])
    ap.add_argument("--jvm-gb", type=int, default=100)
    ap.add_argument("--save-dir", default="experiments/pynl_softmax")
    args = ap.parse_args()

    cfgs = list(POS_MAP) if args.constellations == "all" else args.constellations.split(",")
    os.makedirs(args.save_dir, exist_ok=True)
    log(f"config: {vars(args)}")
    log(f"objective: 8-way softmax (CrossEntropy with_logits over a CONCAT'd 8-vector)")

    neuralogic.set_max_memory_size(args.jvm_gb)
    log(f"neuralogic {neuralogic.__version__}, JVM heap {args.jvm_gb}G, "
        f"JAVA_HOME={os.environ.get('JAVA_HOME')}")

    t0 = time.time()
    train_s, train_m = load(args.dataset_root, cfgs, "train", args.n_train)
    val_s,   val_m   = load(args.dataset_root, cfgs, "val",   args.n_val)
    log(f"loaded {len(train_s)} train / {len(val_s)} val puzzles (1 sample each) "
        f"in {time.time()-t0:.0f}s")

    act = {"leaky_relu": Transformation.LEAKY_RELU, "norm": Transformation.NORM,
           "tanh": Transformation.TANH, "relu": Transformation.RELU}[args.activation]
    model = build_model(args.dim, args.layers, act)
    model.build(Settings(optimizer=Adam(lr=args.lr), error_function=CrossEntropy(with_logits=True)))

    t0 = time.time()
    all_built = list(model.build_dataset(Dataset(train_s + val_s)))   # one call only
    tr, va = all_built[:len(train_s)], all_built[len(train_s):]
    assert len(tr) == len(train_s) and len(va) == len(val_s), "samples merged — query ids clash"
    log(f"grounded + neuralized in {time.time()-t0:.0f}s")

    probe = np.asarray(model.test([va[0]])[0]).flatten()
    log(f"sanity: query output shape {probe.shape} (must be (8,)); "
        f"uniform loss would be ln(8)={np.log(8):.4f}")
    assert probe.shape == (8,), f"expected an 8-vector of logits, got {probe.shape}"

    best, hist = 0.0, []
    log(f"{'epoch':>5} {'loss':>10} {'val acc':>9} {'logit std':>10} {'best':>8}   (chance 12.5%)")
    for ep in range(1, args.epochs + 1):
        res = model.train(tr, epochs=1)
        loss = float(np.mean([float(np.asarray(r[2]).flatten()[0]) for r in res])) if res else float("nan")
        acc, spread, per_cfg = evaluate(model, va, val_m)
        hist.append({"epoch": ep, "loss": loss, "val_acc": acc, "logit_std": spread})
        if acc > best:
            best = acc
            model.save(os.path.join(args.save_dir, "model_best.bin"))
            with open(os.path.join(args.save_dir, "best.json"), "w") as f:
                json.dump({"epoch": ep, "val_acc": acc, "per_constellation": per_cfg,
                           "args": vars(args)}, f, indent=2)
        log(f"{ep:>5} {loss:>10.4f} {acc*100:>8.1f}% {spread:>10.2e} {best*100:>7.1f}%")
        if ep % 25 == 0:
            with open(os.path.join(args.save_dir, "history.json"), "w") as f:
                json.dump(hist, f, indent=2)

    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    log(f"DONE. best val acc {best*100:.2f}%  (torch: center_single 75.4%, I-RAVEN 68.8%)")


if __name__ == "__main__":
    main()
