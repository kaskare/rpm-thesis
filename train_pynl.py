#!/usr/bin/env python3
"""Full-scale PyNeuraLogic I-RAVEN solver.

Scaled-up version of pynl_iraven_minimal.ipynb: object-only relational GNN over
per-candidate graphs, scored one candidate at a time and argmax'd at evaluation.

Target to beat (torch RelGNLayer, experiments/object_vsa_gnn_frontend_gnn_granular_object_only):
  center_single 75.4%   |   I-RAVEN overall 68.8%
"""
import argparse, glob, json, os, sys, time, xml.etree.ElementTree as ET

import numpy as np
import neuralogic
from neuralogic.core import (Model, R, V, Settings, Transformation, Aggregation,
                             Metadata, Combination)
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn import Adam, CrossEntropy

POS_MAP = {
    "center_single": {0: 0},
    "distribute_four": {0: 1, 1: 2, 2: 3, 3: 4},
    "distribute_nine": {i: 5 + i for i in range(9)},
    "left_center_single_right_center_single": {0: 14, 1: 15},
    "up_center_single_down_center_single": {0: 16, 1: 17},
    "in_center_single_out_center_single": {0: 0, 1: 9},
    "in_distribute_four_out_center_single": {0: 0, 1: 18, 2: 19, 3: 20, 4: 21},
}
RELATIONS = ["left_of", "right_of", "above", "below", "same_row", "same_col",
             "same_pos_cross_panel"]
N_POS, N_COL, N_SHP, N_SIZ, N_PNL = 22, 10, 5, 6, 9


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── data ────────────────────────────────────────────────────────────────────
_CANON = {}


def _canonical(config, root=None):
    """(component_index, bbox) -> canonical local slot for this constellation.

    The XML `Position` attribute lists ONLY the slots a panel actually occupies,
    so `all_positions.index(bbox)` is an index into what is present -- its meaning
    shifts with the object count. For any constellation whose panels hold a
    varying number of objects (distribute_four/nine, in_distribute_four_...), that
    destroys position identity: a lone object at grid slot 3 and a lone object at
    grid slot 0 both come out as local index 0.

    Fix: derive the slot from the bbox GEOMETRY, using the union of every bbox
    that appears in that component across the dataset. Verified to recover exactly
    len(POS_MAP[cfg]) slots for all seven constellations.
    """
    if config in _CANON:
        return _CANON[config]
    root = root or os.environ.get("IRAVEN_ROOT", "data/I-RAVEN-10000")
    seen = {}
    for f in sorted(glob.glob(os.path.join(root, config, "*_train.xml")))[:200]:
        for panel in ET.parse(f).getroot()[0]:
            for struct in panel:
                for ci, comp in enumerate(struct):
                    for layout in comp:
                        for b in eval(layout.attrib["Position"]):
                            seen.setdefault(ci, set()).add(tuple(b))
    out, off = {}, 0
    for ci in sorted(seen):
        for k, b in enumerate(sorted(seen[ci])):
            out[(ci, b)] = off + k
        off += len(seen[ci])
    if off != len(POS_MAP[config]):
        raise RuntimeError(f"{config}: canonical map found {off} slots, "
                           f"expected {len(POS_MAP[config])}")
    _CANON[config] = out
    return out


def parse(xml_path, config):
    pos_map = POS_MAP[config]
    canon = _canonical(config, os.path.dirname(os.path.dirname(xml_path)))
    panels = []
    for panel in ET.parse(xml_path).getroot()[0]:
        objs = []
        for struct in panel:
            for ci, comp in enumerate(struct):
                for layout in comp:
                    for e in layout:
                        b = eval(e.attrib["bbox"])
                        local = canon[(ci, tuple(b))]
                        objs.append({"pos": pos_map[local],
                                     "type": int(e.attrib["Type"]) - 1,
                                     "size": int(e.attrib["Size"]),
                                     "color": int(e.attrib["Color"]),
                                     # RAVEN bboxes are CENTRE-based, [y_centre, x_centre, ...]
                                     # (checked against the rendered pixel masks)
                                     "cx": b[1], "cy": b[0]})
        panels.append(objs)
    return panels


def spatial_rels(a, b, eps=1e-4):
    r = []
    if   a["cx"] < b["cx"] - eps: r.append("left_of")
    elif a["cx"] > b["cx"] + eps: r.append("right_of")
    if   a["cy"] < b["cy"] - eps: r.append("above")
    elif a["cy"] > b["cy"] + eps: r.append("below")
    if abs(a["cy"] - b["cy"]) <= eps: r.append("same_row")
    if abs(a["cx"] - b["cx"]) <= eps: r.append("same_col")
    return r


def sample_for(panels, cand, label, gid):
    keep = list(range(8)) + [8 + cand]
    facts, nodes = [R.graph(gid)], {}
    for p in keep:
        grid = p if p < 8 else 8
        for o in panels[p]:
            nid = f"o{p}_{o['pos']}"
            nodes[nid] = (p, o)
            facts += [R.pos(nid, f"p{o['pos']}"), R.color(nid, f"c{o['color']}"),
                      R.shape(nid, f"s{o['type']}"), R.size(nid, f"z{o['size']}"),
                      R.panel(nid, f"g{grid}"),
                      R.is_candidate(nid) if p == 8 + cand else R.is_context(nid)]
    for p in keep:
        ids = [n for n, (q, _) in nodes.items() if q == p]
        for i in ids:
            for j in ids:
                if i != j:
                    facts += [R.get(rel)(i, j) for rel in spatial_rels(nodes[i][1], nodes[j][1])]
    by_pos = {}
    for n, (_, o) in nodes.items():
        by_pos.setdefault(o["pos"], []).append(n)
    for group in by_pos.values():
        for i in group:
            for j in group:
                if i != j:
                    facts.append(R.same_pos_cross_panel(i, j))
    return Sample(R.score(gid)[label], facts)


def load(root, constellations, split, n_per):
    samples, groups = [], []
    for cfg in constellations:
        files = sorted(glob.glob(os.path.join(root, cfg, f"*_{split}.npz")))[:n_per]
        for k, f in enumerate(files):
            panels = parse(f.replace(".npz", ".xml"), cfg)
            target = int(np.load(f)["target"])
            groups.append((list(range(len(samples), len(samples) + 8)), target, cfg))
            samples += [sample_for(panels, c, 1 if c == target else 0, f"{cfg}_{split}{k}_{c}")
                        for c in range(8)]
    return samples, groups


# ── model ───────────────────────────────────────────────────────────────────
def build_model(embed_dim, hidden_dim, layers, act):
    """Mirrors CandidateVsaGNN: embeddings at embed_dim, projected to hidden_dim,
    then `layers` RelGNLayers at hidden_dim (torch: gnn/model.py NodeEncoder -> inp -> gnn_layers)."""
    dim = hidden_dim
    m = Model()
    for i in range(N_POS): m += R.emb_pos(f"p{i}")[embed_dim,]
    for i in range(N_COL): m += R.emb_col(f"c{i}")[embed_dim,]
    for i in range(N_SHP): m += R.emb_shp(f"s{i}")[embed_dim,]
    for i in range(N_SIZ): m += R.emb_siz(f"z{i}")[embed_dim,]
    for i in range(N_PNL): m += R.emb_pnl(f"g{i}")[embed_dim,]

    for fact, emb in [("pos", "emb_pos"), ("color", "emb_col"), ("shape", "emb_shp"),
                      ("size", "emb_siz"), ("panel", "emb_pnl")]:
        m += (R.henc(V.O) <= (R.get(fact)(V.O, V.A), R.get(emb)(V.A)))
    m += (R.henc / 1) | Metadata(transformation=Transformation.IDENTITY)

    # torch `inp`: Linear(embed_dim -> hidden_dim) + LayerNorm + ReLU
    m += (R.h0(V.O) <= R.henc(V.O)[hidden_dim, embed_dim])
    m += (R.h0 / 1) | Metadata(transformation=act)

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

    m += (R.ctx(V.G)  <= (R.graph(V.G), R.is_context(V.O),   R.get(prev)(V.O)[dim, dim])) \
         | Metadata(aggregation=Aggregation.AVG)
    m += (R.cand(V.G) <= (R.graph(V.G), R.is_candidate(V.O), R.get(prev)(V.O)[dim, dim])) \
         | Metadata(aggregation=Aggregation.AVG)
    m += (R.ctx / 1)  | Metadata(transformation=act)
    m += (R.cand / 1) | Metadata(transformation=act)
    m += (R.score(V.G)[1, 2 * dim] <= (R.ctx(V.G)[dim, dim], R.cand(V.G)[dim, dim])) \
         | Metadata(combination=Combination.CONCAT)
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
    ap.add_argument("--constellations", default="center_single",
                    help="comma-separated, or 'all'")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--activation", default="leaky_relu",
                    choices=["leaky_relu", "norm", "tanh", "relu"])
    ap.add_argument("--jvm-gb", type=int, default=300)
    ap.add_argument("--save-dir", default="experiments/pynl_full")
    args = ap.parse_args()

    cfgs = list(POS_MAP) if args.constellations == "all" else args.constellations.split(",")
    os.makedirs(args.save_dir, exist_ok=True)
    log(f"config: {vars(args)}")
    log(f"constellations: {cfgs}")

    neuralogic.set_max_memory_size(args.jvm_gb)
    log(f"neuralogic {neuralogic.__version__}, JVM heap {args.jvm_gb}G, "
        f"JAVA_HOME={os.environ.get('JAVA_HOME')}")

    t0 = time.time()
    train_s, train_g = load(args.dataset_root, cfgs, "train", args.n_train)
    val_s,   val_g   = load(args.dataset_root, cfgs, "val",   args.n_val)
    log(f"loaded {len(train_g)} train / {len(val_g)} val puzzles "
        f"({len(train_s)}+{len(val_s)} samples) in {time.time()-t0:.0f}s")

    act = {"leaky_relu": Transformation.LEAKY_RELU, "norm": Transformation.NORM,
           "tanh": Transformation.TANH, "relu": Transformation.RELU}[args.activation]
    model = build_model(args.embed_dim, args.hidden_dim, args.layers, act)
    model.build(Settings(optimizer=Adam(lr=args.lr), error_function=CrossEntropy()))

    t0 = time.time()
    # build_dataset() may only be called ONCE per built model in 0.9.2
    all_built = list(model.build_dataset(Dataset(train_s + val_s)))
    tr, va = all_built[:len(train_s)], all_built[len(train_s):]
    assert len(tr) == len(train_s) and len(va) == len(val_s), "samples merged — query ids clash"
    log(f"grounded + neuralized in {time.time()-t0:.0f}s")

    best, hist = 0.0, []
    log(f"{'epoch':>5} {'loss':>10} {'val acc':>9} {'out std':>10} {'best':>8}   (chance 12.5%)")
    for ep in range(1, args.epochs + 1):
        res = model.train(tr, epochs=1)
        loss = float(np.mean([float(np.asarray(r[2]).flatten()[0]) for r in res])) if res else float("nan")
        acc, spread, per_cfg = evaluate(model, va, val_g)
        hist.append({"epoch": ep, "loss": loss, "val_acc": acc, "out_std": spread})
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
