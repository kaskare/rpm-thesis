#!/usr/bin/env python3
"""Per-puzzle (dynamic) graphs vs the fixed padded slot graph, both trained natively in PyNeuraLogic.

fixed   : 16 panels x 9 slots, empty slots are `absent` nodes, same-slot edges across panels
          (the structure of train_all_constellations.py, rebuilt per sample).
fixed_panel : fixed + the panel nodes of `dynamic` (absent slots feed their panel too), to separate
          the effect of the per-puzzle structure from the effect of the panel level.
objects : the object level of `dynamic` alone (no panel nodes); the head pools objects.
dynamic : nodes = the objects present + one node per panel. Object edges link objects at the same
          position in two panels (only when both exist); objects feed their panel with SUM (so the
          panel vector carries count and position set); panels exchange messages labelled
          same row / same column / other.
Embedding tables live in the template: a one-hot fact times a [d, n] weight is a table lookup."""
import argparse, json, os, sys, time
from collections import defaultdict

import numpy as np
import neuralogic
from neuralogic.core import Model, R, V, Settings, Transformation, Aggregation, Combination, Metadata
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import CrossEntropy
from neuralogic.nn import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini_raven import dataset, GRIDS, RANGE

N_POS = sum(len(v) for v in GRIDS.values())           # 13 universal positions
SLOTS = max(len(v) for v in GRIDS.values())           # 9
ROLE = 9                                              # context panels 0-7, every candidate = 8
IDENT = Metadata(transformation=Transformation.IDENTITY)


def onehot(i, n):
    v = [0.0] * n
    v[i] = 1.0
    return v


def role(p):
    return min(p, 8)


def pedge_type(i, j):
    i, j = role(i), role(j)
    base = "cc" if i < 8 and j < 8 else ("cx" if j == 8 else "xc")
    if i // 3 == j // 3:
        return base + "_row"
    if i % 3 == j % 3:
        return base + "_col"
    return base


def connected(i, j):
    return i != j and not (i >= 8 and j >= 8)           # candidates never see each other


def obj_facts(name, o, p):
    return [R.ty(name)[onehot(o["type"], RANGE["type"])], R.sz(name)[onehot(o["size"], RANGE["size"])],
            R.co(name)[onehot(o["color"], RANGE["color"])], R.ps(name)[onehot(o["pos"], N_POS)],
            R.rl(name)[onehot(role(p), ROLE)]]


def facts_fixed(pz):
    slots = GRIDS[pz["grid"]]
    f = []
    for p, objs in enumerate(pz["panels"]):
        at = {o["pos"]: o for o in objs}
        for k in range(SLOTS):
            n = f"p{p}_s{k}"
            if k < len(slots) and slots[k] in at:
                f += obj_facts(n, at[slots[k]], p)
            else:
                f.append(R.absent(n))
            if p < 8:
                f.append(R.is_context(n))
            else:
                f.append(R.in_cand(n, f"k{p - 8}"))
    for k in range(SLOTS):
        for i in range(16):
            for j in range(16):
                if connected(i, j):
                    f.append(R.edge(f"p{i}_s{k}", f"p{j}_s{k}", pedge_type(i, j)))
    return f


def panel_facts():
    f = [R.ctx_panel(f"q{p}") if p < 8 else R.cand_panel(f"q{p}", f"k{p - 8}") for p in range(16)]
    return f + [R.pedge(f"q{i}", f"q{j}", pedge_type(i, j)) for i in range(16) for j in range(16) if connected(i, j)]


def facts_fixed_panel(pz):
    return facts_fixed(pz) + [R.in_panel(f"p{p}_s{k}", f"q{p}") for p in range(16) for k in range(SLOTS)] + panel_facts()


def facts_dynamic(pz):
    f = []
    where = defaultdict(list)                            # position -> [(panel, node)]
    for p, objs in enumerate(pz["panels"]):
        for o in objs:
            n = f"o{p}_{o['pos']}"
            f += obj_facts(n, o, p)
            f.append(R.in_panel(n, f"q{p}"))
            where[o["pos"]].append((p, n))
    for occ in where.values():
        for i, a in occ:
            for j, b in occ:
                if connected(i, j):
                    f.append(R.edge(a, b, pedge_type(i, j)))
    return f + panel_facts()


PANELS = ("fixed_panel", "dynamic")                  # arms with a panel level
EDGE_TYPES = [b + t for b in ("cc", "cx", "xc") for t in ("", "_row", "_col")]


def embed(m, d):
    m += (R.h0(V.O) <= (R.ty(V.O)[d, RANGE["type"]], R.sz(V.O)[d, RANGE["size"]],
                        R.co(V.O)[d, RANGE["color"]], R.ps(V.O)[d, N_POS], R.rl(V.O)[d, ROLE]))
    m += (R.h0 / 1) | IDENT


def mp_layer(m, d, prev, cur, edge="edge"):
    m += (R.get(cur)(V.D) <= (R.get(edge)(V.S, V.D, V.T), R.get(prev)(V.S)[d, d],
                              R.get(edge + "_emb")(V.T))) | Metadata(aggregation=Aggregation.AVG)
    m += (R.get(cur)(V.X) <= R.get(prev)(V.X))            # identity residual (a second, weighted
    # self rule with the same body would be merged with this one by the grounder)


def head(m, d, cand_vec, ctx_vec):
    act = Transformation.LEAKY_RELU
    m += (R.ctxp <= ctx_vec) | Metadata(aggregation=Aggregation.AVG)
    m += (R.ctxp / 0) | Metadata(transformation=act)
    m += (R.candp(V.C) <= cand_vec) | Metadata(aggregation=Aggregation.AVG)
    m += (R.candp / 1) | Metadata(transformation=act)
    m += (R.sh(V.C) <= (R.candp(V.C)[d, d], R.ctxp[d, d]))
    m += (R.sh / 1) | Metadata(transformation=act)
    m += (R.score(V.C)[1, d] <= R.sh(V.C))
    m += (R.score(V.C) <= R.cidx(V.C)[1.0].fixed())      # CONCAT-order probe (probe sample only)
    m += (R.score / 1) | IDENT
    m += (R.answer(V.G) <= (R.puzzle(V.G), R.score(V.C))) | Metadata(
        aggregation=Aggregation.CONCAT, combination=Combination.PRODUCT, transformation=Transformation.IDENTITY)
    m += (R.answer / 1) | IDENT


def template(graph, d, layers, pool="panels"):
    m = Model()
    for t in EDGE_TYPES:
        m += R.edge_emb(t)[d,]
        if graph in PANELS:
            m += R.pedge_emb(t)[d,]
    embed(m, d)
    if graph.startswith("fixed"):
        m += (R.h0(V.O) <= R.absent(V.O)[d, 1])             # learned absent vector
    if graph in PANELS:
        m += (R.g0(V.P) <= (R.in_panel(V.O, V.P), R.h0(V.O)[d, d])) | Metadata(aggregation=Aggregation.SUM)
        m += (R.g0 / 1) | Metadata(transformation=Transformation.NORM)
    for k in range(1, layers + 1):
        h, hp = f"h{k}", f"h{k-1}"
        mp_layer(m, d, hp, h)
        if graph in PANELS:
            g, gp = f"g{k}", f"g{k-1}"
            mp_layer(m, d, gp, g, edge="pedge")
            m += (R.get(g)(V.P) <= (R.in_panel(V.O, V.P), R.get(hp)(V.O)[d, d])) | Metadata(aggregation=Aggregation.SUM)
            m += (R.get(h)(V.O) <= (R.in_panel(V.O, V.P), R.get(gp)(V.P)[d, d]))
            m += (R.get(g) / 1) | Metadata(transformation=Transformation.NORM)
        m += (R.get(h) / 1) | Metadata(transformation=Transformation.NORM)
    L = layers
    if graph == "fixed":
        head(m, d, (R.in_cand(V.O, V.C), R.get(f"h{L}")(V.O)[d, d]),
             (R.is_context(V.O), R.get(f"h{L}")(V.O)[d, d]))
    elif pool == "objects" or graph == "objects":                                   # pool object nodes, as the fixed arm does
        head(m, d, (R.in_panel(V.O, V.P), R.cand_panel(V.P, V.C), R.get(f"h{L}")(V.O)[d, d]),
             (R.in_panel(V.O, V.P), R.ctx_panel(V.P), R.get(f"h{L}")(V.O)[d, d]))
    else:
        head(m, d, (R.cand_panel(V.P, V.C), R.get(f"g{L}")(V.P)[d, d]),
             (R.ctx_panel(V.P), R.get(f"g{L}")(V.P)[d, d]))
    return m


def probe():
    return Sample(R.answer("probe")[[0.0] * 8], [R.puzzle("probe")] + [R.cidx(f"k{k}")[float(10 + k)] for k in range(8)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=["fixed", "fixed_panel", "dynamic", "objects"], default="dynamic")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pool", choices=["panels", "objects"], default="panels", help="what the head reads")
    ap.add_argument("--xmx", default="6g", help="Java heap")
    a = ap.parse_args()
    neuralogic.set_jvm_options(["-Xms1g", f"-Xmx{a.xmx}"])
    build = {"fixed": facts_fixed, "fixed_panel": facts_fixed_panel, "dynamic": facts_dynamic,
             "objects": facts_dynamic}[a.graph]
    tr, va = dataset(a.n_train, 0), dataset(a.n_val, 1)
    m = template(a.graph, a.dim, a.layers, a.pool)
    m.build(Settings(optimizer=Adam(lr=a.lr), error_function=CrossEntropy(with_logits=True),
                     iso_value_compression=False))
    t0 = time.time()
    samples = [Sample(R.answer(f"g{i}")[[0.0] * 8], [R.puzzle(f"g{i}")] + build(pz)) for i, pz in enumerate(tr + va)]
    n_facts = np.mean([len(s.example) for s in samples]) if hasattr(samples[0], "example") else float("nan")
    built = list(m.build_dataset(Dataset([probe()] + samples)))
    t_ground = time.time() - t0
    out = np.asarray(list(m.test([built[0]]))[0], dtype=float).flatten()
    order = [int(round(v)) - 10 for v in out]
    assert sorted(order) == list(range(8)), out
    built = built[1:]
    assert len(built) == len(samples), (len(built), len(samples))
    from neuralogic.core.constructs.java_objects import ValueFactory
    vf = ValueFactory()
    for b, pz in zip(built, tr + va):
        b._java_sample.target = vf.get_value([1.0 if order[j] == pz["target"] else 0.0 for j in range(8)])[1]
    b_tr, b_va = built[:len(tr)], built[len(tr):]

    def acc(bs, data):
        per = defaultdict(lambda: [0, 0])
        for o, pz in zip(m.test(bs), data):
            o = np.asarray(o, dtype=float).flatten()
            s = np.zeros(8)
            for j, k in enumerate(order):
                s[k] = o[j]
            for key in ("all", pz["layout_rule"]):
                per[key][0] += int(s.argmax() == pz["target"])
                per[key][1] += 1
        return {k: v[0] / v[1] for k, v in per.items()}

    print(f"{a.graph}: {len(samples)} puzzles grounded in {t_ground:.0f}s "
          f"({1000 * t_ground / len(samples):.0f} ms/puzzle), facts/puzzle {n_facts:.0f}", flush=True)
    hist = []
    for ep in range(1, a.epochs + 1):
        t1 = time.time()
        res = m.train(b_tr, epochs=1)
        loss = float(np.mean([float(np.asarray(r[2]).flatten()[0]) for r in res]))
        t_ep = time.time() - t1
        av, at = acc(b_va, va), acc(b_tr[:300], tr[:300])
        hist.append(dict(epoch=ep, loss=loss, val=av, train=at["all"], sec=t_ep))
        print(f"ep {ep:>2} loss {loss:.3f} train {100*at['all']:.1f} val {100*av['all']:.1f} "
              f"({t_ep:.0f}s) " + " ".join(f"{k} {100*v:.0f}" for k, v in sorted(av.items()) if k != "all"),
              flush=True)
    if a.out:
        json.dump(dict(args=vars(a), ground_s=t_ground, history=hist), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
