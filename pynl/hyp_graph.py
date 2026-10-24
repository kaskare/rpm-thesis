#!/usr/bin/env python3
"""Hypothesis-indexed correspondence graph: the grounder builds one copy of the object graph per
(row, correspondence hypothesis), with edges only where a join finds a partner object.

Facts per puzzle (mini_raven, synthetic): objects with one-hot type/size/colour/position, inrow(O, R, C)
(object O sits in column C of row R; panels 6 and 7 are in all 8 candidate rows), at(O, S),
rk(O, Q) (rank of O in its panel), hyp(H) (the hypotheses of this puzzle's grid).
Background: mv(S1, S2, H, T): in a column pair of type T, slot S1 corresponds to S2 under H
(H = id or a cyclic shift s{k} over the layout's slots; T = fwd, bwd, skip, bskip).

  corr(O1, O2, R, T, H) :- inrow(O1, R, C1), inrow(O2, R, C2), pair(C1, C2, T),
                           at(O1, S1), at(O2, S2), mv(S1, S2, H, T), hyp(H)      exists only if both objects do
  corr(O1, O2, R, T, rank) :- ..., rk(O1, Q), rk(O2, Q), hyp(rank)
  x0(O, R, H)  :- inrow(O, R, C), hyp(H), h0(O), colemb(C)          one copy per (row, hypothesis)
  xk(O2, R, H) :- corr(O1, O2, R, T, H)[d, 1], x{k-1}(O1, R, H)[d, d], etemb(T)       AVG, + residual, NORM
  rowv(R, H)   :- inrow(O, R, C), xL(O, R, H)[d, d]                              AVG
  fit(K, H)    :- cand(K), rowv(r1, H)[d, d], rowv(r2, H)[d, d], rowv(K, H)[d, d]     the SAME H in all three rows
  score(K)     :- fit(K, H)[1, d]                                                MAX (or AVG) over H

Weights are shared by every hypothesis: H is only a variable. --hyps id keeps H = id (same-slot edges,
the GNN's assumption) as the control."""
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

N_POS = sum(len(v) for v in GRIDS.values())
IDENT = Metadata(transformation=Transformation.IDENTITY)
CRISP = Metadata(combination=Combination.MIN, aggregation=Aggregation.MAX, transformation=Transformation.IDENTITY)
PAIRS = {("c0", "c1"): "fwd", ("c1", "c2"): "fwd", ("c1", "c0"): "bwd", ("c2", "c1"): "bwd",
         ("c0", "c2"): "skip", ("c2", "c0"): "bskip"}
SHIFT = {"fwd": 1, "bwd": -1, "skip": 2, "bskip": -2}
ROWS = {"r1": (0, 1, 2), "r2": (3, 4, 5), **{f"k{k}": (6, 7, 8 + k) for k in range(8)}}


def onehot(i, n):
    v = [0.0] * n
    v[i] = 1.0
    return v


def hyps(grid, mode):
    n = len(GRIDS[grid])
    return ["id"] if mode == "id" else ["id"] + [f"s{k}" for k in range(1, n)] + (["rank"] if mode == "all" else [])


def background():
    f = [R.pair(a, b, t) for (a, b), t in PAIRS.items()]
    for slots in GRIDS.values():
        n = len(slots)
        for k in range(n):
            h = "id" if k == 0 else f"s{k}"
            for i, s in enumerate(slots):
                for t, m in SHIFT.items():
                    f.append(R.mv(f"u{s}", f"u{slots[(i + m * k) % n]}", h, t))
    return f


def facts(pz, mode):
    f = [R.hyp(h) for h in hyps(pz["grid"], mode)] + [R.cand(f"k{k}") for k in range(8)]
    for p, objs in enumerate(pz["panels"]):
        for q, o in enumerate(sorted(objs, key=lambda o: o["pos"])):
            n = f"o{p}_{o['pos']}"
            f += [R.ty(n)[onehot(o["type"], RANGE["type"])], R.sz(n)[onehot(o["size"], RANGE["size"])],
                  R.co(n)[onehot(o["color"], RANGE["color"])], R.ps(n)[onehot(o["pos"], N_POS)],
                  R.at(n, f"u{o['pos']}"), R.rk(n, f"q{q}")]
            f += [R.inrow(n, r, f"c{idx.index(p)}") for r, idx in ROWS.items() if p in idx]
    return f


def template(d, layers, pool_h, head="mlp", counts=False):
    m = Model()
    for b in background():
        m += b
    for t in SHIFT:
        m += R.etemb(t)[d,]
    for c in ("c0", "c1", "c2"):
        m += R.colemb(c)[d,]
    O, O1, O2, Rw, C, C1, C2, T, H, S1, S2, Q, K = (V.O, V.O1, V.O2, V.Rw, V.C, V.C1, V.C2, V.T, V.H,
                                                     V.S1, V.S2, V.Q, V.K)
    m += (R.h0(O) <= (R.ty(O)[d, RANGE["type"]], R.sz(O)[d, RANGE["size"]], R.co(O)[d, RANGE["color"]],
                      R.ps(O)[d, N_POS]))
    m += (R.h0 / 1) | IDENT
    body = (R.inrow(O1, Rw, C1), R.inrow(O2, Rw, C2), R.pair(C1, C2, T))
    m += (R.corr(O1, O2, Rw, T, H) <= body + (R.at(O1, S1), R.at(O2, S2), R.mv(S1, S2, H, T), R.hyp(H))) | CRISP
    m += (R.corr(O1, O2, Rw, T, "rank") <= body + (R.rk(O1, Q), R.rk(O2, Q), R.hyp("rank"))) | CRISP
    m += (R.corr / 5) | IDENT
    m += (R.x0(O, Rw, H) <= (R.inrow(O, Rw, C), R.hyp(H), R.h0(O), R.colemb(C)))
    m += (R.x0 / 3) | IDENT
    for k in range(1, layers + 1):
        x, xp = R.get(f"x{k}"), R.get(f"x{k - 1}")
        # corr is a derived scalar: a [d, 1] weight lifts it to a vector (an unweighted scalar in a vector sum breaks backward)
        m += (x(O2, Rw, H) <= (R.corr(O1, O2, Rw, T, H)[d, 1], xp(O1, Rw, H)[d, d], R.etemb(T))) | Metadata(aggregation=Aggregation.AVG)
        m += (x(O, Rw, H) <= xp(O, Rw, H))
        m += (R.get(f"x{k}") / 3) | Metadata(transformation=Transformation.NORM)
    act = Transformation.LEAKY_RELU
    m += (R.rowv(Rw, H) <= (R.inrow(O, Rw, C), R.get(f"x{layers}")(O, Rw, H)[d, d])) | Metadata(aggregation=Aggregation.AVG)
    if counts:   # how many groundings the joins found: correspondences per edge type, objects per column
        m += (R.cov(Rw, H, T) <= R.corr(O1, O2, Rw, T, H)) | Metadata(aggregation=Aggregation.COUNT, transformation=Transformation.IDENTITY)
        m += (R.ncol(Rw, C) <= R.inrow(O, Rw, C)) | Metadata(aggregation=Aggregation.COUNT, transformation=Transformation.IDENTITY)
        for t in SHIFT:
            m += (R.rowv(Rw, H) <= R.cov(Rw, H, t)[d, 1])
        for c in ("c0", "c1", "c2"):
            m += (R.rowv(Rw, H) <= (R.hyp(H), R.ncol(Rw, c)[d, 1]))
    m += (R.rowv / 2) | Metadata(transformation=act)
    agg = Aggregation.MAX if pool_h == "max" else Aggregation.AVG
    if head == "mlp":
        m += (R.fit(K, H) <= (R.cand(K), R.rowv("r1", H)[d, d], R.rowv("r2", H)[d, d], R.rowv(K, H)[d, d]))
        m += (R.fit / 2) | Metadata(transformation=act)
        m += (R.score(K) <= R.fit(K, H)[1, d]) | Metadata(aggregation=agg)
    else:        # analogy as similarity: the candidate row should look like the context rows, under the same H
        m += (R.ctxv(H) <= (R.rowv("r1", H)[d, d], R.rowv("r2", H)[d, d]))
        m += (R.ctxv / 1) | IDENT
        m += (R.candv(K, H) <= (R.cand(K), R.rowv(K, H)[d, d]))
        m += (R.candv / 2) | IDENT
        m += (R.fit(K, H) <= (R.candv(K, H), R.ctxv(H))) | Metadata(combination=Combination.COSSIM, transformation=Transformation.IDENTITY)
        m += (R.fit / 2) | IDENT
        m += (R.score(K) <= R.fit(K, H)[5.0]) | Metadata(aggregation=agg)
    m += (R.score(K) <= R.cidx(K)[1.0].fixed())
    m += (R.score / 1) | IDENT
    m += (R.answer(V.G) <= (R.puzzle(V.G), R.score(K))) | Metadata(
        aggregation=Aggregation.CONCAT, combination=Combination.PRODUCT, transformation=Transformation.IDENTITY)
    m += (R.answer / 1) | IDENT
    return m


def probe():
    return Sample(R.answer("probe")[[0.0] * 8], [R.puzzle("probe")] + [R.cidx(f"k{k}")[float(10 + k)] for k in range(8)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", choices=["id", "shifts", "all"], default="all")
    ap.add_argument("--pool-h", choices=["max", "avg"], default="max")
    ap.add_argument("--head", choices=["mlp", "cos"], default="mlp")
    ap.add_argument("--counts", action="store_true", help="add join counts (COUNT aggregation) to the row vector")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default=None)
    ap.add_argument("--xmx", default="12g")
    a = ap.parse_args()
    neuralogic.set_jvm_options(["-Xms1g", f"-Xmx{a.xmx}"])
    from neuralogic.logging import clear_handlers
    clear_handlers()
    tr, va = dataset(a.n_train, 0), dataset(a.n_val, 1)            # same data as dynamic_graph.py
    m = template(a.dim, a.layers, a.pool_h, a.head, a.counts)
    m.build(Settings(optimizer=Adam(lr=a.lr), error_function=CrossEntropy(with_logits=True),
                     iso_value_compression=False))
    t0 = time.time()
    samples = [Sample(R.answer(f"g{i}")[[0.0] * 8], [R.puzzle(f"g{i}")] + facts(pz, a.hyps)) for i, pz in enumerate(tr + va)]
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

    print(f"hyps={a.hyps} pool={a.pool_h} head={a.head} counts={a.counts}: {len(samples)} puzzles grounded in {t_ground:.0f}s "
          f"({1000 * t_ground / len(samples):.0f} ms/puzzle)", flush=True)
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
