#!/usr/bin/env python3
"""The rules template as a differentiable executor over SOFT perception (I-RAVEN-X 3x3, values 0-10).

Facts are val(P, A, V)[p]: panel P has value V for attribute A with probability p (the top-k values of a
perception distribution). The rules are pynl_symbolic's rule families written once; with MIN as
combination and MAX as aggregation they are crisp logic on 0/1 facts and Goedel fuzzy logic on soft
facts, so the same template scores argmax facts (today's end to end) and full distributions.

  holds(Row, A, Rule)   max over value assignments of min(p): the row satisfies Rule
  sat(K, A, Rule)       min(holds(r1), holds(r2), holds(K)): both context rows show Rule, candidate row K keeps it
  score(K)              sum over (A, Rule) of sat

Perception is simulated: logits = -beta * |v - true| + sigma * N(0, 1) over 0..10, softmax.

  python3 pynl/pynl_soft.py check  --n 200          template == Python on soft facts
  python3 pynl/pynl_soft.py sweep  --n 10000        crisp (argmax) vs soft, over noise levels (Python)
"""
import argparse, gzip, json, math, os, sys, time
from itertools import permutations, product

import numpy as np

ATTRS = ("type", "size", "color")
STEPS = (-2, -1, 1, 2)
RULES = ["const"] + [f"prog{k:+d}" for k in STEPS] + ["add", "sub", "dist3"]
NV = 11                       # values 0..10 in this file
VALS = range(NV)
DATA = "/mnt/project-files/iravenx/center_single_shuffle_n_3_maxval_10.json.gz"


def rows_of():
    r = {"r1": (0, 1, 2), "r2": (3, 4, 5)}
    r.update({f"k{k}": (6, 7, 8 + k) for k in range(8)})
    return r


def load(n, path=DATA):
    d = json.load(gzip.open(path, "rt"))
    out = []
    for key in sorted(d, key=int)[:n]:
        s = d[key]
        vals = np.array([[int(p[0][a.capitalize()]) for a in ATTRS] for p in s["rpm"]])   # (16, 3)
        out.append((vals, int(s["target"])))
    return out


def perceive(vals, beta, sigma, rng, k=3):
    """-> probs (16, 3, 10) and top-k sparse facts {(p, a): [(v, prob), ...]}"""
    v = np.arange(NV)
    logits = -beta * np.abs(v[None, None, :] - vals[:, :, None]) + sigma * rng.standard_normal(vals.shape + (NV,))
    pr = np.exp(logits - logits.max(-1, keepdims=True))
    pr /= pr.sum(-1, keepdims=True)
    top = {}
    for p in range(16):
        for a in range(3):
            idx = np.argsort(-pr[p, a])[:k]
            top[p, a] = [(int(i), float(pr[p, a, i])) for i in idx]
    return pr, top


def crisp(top):
    return {pa: [(c[0][0], 1.0)] for pa, c in top.items()}


# ── Python reference: the same Goedel semantics ──────────────────────────────
def rel(rule, x, y, z):
    if rule == "const": return x == y == z
    if rule.startswith("prog"):
        k = int(rule[4:]); return y == x + k and z == y + k
    if rule == "add": return z == x + y
    if rule == "sub": return x == y + z


def row_holds(facts, idx, a, r1idx, tnorm, agg=max):
    """{rule: truth} for one row and attribute."""
    F = [facts[p, a] for p in idx]
    h = {r: 0.0 for r in RULES}
    for (x, px), (y, py), (z, pz) in product(*F):
        t = tnorm((px, py, pz))
        for r in RULES[:-1]:
            if rel(r, x, y, z):
                h[r] = agg(h[r], t)
        # dist3: the same multiset as context row 1, three distinct values
        if len({x, y, z}) == 3:
            for (u, pu), (w, pw), (q, pq) in product(*[facts[p, a] for p in r1idx]):
                if sorted((u, w, q)) == sorted((x, y, z)):
                    h["dist3"] = agg(h["dist3"], tnorm((px, py, pz, pu, pw, pq)))
    return h


def python_scores(facts, tnorm=min, agg=max):
    rows = rows_of()
    s = np.zeros(8)
    for a in range(3):
        H = {r: row_holds(facts, idx, a, rows["r1"], tnorm, agg) for r, idx in rows.items()}
        for k in range(8):
            s[k] += sum(tnorm((H["r1"][r], H["r2"][r], H[f"k{k}"][r])) for r in RULES)
    return s


def acc(scores, y):
    best = np.flatnonzero(scores >= scores.max() - 1e-9)
    return (y in best) / len(best)


# ── the PyNeuraLogic template ────────────────────────────────────────────────
def build_template(sem="godel", concat=False):
    """sem: godel = MIN/MAX (fuzzy logic); prob = PRODUCT/SUM (probability that the relation holds,
    exact for one row when panels are independent). concat: one output out = the 8 scores, plus
    cidx(K) inputs to recover the CONCAT order (torch wrapper)."""
    from neuralogic.core import R, V, Model, Metadata, Transformation, Aggregation, Combination
    m = Model()
    for x in VALS:
        for k in STEPS:
            if x + k in VALS:
                m += R.step(f"v{x}", f"v{x + k}", f"s{k:+d}")
        for y in VALS:
            if x + y in VALS:
                m += R.plus(f"v{x}", f"v{y}", f"v{x + y}")
            for z in VALS:
                if len({x, y, z}) == 3:
                    for u in permutations((x, y, z)):
                        m += R.perm3(f"v{x}", f"v{y}", f"v{z}", *(f"v{w}" for w in u))
    fuzzy = Metadata(combination=Combination.MIN if sem == "godel" else Combination.PRODUCT,
                     transformation=Transformation.IDENTITY,
                     aggregation=Aggregation.MAX if sem == "godel" else Aggregation.SUM)
    Rw, K, A, Ru = V.Rw, V.K, V.A, V.Ru
    P1, P2, P3, Q1, Q2, Q3 = V.P1, V.P2, V.P3, V.Q1, V.Q2, V.Q3
    X, Y, Z, U, W, T = V.X, V.Y, V.Z, V.U, V.W, V.T
    row = R.row(Rw, P1, P2, P3)
    val = lambda P, v: R.val(P, A, v)
    rule = lambda name, *body: (R.holds(Rw, A, name) <= (row, val(P1, X), val(P2, Y), val(P3, Z)) + body) | fuzzy

    m += (R.holds(Rw, A, "const") <= (row, val(P1, X), val(P2, X), val(P3, X))) | fuzzy
    for k in STEPS:
        m += rule(f"prog{k:+d}", R.step(X, Y, f"s{k:+d}"), R.step(Y, Z, f"s{k:+d}"))
    m += rule("add", R.plus(X, Y, Z))
    m += rule("sub", R.plus(Y, Z, X))
    m += rule("dist3", R.row("r1", Q1, Q2, Q3), val(Q1, U), val(Q2, W), val(Q3, T), R.perm3(U, W, T, X, Y, Z))
    m += (R.holds / 3) | Metadata(transformation=Transformation.IDENTITY)
    m += (R.sat(K, A, Ru) <= (R.holds("r1", A, Ru), R.holds("r2", A, Ru), R.holds(K, A, Ru), R.cand(K))) | fuzzy
    m += (R.sat / 3) | Metadata(transformation=Transformation.IDENTITY)
    m += (R.score(K) <= R.sat(K, A, Ru)) | Metadata(transformation=Transformation.IDENTITY, aggregation=Aggregation.SUM)
    # bias 0; learnable only under the torch wrapper (its backward needs at least one template weight, lr 0)
    m += (R.score(K) <= (R.cand(K)[0.0] if concat else R.cand(K)[0.0].fixed())) | Metadata(transformation=Transformation.IDENTITY)
    m += (R.score / 1) | Metadata(transformation=Transformation.IDENTITY)
    if concat:
        m += (R.score(K) <= R.cidx(K)) | Metadata(transformation=Transformation.IDENTITY)
        m += (R.out <= R.score(K)) | Metadata(aggregation=Aggregation.CONCAT, transformation=Transformation.IDENTITY)
        m += (R.out / 0) | Metadata(transformation=Transformation.IDENTITY)
    return m


def facts_of(facts):
    from neuralogic.core import R
    F = [R.row(r, *(f"p{p}" for p in idx)) for r, idx in rows_of().items()]
    F += [R.cand(f"k{k}") for k in range(8)]
    F += [R.val(f"p{p}", ATTRS[a], f"v{v}")[pv] for (p, a), lst in facts.items() for v, pv in lst]
    return F


def template_scores(model, facts_list):
    from neuralogic.core import R
    from neuralogic.dataset import Dataset, Sample
    out = []
    for f in facts_list:      # one query per candidate keeps the output order explicit
        ds = Dataset([Sample(R.score(f"k{k}")[0.0], facts_of(f)) for k in range(8)])
        out.append(np.array([float(np.asarray(o).flatten()[0]) for o in model.test(model.build_dataset(ds))]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "sweep"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=3, help="top-k values per panel and attribute kept as facts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = load(args.n)

    if args.mode == "check":
        from neuralogic.logging import clear_handlers
        clear_handlers()
        from neuralogic.core import Settings
        from neuralogic.nn import Adam
        model = build_template()
        model.build(Settings(optimizer=Adam(lr=0.0), iso_value_compression=False, chain_pruning=False))
        rng = np.random.default_rng(args.seed)
        fl, ref, ys = [], [], []
        for vals, y in data:
            _, top = perceive(vals, beta=1.0, sigma=1.0, rng=rng, k=args.k)
            fl.append(top); ref.append(python_scores(top)); ys.append(y)
        t0 = time.time()
        out = template_scores(model, fl)
        d = np.abs(np.stack(out) - np.stack(ref))
        print(f"{len(fl)} puzzles in {time.time() - t0:.0f}s; template == Python on {(d < 1e-6).sum()}/{d.size} "
              f"candidates, max |diff| {d.max():.2g}; soft acc template {100 * np.mean([acc(o, y) for o, y in zip(out, ys)]):.2f}%")
        return

    print("beta sigma | argmax wrong/panel-attr | puzzles fully perceived | crisp (argmax) | soft min/max | product/max | product/sum | oracle")
    for beta, sigma in [(4, 0.5), (3, 0.5), (2.5, 1.0), (2, 1.0), (1.5, 1.0), (1, 1.0), (1, 1.5), (0.5, 1.0)]:
        rng = np.random.default_rng(args.seed)
        wrong, full, a_c, a_s, a_p, a_q, a_o = [], [], [], [], [], [], []
        for vals, y in data:
            pr, top = perceive(vals, beta, sigma, rng, k=args.k)
            am = pr.argmax(-1)
            wrong.append(np.mean(am != vals)); full.append(np.all(am == vals))
            a_c.append(acc(python_scores(crisp(top)), y))
            a_s.append(acc(python_scores(top, min), y))
            a_p.append(acc(python_scores(top, math.prod), y))
            a_q.append(acc(python_scores(top, math.prod, lambda u, v: u + v), y))
            a_o.append(acc(python_scores({(p, a): [(int(vals[p, a]), 1.0)] for p in range(16) for a in range(3)}), y))
        print(f"{beta:>4} {sigma:>5} | {100 * np.mean(wrong):6.2f}% | {100 * np.mean(full):6.1f}% | "
              f"{100 * np.mean(a_c):6.2f} | {100 * np.mean(a_s):6.2f} | {100 * np.mean(a_p):6.2f} | {100 * np.mean(a_q):6.2f} | {100 * np.mean(a_o):6.2f}", flush=True)


if __name__ == "__main__":
    main()
