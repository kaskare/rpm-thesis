#!/usr/bin/env python3
"""Train small, test big: the primitives template on rows of ANY length.

Trained on I-RAVEN (3x3 grids, values 0-9), tested unchanged on I-RAVEN-X (IBM's text benchmark:
the last 3 rows of a 10x10 grid, values 0-999, optional confounder attributes).

The vocabulary is pynl_symbolic's prims template, with the numeric relations written over row
POSITIONS instead of a fixed row(R, P1, P2, P3):

  facts (per puzzle)  first/second/penult/last(Row, P)          row structure, any length
                      val(P, C, A, V)                           as in pynl_symbolic
                      full(Row, C, A)                           every panel of the row has a value
                      msig(Row, C, A, M), uniform(M)            the row's sorted values as one constant; all equal
                      plus_to(Row, C, A, Z, K)                  Z = (sum of all but the last) + K
                      minus_to(Row, C, A, X, K)                 X = (sum of all but the first) - K
                      step(X, Y, K)                             Y = X + K, for numbers in the same row
                                                                (so values are not limited to a 0-9 table)
  relations           eq_ab first == second        eq_bc penult == last        eq_ac first == last
                      eq_abc every panel equal
                      step_ab{k} second == first + k                   step_bc{k} last == penult + k
                      plus{k}  last == sum of the others + k
                      minus{k} last == first - sum of the middle + k
                      msame    the same multiset of values as context row 1
On rows of three these are exactly pynl_symbolic.NUM_PRIMS (check mode verifies it on I-RAVEN), and
the analogy rule pres(K, C, A, Prim) is unchanged. Position and slot-tuple primitives (I-RAVEN only)
are pynl_symbolic's.

  --weights per_attr  one weight per (attribute, primitive), as pynl_symbolic
  --weights shared    one weight per primitive, shared by every numeric attribute of the puzzle; the
                      model is not told which attributes matter, so it also scores confounders

  check  unit weights: template score == Python count, on I-RAVEN val 300+ and on I-RAVEN-X
  train  on I-RAVEN train, best epoch on I-RAVEN val 0..n-1; also logs, never selects on, accuracy on
         I-RAVEN-X dev puzzles (--dev, from puzzle 1000 on, apart from the test puzzles)
  test   the best and the last checkpoint, once each: I-RAVEN test and each --iravenx file
         (first --n-iravenx puzzles)
"""
import argparse, json, os, sys, time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path.insert(0, ROOT)
from rpm_symbolic import read_puzzle, files, CONSTELLATIONS
import pynl_symbolic as PS
from pynl_symbolic import ATTRS_NUM, STEPS, OFFS, NUM_PRIMS, SET_PRIMS, kname

NAMES = ["r1", "r2"] + PS.CAND_ROWS


# ── puzzles as rows of panels ─────────────────────────────────────────────────
def iraven_puzzle(panels):
    """(rows, vals, extra facts) of an I-RAVEN puzzle; extra = pynl_symbolic's facts minus its
    numeric background (needed by the position and slot-tuple primitives)."""
    rows = {"r1": [0, 1, 2], "r2": [3, 4, 5]}
    rows.update({f"k{k}": [6, 7, 8 + k] for k in range(8)})
    T = PS.puzzle_tuples(panels, "g")
    vals = {(int(t[1][1:]), t[2], t[3]): int(t[4][1:]) for t in T if t[0] == "val"}
    extra = [t for t in T if t[0] not in ("val", "puzzle")]
    return rows, vals, extra


def iravenx_attrs(rpm):
    """Type, Size, Color and any confounders; Angle is not shown (IBM's default, angle: false)."""
    return [a for a in rpm[0][0] if a != "Angle"]


def iravenx_puzzle(rpm, nshow=3):
    """I-RAVEN-X: the last nshow rows of an n x n grid; one object per panel (component c0)."""
    n = int(round((len(rpm) - 7) ** 0.5))
    rows = {f"r{i + 1}": list(range(r * n, (r + 1) * n)) for i, r in enumerate(range(n - nshow, n - 1))}
    last = list(range((n - 1) * n, n * n - 1))
    rows.update({f"k{k}": last + [n * n - 1 + k] for k in range(8)})
    shown = sorted({p for ps in rows.values() for p in ps})
    vals = {(p, "c0", a.lower()): int(rpm[p][0][a]) for p in shown for a in iravenx_attrs(rpm)}
    return rows, vals, [("cand", f"k{k}") for k in range(8)]


def row_values(rows, vals):
    """{(row, C, A): list of values (None where the panel has no single value)}"""
    ca = {(c, a) for (_, c, a) in vals}
    return {(r, c, a): [vals.get((p, c, a)) for p in ps] for r, ps in rows.items() for c, a in ca}


def puzzle_tuples(rows, vals, extra, gid):
    T = [("puzzle", gid)] + list(extra)
    T += [("val", f"p{p}", c, a, f"v{v}") for (p, c, a), v in vals.items()]
    T += [("numattr", a) for a in sorted({a for (_, _, a) in vals})]
    for r, ps in rows.items():
        P = [f"p{p}" for p in ps]
        T += [("first", r, P[0]), ("second", r, P[1]), ("penult", r, P[-2]), ("last", r, P[-1])]
    # whole-row facts come from Python: recursion along the row (running sums, all-equal) is exact
    # too, but its grounding grows too fast once confounders fill the value range (47 s for one
    # puzzle with a single confounder); joining a sum fact with an offset fact (sum(S), off(S, Z, K))
    # made the grounder's subsumption engine stall on some puzzles, hence plus_to / minus_to
    for (r, c, a), v in row_values(rows, vals).items():
        if None in v:
            continue
        sig = "u" + "_".join(map(str, sorted(v)))
        T += [("full", r, c, a), ("msig", r, c, a, sig)] + ([("uniform", sig)] if len(set(v)) == 1 else [])
        U = set(v)
        T += [("step", f"v{x}", f"v{x + k}", kname(k)) for x in U for k in STEPS if x + k in U]
        T += [("plus_to", r, c, a, f"v{sum(v[:-1]) + k}", kname(k)) for k in OFFS if sum(v[:-1]) + k >= 0]
        T += [("minus_to", r, c, a, f"v{sum(v[1:]) - k}", kname(k)) for k in OFFS if sum(v[1:]) - k >= 0]
    return sorted(set(T), key=str)


# ── the same vocabulary in plain Python: the exactness reference ───────────────
def num_holds(v, v1):
    """NUM_PRIMS on one row of any length; v1 = context row 1 (for msame)."""
    h = set()
    if None in v:
        return h
    a, b, y, z = v[0], v[1], v[-2], v[-1]
    if a == b: h.add("eq_ab")
    if y == z: h.add("eq_bc")
    if a == z: h.add("eq_ac")
    if len(set(v)) == 1: h.add("eq_abc")
    for k in STEPS:
        if b == a + k: h.add(f"step_ab{k:+d}")
        if z == y + k: h.add(f"step_bc{k:+d}")
    for k in OFFS:
        if z == sum(v[:-1]) + k: h.add(f"plus{k:+d}")
        if z == a - sum(v[1:-1]) + k: h.add(f"minus{k:+d}")
    if None not in v1 and sorted(v) == sorted(v1): h.add("msame")
    return h


def python_counts(rows, vals, weights):
    """Per candidate: (component, attribute, primitive) relations held by every context row and the
    candidate's row, over the attributes the template has weights for."""
    rv = row_values(rows, vals)
    ctx = [r for r in rows if r.startswith("r")]
    counts = np.zeros(8)
    for (r, c, a) in rv:
        if r != "r1" or (weights == "per_attr" and a not in ATTRS_NUM):
            continue
        common = set.intersection(*(num_holds(rv[q, c, a], rv["r1", c, a]) for q in ctx))
        for k in range(8):
            counts[k] += len(common & num_holds(rv[f"k{k}", c, a], rv["r1", c, a]))
    return counts


def reference(item, weights):
    kind, payload = item["kind"], item["payload"]
    if kind == "iraven":                 # unit weights count every attribute in both variants
        return PS.python_counts(payload)
    return python_counts(*iravenx_puzzle(payload)[:2], weights)


# ── the PyNeuraLogic template ─────────────────────────────────────────────────
def build_template(weights="per_attr", fixed_unit_weights=False):
    from neuralogic.core import R, V as Var, Transformation, Aggregation, Metadata, Combination
    from neuralogic.core import Model
    m = Model()
    Rw, P, L, P1, P2, P3, Cc, A = Var.Rw, Var.P, Var.L, Var.P1, Var.P2, Var.P3, Var.Cc, Var.A
    X, Y, Z, S, K, Pr = Var.X, Var.Y, Var.Z, Var.S, Var.K, Var.Pr
    U, W2, W3 = Var.U, Var.W2, Var.W3
    val = lambda P_, v: R.val(P_, Cc, A, v)
    ident = Metadata(transformation=Transformation.IDENTITY)
    crisp = Metadata(combination=Combination.MIN, transformation=Transformation.IDENTITY,
                     aggregation=Aggregation.MAX)

    def H(prim, *body):
        nonlocal m
        m += (R.holds(Rw, Cc, A, prim) <= body) | crisp

    # numeric primitives over row positions, generic over the attribute A
    full = R.full(Rw, Cc, A)          # as pynl_symbolic: no numeric relation on a row with a gap
    H("eq_ab", full, R.first(Rw, P1), R.second(Rw, P2), val(P1, X), val(P2, X))
    H("eq_bc", full, R.penult(Rw, P1), R.last(Rw, P2), val(P1, X), val(P2, X))
    H("eq_ac", full, R.first(Rw, P1), R.last(Rw, P2), val(P1, X), val(P2, X))
    H("eq_abc", R.msig(Rw, Cc, A, S), R.uniform(S))
    for k in STEPS:
        H(f"step_ab{k:+d}", full, R.first(Rw, P1), R.second(Rw, P2), val(P1, X), val(P2, Y), R.step(X, Y, kname(k)))
        H(f"step_bc{k:+d}", full, R.penult(Rw, P1), R.last(Rw, P2), val(P1, X), val(P2, Y), R.step(X, Y, kname(k)))
    for k in OFFS:
        H(f"plus{k:+d}", R.plus_to(Rw, Cc, A, Z, kname(k)), R.last(Rw, L), val(L, Z))
        # last == first - middle + k  <=>  first == (middle + last) - k
        H(f"minus{k:+d}", R.minus_to(Rw, Cc, A, X, kname(k)), R.first(Rw, P), val(P, X))
    H("msame", R.msig(Rw, Cc, A, S), R.msig("r1", Cc, A, S))

    # slot-wise identity and set primitives: pynl_symbolic's, on I-RAVEN's rows of three
    row = R.row(Rw, P1, P2, P3)
    m += (R.holds(Rw, Cc, A, "tup_eq") <= (row, R.tup(P1, Cc, A, X), R.tup(P2, Cc, A, X), R.tup(P3, Cc, A, X))) | crisp
    occ = lambda P_, s: R.occset(P_, Cc, s)

    def SP(prim, *body):
        nonlocal m
        m += (R.holds(Rw, Cc, "position", prim) <= (row,) + body) | crisp

    SP("seq_ab", occ(P1, X), occ(P2, X))
    SP("seq_bc", occ(P2, Y), occ(P3, Y))
    SP("s_union", occ(P1, X), occ(P2, Y), occ(P3, Z), R.s_union(X, Y, Z))
    SP("s_minus", occ(P1, X), occ(P2, Y), occ(P3, Z), R.s_minus(X, Y, Z))
    SP("s_inter", occ(P1, X), occ(P2, Y), occ(P3, Z), R.s_inter(X, Y, Z))
    for k in STEPS:
        SP(f"shift_ab{k:+d}", occ(P1, X), occ(P2, Y), R.s_shift(X, Y, kname(k)))
        SP(f"shift_bc{k:+d}", occ(P2, Y), occ(P3, Z), R.s_shift(Y, Z, kname(k)))
    m += (R.holds(Rw, Cc, "position", "msame") <= (row, R.row("r1", Var.Q1, Var.Q2, Var.Q3),
                                                   occ(P1, X), occ(P2, Y), occ(P3, Z),
                                                   occ(Var.Q1, U), occ(Var.Q2, W2), occ(Var.Q3, W3),
                                                   R.sperm3(U, W2, W3, X, Y, Z))) | crisp
    m += (R.holds / 4) | ident

    wt = (lambda atom: atom[1.0].fixed()) if fixed_unit_weights else (lambda atom: atom[0.1])
    lin = Metadata(transformation=Transformation.IDENTITY, aggregation=Aggregation.SUM)
    # analogy: a relation both context rows show and the candidate row preserves
    m += (R.pres(K, Cc, A, Pr) <= (R.holds("r1", Cc, A, Pr), R.holds("r2", Cc, A, Pr),
                                   R.holds(K, Cc, A, Pr), R.cand(K))) | crisp
    m += (R.pres / 4) | ident
    if weights == "per_attr":
        for a in ATTRS_NUM:
            for p in NUM_PRIMS:
                m += (R.score(K) <= wt(R.pres(K, Cc, a, p))) | lin
    else:   # one weight per primitive, summed over components AND numeric attributes
        m += (R.npres(K, Cc, A, Pr) <= (R.pres(K, Cc, A, Pr), R.numattr(A))) | crisp
        m += (R.npres / 4) | ident
        for p in NUM_PRIMS:
            m += (R.score(K) <= wt(R.npres(K, Cc, A, p))) | lin
    for a in ("type", "size", "color"):
        m += (R.score(K) <= wt(R.pres(K, Cc, a, "tup_eq"))) | lin
    for p in SET_PRIMS:
        m += (R.score(K) <= wt(R.pres(K, Cc, "position", p))) | lin

    # as pynl_symbolic: CONCAT-order probe, bias, logits, one answer(G) vector per puzzle
    m += (R.score(K) <= R.cidx(K)[1.0].fixed()) | ident
    bias = R.cand(K)[0.0].fixed() if fixed_unit_weights else R.cand(K)[0.0]
    m += (R.score(K) <= bias) | ident
    m += (R.score / 1) | ident
    G = Var.G
    m += (R.answer(G) <= (R.puzzle(G), R.score(K))) | Metadata(aggregation=Aggregation.CONCAT,
                                                              combination=Combination.PRODUCT,
                                                              transformation=Transformation.IDENTITY)
    m += (R.answer / 1) | ident
    return m


# ── data ──────────────────────────────────────────────────────────────────────
def load_iraven(split, n, start, root, cfgs=CONSTELLATIONS):
    out = []
    for cfg in cfgs:
        for f in files(cfg, split, root=root)[start:start + n if n else None]:
            panels, _ = read_puzzle(f.replace(".npz", ".xml"), cfg)
            out.append({"group": cfg, "kind": "iraven", "payload": panels, "y": int(np.load(f)["target"])})
    return out


def load_iravenx(path, n, start=0):
    import gzip
    d = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    tag = os.path.basename(path).split(".json")[0]
    return [{"group": tag, "kind": "iravenx", "payload": d[str(i)]["rpm"], "y": int(d[str(i)]["target"])}
            for i in range(start, min(start + n, len(d)))]


def make_sample(item, gid, R, Sample):
    pz = iraven_puzzle(item["payload"]) if item["kind"] == "iraven" else iravenx_puzzle(item["payload"])
    facts = [R.get(t[0])(*t[1:]) for t in puzzle_tuples(*pz, gid)]
    return Sample(R.answer(gid)[[0.0] * 8], facts)       # targets are set after the order is known


def accuracy(scores, data):
    per = defaultdict(lambda: [0.0, 0])
    for sc, it in zip(scores, data):
        top = np.flatnonzero(np.isclose(sc, sc.max()))
        per[it["group"]][0] += (1.0 / len(top)) if it["y"] in top else 0.0
        per[it["group"]][1] += 1
    tot = sum(v[0] for v in per.values()) / max(1, sum(v[1] for v in per.values()))
    return tot, {k: v[0] / v[1] for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "train", "test"])
    ap.add_argument("--weights", default="shared", choices=["shared", "per_attr"])
    ap.add_argument("--root", default="data/I-RAVEN-10000", help="I-RAVEN folder (relative to the repo root)")
    ap.add_argument("--iravenx", default="data/iravenx/center_single_shuffle_n_10_maxval_1000.json",
                    help="comma-separated I-RAVEN-X files (.json or .json.gz)")
    ap.add_argument("--n-iravenx", type=int, default=500, help="first N puzzles of each file (IBM: 500)")
    ap.add_argument("--dev", default="data/iravenx/center_single_shuffle_n_10_maxval_1000.json",
                    help="train mode: I-RAVEN-X file whose puzzles 1000.. are logged per epoch ('' = none)")
    ap.add_argument("--n-dev", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=300, help="I-RAVEN puzzles per constellation")
    ap.add_argument("--n-val", type=int, default=300, help="I-RAVEN val 0..n-1 per constellation")
    ap.add_argument("--n-test", type=int, default=0, help="I-RAVEN test puzzles per constellation (0 = all)")
    ap.add_argument("--n-check", type=int, default=20, help="check mode: puzzles per constellation / file")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--jvm-gb", type=int, default=40)
    ap.add_argument("--chunk", type=int, default=1000, help="test mode: puzzles per build")
    ap.add_argument("--save-dir", default="", help="default experiments/transfer_<weights>")
    args = ap.parse_args()
    args.save_dir = args.save_dir or os.path.join(ROOT, f"experiments/transfer_{args.weights}")
    irx = [p for p in args.iravenx.split(",") if p]
    irx = [p if os.path.isabs(p) else os.path.join(ROOT, p) for p in irx]

    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn import Adam, CrossEntropy
    from neuralogic.core.constructs.java_objects import ValueFactory

    t0 = time.time()

    def grounded(model, data, prefix):
        """build the probe + samples once, read the CONCAT order, set the one-hot targets"""
        samples = [make_sample(it, f"{prefix}{i}", R, Sample) for i, it in enumerate(data)]
        built = list(model.build_dataset(Dataset([PS.probe_sample(R, Sample)] + samples)))
        return built[1:], PS.read_order(model, built[0])

    def set_targets(built, data, order):
        vf = ValueFactory()
        for b, it in zip(built, data):          # `target` is read-only in Python; set it in Java
            b._java_sample.target = vf.get_value([1.0 if order[j] == it["y"] else 0.0 for j in range(8)])[1]

    def scores_of(model, built, order):
        out = [PS.unpermute(np.asarray(o, dtype=float).flatten(), order) for o in model.test(built)]
        assert len(out) == len(built), f"{len(out)} outputs for {len(built)} samples"
        return out

    def fresh(lr, unit=False):
        model = build_template(args.weights, fixed_unit_weights=unit)
        # iso_value_compression MUST be off (see pynl_symbolic: equal initial values freeze scores)
        model.build(Settings(optimizer=Adam(lr=lr), error_function=CrossEntropy(with_logits=True),
                             iso_value_compression=False))
        return model

    if args.mode == "check":
        data = load_iraven("val", args.n_check, 300, args.root)
        for p in irx:
            data += load_iravenx(p, args.n_check)
        model = fresh(0.0, unit=True)
        built, order = grounded(model, data, "g")
        out = np.stack(scores_of(model, built, order))
        ref = np.stack([reference(it, args.weights) for it in data])
        diff = np.abs(out - ref)
        print(f"{len(data)} puzzles in {time.time() - t0:.0f}s; template score == Python count on "
              f"{int((diff < 1e-6).sum())}/{diff.size} candidates; max |diff| = {diff.max():.3g}")
        for g in dict.fromkeys(it["group"] for it in data):
            idx = [i for i, it in enumerate(data) if it["group"] == g]
            print(f"  {g:<45} {int((diff[idx] < 1e-6).sum())}/{diff[idx].size}")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = {c: os.path.join(args.save_dir, f"template_{c}.bin") for c in ("best", "last")}
    if args.mode == "test":
        best = json.load(open(os.path.join(args.save_dir, "best.json")))
        assert best["args"]["weights"] == args.weights, "checkpoint was trained with other --weights"
        sets = [("I-RAVEN test", load_iraven("test", args.n_test, 0, args.root))]
        sets += [(os.path.basename(p), load_iravenx(p, args.n_iravenx)) for p in irx]
        res = {"weights": args.weights, "best_epoch": best["epoch"], "last_epoch": best["args"]["epochs"]}
        for name, data in sets:
            scores = {c: [] for c in ckpt}
            for i in range(0, len(data), args.chunk):   # a fresh build per chunk: build_dataset runs once per model
                model = fresh(0.0)
                built, order = grounded(model, data[i:i + args.chunk], "e")   # order read before load: bias is 0
                for c, path in ckpt.items():
                    model.load(path)
                    scores[c] += scores_of(model, built, order)
            res[name] = {"n": len(data)}
            for c in ckpt:
                acc, per = accuracy(scores[c], data)
                res[name][c] = {"acc": acc, "per": per}
            print(f"{name:<55} best {100 * res[name]['best']['acc']:6.2f}%  last {100 * res[name]['last']['acc']:6.2f}%"
                  f"  (n={len(data)})", flush=True)
        json.dump(res, open(os.path.join(args.save_dir, "test.json"), "w"), indent=1)
        return

    # ---- train: softmax cross-entropy over the 8 candidates, on I-RAVEN only ----
    tr = load_iraven("train", args.n_train, 0, args.root)
    va = load_iraven("val", args.n_val, 0, args.root)
    dev = load_iravenx(os.path.join(ROOT, args.dev), args.n_dev, start=1000) if args.dev else []
    model = fresh(args.lr)
    built, order = grounded(model, tr + va + dev, "t")
    set_targets(built, tr + va + dev, order)
    b_tr, b_va, b_dev = built[:len(tr)], built[len(tr):len(tr) + len(va)], built[len(tr) + len(va):]
    print(f"{len(tr)} train / {len(va)} val / {len(dev)} I-RAVEN-X dev puzzles grounded in "
          f"{time.time() - t0:.0f}s; CONCAT order {order}", flush=True)

    def evaluate():
        acc, per = accuracy(scores_of(model, b_va, order), va)
        return acc, per, accuracy(scores_of(model, b_dev, order), dev)[0] if dev else float("nan")

    def save(ep, acc, per):
        model.save(ckpt["best"])
        json.dump({"epoch": ep, "val_acc": acc, "per_constellation": per, "args": vars(args),
                   "concat_order": order}, open(os.path.join(args.save_dir, "best.json"), "w"), indent=1)

    best, per, dacc = evaluate()
    save(0, best, per)                    # epoch 0 can be the best one
    hist = [{"epoch": 0, "val_acc": best, "per": per, "dev_acc": dacc}]
    print(f"epoch   0  (initial weights)  val {100 * best:.2f}%  I-RAVEN-X dev {100 * dacc:.1f}%", flush=True)
    for ep in range(1, args.epochs + 1):
        res = model.train(b_tr, epochs=1)
        loss = float(np.mean([float(np.asarray(r[2]).flatten()[0]) for r in res])) if res else float("nan")
        acc, per, dacc = evaluate()
        hist.append({"epoch": ep, "loss": loss, "val_acc": acc, "per": per, "dev_acc": dacc})
        if acc > best:
            best = acc
            save(ep, acc, per)
        print(f"epoch {ep:>3}  loss {loss:.4f}  val {100 * acc:.2f}%  best {100 * best:.2f}%  "
              f"I-RAVEN-X dev {100 * dacc:.1f}%  " + "  ".join(f"{k[:6]} {100 * v:.0f}" for k, v in per.items()),
              flush=True)
    model.save(ckpt["last"])
    json.dump(hist, open(os.path.join(args.save_dir, "history.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
