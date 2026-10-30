#!/usr/bin/env python3
"""The value-axis model (notes §27): I-RAVEN-X with every attribute VALUE a constant of the template, so that
the template, like a CNN along the value axis, names no value and is the same at every value.

Facts (per puzzle, I-RAVEN-X center_single: one object per panel):
  x(P, A, v17)          panel P has value 17 for attribute A (type, size, colour); the fact is 1.0
  next(P, Q), col(P, first|mid|last), inrow(P, Rw), full(Rw), qlast(P)   the row structure (puzzle_facts)
Background facts (values v0..vM, M = maxval, the generator draws up to maxval inclusive):
  off(U, V, ok)         V = U + k, k in -K..K (the convolution's offsets)
  plus(U, W, V)         U + W = V within the range (addition as a relation)
  val(V)                every value

Template (weights shared across attributes and value positions; the attribute and the value are variables):
  h(P, A, V)    :- x(P, A, U), off(U, V, ok)          one weight per offset k: a local kernel along the value
                                                     axis; h encodes "V is k above panel P's value"
  pv(P, Q, A, V):- next(P, Q), h(P, A, V); h(Q, A, V)  neighbouring panels seen at the same value, tanh
  pd(P, Q, A)   :- pv(P, Q, A, V), summed over V       the pair's offset, pooled over values: shift-invariant
  rd(Rw, A)     :- the row's pairs, averaged           the row's step pattern
  acc(Q, A, V)  :- next(P, Q), acc(P, A, U), x(Q, A, W), plus(U, W, V)    the running sum along a row, exact
                   (product of the two operands' values: a convolution of value distributions); accm starts
                   at the second column (the minus form c1 = c2 + ... + cn)
  pchk, mchk    :- does a complete row's last (first) value equal the sum of the others: 1 or no atom
  sa(A, V), ma(A, V)   the question row's sum, and its first value minus the middle sum: plus / minus answers
  pr(Rw, A, V)  :- how many panels of the row have value V (counts); nd, ov: distinct values, row overlap
  ctx(A)        :- complete rows' and the question row's descriptors + the checks: which rule, value-free
  o(A, V)       :- ctx(A) at every V + h(qlast, A, V) + counts at V + sa, ma at V: two tanh layers shared
                   across V (a 1x1 convolution), then one score per value and a softmax over the values

Shift symmetry: rename every value v -> v+k (within range). No rule names a value, and off, val and next are
preserved by the renaming, so every atom except those built on plus moves with the values (the theorem of notes
§21.2): h, pd, rd, the counts and the output are equivariant, the descriptors invariant. plus is NOT preserved
(u+k + w+k != (u+w)+k): the plus pathway follows the per-operand shifts instead (shift one operand by k, the sum
moves by k), which is what addition does. --no-plus drops it, and the model is then exactly shift-equivariant.

--find-zero: addition without the plus table, the zero found per puzzle (notes §27). The plus table marks where
zero is (v0 adds nothing); the shift structure alone does not, and addition needs it. So the background is only
the value axis over its whole range, sh(U, V, sK) for V = U + K (K = 0..maxval: zero lies at or below every
value, so a value is a shift up from it), and every candidate zero Z is carried along, apart: a value W is the
shift from Z to W, and a running sum moves by it, so acc(P, A, Z, V) is the row's sum counted from Z. A complete row checks out under Z or not; the zero is the one
under which both complete rows check out, and the answers are read under it. Nothing in the template names a
value: the model is shift-equivariant, a shifted Arithmetic puzzle checks out under the shifted zero.

    python3 pynl/value_axis.py --iravenx data/iravenx/center_single_shuffle_n_10_maxval_50.json --n 6000 \\
        --train-epochs 60 --save-dir experiments/valaxis_n10_m50 --jvm-gb 64
    python3 pynl/value_axis.py --check-shift          # random weights: exact shift equivariance
Evaluation: matched and exact through bench_hub_grounding.evaluate (the value distributions are turned into
thermometer codes), and the exact value per rule by argmax.
"""
import argparse, ast, collections, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_hub_grounding as B

ATTRS = ("type", "size", "color")
OFFSETS = (-2, -1, 0, 1, 2)
RULES = ("Constant", "Progression", "Distribute_Three", "Arithmetic")
oname = lambda k: f"o{'m' if k < 0 else 'p'}{abs(k)}"
sname = lambda k: f"s{'m' if k < 0 else 'p'}{abs(k)}"       # a shift by k along the whole axis


def build_value_template(d, maxval, plus=True, per_value=False, find_zero=False, chain_rules=False):
    """per_value=True: the ablation of the value symmetry (notes §27). Every rule with weights whose atoms carry
    a value is written once per value constant, with its own weights: the kernel, the comparison at a value, the
    pooling, and the scoring layers. So the template names every value, like the facet model's weight per code
    bit, and no longer moves with the values. The value-free part (rd, ctx), the sums and the counts are kept.
    find_zero=True (with plus): no plus table; the sums are counted from every candidate zero, and the zero is
    the one under which both complete rows check out (see the module docstring). The sums carry no weights:
    by default they come in as facts from zero_chains(); chain_rules=True derives them with rules instead (the
    same network, but PyNeuraLogic's grounder takes ~40 s a puzzle on them)."""
    from neuralogic.core import Model, R, V, Transformation, Aggregation, Metadata, Combination
    m = Model()
    S = R.hidden
    meta = lambda **kw: Metadata(**kw)
    tanh, ident = meta(transformation=Transformation.TANH), meta(transformation=Transformation.IDENTITY)
    vals = [f"v{i}" for i in range(maxval + 1)]
    vs = vals if per_value else [V.V]          # the value of a rule: a constant per value, or one variable
    # the question row's last context panel, named in the rules that read the sums: joined as qlast(P) with
    # acc or accm, PyNeuraLogic's bottom-up grounder takes minutes to hours on long runs of small values
    # (rows of 14 and more); the grounded network is the same
    ql = f"p{B.ROWS['r3'][-1]}"
    # background knowledge: the value axis (offsets), and addition as a relation
    m += [R.val(v) for v in vals]
    m += [R.off(vals[i], vals[i + k], oname(k)) for i in range(maxval + 1) for k in OFFSETS if 0 <= i + k <= maxval]
    if plus and not find_zero:
        m += [R.plus(vals[u], vals[w], vals[u + w]) for u in range(maxval + 1) for w in range(maxval + 1 - u)]
    if plus and find_zero:                     # the axis over its whole range, upwards; no zero marked
        m += [R.sh(vals[u], vals[v], sname(v - u)) for u in range(maxval + 1) for v in range(u, maxval + 1)]

    # the convolution along the value axis: one weight per offset, the same at every value
    for k in OFFSETS:
        for i, v in enumerate(vs):
            if not per_value or 0 <= i - k <= maxval:
                m += R.h(V.P, V.A, v) <= (R.x(V.P, V.A, V.U)[d, 1], S.off(V.U, v, oname(k)))
    m += (R.h / 3) | ident
    # neighbouring panels compared at every value, pooled over the values: the pair's offset
    for v in vs:
        m += R.pv(V.P, V.Q, V.A, v) <= (S.next(V.P, V.Q), R.h(V.P, V.A, v)[d, d])
        m += R.pv(V.P, V.Q, V.A, v) <= (S.next(V.P, V.Q), R.h(V.Q, V.A, v)[d, d])
    m += (R.pv / 4) | tanh
    for v in vs:
        m += (R.pd(V.P, V.Q, V.A) <= R.pv(V.P, V.Q, V.A, v)[d, d]) | meta(aggregation=Aggregation.SUM)
    m += (R.pd / 3) | tanh
    m += (R.rd(V.Rw, V.A) <= (S.inrow(V.P, V.Rw), R.pd(V.P, V.Q, V.A)[d, d])) | meta(aggregation=Aggregation.AVG)
    m += (R.rd / 2) | tanh

    # values per row: counts, distinct values, overlap of a complete row with the question row
    m += (R.pr(V.Rw, V.A, V.V) <= (S.inrow(V.P, V.Rw), R.x(V.P, V.A, V.V))) | meta(aggregation=Aggregation.SUM)
    m += (R.pr / 3) | ident
    m += (R.pb(V.Rw, V.A, V.V) <= (S.inrow(V.P, V.Rw), R.x(V.P, V.A, V.V))) | meta(aggregation=Aggregation.MAX)
    m += (R.pb / 3) | ident
    m += (R.nd(V.Rw, V.A) <= R.pb(V.Rw, V.A, V.V)) | meta(aggregation=Aggregation.SUM)
    m += (R.nd / 2) | ident
    m += (R.ov(V.Rw, V.A) <= (S.full(V.Rw), R.pb(V.Rw, V.A, V.V), R.pb("r3", V.A, V.V))) | meta(
        combination=Combination.ELPRODUCT, aggregation=Aggregation.SUM)
    m += (R.ov / 2) | ident

    if plus and not find_zero:
        # running sums along each row: exact, a product of the operands' values summed over plus (a convolution)
        prod = meta(combination=Combination.ELPRODUCT, aggregation=Aggregation.SUM)
        m += R.acc(V.P, V.A, V.V) <= (S.col(V.P, "first"), R.x(V.P, V.A, V.V))
        m += (R.acc(V.Q, V.A, V.V) <= (S.next(V.P, V.Q), R.acc(V.P, V.A, V.U), R.x(V.Q, V.A, V.W),
                                       S.plus(V.U, V.W, V.V))) | prod
        m += (R.acc / 3) | ident
        m += R.accm(V.Q, V.A, V.V) <= (S.next(V.P, V.Q), S.col(V.P, "first"), R.x(V.Q, V.A, V.V))
        m += (R.accm(V.Q, V.A, V.V) <= (S.next(V.P, V.Q), R.accm(V.P, V.A, V.U), R.x(V.Q, V.A, V.W),
                                        S.plus(V.U, V.W, V.V))) | prod
        m += (R.accm / 3) | ident
        # a complete row satisfies plus (last = sum of the rest) or minus (first = sum of the rest): 1, or no atom
        m += (R.pchk(V.Rw, V.A) <= (S.full(V.Rw), S.inrow(V.L, V.Rw), S.col(V.L, "last"), S.next(V.P, V.L),
                                    R.acc(V.P, V.A, V.V), R.x(V.L, V.A, V.V))) | prod
        m += (R.mchk(V.Rw, V.A) <= (S.full(V.Rw), S.inrow(V.L, V.Rw), S.col(V.L, "last"), R.accm(V.L, V.A, V.V),
                                    S.inrow(V.F, V.Rw), S.col(V.F, "first"), R.x(V.F, V.A, V.V))) | prod
        m += (R.pchk / 2) | ident
        m += (R.mchk / 2) | ident
        # the question row's answers under plus and minus
        m += R.sa(V.A, V.V) <= R.acc(ql, V.A, V.V)
        m += (R.ma(V.A, V.V) <= (R.accm(ql, V.A, V.W), S.inrow(V.F, "r3"), S.col(V.F, "first"),
                                 R.x(V.F, V.A, V.U), S.plus(V.V, V.W, V.U))) | prod
        m += (R.sa / 2) | ident
        m += (R.ma / 2) | ident
    if plus and find_zero:
        # every candidate zero Z kept apart: a value W is the shift K from Z up to W, and the running sum moves by
        # it (zero_chains() computes the same atoms)
        prod = meta(combination=Combination.ELPRODUCT, aggregation=Aggregation.SUM)
        if chain_rules:
            m += (R.amt(V.Q, V.A, V.Z, V.K) <= (R.x(V.Q, V.A, V.W), S.sh(V.Z, V.W, V.K))) | prod
            m += (R.amt / 4) | ident
            m += (R.acc(V.P, V.A, V.Z, V.V) <= (S.col(V.P, "first"), R.x(V.P, V.A, V.V), S.val(V.Z))) | prod
            m += (R.stp(V.Q, V.A, V.Z, V.U, V.K) <= (S.next(V.P, V.Q), R.acc(V.P, V.A, V.Z, V.U),
                                                    R.amt(V.Q, V.A, V.Z, V.K))) | prod
            m += (R.acc(V.Q, V.A, V.Z, V.V) <= (R.stp(V.Q, V.A, V.Z, V.U, V.K), S.sh(V.U, V.V, V.K))) | prod
            m += (R.stp / 5) | ident
            m += (R.acc / 4) | ident
            m += (R.accm(V.Q, V.A, V.Z, V.V) <= (S.next(V.P, V.Q), S.col(V.P, "first"), R.x(V.Q, V.A, V.V),
                                                 S.val(V.Z))) | prod
            m += (R.stpm(V.Q, V.A, V.Z, V.U, V.K) <= (S.next(V.P, V.Q), R.accm(V.P, V.A, V.Z, V.U),
                                                     R.amt(V.Q, V.A, V.Z, V.K))) | prod
            m += (R.accm(V.Q, V.A, V.Z, V.V) <= (R.stpm(V.Q, V.A, V.Z, V.U, V.K), S.sh(V.U, V.V, V.K))) | prod
            m += (R.stpm / 5) | ident
            m += (R.accm / 4) | ident
        # a complete row checks out under Z or not; the zero is the Z both complete rows check out under
        m += (R.pchk(V.Rw, V.A, V.Z) <= (S.full(V.Rw), S.inrow(V.L, V.Rw), S.col(V.L, "last"), S.next(V.P, V.L),
                                         R.acc(V.P, V.A, V.Z, V.V), R.x(V.L, V.A, V.V))) | prod
        m += (R.mchk(V.Rw, V.A, V.Z) <= (S.full(V.Rw), S.inrow(V.L, V.Rw), S.col(V.L, "last"),
                                         R.accm(V.L, V.A, V.Z, V.V), S.inrow(V.F, V.Rw), S.col(V.F, "first"),
                                         R.x(V.F, V.A, V.V))) | prod
        m += (R.pchk / 3) | ident
        m += (R.mchk / 3) | ident
        m += (R.zp(V.A, V.Z) <= (R.pchk("r1", V.A, V.Z), R.pchk("r2", V.A, V.Z))) | prod
        m += (R.zm(V.A, V.Z) <= (R.mchk("r1", V.A, V.Z), R.mchk("r2", V.A, V.Z))) | prod
        m += (R.zp / 2) | ident
        m += (R.zm / 2) | ident
        m += (R.fzp(V.A) <= R.zp(V.A, V.Z)) | meta(aggregation=Aggregation.MAX)       # a zero was found
        m += (R.fzm(V.A) <= R.zm(V.A, V.Z)) | meta(aggregation=Aggregation.MAX)
        m += (R.fzp / 1) | ident
        m += (R.fzm / 1) | ident
        # the question row's answers under the zero found: its sum, and its first value minus the middle sum
        # (V = U - W + Z: V is U moved back by the shift from Z to W)
        m += (R.sa(V.A, V.V) <= (R.zp(V.A, V.Z), R.acc(ql, V.A, V.Z, V.V))) | prod
        m += (R.mam(V.A, V.K) <= (R.zm(V.A, V.Z), R.accm(ql, V.A, V.Z, V.W), S.sh(V.Z, V.W, V.K))) | prod
        m += (R.ma(V.A, V.V) <= (R.mam(V.A, V.K), S.inrow(V.F, "r3"), S.col(V.F, "first"), R.x(V.F, V.A, V.U),
                                 S.sh(V.V, V.U, V.K))) | prod
        m += (R.mam / 2) | ident
        m += (R.sa / 2) | ident
        m += (R.ma / 2) | ident

    # which rule: value-free descriptors of the complete rows and of the question row
    avg = meta(aggregation=Aggregation.AVG)
    m += (R.ctx(V.A) <= (S.full(V.Rw), R.rd(V.Rw, V.A)[d, d])) | avg
    m += R.ctx(V.A) <= R.rd("r3", V.A)[d, d]
    m += (R.ctx(V.A) <= (S.full(V.Rw), R.nd(V.Rw, V.A)[d, 1])) | avg
    m += R.ctx(V.A) <= R.nd("r3", V.A)[d, 1]
    m += (R.ctx(V.A) <= (S.full(V.Rw), R.ov(V.Rw, V.A)[d, 1])) | avg
    if plus and not find_zero:
        m += (R.ctx(V.A) <= (S.full(V.Rw), R.pchk(V.Rw, V.A)[d, 1])) | avg
        m += (R.ctx(V.A) <= (S.full(V.Rw), R.mchk(V.Rw, V.A)[d, 1])) | avg
    if plus and find_zero:
        m += R.ctx(V.A) <= R.fzp(V.A)[d, 1]
        m += R.ctx(V.A) <= R.fzm(V.A)[d, 1]
    m += R.ctx(V.A) <= (S.attr(V.A), R.one[d, 1])                      # bias
    m += (R.ctx / 1) | tanh
    m += R.ctx2(V.A) <= R.ctx(V.A)[d, d]
    m += R.ctx2(V.A) <= (S.attr(V.A), R.one[d, 1])
    m += (R.ctx2 / 1) | tanh

    # the answer at every value: the same two layers at every V (a 1x1 convolution), then a softmax over V
    for v in vs:
        m += R.o(V.A, v) <= (S.val(v), R.ctx2(V.A)[d, d])
        m += R.o(V.A, v) <= (S.qlast(V.P), R.h(V.P, V.A, v)[d, d])
        m += (R.o(V.A, v) <= (S.full(V.Rw), R.pr(V.Rw, V.A, v)[d, 1])) | meta(aggregation=Aggregation.SUM)
        m += R.o(V.A, v) <= R.pr("r3", V.A, v)[d, 1]
        if plus:
            m += R.o(V.A, v) <= R.sa(V.A, v)[d, 1]
            m += R.o(V.A, v) <= R.ma(V.A, v)[d, 1]
    m += (R.o / 2) | tanh
    for v in vs:
        m += R.o2(V.A, v) <= R.o(V.A, v)[d, d]
        m += R.o2(V.A, v) <= (S.val(v), R.one[d, 1])
    m += (R.o2 / 2) | tanh
    for v in vs:
        m += R.s(V.A, v) <= R.o2(V.A, v)[1, d]
        m += R.s(V.A, v) <= (S.val(v), R.one[1, 1])
    m += (R.s / 2) | ident
    for a in ATTRS:
        m += (R.dist(a) <= tuple(R.s(a, v) for v in vals)) | meta(combination=Combination.CONCAT,
                                                                 transformation=Transformation.SOFTMAX)
    m += (R.out(V.G) <= (S.puzzle(V.G),) + tuple(R.dist(a) for a in ATTRS)) | meta(
        combination=Combination.CONCAT, transformation=Transformation.IDENTITY)
    m += (R.dist / 1) | ident
    m += (R.out / 1) | ident
    return m


def value_facts(panels, gid):
    from neuralogic.core import R
    facts = B.puzzle_facts(panels, gid, col_roles=True, fold=True) + [R.one] + [R.attr(a) for a in ATTRS]
    for p in range(B.N_CTX):
        o = panels[p][0]
        facts += [R.x(f"p{p}", a, f"v{o[a]}") for a in ATTRS]
    return facts


def zero_chains(panels, maxval):
    """The running sums of every row counted from every candidate zero Z, as facts acc(P, A, Z, V) (from the
    first panel) and accm(P, A, Z, V) (from the second): what build_value_template(find_zero=True,
    chain_rules=True) derives. A panel's value W moves the sum up by W - Z, so it needs Z <= W, and the sum
    stops where it leaves the axis."""
    from neuralogic.core import R
    facts = []
    for a in ATTRS:
        for ps in B.ROWS.values():
            for pred, start in ((R.acc, 0), (R.accm, 1)):
                if start >= len(ps):
                    continue
                for z in range(maxval + 1):
                    s = panels[ps[start]][0][a]
                    facts.append(pred(f"p{ps[start]}", a, f"v{z}", f"v{s}"))
                    for p in ps[start + 1:]:
                        w = panels[p][0][a]
                        if w < z or s + w - z > maxval:
                            break
                        s += w - z
                        facts.append(pred(f"p{p}", a, f"v{z}", f"v{s}"))
    return facts


def value_target(panel, maxval):
    t = np.zeros((len(ATTRS), maxval + 1))
    for i, a in enumerate(ATTRS):
        t[i, panel[0][a]] = 1.0
    return t.flatten()


def to_codes(out, panels, maxval):
    """The value distributions -> the facet model's output space (thermometer codes P(value > j), the count and
    occupancy of the one centred object), so bench_hub_grounding.evaluate scores it like the facet model."""
    p = np.asarray(out, dtype=float).reshape(len(ATTRS), maxval + 1)
    therm = [np.array([p[i, j + 1:].sum() for j in range(B.THERM)]) for i in range(len(ATTRS))]
    occ = np.zeros(B.N_POS)
    occ[panels[B.N_CTX - 1][0]["pos"]] = 1
    return np.concatenate(therm + [np.array(B.code(1)), occ])


def score(outs, data, rules, maxval):
    """matched and exact as in the facet model, the exact value per rule by argmax, and how often the generated
    value is the answer's (all three attributes)."""
    per_puzzle = [{"k0": to_codes(o, pz, maxval)} for o, (pz, _, _) in zip(outs, data)]
    _, exact, matched, ties, _ = B.evaluate(per_puzzle, data, "therm")
    hits, allv = collections.defaultdict(lambda: [0, 0]), 0
    for o, (pz, y, _), r in zip(outs, data, rules):
        p = np.asarray(o, dtype=float).reshape(len(ATTRS), maxval + 1)
        ok = True
        for i, a in enumerate(ATTRS):
            h = hits[r[a.capitalize()]]
            good = int(np.argmax(p[i]) == pz[B.N_CTX + y][0][a])
            h[0] += good
            h[1] += 1
            ok &= bool(good)
        allv += ok
    return matched, exact, ties, allv, hits


def rules_of(path, start, n):
    raw = B._IRAVENX[path]
    lit = lambda r: ast.literal_eval(r) if isinstance(r, str) else r
    return [lit(raw[str(i)]["rules"])[0] for i in range(start, start + n)]


def maxval_of(path):
    return int(os.path.basename(path).split("maxval_")[1].split(".")[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iravenx", default="data/iravenx/center_single_shuffle_n_10_maxval_50.json")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--n-sel", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-epochs", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--no-plus", action="store_true", help="without the plus relation (exactly shift-equivariant)")
    ap.add_argument("--find-zero", action="store_true",
                    help="no plus table: the zero is found per puzzle, as the one under which both complete rows check out")
    ap.add_argument("--per-value", action="store_true",
                    help="weights per value instead of per offset: the ablation of the value symmetry")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--jvm-gb", type=int, default=16)
    ap.add_argument("--check-shift", action="store_true", help="random weights: outputs of shifted puzzles")
    ap.add_argument("--eval", default=None, help="score this run directory on --iravenx (any value range)")
    args = ap.parse_args()

    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn import Adam, MSE

    first = B.load_iravenx(args.iravenx, 0, 1)[0][0]
    maxval = maxval_of(args.iravenx)
    B.configure(n_cols=(len(first) - 7) // 3, maxval=maxval)
    if args.eval:
        a = json.load(open(os.path.join(args.eval, "best.json")))["args"]
        args.dim, args.no_plus, args.per_value = a["dim"], a["no_plus"], a.get("per_value", False)
        args.find_zero = a.get("find_zero", False)
        if args.per_value and maxval != maxval_of(a["iravenx"]):
            raise SystemExit(f"{args.eval}: weights per value are tied to values 0..{maxval_of(a['iravenx'])}")
    if args.find_zero and (args.no_plus or args.per_value):
        raise SystemExit("--find-zero goes with the sums (not --no-plus) and shared weights (not --per-value)")
    model = build_value_template(args.dim, maxval, plus=not args.no_plus, per_value=args.per_value,
                                 find_zero=args.find_zero)
    model.build(Settings(optimizer=Adam(lr=args.lr), error_function=MSE(), iso_value_compression=False,
                         chain_pruning=False))
    n_w = sum(np.asarray(v).size for v in model.state_dict()["weights"].values())

    def samples_of(data, tag):
        return [Sample(R.out(f"{tag}{i}")[value_target([pz[B.N_CTX + y][0]], maxval).tolist()],
                       value_facts(pz, f"{tag}{i}") + (zero_chains(pz, maxval) if args.find_zero else []))
                for i, (pz, y, _) in enumerate(data)]

    if args.check_shift:
        check_shift(model, samples_of, maxval, args.iravenx,
                    "without plus" if args.no_plus else "zero found per puzzle" if args.find_zero else "with plus")
        return
    if args.eval:
        model.load(os.path.join(args.eval, "template_best.bin"))
        data = B.load_iravenx(args.iravenx, 6300, args.n_val)
        outs = model.test(list(model.build_dataset(Dataset(samples_of(data, "u")))))
        report(f"{args.eval} on {os.path.basename(args.iravenx)}", outs, data, rules_of(args.iravenx, 6300, args.n_val), maxval)
        return

    t = time.time()
    sets = {"train": B.load_iravenx(args.iravenx, 0, args.n), "sel": B.load_iravenx(args.iravenx, 6000, args.n_sel),
            "unseen": B.load_iravenx(args.iravenx, 6300, args.n_val)}
    rules = {"train": rules_of(args.iravenx, 0, args.n), "sel": rules_of(args.iravenx, 6000, args.n_sel),
             "unseen": rules_of(args.iravenx, 6300, args.n_val)}
    built = {k: list(model.build_dataset(Dataset(samples_of(s, k)))) for k, s in sets.items()}
    print(f"value-axis model{'' if not args.no_plus else ', no plus'}{', weights per value' if args.per_value else ''}{', zero found per puzzle' if args.find_zero else ''}: dim {args.dim}, {n_w} weights, values 0..{maxval}, "
          f"rows of {len(B.COLS)}; {args.n} training puzzles; grounded in {time.time() - t:.0f}s", flush=True)

    best, hist = (-1, 0), []
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    t = time.time()
    for ep in range(args.eval_every, args.train_epochs + 1, args.eval_every):
        model.train(built["train"], epochs=args.eval_every)
        res = {}
        for k in ("sel", "unseen"):
            matched, exact, ties, allv, hits = score(model.test(built[k]), sets[k], rules[k], maxval)
            res[k] = {"matched": matched / len(sets[k]), "exact": exact["all"] / len(sets[k]),
                      "per_rule": {r: hits[r][0] / max(1, hits[r][1]) for r in RULES}}
        hist.append({"epoch": ep, **res})
        if res["sel"]["matched"] > best[0]:
            best = (res["sel"]["matched"], ep)
            if args.save_dir:
                model.save(os.path.join(args.save_dir, "template_best.bin"))
                json.dump({"epoch": ep, "args": vars(args), "weights": n_w, **hist[-1]},
                          open(os.path.join(args.save_dir, "best.json"), "w"), indent=1)
        if args.save_dir:
            json.dump(hist, open(os.path.join(args.save_dir, "history.json"), "w"), indent=1)
        u = res["unseen"]
        print(f"epoch {ep:>4}  sel matched {100 * res['sel']['matched']:5.1f}  |  unseen matched {100 * u['matched']:5.1f}"
              f"  exact {100 * u['exact']:5.1f}  per rule " + " / ".join(f"{100 * u['per_rule'][r]:.1f}" for r in RULES)
              + f"  ({time.time() - t:.0f}s)", flush=True)
    h = next(x for x in hist if x["epoch"] == best[1])["unseen"]
    print(f"best on selection (epoch {best[1]}): unseen matched {100 * h['matched']:.1f}%, exact {100 * h['exact']:.1f}%, "
          f"exact value per rule " + " / ".join(f"{r} {100 * h['per_rule'][r]:.1f}" for r in RULES), flush=True)


def report(label, outs, data, rules, maxval):
    matched, exact, ties, allv, hits = score(outs, data, rules, maxval)
    n = len(data)
    print(f"{label}: matched {100 * matched / n:.1f}%, exact {100 * exact['all'] / n:.1f}% (ties {ties}); all three "
          f"values right {100 * allv / n:.1f}%; exact value per rule "
          + " / ".join(f"{r} {100 * hits[r][0] / max(1, hits[r][1]):.1f} (of {hits[r][1]})" for r in RULES), flush=True)


def check_shift(model, samples_of, maxval, path, what):
    """Random weights: every context value of one attribute moved by k, in puzzles whose values stay at least K
    from the ends of the range before and after (the kernel is cut off at the ends, as a CNN's at its padding).
    The output for that attribute should move by k and the others stay the same; with plus, only as long as
    the plus atoms do not change; with the zero found, as long as the shifted zero is still on the axis."""
    from neuralogic.dataset import Dataset
    data = B.load_iravenx(path, 6300, 300)
    K = max(OFFSETS)
    print(f"random weights, {what}; max |output difference| after rolling the "
          f"original output by k", flush=True)
    for k in (1, 3, -2):
        for i, a in enumerate(ATTRS):
            base, shifted = [], []
            for pz, y, c in data:
                vs = [o[a] for p in pz[:B.N_CTX] for o in p]
                if min(vs) - K + min(k, 0) < 0 or max(vs) + K + max(k, 0) > maxval:
                    continue
                base.append((pz, y, c))
                shifted.append(([[{**o, a: o[a] + k} for o in p] for p in pz], y, c))
            if not base:
                continue
            o0 = model.test(list(model.build_dataset(Dataset(samples_of(base, "b")))))
            o1 = model.test(list(model.build_dataset(Dataset(samples_of(shifted, "s")))))
            diff, ctrl = [], []
            for x0, x1 in zip(o0, o1):
                p0 = np.asarray(x0, dtype=float).reshape(len(ATTRS), maxval + 1)
                p1 = np.asarray(x1, dtype=float).reshape(len(ATTRS), maxval + 1)
                ctrl.append(np.abs(p0 - p1).max())                  # control: not rolled
                p0[i] = np.roll(p0[i], k)
                diff.append(np.abs(p0 - p1).max())
            diff = np.array(diff)
            print(f"  {a} {k:+d}: {len(base):3d} puzzles, max {diff.max():.1e}, exact (< 1e-12) in "
                  f"{int((diff < 1e-12).sum())}; not rolled (control): mean {np.mean(ctrl):.1e}", flush=True)


if __name__ == "__main__":
    main()
