#!/usr/bin/env python3
"""Symbolic I-RAVEN solver in PyNeuraLogic: PRIMITIVES model.

Unlike train_all_constellations.py (message passing over binary edges with vector
embeddings), this template reasons over SYMBOLS:

  facts (per puzzle)     row(R, P1, P2, P3)            r1, r2 = context rows; k0..k7 = candidate rows
                         val(P, C, A, V)               panel P, component C, attribute A, value V
                                                       (type/size/color only when the panel is uniform)
                         tup(P, C, A, T)               the panel's slot->value assignment, as one constant
                         occset(P, C, M)               the set of occupied slots, as one constant
                         s_union/s_minus/s_inter(M1, M2, M3), s_shift(M1, M2, K)
                                                       set algebra over the sets present in the puzzle
  background (template)  step(X, Y, K)                 Y = X + K,       K in -2,-1,+1,+2
                         plus(X, Y, Z, K)              Z = X + Y + K,   K in -1,0,+1
                         minus(X, Y, Z, K)             Z = X - Y + K,   K in -1,0,+1
                         perm3(X, Y, Z, U, V, W)       (U,V,W) is a permutation of (X,Y,Z)

  rules derive holds(Row, C, A, Prim) for a generic vocabulary of row relations (equalities,
  steps, sums/differences with small offsets, the same multiset as row 1, slot-wise identity,
  set equality/union/difference/intersection/shift). The analogy rule
        pres(K, C, A, Prim) :- holds(r1, C, A, Prim), holds(r2, C, A, Prim), holds(K, C, A, Prim)
  keeps the relations that both context rows show and candidate row K preserves, and
        score(K) :- pres(K, C, a, prim)      one LEARNED weight per (attribute, primitive)
  scores the candidate. No RAVEN rule is named anywhere; the weights decide what matters.

Each candidate is one binary sample (sigmoid + cross-entropy); prediction = argmax over the 8.
"""
import argparse, random, json, os, sys, time
from collections import Counter, defaultdict
from itertools import permutations

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path.insert(0, ROOT)
from rpm_symbolic import read_puzzle, files, CONSTELLATIONS

ATTRS_NUM = ("type", "size", "color", "number")
STEPS = (-2, -1, 1, 2)
OFFS = (-1, 0, 1)
V = range(0, 10)

NUM_PRIMS = (["eq_ab", "eq_bc", "eq_ac", "eq_abc", "msame"]
             + [f"step_ab{k:+d}" for k in STEPS] + [f"step_bc{k:+d}" for k in STEPS]
             + [f"plus{k:+d}" for k in OFFS] + [f"minus{k:+d}" for k in OFFS])
SET_PRIMS = (["seq_ab", "seq_bc", "s_union", "s_minus", "s_inter", "msame"]
             + [f"shift_ab{k:+d}" for k in STEPS] + [f"shift_bc{k:+d}" for k in STEPS])
TUP_PRIMS = ["tup_eq"]


def kname(k):
    return f"k{'p' if k > 0 else 'm' if k < 0 else 'z'}{abs(k)}"


def scalar(vals):
    return vals[0] if vals and len(set(vals)) == 1 else None


# ── per-puzzle symbolic facts (plain tuples; turned into atoms by the caller) ──
CAND_ROWS = [f"k{k}" for k in range(8)]


def rows_of(columns=False):
    """context rows r1, r2 (and with columns, context columns c1, c2) and candidate rows k0..k7"""
    rows = {"r1": (0, 1, 2), "r2": (3, 4, 5)}
    if columns:
        rows.update({"c1": (0, 3, 6), "c2": (1, 4, 7)})
    rows.update({f"k{k}": (6, 7, 8 + k) for k in range(8)})
    return rows


def puzzle_tuples(panels, gid, columns=False, multiset=False):
    """All facts of one puzzle: context rows r1, r2 (+ columns c1, c2) and the 8 candidate rows
    k0..k7. `puzzle(gid)` binds the unique query answer(gid). With multiset, ms(P, C, A, U) names
    each panel's multiset of values of A as one constant."""
    rows = rows_of(columns)
    T = [("row", r, *(f"p{i}" for i in idx)) for r, idx in rows.items()]
    T += [("cand", r) for r in CAND_ROWS] + [("puzzle", gid)]
    for p in range(16):
        for ci, comp in enumerate(panels[p]):
            P, C = f"p{p}", f"c{ci}"
            for i, a in enumerate(("type", "size", "color")):
                v = scalar(comp[a])
                if v is not None:
                    T.append(("val", P, C, a, f"v{v}"))
                tup = "_".join(f"{sl}-{comp['slots'][sl][i]}" for sl in sorted(comp["slots"]))
                T.append(("tup", P, C, a, f"t{tup}"))
                if multiset and comp[a]:
                    T.append(("ms", P, C, a, "u" + "_".join(map(str, comp[a]))))
            T.append(("val", P, C, "number", f"v{comp['number']}"))
            T.append(("occset", P, C, setname(ci, comp["position"])))
    for ci in range(len(panels[0])):
        n = panels[0][ci]["nslots"]
        sets = {panels[p][ci]["position"] for p in range(16)}
        for A in sets:
            for B in sets:
                for Cs in sets:
                    if Cs == (A | B): T.append(("s_union", setname(ci, A), setname(ci, B), setname(ci, Cs)))
                    if Cs == (A - B): T.append(("s_minus", setname(ci, A), setname(ci, B), setname(ci, Cs)))
                    if Cs == (A & B): T.append(("s_inter", setname(ci, A), setname(ci, B), setname(ci, Cs)))
                for kk in STEPS:
                    if B == frozenset((x + kk) % n for x in A):
                        T.append(("s_shift", setname(ci, A), setname(ci, B), kname(kk)))
        triples = {tuple(setname(ci, panels[i][ci]["position"]) for i in idx) for idx in rows.values()}
        for t in triples:
            for u in triples:
                if sorted(t) == sorted(u):
                    T.append(("sperm3",) + t + u)
        for A_ in sets:
            for B_ in sets:
                if A_ != B_:
                    T.append(("sneq", setname(ci, A_), setname(ci, B_)))
    return T


def setname(ci, S):
    return f"m{ci}_{'_'.join(map(str, sorted(S)))}" if S else f"m{ci}_empty"


def background_tuples():
    B = []
    for x in V:
        for k in STEPS:
            if x + k in V:
                B.append(("step", f"v{x}", f"v{x + k}", kname(k)))
        for y in V:
            for k in OFFS:
                if x + y + k in V:
                    B.append(("plus", f"v{x}", f"v{y}", f"v{x + y + k}", kname(k)))
                if x - y + k in V:
                    B.append(("minus", f"v{x}", f"v{y}", f"v{x - y + k}", kname(k)))
    for x in V:
        for y in V:
            for z in V:
                for u in set(permutations((x, y, z))):
                    B.append(("perm3", f"v{x}", f"v{y}", f"v{z}") + tuple(f"v{w}" for w in u))
    return B


# ── the same vocabulary in plain Python: the exactness reference ───────────────
def python_counts(panels):
    """Per candidate: number of (component, attribute, primitive) relations held by
    both context rows and the candidate's row -- exactly what the template counts
    when every weight is 1."""
    rows = rows_of()
    prim = row_prim_sets(panels, rows)
    counts = np.zeros(8)
    for ci in range(len(panels[0])):
        common = prim["r1"][ci] & prim["r2"][ci]
        for k in range(8):
            counts[k] += len(common & prim[f"k{k}"][ci])
    return counts


def row_prim_sets(panels, rows):
    """{row: {component: set of (attribute, primitive)}} -- the primitive vocabulary in Python."""
    def num_holds(vals, row_vals1):
        a, b, c = vals
        h = set()
        if None in (a, b, c):
            return h
        if a == b: h.add("eq_ab")
        if b == c: h.add("eq_bc")
        if a == c: h.add("eq_ac")
        if a == b == c: h.add("eq_abc")
        for k in STEPS:
            if b == a + k: h.add(f"step_ab{k:+d}")
            if c == b + k: h.add(f"step_bc{k:+d}")
        for k in OFFS:
            if c == a + b + k and 0 <= c <= 9: h.add(f"plus{k:+d}")
            if c == a - b + k and 0 <= c <= 9: h.add(f"minus{k:+d}")
        if None not in row_vals1 and sorted(vals) == sorted(row_vals1): h.add("msame")
        return h

    def set_holds(S, S1, n):
        A, B, C = S
        h = set()
        if A == B: h.add("seq_ab")
        if B == C: h.add("seq_bc")
        if C == (A | B): h.add("s_union")
        if C == (A - B): h.add("s_minus")
        if C == (A & B): h.add("s_inter")
        for k in STEPS:
            if B == frozenset((x + k) % n for x in A): h.add(f"shift_ab{k:+d}")
            if C == frozenset((x + k) % n for x in B): h.add(f"shift_bc{k:+d}")
        if sorted(map(sorted, S)) == sorted(map(sorted, S1)): h.add("msame")
        return h

    out = {r: {} for r in rows}
    for ci in range(len(panels[0])):
        n = panels[0][ci]["nslots"]
        for r, idx in rows.items():
            hs = set()
            for a in ATTRS_NUM:
                get = (lambda i: panels[i][ci]["number"]) if a == "number" else \
                      (lambda i, a=a: scalar(panels[i][ci][a]))
                v = [get(i) for i in idx]
                v1 = [get(i) for i in rows["r1"]]
                hs |= {(a, p) for p in num_holds(v, v1)}
            for a in ("type", "size", "color"):
                k_ = ("type", "size", "color").index(a)
                tu = [tuple(sorted((s, panels[i][ci]["slots"][s][k_]) for s in panels[i][ci]["slots"])) for i in idx]
                if tu[0] == tu[1] == tu[2]:
                    hs.add((a, "tup_eq"))
            S = [panels[i][ci]["position"] for i in idx]
            S1 = [panels[i][ci]["position"] for i in rows["r1"]]
            hs |= {("position", p) for p in set_holds(S, S1, n)}
            out[r][ci] = hs
    return out


RULES_ENTITY = ["const"] + [f"prog{k:+d}" for k in STEPS] + ["add", "sub", "dist3"]
RULES_POSITION = ["const"] + [f"shift{k:+d}" for k in STEPS] + ["union", "diff", "dist3"]
ADD_PRIM = {"size": "plus+1", "color": "plus+0", "number": "plus+0"}     # hand-coded semantics
SUB_PRIM = {"size": "minus-1", "color": "minus+0", "number": "minus+0"}


def python_rule_counts(panels, columns=False, multiset=False):
    """Per candidate: number of (component, attribute, rule) where the rule holds on context
    rows 1 and 2 and on the candidate's row -- exactly what the rules template counts with
    unit weights. Rules are defined as compositions of the primitives. With columns,
    Distribute_Three must also hold on columns 1 and 2 (msame is relative to row 1, so
    both columns hold row 1's three values, as in the generator's Latin square). With multiset,
    Constant also holds when the row's three panels have the same multiset of values."""
    rows = rows_of(columns)
    prim = row_prim_sets(panels, rows)
    counts = np.zeros(8)
    for ci in range(len(panels[0])):
        RH = {}
        for r, idx in rows.items():
            P = prim[r][ci]
            rh = set()
            for a in ATTRS_NUM:
                get = (lambda i: panels[i][ci]["number"]) if a == "number" else \
                      (lambda i, a=a: scalar(panels[i][ci][a]))
                vals = [get(i) for i in idx]
                if (a, "eq_abc") in P or (a, "tup_eq") in P: rh.add((a, "const"))
                if multiset and a != "number" and panels[idx[0]][ci][a] and \
                        all(panels[i][ci][a] == panels[idx[0]][ci][a] for i in idx):
                    rh.add((a, "const"))
                for k in STEPS:
                    if (a, f"step_ab{k:+d}") in P and (a, f"step_bc{k:+d}") in P: rh.add((a, f"prog{k:+d}"))
                if a in ADD_PRIM and (a, ADD_PRIM[a]) in P: rh.add((a, "add"))
                if a in SUB_PRIM and (a, SUB_PRIM[a]) in P: rh.add((a, "sub"))
                if (a, "msame") in P and None not in vals and len(set(vals)) == 3: rh.add((a, "dist3"))
            sets = [panels[i][ci]["position"] for i in idx]
            if ("position", "seq_ab") in P and ("position", "seq_bc") in P: rh.add(("position", "const"))
            for k in STEPS:
                if ("position", f"shift_ab{k:+d}") in P and ("position", f"shift_bc{k:+d}") in P:
                    rh.add(("position", f"shift{k:+d}"))
            if ("position", "s_union") in P: rh.add(("position", "union"))
            if ("position", "s_minus") in P: rh.add(("position", "diff"))
            if ("position", "msame") in P and len(set(sets)) == 3: rh.add(("position", "dist3"))
            RH[r] = rh
        common = RH["r1"] & RH["r2"]
        if columns:
            common = {ar for ar in common if ar[1] != "dist3" or ar in RH["c1"] and ar in RH["c2"]}
        for k in range(8):
            counts[k] += len(common & RH[f"k{k}"])
    return counts


# ── the PyNeuraLogic template ─────────────────────────────────────────────────
INIT_SEED = None      # set by --init-seed: rule weights start uniform in [0, 0.2] instead of all at 0.1


def build_template(fixed_unit_weights=False, scoring="prims", columns=False, multiset=False):
    from neuralogic.core import R, V as Var, Transformation, Aggregation, Metadata, Combination
    from neuralogic.core import Model
    m = Model()
    for t in background_tuples():
        m += R.get(t[0])(*t[1:])
    X, Y, Z, U, W2, W3 = Var.X, Var.Y, Var.Z, Var.U, Var.W2, Var.W3
    Rw, P1, P2, P3, Cc, A = Var.Rw, Var.P1, Var.P2, Var.P3, Var.Cc, Var.A
    Q1, Q2, Q3, K = Var.Q1, Var.Q2, Var.Q3, Var.K
    row = R.row(Rw, P1, P2, P3)
    val = lambda P, v: R.val(P, Cc, A, v)

    crisp = Metadata(combination=Combination.MIN, transformation=Transformation.IDENTITY,
                     aggregation=Aggregation.MAX)

    def H(prim, *body):
        nonlocal m
        m += (R.holds(Rw, Cc, A, prim) <= (row,) + body) | crisp

    # numeric primitives, generic over the attribute A
    H("eq_ab", val(P1, X), val(P2, X))
    H("eq_bc", val(P2, Y), val(P3, Y))
    H("eq_ac", val(P1, X), val(P3, X))
    H("eq_abc", val(P1, X), val(P2, X), val(P3, X))
    for k in STEPS:
        H(f"step_ab{k:+d}", val(P1, X), val(P2, Y), R.step(X, Y, kname(k)))
        H(f"step_bc{k:+d}", val(P2, Y), val(P3, Z), R.step(Y, Z, kname(k)))
    for k in OFFS:
        H(f"plus{k:+d}", val(P1, X), val(P2, Y), val(P3, Z), R.plus(X, Y, Z, kname(k)))
        H(f"minus{k:+d}", val(P1, X), val(P2, Y), val(P3, Z), R.minus(X, Y, Z, kname(k)))
    # the same multiset of values as context row 1 (across rows: a 6-ary relation)
    m += (R.holds(Rw, Cc, A, "msame") <= (row, R.row("r1", Q1, Q2, Q3),
                                          val(P1, X), val(P2, Y), val(P3, Z),
                                          val(Q1, U), val(Q2, W2), val(Q3, W3),
                                          R.perm3(U, W2, W3, X, Y, Z))) | crisp
    # slot-wise identity of an object attribute
    m += (R.holds(Rw, Cc, A, "tup_eq") <= (row, R.tup(P1, Cc, A, X), R.tup(P2, Cc, A, X), R.tup(P3, Cc, A, X))) | crisp
    # set primitives over occupied-slot sets (attribute constant 'position')
    occ = lambda P, s: R.occset(P, Cc, s)

    def S(prim, *body):
        nonlocal m
        m += (R.holds(Rw, Cc, "position", prim) <= (row,) + body) | crisp

    S("seq_ab", occ(P1, X), occ(P2, X))
    S("seq_bc", occ(P2, Y), occ(P3, Y))
    S("s_union", occ(P1, X), occ(P2, Y), occ(P3, Z), R.s_union(X, Y, Z))
    S("s_minus", occ(P1, X), occ(P2, Y), occ(P3, Z), R.s_minus(X, Y, Z))
    S("s_inter", occ(P1, X), occ(P2, Y), occ(P3, Z), R.s_inter(X, Y, Z))
    for k in STEPS:
        S(f"shift_ab{k:+d}", occ(P1, X), occ(P2, Y), R.s_shift(X, Y, kname(k)))
        S(f"shift_bc{k:+d}", occ(P2, Y), occ(P3, Z), R.s_shift(Y, Z, kname(k)))
    m += (R.holds(Rw, Cc, "position", "msame") <= (row, R.row("r1", Q1, Q2, Q3),
                                                   occ(P1, X), occ(P2, Y), occ(P3, Z),
                                                   occ(Q1, U), occ(Q2, W2), occ(Q3, W3),
                                                   R.sperm3(U, W2, W3, X, Y, Z))) | crisp
    # every holds/1 atom is crisp: 1 if derivable (MAX over duplicate derivations)
    m += (R.holds / 4) | Metadata(transformation=Transformation.IDENTITY)
    # trainable weights start small (0.1): summed evidence reaches 30-40 in two-component puzzles,
    # and small logits keep the initial softmax soft enough to give useful gradients
    wt = (lambda atom: atom[1.0].fixed()) if fixed_unit_weights else (lambda atom: atom[0.1])
    if INIT_SEED is not None and not fixed_unit_weights:
        rng = random.Random(INIT_SEED)
        wt = lambda atom: atom[rng.uniform(0.0, 0.2)]
    lin = Metadata(transformation=Transformation.IDENTITY, aggregation=Aggregation.SUM)

    if scoring == "prims":
        # analogy: a relation both context rows show and the candidate row preserves
        Pr = Var.Pr
        m += (R.pres(K, Cc, A, Pr) <= (R.holds("r1", Cc, A, Pr), R.holds("r2", Cc, A, Pr),
                                       R.holds(K, Cc, A, Pr), R.cand(K))) | crisp
        m += (R.pres / 4) | Metadata(transformation=Transformation.IDENTITY)
        # one weight per (attribute, primitive), summed over components
        for a in ATTRS_NUM:
            for p in NUM_PRIMS:
                m += (R.score(K) <= wt(R.pres(K, Cc, a, p))) | lin
        for a in ("type", "size", "color"):
            m += (R.score(K) <= wt(R.pres(K, Cc, a, "tup_eq"))) | lin
        for p in SET_PRIMS:
            m += (R.score(K) <= wt(R.pres(K, Cc, "position", p))) | lin

    else:  # scoring == "rules": RAVEN's rule families, hand-coded as compositions of primitives
        for a, pk in ADD_PRIM.items():
            m += R.add_prim(a, pk)
        for a, pk in SUB_PRIM.items():
            m += R.sub_prim(a, pk)
        for x in V:
            for y in V:
                if x != y:
                    m += R.neq(f"v{x}", f"v{y}")
        Ru, Pk = Var.Ru, Var.Pk
        hold = lambda prim, a=A: R.holds(Rw, Cc, a, prim)

        def RH(rule, *body, a=A):
            nonlocal m
            m += (R.rh(Rw, Cc, a, rule) <= body) | crisp

        RH("const", hold("eq_abc"))
        RH("const", hold("tup_eq"))
        if multiset:      # the same multiset of values in all three panels (objects moved with them)
            RH("const", row, R.ms(P1, Cc, A, X), R.ms(P2, Cc, A, X), R.ms(P3, Cc, A, X))
        for k in STEPS:
            RH(f"prog{k:+d}", hold(f"step_ab{k:+d}"), hold(f"step_bc{k:+d}"))
        RH("add", R.holds(Rw, Cc, A, Pk), R.add_prim(A, Pk))
        RH("sub", R.holds(Rw, Cc, A, Pk), R.sub_prim(A, Pk))
        RH("dist3", hold("msame"), row, val(P1, X), val(P2, Y), val(P3, Z),
           R.neq(X, Y), R.neq(Y, Z), R.neq(X, Z))
        pos = "position"
        RH("const", hold("seq_ab", pos), hold("seq_bc", pos), a=pos)
        for k in STEPS:
            RH(f"shift{k:+d}", hold(f"shift_ab{k:+d}", pos), hold(f"shift_bc{k:+d}", pos), a=pos)
        RH("union", hold("s_union", pos), a=pos)
        RH("diff", hold("s_minus", pos), a=pos)
        RH("dist3", hold("msame", pos), row, occ(P1, X), occ(P2, Y), occ(P3, Z),
           R.sneq(X, Y), R.sneq(Y, Z), R.sneq(X, Z), a=pos)
        m += (R.rh / 4) | Metadata(transformation=Transformation.IDENTITY)
        # abduction: a rule shown by BOTH context rows, checked on the candidate's row
        ctx = (R.rh("r1", Cc, A, Ru), R.rh("r2", Cc, A, Ru), R.rh(K, Cc, A, Ru), R.cand(K))
        if not columns:
            m += (R.sat(K, Cc, A, Ru) <= ctx) | crisp
        else:   # Distribute_Three must also be shown by context columns 1 and 2
            for rule in sorted(set(RULES_ENTITY + RULES_POSITION) - {"dist3"}):
                m += R.rowrule(rule)
            m += (R.sat(K, Cc, A, Ru) <= ctx + (R.rowrule(Ru),)) | crisp
            m += (R.sat(K, Cc, A, "dist3") <= (R.rh("r1", Cc, A, "dist3"), R.rh("r2", Cc, A, "dist3"),
                                               R.rh("c1", Cc, A, "dist3"), R.rh("c2", Cc, A, "dist3"),
                                               R.rh(K, Cc, A, "dist3"), R.cand(K))) | crisp
        m += (R.sat / 4) | Metadata(transformation=Transformation.IDENTITY)
        # one weight per (attribute, rule family), summed over components
        for a in ATTRS_NUM:
            for rule in RULES_ENTITY:
                m += (R.score(K) <= wt(R.sat(K, Cc, a, rule))) | lin
        for rule in RULES_POSITION:
            m += (R.score(K) <= wt(R.sat(K, Cc, "position", rule))) | lin

    # CONCAT-order probe: fires only in the synthetic probe sample (the only one with cidx facts)
    m += (R.score(K) <= R.cidx(K)[1.0].fixed()) | Metadata(transformation=Transformation.IDENTITY)
    # bias: also guarantees score(K) is derivable -- an underivable query is silently dropped
    bias = R.cand(K)[0.0].fixed() if fixed_unit_weights else R.cand(K)[0.0]
    m += (R.score(K) <= bias) | Metadata(transformation=Transformation.IDENTITY)
    m += (R.score / 1) | Metadata(transformation=Transformation.IDENTITY)      # logits
    G = Var.G
    # PRODUCT: the binding atom puzzle(G) (value 1) must not be ADDED to each candidate's score
    m += (R.answer(G) <= (R.puzzle(G), R.score(K))) | Metadata(aggregation=Aggregation.CONCAT,
                                                              combination=Combination.PRODUCT,
                                                              transformation=Transformation.IDENTITY)
    m += (R.answer / 1) | Metadata(transformation=Transformation.IDENTITY)
    return m


def probe_sample(R, Sample):
    """Synthetic sample in which candidate k scores exactly 10+k. Built in the SAME dataset as
    the real puzzles: CONCAT order is consistent within a model but not across models."""
    facts = [R.puzzle("probe")] + [R.cand(f"k{k}") for k in range(8)] + \
            [R.cidx(f"k{k}")[float(10 + k)] for k in range(8)]
    return Sample(R.answer("probe")[[0.0] * 8], facts)


def read_order(model, built_probe):
    out = np.asarray(list(model.test([built_probe]))[0], dtype=float).flatten()
    order = [int(round(v)) - 10 for v in out]       # order[j] = candidate at output position j
    assert sorted(order) == list(range(8)), f"cannot recover CONCAT order: {out}"
    return order


# ── samples, check, training ──────────────────────────────────────────────────
def make_sample(panels, y, gid, order, R, Sample, columns=False, multiset=False):
    """One sample per puzzle: query answer(gid) = one-hot of the answer, in CONCAT order."""
    target = [1.0 if order[j] == y else 0.0 for j in range(8)]
    facts = [R.get(t[0])(*t[1:]) for t in puzzle_tuples(panels, gid, columns, multiset)]
    return Sample(R.answer(gid)[target], facts)


def unpermute(vec, order):
    """CONCAT-order vector -> scores for candidates 0..7."""
    s = np.zeros(8)
    for j, k in enumerate(order):
        s[k] = vec[j]
    return s


def load_split(cfgs, split, n, start=0, root="data/I-RAVEN-10000"):
    data = []
    for cfg in cfgs:
        for f in files(cfg, split, root=root)[start:start + n]:
            panels, _ = read_puzzle(f.replace(".npz", ".xml"), cfg)
            data.append((cfg, f, panels, int(np.load(f)["target"])))
    return data


def accuracy(scores, data):
    per = defaultdict(lambda: [0.0, 0])
    for (cfg, _, _, y), s in zip(data, scores):
        top = np.flatnonzero(s == s.max())
        per[cfg][0] += (1.0 / len(top)) if y in top else 0.0
        per[cfg][1] += 1
    tot = sum(v[0] for v in per.values()) / sum(v[1] for v in per.values())
    return tot, {k: v[0] / v[1] for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "train"])
    ap.add_argument("--model", default="prims", choices=list(OPTIONS),
                    help="prims: generic primitives + analogy; rules: RAVEN's rule families; "
                         "rules_cols: rules with Distribute_Three also checked down the columns; "
                         "rules_cols_ms: rules_cols with Constant also as the same multiset per panel")
    ap.add_argument("--n-train", type=int, default=100, help="puzzles per constellation")
    ap.add_argument("--n-val", type=int, default=100, help="unseen validation puzzles per constellation")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--constellations", default="all")
    ap.add_argument("--jvm-gb", type=int, default=30)
    ap.add_argument("--save-dir", default=os.path.join(ROOT, "experiments/pynl_symbolic_prims"))
    ap.add_argument("--init-seed", type=int, default=None,
                    help="random initial rule weights (uniform 0..0.2) instead of all 0.1")
    ap.add_argument("--root", default="data/I-RAVEN-10000", help="dataset folder, relative to the repo root")
    args = ap.parse_args()
    opt = OPTIONS[args.model]
    global INIT_SEED
    INIT_SEED = args.init_seed

    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn import Adam, CrossEntropy

    cfgs = CONSTELLATIONS if args.constellations == "all" else args.constellations.split(",")
    t0 = time.time()
    order = None                                    # set from the in-model probe below

    def scores_of(model, built):
        return [unpermute(np.asarray(o, dtype=float).flatten(), order) for o in model.test(built)]

    if args.mode == "check":
        # weights fixed at 1, identity output: the template must count exactly what Python counts
        data = load_split(cfgs, "val", args.n_val, start=300, root=args.root)
        model = TEMPLATES[args.model](fixed_unit_weights=True)
        model.build(Settings(optimizer=Adam(lr=0.0), error_function=CrossEntropy(with_logits=True),
                             iso_value_compression=False))
        # targets are irrelevant in check mode, so samples can be built before the order is known
        samples = [make_sample(p_, y, f"g{i}", list(range(8)), R, Sample, **opt) for i, (_, _, p_, y) in enumerate(data)]
        built = list(model.build_dataset(Dataset([probe_sample(R, Sample)] + samples)))
        order = read_order(model, built[0])
        print(f"CONCAT order (output position -> candidate): {order}", flush=True)
        out = scores_of(model, built[1:])
        assert len(out) == len(samples), f"{len(out)} outputs for {len(samples)} samples"
        ref = np.stack([REFERENCE[args.model](p_) for _, _, p_, _ in data])
        diff = np.abs(np.stack(out) - ref)
        print(f"{len(data)} puzzles in {time.time() - t0:.0f}s")
        print(f"template score == Python count on {int((diff < 1e-6).sum())}/{diff.size} candidates;"
              f" max |diff| = {diff.max():.3g}")
        acc, per = accuracy(out, data)
        print(f"unweighted accuracy (ties shared): {100 * acc:.2f}%  " +
              "  ".join(f"{k[:10]} {100 * v:.0f}%" for k, v in per.items()))
        return

    # ---- train: softmax cross-entropy over the 8 candidates ----
    os.makedirs(args.save_dir, exist_ok=True)
    tr = load_split(cfgs, "train", args.n_train, root=args.root)
    va = load_split(cfgs, "val", args.n_val, start=300, root=args.root)   # unseen; never used to pick anything else
    model = TEMPLATES[args.model]()
    # iso_value_compression MUST be off: it merges neurons whose initial VALUES coincide, and with
    # every weight starting equal, a candidate's score can equal a constant fact's 1.0 -- that
    # candidate's score is then frozen for the whole of training (found the hard way: training
    # accuracy collapsed from 98% to 62%)
    model.build(Settings(optimizer=Adam(lr=args.lr), error_function=CrossEntropy(with_logits=True),
                         iso_value_compression=False))
    # the one-hot targets depend on the CONCAT order, which is only known after building; build
    # with placeholder targets, read the order from the probe, then set each sample's target
    s_tr = [make_sample(p_, y, f"t{i}", list(range(8)), R, Sample, **opt) for i, (_, _, p_, y) in enumerate(tr)]
    s_va = [make_sample(p_, y, f"v{i}", list(range(8)), R, Sample, **opt) for i, (_, _, p_, y) in enumerate(va)]
    built = list(model.build_dataset(Dataset([probe_sample(R, Sample)] + s_tr + s_va)))  # only once
    order = read_order(model, built[0])
    print(f"CONCAT order (output position -> candidate): {order}", flush=True)
    built = built[1:]
    from neuralogic.core.constructs.java_objects import ValueFactory
    vf = ValueFactory()
    for b, (_, _, _, y) in zip(built, tr + va):          # `target` is read-only in Python; set it in Java
        b._java_sample.target = vf.get_value([1.0 if order[j] == y else 0.0 for j in range(8)])[1]
    b_tr, b_va = built[:len(s_tr)], built[len(s_tr):]
    assert len(b_tr) == len(s_tr) and len(b_va) == len(s_va)
    print(f"{len(tr)} train / {len(va)} val puzzles grounded in {time.time() - t0:.0f}s", flush=True)

    acc0, per0 = accuracy(scores_of(model, b_va), va)
    print(f"epoch   0  (initial weights)  val {100 * acc0:.2f}%  " +
          "  ".join(f"{k[:6]} {100 * v:.0f}" for k, v in per0.items()), flush=True)
    best, hist = acc0, [{"epoch": 0, "val_acc": acc0, "per": per0}]
    for ep in range(1, args.epochs + 1):
        res = model.train(b_tr, epochs=1)
        loss = float(np.mean([float(np.asarray(r[2]).flatten()[0]) for r in res])) if res else float("nan")
        acc, per = accuracy(scores_of(model, b_va), va)
        hist.append({"epoch": ep, "loss": loss, "val_acc": acc, "per": per})
        if acc > best:
            best = acc
            model.save(os.path.join(args.save_dir, "template_best.bin"))
            json.dump({"epoch": ep, "val_acc": acc, "per_constellation": per, "args": vars(args),
                       "concat_order": order}, open(os.path.join(args.save_dir, "best.json"), "w"), indent=1)
        print(f"epoch {ep:>3}  loss {loss:.4f}  val {100 * acc:.2f}%  best {100 * best:.2f}%  " +
              "  ".join(f"{k[:6]} {100 * v:.0f}" for k, v in per.items()), flush=True)
    json.dump(hist, open(os.path.join(args.save_dir, "history.json"), "w"), indent=1)


# rules variants: Distribute_Three also down columns 1 and 2 (as NVSA); Constant also as the same
# multiset of values in each panel. Samples and template of a model take the same options.
OPTIONS = {"prims": {}, "rules": {}, "rules_cols": {"columns": True},
           "rules_cols_ms": {"columns": True, "multiset": True}}
TEMPLATES = {"prims": lambda **kw: build_template(scoring="prims", **kw),
             **{m: lambda m=m, **kw: build_template(scoring="rules", **OPTIONS[m], **kw)
                for m in ("rules", "rules_cols", "rules_cols_ms")}}
REFERENCE = {"prims": python_counts,
             **{m: lambda p, m=m: python_rule_counts(p, **OPTIONS[m]) for m in ("rules", "rules_cols", "rules_cols_ms")}}


if __name__ == "__main__":
    main()
