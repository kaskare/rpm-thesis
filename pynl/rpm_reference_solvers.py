#!/usr/bin/env python3
"""Two non-learned symbolic solvers over GROUND-TRUTH attributes, as reference points.

primitives   A generic vocabulary of relations over a row of three values (equality,
             steps of +-1/+-2, c = a +- b + k, the same multiset as the other rows) and over a
             row of three slot sets (equality, union, difference, cyclic shift), plus slot-wise
             equality of object attributes. Generic analogy principle: every primitive that
             holds on BOTH context rows should also hold on the answer row. Nothing here names a
             RAVEN rule.

rules        RAVEN's four rule families with their exact semantics in this dataset (measured
             against the XML annotations): Constant, Progression(+-1/+-2), Arithmetic
             (add/sub; Size with a +-1 index offset; Position as union/difference),
             Distribute_Three. A rule is abduced if it holds on rows 1 and 2; each candidate
             scores one point per (component, attribute) whose abduced rule it satisfies.

rules+cols   rules, with Distribute_Three also checked down the columns, as NVSA does: columns
             1 and 2 (panels 0,3,6 and 1,4,7) must hold the same three values as rows 1 and 2.
             The generator lays the three values out as a Latin square, so this never rejects
             a true Distribute_Three; it only drops rows that share a multiset by chance.

rules+cols+ms  rules+cols, with Constant also holding when every panel of the row has the same
             multiset of values. In grids whose objects differ, a Position rule (progression,
             Distribute_Three) moves the objects with their attributes, so no slot-wise match
             exists; these are exactly the Constant attributes I-RAVEN modifies in candidates
             (SRAN sampling.py, sample_attr_avail). Like the generator (SRAN const.py) and the
             PyNeuraLogic rules template, it has no Arithmetic on Type.

All report expected accuracy (ties share credit) on the validation split.

Usage:  python pynl/rpm_reference_solvers.py --split val --n 2000
"""
import argparse, os, sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path.insert(0, ROOT)
from rpm_symbolic import read_puzzle, files, CONSTELLATIONS

ATTR_IDX = {"type": 0, "size": 1, "color": 2}
STEPS = (-2, -1, 1, 2)


def scalar(vals):
    return vals[0] if vals and len(set(vals)) == 1 else None


# ── primitive vocabulary ──────────────────────────────────────────────────────
def num_prims(a, b, c):
    """generic relations over a row of three numbers (None = undefined)."""
    if None in (a, b, c):
        return set()
    p = set()
    if a == b: p.add("a=b")
    if b == c: p.add("b=c")
    if a == c: p.add("a=c")
    for k in STEPS:
        if b - a == k: p.add(f"b=a{k:+d}")
        if c - b == k: p.add(f"c=b{k:+d}")
    for k in (-1, 0, 1):
        if c == a + b + k: p.add(f"c=a+b{k:+d}")
        if c == a - b + k: p.add(f"c=a-b{k:+d}")
    return p


def set_prims(A, B, C, n):
    p = set()
    if A == B: p.add("A=B")
    if B == C: p.add("B=C")
    if C == (A | B): p.add("C=AuB")
    if C == (A - B): p.add("C=A-B")
    if C == (A & B): p.add("C=AnB")
    for k in STEPS:
        sh = lambda S: frozenset((x + k) % n for x in S)
        if B == sh(A): p.add(f"B=shift(A,{k:+d})")
        if C == sh(B): p.add(f"C=shift(B,{k:+d})")
    return p


def slot_prims(row, ci):
    """attribute equal at every slot occupied in all three panels of the row."""
    shared = set.intersection(*[set(p[ci]["slots"]) for p in row])
    if not shared:
        return set()
    p = set()
    for a, i in ATTR_IDX.items():
        if all(len({q[ci]["slots"][s][i] for q in row}) == 1 for s in shared):
            p.add(f"slotwise_equal_{a}")
    return p


def row_uniform(row, ci, attr):
    """every object in all three panels of the row has the same value of attr."""
    vals = {v for q in row for v in q[ci][attr]}
    return len(vals) == 1


def row_prims(row, ci):
    p = {f"row_uniform_{a}" for a in ATTR_IDX if row_uniform(row, ci, a)}
    for a in ATTR_IDX:
        p |= {f"{a}:{r}" for r in num_prims(*[scalar(q[ci][a]) for q in row])}
    p |= {f"number:{r}" for r in num_prims(*[q[ci]["number"] for q in row])}
    p |= {f"position:{r}" for r in set_prims(*[q[ci]["position"] for q in row], row[0][ci]["nslots"])}
    p |= slot_prims(row, ci)
    return p


def multiset_key(row, ci, view):
    if view == "position":
        vals = [q[ci]["position"] for q in row]
        return tuple(sorted(vals, key=lambda s: sorted(s)))
    if view == "number":
        return tuple(sorted(q[ci]["number"] for q in row))
    vals = [scalar(q[ci][view]) for q in row]
    return None if None in vals else tuple(sorted(vals))


def solve_primitives(panels):
    ncomp = len(panels[0])
    r1, r2 = panels[0:3], panels[3:6]
    scores = np.zeros(8)
    for ci in range(ncomp):
        common = row_prims(r1, ci) & row_prims(r2, ci)
        views = ("type", "size", "color", "number", "position")
        same_set = {v for v in views if multiset_key(r1, ci, v) is not None and
                    multiset_key(r1, ci, v) == multiset_key(r2, ci, v)}
        for c in range(8):
            r3 = (panels[6], panels[7], panels[8 + c])
            scores[c] += len(common & row_prims(r3, ci))
            scores[c] += sum(multiset_key(r3, ci, v) == multiset_key(r1, ci, v) for v in same_set)
    return scores


# ── rule families ─────────────────────────────────────────────────────────────
def entity_rule_holds(rule, attr, row, ci, multiset=False):
    vals = [scalar(q[ci][attr]) for q in row]
    if rule == "constant":                        # one value across every object in the row,
        if row_uniform(row, ci, attr):            # or (fixed layout) the same value slot by slot
            return True
        if multiset and row[0][ci][attr] and row[0][ci][attr] == row[1][ci][attr] == row[2][ci][attr]:
            return True                           # or the same multiset in every panel
        return slot_prims(row, ci) >= {f"slotwise_equal_{attr}"} and \
               all(q[ci]["position"] == row[0][ci]["position"] for q in row)
    if None in vals:
        return False
    a, b, c = vals
    if rule.startswith("prog"):
        k = int(rule[4:]); return b == a + k and c == b + k
    off = 1 if attr == "size" else 0
    if rule == "add": return c == a + b + off
    if rule == "sub": return c == a - b - off
    return False


def number_rule_holds(rule, row, ci):
    a, b, c = [q[ci]["number"] for q in row]
    if rule == "constant": return a == b == c
    if rule.startswith("prog"):
        k = int(rule[4:]); return b == a + k and c == b + k
    if rule == "add": return c == a + b
    if rule == "sub": return c == a - b
    return False


def position_rule_holds(rule, row, ci):
    A, B, C = [q[ci]["position"] for q in row]
    n = row[0][ci]["nslots"]
    if rule == "constant": return A == B == C
    if rule.startswith("shift"):
        k = int(rule[5:]); sh = lambda S: frozenset((x + k) % n for x in S)
        return B == sh(A) and C == sh(B)
    if rule == "union": return C == (A | B)
    if rule == "diff": return C == (A - B)
    return False


ENTITY_RULES = ["constant"] + [f"prog{k:+d}" for k in STEPS] + ["add", "sub"]
NUMBER_RULES = ["constant"] + [f"prog{k:+d}" for k in STEPS] + ["add", "sub"]
POSITION_RULES = ["constant"] + [f"shift{k:+d}" for k in STEPS] + ["union", "diff"]


def solve_rules(panels, columns=False, multiset=False):
    ncomp = len(panels[0])
    r1, r2 = panels[0:3], panels[3:6]
    c1, c2 = panels[0:7:3], panels[1:8:3]
    scores = np.zeros(8)
    for ci in range(ncomp):
        groups = [(a, lambda rule, row, a=a: entity_rule_holds(rule, a, row, ci, multiset),
                   [r for r in ENTITY_RULES if not (multiset and a == "type" and r in ("add", "sub"))])
                  for a in ATTR_IDX]
        groups.append(("number", lambda rule, row: number_rule_holds(rule, row, ci), NUMBER_RULES))
        groups.append(("position", lambda rule, row: position_rule_holds(rule, row, ci), POSITION_RULES))
        for view, holds, rules in groups:
            active = [r for r in rules if holds(r, r1) and holds(r, r2)]
            dist3 = multiset_key(r1, ci, view) is not None and \
                    multiset_key(r1, ci, view) == multiset_key(r2, ci, view) and \
                    len(set(multiset_key(r1, ci, view))) == 3
            if columns:
                dist3 = dist3 and multiset_key(c1, ci, view) == multiset_key(r1, ci, view) == \
                                  multiset_key(c2, ci, view)
            if not active and not dist3:
                continue                     # nothing abduced: this attribute is noise here
            for c in range(8):
                r3 = (panels[6], panels[7], panels[8 + c])
                ok = any(holds(r, r3) for r in active) or \
                     (dist3 and multiset_key(r3, ci, view) == multiset_key(r1, ci, view))
                scores[c] += ok
    return scores


# ── evaluation ────────────────────────────────────────────────────────────────
SOLVERS = (("primitives", solve_primitives), ("rules", solve_rules),
           ("rules+cols", lambda p: solve_rules(p, columns=True)),
           ("rules+cols+ms", lambda p: solve_rules(p, columns=True, multiset=True)))


def _eval(job):
    cfg, split, n, root = job
    res = Counter()
    for f in files(cfg, split, n, root):
        y = int(np.load(f)["target"])
        panels, _ = read_puzzle(f.replace(".npz", ".xml"), cfg)
        for name, solver in SOLVERS:
            s = solver(panels)
            top = np.flatnonzero(s == s.max())
            res[name] += (1.0 / len(top)) if y in top else 0.0
            res[name + "_strict"] += len(top) == 1 and top[0] == y
        res["n"] += 1
    return cfg, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--root", default="data/I-RAVEN-10000", help="dataset folder, relative to the repo root")
    args = ap.parse_args()
    with Pool(len(CONSTELLATIONS)) as pool:
        out = dict(pool.map(_eval, [(c, args.split, args.n, args.root) for c in CONSTELLATIONS]))
    print(f"Reference symbolic solvers on GROUND-TRUTH attributes, {args.split} split, up to {args.n}/constellation")
    print("(expected accuracy, ties share credit; strict = unique correct argmax)\n")
    names = [name for name, _ in SOLVERS]
    line = lambda r, n: "".join(f" {100*r[k]/n:>10.2f}% {100*r[k + '_strict']/n:>6.1f}%" for k in names)
    print(f"{'constellation':<40} {'n':>5}" + "".join(f" {k:>11} {'strict':>7}" for k in names))
    tot = Counter()
    for cfg in CONSTELLATIONS:
        r = out[cfg]; tot.update(r)
        print(f"{cfg:<40} {r['n']:>5}" + line(r, r["n"]))
    print("-" * (46 + 20 * len(names)))
    print(f"{'OVERALL':<40} {tot['n']:>5}" + line(tot, tot["n"]))


if __name__ == "__main__":
    main()
