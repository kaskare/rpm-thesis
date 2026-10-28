#!/usr/bin/env python3
"""Which symmetries do the I-RAVEN and I-RAVEN-X data have? (notes §22)

A transformation g of a complete matrix (answer filled in) is
  - a task symmetry for a rule family if every rule instance of that family still holds after g, with some
    parameter (the answer then moves with g);
  - a data symmetry if, in addition, the generator produces g.x as often as x.
Part 1 prints, per g and rule family, the share of rule instances that survive g (the task question).
Part 2 prints the generator statistics that decide the data question, and the structure of the candidate sets.
Transformations that move the missing cell (column reversal) are only meaningful on complete matrices.

    python3 pynl/dataset_symmetry_check.py [--n 1000]
"""
import argparse, functools, glob, json, os, sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rpm_symbolic import read_puzzle, files, CONSTELLATIONS
from gnn.data import canonical_positions

FAMILIES = ("Constant", "Progression", "Arithmetic", "Distribute_Three")
STEPS = (-2, -1, 1, 2)
VMAX = {"type": 4, "size": 5, "color": 9}               # I-RAVEN levels (type: triangle .. circle = 0 .. 4)


# ── rule families on a 3 x k matrix of values (rows = rows of the puzzle) ───────────────────────────────
def holds(fam, M, kind="value", nslots=None, off=0, any_step=False):
    """kind: "value" (numbers) or "set" (position sets over nslots slots). off: Size's +1 in I-RAVEN."""
    k = len(M[0])
    if fam == "Constant":
        return all(len(set(r)) == 1 for r in M)
    if fam == "Progression":
        if kind == "set":
            sh = lambda S, d: frozenset((x + d) % nslots for x in S)
            return any(all(r[j + 1] == sh(r[j], d) for r in M for j in range(k - 1)) for d in STEPS)
        return any(all(r[j + 1] - r[j] == d for r in M for j in range(k - 1)) for d in STEPS)
    if fam == "Arithmetic":
        if kind == "set":
            return all(r[2] == r[0] | r[1] for r in M) or all(r[2] == r[0] - r[1] for r in M)
        if k == 3:
            return all(r[2] == r[0] + r[1] + off for r in M) or all(r[2] == r[0] - r[1] - off for r in M)
        return all(r[-1] == sum(r[:-1]) for r in M) or all(r[0] == sum(r[1:]) for r in M)
    if fam == "Distribute_Three":                       # rows are cyclic shifts of row 1, one step per row
        if len(set(M[0])) != k:
            return False
        roll = lambda r, s: tuple(r[(j - s) % k] for j in range(k))
        steps = range(1, k) if any_step else (1, -1)
        return any(all(tuple(M[i]) == roll(tuple(M[0]), i * s) for i in range(len(M))) for s in steps)
    raise ValueError(fam)


def permute(M, rows=None, cols=None):
    rows = rows or range(len(M))
    cols = cols or range(len(M[0]))
    return [[M[r][c] for c in cols] for r in rows]


# ── I-RAVEN ───────────────────────────────────────────────────────────────────────────────────────────
RULE_NAMES = ["Constant", "Progression_One", "Progression_Mone", "Progression_Two", "Progression_Mtwo",
              "Arithmetic_Plus", "Arithmetic_Minus", "Distribute_Three_Left", "Distribute_Three_Right"]
POSNUM_NAMES = ["Constant", "Progression_One_Pos", "Progression_Mone_Pos", "Progression_One_Num",
                "Progression_Mone_Num", "Progression_Two_Pos", "Progression_Mtwo_Pos", "Progression_Two_Num",
                "Progression_Mtwo_Num", "Arithmetic_Plus_Pos", "Arithmetic_Minus_Pos", "Arithmetic_Plus_Num",
                "Arithmetic_Minus_Num", "Distribute_Three_Left_Pos", "Distribute_Three_Right_Pos",
                "Distribute_Three_Left_Num", "Distribute_Three_Right_Num"]


@functools.lru_cache(None)
def slot_maps(cfg, ci, nslots):
    """the 8 symmetries of the square, as permutations of the component's slots (grid layouts only)."""
    canon = canonical_positions(cfg, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "data/I-RAVEN-10000"))
    xy = {k: (b[0], b[1]) for (c, b), k in canon.items() if c == ci}
    k0 = min(xy)
    xy = {k - k0: v for k, v in xy.items()}
    cy = np.mean([v[0] for v in xy.values()]); cx = np.mean([v[1] for v in xy.values()])
    ops = {"slots L-R": lambda y, x: (y, -x), "slots U-D": lambda y, x: (-y, x), "slots 180": lambda y, x: (-y, -x),
           "slots 90": lambda y, x: (x, -y), "slots diag": lambda y, x: (x, y)}
    maps = {}
    for name, op in ops.items():
        img = {k: op(y - cy, x - cx) for k, (y, x) in xy.items()}
        maps[name] = {k: min(xy, key=lambda j: (xy[j][0] - cy - img[k][0]) ** 2 + (xy[j][1] - cx - img[k][1]) ** 2)
                      for k in xy}
    return maps


def iraven(n):
    surv = defaultdict(Counter)                         # (transform, family, view kind) -> Counter(ok, n)
    params = Counter()
    white = Counter()                                   # Arithmetic-Color: C12 == 0 vs C22 == 0
    operands = defaultdict(list)                        # Arithmetic plus: (attr, column) -> values
    comp_stats = defaultdict(Counter)
    cand = Counter()
    targets = Counter()
    for cfg in CONSTELLATIONS:
        for f in files(cfg, "train", n):
            y = int(np.load(f)["target"])
            targets[y] += 1
            panels, rules = read_puzzle(f.replace(".npz", ".xml"), cfg)
            full = panels[:8] + [panels[8 + y]]
            for ci, comp_rules in enumerate(rules):
                rz = np.load(f.replace(".npz", f"_rule_comp{ci}.npz"))
                for key in ("type_rule", "size_rule", "color_rule"):
                    params[(key[:-5], RULE_NAMES[int(rz[key])])] += 1
                params[("number/position", POSNUM_NAMES[int(rz["pos_num_rule"])])] += 1
                for attr, fam in comp_rules.items():
                    if fam not in FAMILIES:
                        continue
                    views = {"Number/Position": ["number", "position"], "Number": ["number"],
                             "Position": ["position"]}.get(attr, [attr.lower()])
                    for view in views:
                        vals = [full[p][ci][view] for p in range(9)]
                        if view in VMAX:
                            if any(len(set(v)) != 1 for v in vals):
                                continue                # non-uniform panel: the attribute is noise here
                            vals = [v[0] for v in vals]
                        M = [vals[0:3], vals[3:6], vals[6:9]]
                        kind = "set" if view == "position" else "value"
                        tag = f"set{full[0][ci]['nslots']}" if kind == "set" else kind
                        nsl = full[0][ci]["nslots"]
                        kw = dict(kind=kind, nslots=nsl, off=1 if view == "size" else 0)
                        if not holds(fam, M, **kw):
                            surv[("(original)", fam, tag)]["bad"] += 1
                            continue
                        comp_stats[(cfg, ci)][(view, fam)] += 1
                        if cfg in CONSTELLATIONS[3:] and view in VMAX:
                            comp_stats[(cfg, ci)][(view, "sum")] += sum(vals)
                            comp_stats[(cfg, ci)][(view, "count")] += 9
                        T = {"rows 1<>2": permute(M, rows=[1, 0, 2]), "cols 1<>2": permute(M, cols=[1, 0, 2]),
                             "panels transposed": [list(c) for c in zip(*M)],
                             "reverse cols*": permute(M, cols=[2, 1, 0])}
                        if view in VMAX:
                            T["reflect values"] = [[VMAX[view] - v for v in r] for r in M]
                        if view == "position" and nsl in (4, 9):
                            for name, g in slot_maps(cfg, ci, nsl).items():
                                T[name] = [[frozenset(g[s] for s in S) for S in r] for r in M]
                        for name, M2 in T.items():
                            s = surv[(name, fam, tag)]
                            s["n"] += 1
                            s["ok"] += holds(fam, M2, **kw)
                        if fam == "Arithmetic" and view == "color":
                            white["n"] += 1
                            white["C12=0"] += M[0][1] == 0
                            white["C22=0"] += M[1][1] == 0
                        if fam == "Arithmetic" and view in ("size", "color") and \
                                all(r[2] == r[0] + r[1] + (view == "size") for r in M):
                            for r in M:
                                operands[(view, 1)].append(r[0]); operands[(view, 2)].append(r[1])
            # candidate sets: a product of per-view value sets means a group acts transitively on them
            keys = []                                   # number and position change together: one coordinate
            for p in panels[8:]:
                keys.append(tuple(x for ci, c in enumerate(p) for x in       # value sets, not per-object tuples
                                  ((c["number"], tuple(sorted(c["position"]))), *(frozenset(c[v]) for v in VMAX))))
            nv = [len({k[j] for k in keys}) for j in range(len(keys[0]))]
            prod = len(set(keys)) == 8 and np.prod(nv) == 8
            cand[cfg, "n"] += 1
            cand[cfg, "product"] += prod
            # candidates alone: the one(s) sharing the most coordinate values with the other seven
            score = [sum(a == b for j in range(8) if j != i for a, b in zip(keys[i], keys[j])) for i in range(8)]
            top = [i for i in range(8) if score[i] == max(score)]
            cand[cfg, prod, "mode"] += (y in top) / len(top)
            cand[cfg, prod, "n"] += 1
    return surv, params, white, operands, comp_stats, cand, targets


# ── I-RAVEN-X ─────────────────────────────────────────────────────────────────────────────────────────
def iravenx(path, n):
    with open(path) as f:
        d = json.load(f)
    m = int(path.split("maxval_")[1].split(".")[0])
    surv = defaultdict(Counter)
    params = Counter()
    zeros = defaultdict(Counter)                        # Arithmetic: (sign, column) -> zero operands
    vals = defaultdict(list)                            # (attr, rule) -> values in the shown rows
    cand = Counter()
    targets = Counter()
    for i in range(n):
        s = d[str(i)]
        k = int(round((len(s["rpm"]) - 7) ** 0.5))
        y = int(s["target"])
        targets[y] += 1
        cells = s["rpm"][:k * k - 1] + [s["rpm"][k * k - 1 + y]]
        for attr in ("Type", "Size", "Color"):
            fam = s["rules"][0][attr]
            full = np.array([int(c[0][attr]) for c in cells]).reshape(k, k)
            M = full[-3:].tolist()                      # the three rows that are shown
            vals[(attr, fam)] += sum(M, [])
            if fam == "Progression":
                params[(fam, int(full[0][1] - full[0][0]))] += 1
            if fam == "Distribute_Three":
                params[(fam, next(st for st in (1, -1) if list(full[1]) == list(np.roll(full[0], st))))] += 1
            if fam == "Arithmetic":
                sign = "+" if all(r[-1] == sum(r[:-1]) for r in full) else "-"
                params[(fam, sign)] += 1
                ops = range(k - 1) if sign == "+" else range(1, k)
                for r in full:
                    for j in ops:
                        zeros[(sign, j)]["n"] += 1
                        zeros[(sign, j)]["zero"] += r[j] == 0
            assert holds(fam, M), (i, attr, fam)
            T = {"rows 1<>2": permute(M, rows=[1, 0, 2]), "cols 1<>2": permute(M, cols=[1, 0] + list(range(2, k))),
                 "reverse cols*": permute(M, cols=list(range(k))[::-1]),
                 "reflect values": [[m - 1 - v for v in r] for r in M]}
            if k > 3:                                   # in 3x3, column 3 holds the missing cell
                T["cols 2<>3"] = permute(M, cols=[0, 2, 1] + list(range(3, k)))
            for name, M2 in T.items():
                for any_step in (False, True):
                    if any_step and fam != "Distribute_Three":
                        continue
                    sv = surv[(name, fam + (" (any step)" if any_step else ""), "value")]
                    sv["n"] += 1
                    sv["ok"] += holds(fam, M2, any_step=any_step)
        C = np.array([[int(p[0][a]) for a in ("Type", "Size", "Color")] for p in s["rpm"][k * k - 1:]])
        nv = [len(set(C[:, a])) for a in range(3)]
        cand["n"] += 1
        cand["product"] += len({tuple(c) for c in C}) == 8 and np.prod(nv) == 8
        cand["answer", "larger"] += sum(C[y, a] > C[:, a].min() for a in range(3) if nv[a] == 2)
        cand["answer", "pairs"] += sum(nv[a] == 2 for a in range(3))
    return m, k, surv, params, zeros, vals, cand, targets


def pct(a, b):
    return f"{100 * a / b:5.1f}" if b else "  n/a"


def print_survival(surv, title):
    print(f"\n{title}: share of rule instances still valid after the transformation (%)")
    names = sorted({t for t, _, _ in surv if t != "(original)"}, key=lambda t: (t.endswith("*"), t))
    fams = sorted({(f, k) for t, f, k in surv if t != "(original)"})
    print(f"{'':36s}" + "".join(f"{t:>18s}" for t in names))
    for fam, kind in fams:
        line = f"{fam + (f' [{kind[3:]}-slot positions]' if kind.startswith('set') else ''):36s}"
        for t in names:
            c = surv.get((t, fam, kind))
            line += f"{pct(c['ok'], c['n']) if c else '':>18s}"
        print(line + f"   (n={max(surv[(t, fam, kind)]['n'] for t in names)})")
    print("  * moves the missing cell: a symmetry of complete matrices only")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="I-RAVEN train puzzles per constellation")
    ap.add_argument("--nx", type=int, default=6000, help="I-RAVEN-X puzzles (train part)")
    ap.add_argument("--only", choices=("iraven", "iravenx"))
    args = ap.parse_args()
    if args.only != "iravenx":
        report_iraven(args.n)
    if args.only != "iraven":
        report_iravenx(args.nx)


def report_iraven(n):

    surv, params, white, operands, comp_stats, cand, targets = iraven(n)
    print_survival(surv, f"I-RAVEN, {n} train puzzles x 7 constellations")
    print("  original instances failing their own label (non-uniform noise excluded):",
          sum(c["bad"] for (t, _, _), c in surv.items() if t == "(original)"))
    print("\nI-RAVEN generator parameters (all components):")
    for (attr, name), c in sorted(params.items()):
        print(f"  {attr:16s}{name:30s}{c:6d}")
    print(f"  Arithmetic on colour: C12 = 0 in {white['C12=0']}/{white['n']}, C22 = 0 in {white['C22=0']}/{white['n']}")
    for v in ("size", "color"):
        a, b = np.mean(operands[(v, 1)]), np.mean(operands[(v, 2)])
        print(f"  Arithmetic plus on {v}: mean column 1 = {a:.2f}, column 2 = {b:.2f}")
    print("  two-component layouts (rule counts and mean value per component):")
    for cfg in CONSTELLATIONS[3:]:
        for ci in (0, 1):
            st = comp_stats[(cfg, ci)]
            fams = Counter({(v, f): c for (v, f), c in st.items() if f in FAMILIES})
            means = {v: st[(v, "sum")] / st[(v, "count")] for v in VMAX if st[(v, "count")]}
            fam_by_view = {v: {f[:4]: c for (vv, f), c in fams.items() if vv == v} for v in ("type", "size", "color")}
            print(f"    {cfg[:22]:22s} comp {ci}: {fam_by_view}" + (f"  means {means}" if means else ""))
    print("  candidate sets that are a product of per-attribute value sets (%): " +
          "  ".join(f"{cfg[:12]} {pct(cand[cfg, 'product'], cand[cfg, 'n'])}" for cfg in CONSTELLATIONS))
    for prod in (True, False):
        print(f"  candidates-only mode guess, {'product' if prod else 'other'} sets (%, chance 12.5): " +
              "  ".join(f"{cfg[:12]} {pct(cand[cfg, prod, 'mode'], cand[cfg, prod, 'n'])}" for cfg in CONSTELLATIONS))
    print(f"  answer index: {dict(sorted(targets.items()))}")


def report_iravenx(nx):
    for path in sorted(glob.glob("data/iravenx/center_single_shuffle_n_*_maxval_*.json")):
        m, k, surv, params, zeros, vals, cand, targets = iravenx(path, nx)
        print_survival(surv, f"\nI-RAVEN-X {k}x{k} (3 rows shown), values 0-{m - 1}, {nx} puzzles")
        print("  parameters:", dict(sorted(params.items(), key=str)))
        if k > 3:
            print("  Arithmetic operands equal to 0, by column (%):")
            for sign in "+-":
                cols = sorted(j for s, j in zeros if s == sign)
                print(f"    {sign}: " + " ".join(f"c{j + 1}:{pct(zeros[(sign, j)]['zero'], zeros[(sign, j)]['n'])}"
                                            for j in cols))
        print("  mean value in the shown rows (range midpoint %.1f):" % ((m - 1) / 2))
        for attr in ("Type", "Size", "Color"):
            print(f"    {attr:6s}" + "  ".join(f"{fam[:5]} {np.mean(vals[(attr, fam)]):5.1f} (max {max(vals[(attr, fam)])})"
                                          for fam in FAMILIES if vals[(attr, fam)]))
        rf = {a: Counter() for a in ("Type", "Size", "Color")}
        for (a, fam), v in vals.items():
            rf[a][fam] = len(v)
        print("  rule frequency:", {a: {f[:5]: c // (3 * k) for f, c in sorted(r.items())} for a, r in rf.items()})
        print(f"  candidate sets: product of per-attribute value pairs {pct(cand['product'], cand['n'])}%; "
              f"the answer holds the larger value of its pair {pct(cand['answer', 'larger'], cand['answer', 'pairs'])}%")
        print(f"  answer index: {dict(sorted(targets.items()))}")


if __name__ == "__main__":
    main()
