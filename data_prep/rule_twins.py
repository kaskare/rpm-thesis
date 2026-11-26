#!/usr/bin/env python3
"""Is Arithmetic hard because it has no value symmetry? I-RAVEN-X puzzles with Arithmetic swapped for a test rule
(docs/ARITHMETIC.md). Rows of 3, values 0..maxval; everything else follows IBM's generator as measured on
center_single_shuffle_n_3_maxval_10.json: Constant one value per row; Progression step +-1 or +-2, the same in all rows;
Distribute three values, rows cyclic shifts of the first; Type never takes the fourth rule; the eight candidates are
the two values of each attribute, all combinations.

The fourth rule, written a, b, c along a row:
  sum       c = a + b, a, b >= 1              the control: no value symmetry; operand swap
  midpoint  c = (a + b) / 2, a != b           as many table entries, shift and reflection symmetric; operand swap.
                                              The only rule c = pa + qb with both symmetries (p + q = 1, p = q)
  mirror    c = 2a - b, a != b                midpoint's triples with the columns moved: the answer is no longer
                                              between the operands, so errors grow as they do in sum; no operand swap
  lookup    b = g(a), c = g(b)                g a fixed table on the values with no symmetry: far fewer entries

g is not a permutation: a permutation commutes with its own powers, so relabelling by g would keep the rule (a single
cycle is a Progression under scrambled labels). Two values with no predecessor and the same image could also swap, so
g is searched until colour refinement separates every value, which leaves only the identity.

Files keep IBM's format and name, under data/iravenx_twins/<rule>/, and the fourth rule keeps the name "Arithmetic",
so every model and scorer runs on them unchanged. "slot" in each puzzle names the real rule.

    python3 data_prep/rule_twins.py --maxval 50 --write   # the four files, then the checks
    python3 data_prep/rule_twins.py --maxval 50           # the checks only
"""
import argparse, collections, json, os, random

TWINS = ("sum", "midpoint", "mirror", "lookup")
ATTRS = ("Type", "Size", "Color")
THREE = ("Constant", "Progression", "Distribute_Three")
PUZZLES = 10000                                       # 0-5999 train, 6000-6299 selection, 6300-7299 unseen


def refine(g):
    """Colour refinement on the graph of g: the classes no automorphism can mix."""
    n = len(g)
    pre = [[u for u in range(n) if g[u] == v] for v in range(n)]
    col = [len(pre[v]) for v in range(n)]
    while True:
        sig = [(col[v], col[g[v]], tuple(sorted(col[u] for u in pre[v]))) for v in range(n)]
        ids = {s: i for i, s in enumerate(sorted(set(sig)))}
        new = [ids[s] for s in sig]
        if len(set(new)) == len(set(col)):
            return new
        col = new


def lookup_table(maxval, seed=0):
    """A table with no symmetry and no fixed point, moving wrongly classed values until every value stands alone."""
    rng = random.Random(seed)
    n = maxval + 1
    g = [rng.choice([w for w in range(n) if w != v]) for v in range(n)]
    for _ in range(200000):
        col = refine(g)
        same = collections.Counter(col)
        if max(same.values()) == 1:
            return g
        v = rng.choice([v for v in range(n) if same[col[v]] > 1])
        g[v] = rng.choice([w for w in range(n) if w != v])
    raise SystemExit("no table found")


def table(rule, m, g):
    """Every row of the fourth rule, as (a, b, c)."""
    r = range(m + 1)
    if rule == "sum":
        return [(a, b, a + b) for a in r for b in r if a >= 1 and b >= 1 and a + b <= m]
    if rule == "midpoint":
        return [(a, b, (a + b) // 2) for a in r for b in r if a != b and (a + b) % 2 == 0]
    if rule == "mirror":
        return [(a, b, 2 * a - b) for a in r for b in r if a != b and 0 <= 2 * a - b <= m]
    return [(a, g[a], g[g[a]]) for a in r]


def keeps(rows, fn, m):
    """Of the rows the change keeps in range, the share that still follow the rule."""
    known = set(rows)
    moved = [tuple(fn(list(row))) for row in rows]
    moved = [t for t in moved if all(0 <= v <= m for v in t)]
    return sum(t in known for t in moved), len(moved)


def row_constant(rng, m):
    return [[v] * 3 for v in (rng.randint(0, m) for _ in range(3))]


def row_progression(rng, m):
    step = rng.choice((-2, -1, 1, 2))
    lo, hi = max(0, -2 * step), min(m, m - 2 * step)
    return [[s, s + step, s + 2 * step] for s in (rng.randint(lo, hi) for _ in range(3))]


def row_distribute(rng, m):
    a, b, c = rng.sample(range(m + 1), 3)
    return [[a, b, c], [c, a, b], [b, c, a]] if rng.random() < 0.5 else [[a, b, c], [b, c, a], [c, a, b]]


def puzzle(rng, m, rows4, slot):
    rules, vals = {"Number/Position": "Constant"}, {}
    for a in ATTRS:
        rules[a] = rng.choice(THREE if a == "Type" else THREE + ("Arithmetic",))
        vals[a] = {"Constant": row_constant, "Progression": row_progression, "Distribute_Three": row_distribute}[
            rules[a]](rng, m) if rules[a] != "Arithmetic" else [list(t) for t in rng.sample(rows4, 3)]
    panel = lambda t, s, c: [{"Type": str(t), "Size": str(s), "Color": str(c), "Angle": str(rng.randint(0, 9))}]
    ctx = [panel(*(vals[a][r][j] for a in ATTRS)) for r in range(3) for j in range(3) if (r, j) != (2, 2)]
    right = [vals[a][2][2] for a in ATTRS]
    wrong = [rng.choice([v for v in range(m + 1) if v != x]) for x in right]
    cands = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    rng.shuffle(cands)
    y = cands.index((0, 0, 0))
    return {"rules": [rules], "rpm": ctx + [panel(*((right, wrong)[w][i] for i, w in enumerate(c))) for c in cands],
            "target": str(y), "slot": slot}


def path_of(rule, m):
    return f"data/iravenx_twins/{rule}/center_single_shuffle_n_3_maxval_{m}.json"


def question_rows(d, lo, hi):
    """The fourth rule's rows and question rows of puzzles lo..hi-1, per attribute."""
    for i in range(lo, hi):
        s = d[str(i)]
        full = s["rpm"][:8] + [s["rpm"][8 + int(s["target"])]]
        for a in ("Size", "Color"):
            if s["rules"][0][a] == "Arithmetic":
                v = [int(p[0][a]) for p in full]
                yield v[0:3], v[3:6], v[6:9]


def references(rule, m):
    """Two scores with nothing trained, on the unseen puzzles. Memorising: the question row is answered only if its
    operands were met in the first n training puzzles, which is what 'as hard as sum' means here: no generalising,
    only coverage. Differences: the likeliest step from b to c given b - a, fit on the training puzzles: the most a
    rule can get that names no value and reads only the question row."""
    d = json.load(open(path_of(rule, m)))
    unseen = list(question_rows(d, 6300, 7300))
    key = (lambda q: q[1]) if rule == "lookup" else (lambda q: (q[0], q[1]))
    res = {}
    for n in (100, 300, 600, 1000, 3000, 6000):
        seen = {}
        for rows in question_rows(d, 0, n):
            for q in rows:
                seen[key(q)] = q[2]
        res[n] = 100 * sum(seen.get(key(q[2])) == q[2][2] for q in unseen) / len(unseen)
    steps = collections.defaultdict(collections.Counter)
    for rows in question_rows(d, 0, 6000):
        for q in rows:
            steps[q[1] - q[0]][q[2] - q[1]] += 1
    diff = sum(steps[q[2][1] - q[2][0]].most_common(1)[0][0] == q[2][2] - q[2][1]
               for q in unseen if steps[q[2][1] - q[2][0]])
    return len(unseen), res, 100 * diff / len(unseen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maxval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    m = args.maxval
    g = lookup_table(m, args.seed)
    rows = {r: table(r, m, g) for r in TWINS}
    assert len(set(refine(g))) == m + 1
    print(f"lookup table on 0..{m}: {len(set(g))} values in its image, no fixed point, every value alone after refinement")
    rand = list(range(m + 1))
    random.Random(1).shuffle(rand)
    changes = {"shift +1": lambda t: [v + 1 for v in t], "shift +7": lambda t: [v + 7 for v in t],
               "reflect": lambda t: [m - v for v in t], "relabel": lambda t: [rand[v] for v in t],
               "swap a, b": lambda t: [t[1], t[0], t[2]]}
    print(f"\nrows kept by each change, %\n{'rule':10}{'rows':>6}" + "".join(f"{c:>11}" for c in changes))
    for r in TWINS:
        cells = [keeps(rows[r], fn, m) for fn in changes.values()]
        print(f"{r:10}{len(rows[r]):>6}" + "".join(f"{100 * h / max(n, 1):>11.1f}" for h, n in cells))
    if args.write:
        for r in TWINS:
            rng = random.Random(f"{args.seed}{r}")
            os.makedirs(os.path.dirname(path_of(r, m)), exist_ok=True)
            json.dump({str(i): puzzle(rng, m, rows[r], r) for i in range(PUZZLES)}, open(path_of(r, m), "w"))
    if all(os.path.exists(path_of(r, m)) for r in TWINS):
        print(f"\nmemorising and differences only, exact % on the unseen puzzles' question rows")
        for r in TWINS:
            n, res, diff = references(r, m)
            print(f"{r:10}{n:>5} rows  memorising " + " ".join(f"{k}: {v:.1f}" for k, v in res.items())
                  + f"  differences {diff:.1f}")


if __name__ == "__main__":
    main()
