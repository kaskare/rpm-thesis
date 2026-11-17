#!/usr/bin/env python3
"""The statements of docs/thesis/ch3_pricing_proofs.md that can be checked without training, on the rule twins
(rows of 3, values 0..50). Nothing is trained; the counting learners are those of analysis/rule_orbits.py.

  1. Which tables stay valid when a row is moved: every panel +s, every panel times k, one operand with the answer.
  2. Coverage from the table alone. With rows drawn evenly from the table, the expected share of unseen rows whose
     orbit n training rows reach is  sum over orbits of P(orbit) * (1 - (1 - P(orbit))^n).  Printed next to the share
     the actual training sets reach (the counting learner's score), and as the rows needed for 90%.
  3. The ceiling of a rule under a move its rows do not keep: one row right per orbit.
  4. A function that is equivariant to shifting the whole puzzle and solves sum: c3 = b3 + (c1 - b1) + (a3 - a1).
     So the ceiling on sum belongs to learners that answer from the question row, not to the puzzle-wide shift.

    python3 analysis/pricing_checks.py
"""
import collections, json

import rule_orbits as RO

DATA = "data/iravenx_twins/{}/center_single_shuffle_n_3_maxval_50.json"
M = 50
KINDS = ("table", "diagonal", "doubling", "scaling")
NAMES = {"table": "nothing (one answer per pair)", "diagonal": "every panel +s", "doubling": "+s and x2",
         "scaling": "+s and xk", "operand": "one operand with the answer"}


def kept(rows, move):
    """Of the moved rows that stay in 0..M, how many are rows of the table again."""
    known = set(rows)
    moved = [t for t in (move(r) for r in rows) if all(0 <= v <= M for v in t)]
    return sum(t in known for t in moved), len(moved)


def expected(shares, n):
    return 100 * sum(p * (1 - (1 - p) ** n) for p in shares)


def main():
    tables = {r: RO.table_rows(r) for r in ("sum", "midpoint", "mirror")}

    print("1. rows that are rows of the table again after a move (of those that stay in range)")
    moves = {"every panel +s": [lambda r, s=s: (r[0] + s, r[1] + s, r[2] + s) for s in range(-M, M + 1) if s],
             "every panel times k": [lambda r, k=k: (k * r[0], k * r[1], k * r[2]) for k in (2, 3, 5)],
             "one operand with the answer": [lambda r, s=s: (r[0] + s, r[1], r[2] + s) for s in range(-M, M + 1) if s]}
    for rule, rows in tables.items():
        out = []
        for name, fs in moves.items():
            ok, tot = map(sum, zip(*(kept(rows, f) for f in fs)))
            out.append(f"{name}: {ok} of {tot} ({100 * ok / tot:.0f}%)")
        print(f"  {rule:9s} " + "; ".join(out))

    print("\n2. midpoint: share of unseen rows whose orbit the training rows reach, expected from the table and reached")
    rows, test = tables["midpoint"], RO.rows(DATA.format("midpoint"), 6300, 1000, False)
    shares = {}
    for kind in KINDS:
        size = collections.Counter(RO.ORBIT[kind](a, b) for a, b, _ in rows)
        shares[kind] = [c / len(rows) for c in size.values()]
    print(f"  {'built in':32s} {'orbits':>6s}   " + "   ".join(f"{n:>4d} puzzles" for n in (100, 600, 6000)) + "   rows for 90%")
    for kind in KINDS:
        cells = []
        for n in (100, 600, 6000):
            train = RO.rows(DATA.format("midpoint"), 0, n, False)
            cells.append(f"{expected(shares[kind], len(train)):5.1f} / {RO.score(RO.fit(train, kind), test, kind):5.1f}")
        need = next(n for n in range(1, 20000) if expected(shares[kind], n) >= 90)
        print(f"  {NAMES[kind]:32s} {len(shares[kind]):>6d}   " + "   ".join(cells) + f"   {need:>6d}")
    counts = [len(RO.rows(DATA.format("midpoint"), 0, n, False)) for n in (100, 600, 6000)]
    print(f"  cells: expected / reached by the actual training set. Training question rows: {counts} at 100, 600, 6000 puzzles")

    print("\n3. ceilings: with one answer per orbit, the rows of the whole table that can be right")
    for rule, rows in tables.items():
        out = []
        for kind in ("diagonal", "doubling", "scaling", "operand"):
            right = sum(1 for a, b, c in rows if RO.PREDICT[kind](a, b, RO.fit(rows, kind)[RO.ORBIT[kind](a, b)]) == c)
            out.append(f"{NAMES[kind]}: {right} of {len(rows)} ({100 * right / len(rows):.1f}%)")
        print(f"  {rule:9s} " + "; ".join(out))

    print("\n4. sum from differences alone, over the whole puzzle: c3 = b3 + (c1 - b1) + (a3 - a1)")
    d = json.load(open(DATA.format("sum")))
    hit = tot = moved = 0
    for i in range(6300, 7300):
        p = d[str(i)]
        cells = p["rpm"][:8] + [p["rpm"][8 + int(p["target"])]]
        for a in RO.ATTRS:
            if p["rules"][0][a] != "Arithmetic":
                continue
            v = [int(c[0][a]) for c in cells]
            f = lambda v: v[7] + (v[2] - v[1]) + (v[6] - v[0])
            tot += 1
            hit += int(f(v) == v[8])
            moved += int(all(f([x + s for x in v]) == f(v) + s for s in (-7, 3, 40)))   # follows a shift of all panels
    print(f"  right on {hit} of {tot} unseen sum rows; moves with a shift of the whole puzzle in {moved} of {tot}")


if __name__ == "__main__":
    main()
