#!/usr/bin/env python3
"""The rule twins with values read as numbers: a fit with no training (docs/ARITHMETIC.md, "Two checks without
training").

For every attribute of a puzzle, c = p*a + q*b is fitted on the puzzle's two complete rows by least squares, and p and
q are applied to the question row. Nothing is learned from other puzzles. This is the far end of the built-in moves in
analysis/rule_orbits.py: the linear form of the rule is given, and the two complete rows fix its two numbers. It shows
what the templates pay for treating values as constants with an order.

    python3 analysis/twins_linear_fit.py [--rules sum midpoint mirror lookup]

Prints, per twin dataset, the exact % on the unseen puzzles 6300..7299 for each rule (the fourth rule under the
dataset's own name).
"""
import argparse, collections, json

import numpy as np

ATTRS = ("Type", "Size", "Color")
DATA = "data/iravenx_twins/{}/center_single_shuffle_n_3_maxval_50.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="+", default=["sum", "midpoint", "mirror", "lookup"])
    ap.add_argument("--start", type=int, default=6300)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    print("exact % with no training: c = p*a + q*b fitted on the two complete rows, applied to the question row")
    for data in args.rules:
        d = json.load(open(DATA.format(data)))
        hit, tot = collections.Counter(), collections.Counter()
        miss = unfixed = 0                         # misses on the linear rules; those whose two rows do not fix p and q
        for i in range(args.start, args.start + args.n):
            p = d[str(i)]
            cells = p["rpm"][:8] + [p["rpm"][8 + int(p["target"])]]
            for a in ATTRS:
                v = [float(c[0][a]) for c in cells]
                rows = np.array([v[0:2], v[3:5]])
                pq = np.linalg.lstsq(rows, np.array([v[2], v[5]]), rcond=None)[0]
                rule = data if p["rules"][0][a] == "Arithmetic" else p["rules"][0][a]
                right = round(float(pq @ np.array(v[6:8]))) == int(v[8])
                tot[rule] += 1
                hit[rule] += int(right)
                if not right and rule not in ("Distribute_Three", "lookup"):
                    miss += 1
                    unfixed += int(np.linalg.matrix_rank(rows) < 2)
        print(f"  {data:9s} " + "  ".join(f"{r} {100 * hit[r] / tot[r]:.1f} ({hit[r]} of {tot[r]})" for r in sorted(tot))
              + f"  | misses on the linear rules: {miss}, of them with rows that do not fix p and q: {unfixed}")


if __name__ == "__main__":
    main()
