#!/usr/bin/env python3
"""Arithmetic with one operator: I-RAVEN-X 3x10 keeping only the puzzles whose Arithmetic attributes all use plus, or all
use minus (docs/ARITHMETIC.md). Along a row, plus is last = sum of the others, minus is last = first - sum of the middle.
Nothing is generated: the other rules stay, and a puzzle with a minus attribute leaves the plus file, also when its other
attribute uses plus. Puzzles are packed within each range and the ranges keep their starts (train 0, selection 6000,
unseen 6300), so every model and scorer runs unchanged with the counts printed here. The test range is not copied.

    python3 data_prep/arith_op_data.py
"""
import json, os

SRC = "data/iravenx/center_single_shuffle_n_10_maxval_50.json"
RANGES = ((0, 6000), (6000, 6300), (6300, 7300))      # train, selection, unseen


def path_of(op):
    return f"data/iravenx_ops/{op}/{os.path.basename(SRC)}"


def ops(s, n=10):
    """The operator of each Arithmetic attribute of a puzzle, read from its three complete rows."""
    shown = s["rpm"][(n - 3) * n: n * n - 1] + [s["rpm"][n * n - 1 + int(s["target"])]]
    found = []
    for attr in ("Size", "Color"):                    # Type never takes Arithmetic
        if s["rules"][0][attr] == "Arithmetic":
            v = [int(p[0][attr]) for p in shown]
            rows = [v[i:i + n] for i in (0, n, 2 * n)]
            plus = all(r[-1] == sum(r[:-1]) for r in rows)
            minus = all(r[-1] == r[0] - sum(r[1:-1]) for r in rows)
            assert plus != minus
            found.append("plus" if plus else "minus")
    return found


def main():
    d = json.load(open(SRC))
    for op in ("plus", "minus"):
        out, counts = {}, []
        for lo, hi in RANGES:
            kept = [d[str(i)] for i in range(lo, hi) if all(o == op for o in ops(d[str(i)]))]
            out.update({str(lo + j): s for j, s in enumerate(kept)})
            counts.append(len(kept))
        os.makedirs(os.path.dirname(path_of(op)), exist_ok=True)
        json.dump(out, open(path_of(op), "w"))
        print(f"{op}: {counts[0]} train, {counts[1]} selection, {counts[2]} unseen")


if __name__ == "__main__":
    main()
