#!/usr/bin/env python3
"""The rule twins over seeds (docs/ARITHMETIC.md, "Three seeds", "The scaling kernel" and "The rule gate"): the shared,
per-value and plus models, the shared model with rows read in their own unit (doubled, scaled), and the shared and
scaled models with the answer gated by the rule descriptor (sharedgate, scaledgate) or with a row descriptor that reads
the steps (sharedsteps, doubledsteps, scaledsteps; "The row descriptor reads the steps"), on sum and midpoint at 100 and 600
puzzles, runs experiments/twins/<run>_<rule>_n<N>_selexact_s<seed> (jobs/pynl/submit_rule_twins.sh with SEED and
SELECT=exact).

One rule for every cell, fixed before the runs: the epoch with the highest whole-puzzle exact on the selection puzzles,
the first one at that maximum; the cell is the exact % on the test rule on the unseen puzzles at that epoch. The seeded
runs save that epoch's checkpoint. "first run" is the earlier unseeded run under the same rule, read from its history.

    python3 analysis/twins_seeds.py                                   # the table: per seed, mean and sd
    python3 analysis/twins_seeds.py --dumps experiments/twins_eval    # also: rows by whether training covered them

--dumps reads the generated values of the saved checkpoints (jobs/pynl/submit_twins_dump.sh) and splits the unseen
midpoint rows by what the training question rows covered: the pair (a, b) for weights per value; for the models with the
shift, the first step b - a, and among the rows with a new first step, its odd part (the first step without its powers of
two: the orbit under doubling). A second split: the rows whose first two panels look like a Progression row's in the
model's unit, and how many of them the model continued as a Progression. A third, for the scaled models: midpoint and
Progression rows by whether the question row shares a value with a complete row (does the answer follow the context?).
A fourth ("Two checks without training"): the unseen midpoint rows by how many training question rows fall in their
orbit, the first step for the shared model and the pair (a, b) for weights per value, seeds pooled.
"""
import argparse, collections, json, os

import numpy as np

ROOT = "experiments/twins"
ATTRS = ("Type", "Size", "Color")
DATA = "data/iravenx_twins/{}/center_single_shuffle_n_3_maxval_50.json"


def picked(run):
    """(epoch, exact % on the test rule, unseen) at the epoch picked by exact on selection, or None if no history."""
    path = os.path.join(ROOT, run, "history.json")
    if not os.path.exists(path):
        return None
    hist = json.load(open(path))
    top = max(h["sel"]["exact"] for h in hist)
    h = next(h for h in hist if h["sel"]["exact"] == top)
    return h["epoch"], 100 * h["unseen"]["per_rule"]["Arithmetic"], len(hist)


def test_rows(rule, start, n):
    """(puzzle, attribute index, a, b, c) for every attribute of the question row that follows the test rule."""
    d = json.load(open(DATA.format(rule)))
    out = []
    for i in range(start, start + n):
        p = d[str(i)]
        rpm, r = p["rpm"], p["rules"][0]
        ans = rpm[8 + int(p["target"])][0]
        out += [(i - start, k, int(rpm[6][0][a]), int(rpm[7][0][a]), int(ans[a])) for k, a in enumerate(ATTRS)
                if r[a] == "Arithmetic"]
    return out


def all_rows(rule, start, n):
    """(puzzle, attribute index, the attribute's rule, its nine values with the answer last) for every attribute."""
    d = json.load(open(DATA.format(rule)))
    out = []
    for i in range(start, start + n):
        p = d[str(i)]
        cells = p["rpm"][:8] + [p["rpm"][8 + int(p["target"])]]
        out += [(i - start, k, p["rules"][0][a], [int(c[0][a]) for c in cells]) for k, a in enumerate(ATTRS)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--dumps", help="directory with <run>.json from jobs/pynl/submit_twins_dump.sh")
    args = ap.parse_args()

    print("exact % on the test rule, unseen puzzles; epoch picked by exact on selection")
    print(f"{'model':12s} {'rule':9s} {'n':>4s}  " + "  ".join(f"seed {s}" for s in args.seeds) + "    mean ± sd   first run")
    for model in ("shared", "pervalue", "lplus", "doubled", "scaled", "sharedgate", "scaledgate", "sharedsteps",
                  "doubledsteps", "scaledsteps"):
        for rule in ("sum", "midpoint"):
            for n in (100, 600):
                cells = [picked(f"{model}_{rule}_n{n}_selexact_s{s}") for s in args.seeds]
                first = picked(f"{model}_{rule}_n{n}") or (None, float("nan"), 0)
                vals = [c[1] for c in cells if c]
                if not vals:
                    continue
                done = [c for c in cells if c]
                part = "" if all(c[2] == 20 for c in done) and len(done) == len(cells) else "  (unfinished)"
                stat = f"{np.mean(vals):5.1f} ± {np.std(vals, ddof=1):4.1f}" if len(vals) > 1 else "      –     "
                print(f"{model:12s} {rule:9s} {n:>4d}  " + "  ".join(f"{c[1]:6.1f}" if c else "     –" for c in cells)
                      + f"   {stat}   {first[1]:6.1f}" + part)

    if not args.dumps:
        return
    odd = lambda d: d // (d & -d)                             # a step without its powers of two
    print("\nunseen midpoint rows right, by what the training question rows covered")
    for model in ("shared", "doubled", "scaled", "sharedgate", "scaledgate", "sharedsteps", "doubledsteps",
                  "scaledsteps", "pervalue"):
        for n in (100, 600):
            unseen, train = test_rows("midpoint", 6300, 1000), test_rows("midpoint", 0, n)
            if model == "pervalue":
                pairs = {(a, b) for _, _, a, b, _ in train}
                groups = {"pair seen": lambda a, b: (a, b) in pairs, "pair new": lambda a, b: (a, b) not in pairs}
            else:
                steps, odds = {b - a for _, _, a, b, _ in train}, {odd(b - a) for _, _, a, b, _ in train}
                groups = {"first step seen": lambda a, b: b - a in steps,
                          "step new, its odd part seen": lambda a, b: b - a not in steps and odd(b - a) in odds,
                          "step and odd part new": lambda a, b: odd(b - a) not in odds}
            for s in args.seeds:
                run = f"{model}_midpoint_n{n}_selexact_s{s}"
                path = os.path.join(args.dumps, run + ".json")
                if not os.path.exists(path):
                    continue
                gen = json.load(open(path))
                out, tot = [], 0
                for name, member in groups.items():
                    rows = [(i, k, c) for i, k, a, b, c in unseen if member(a, b)]
                    hit = sum(int(gen[i][k] == c) for i, k, c in rows)
                    tot += hit
                    out.append(f"{name} {hit} of {len(rows)}")
                print(f"  {run}: " + "; ".join(out) + f"; all {100 * tot / len(unseen):.1f}%")

    # A question row whose first two panels look, in the model's unit, like those of a Progression row (the data's
    # Progression steps are +-1 and +-2): first step +-2 for the shared model, a power of two for doubled, every row for
    # scaled. "as Progression": the model generated 2b - a, the row continued by its first step.
    like = {"shared": lambda d: abs(d) <= 2, "doubled": lambda d: abs(odd(d)) == 1, "scaled": lambda d: True}
    like.update(sharedgate=like["shared"], scaledgate=like["scaled"], sharedsteps=like["shared"],
                doubledsteps=like["doubled"], scaledsteps=like["scaled"])
    print("\nunseen midpoint rows right, by whether the row starts like a Progression row in the model's unit")
    for model in like:
        for n in (100, 600):
            unseen = test_rows("midpoint", 6300, 1000)
            for s in args.seeds:
                run = f"{model}_midpoint_n{n}_selexact_s{s}"
                path = os.path.join(args.dumps, run + ".json")
                if not os.path.exists(path):
                    continue
                gen = json.load(open(path))
                out = []
                for name, want in (("starts like a Progression", True), ("other rows", False)):
                    rows = [(i, k, a, b, c) for i, k, a, b, c in unseen if like[model](b - a) == want]
                    hit = sum(int(gen[i][k] == c) for i, k, a, b, c in rows)
                    prog = sum(int(gen[i][k] == (2 * b - a) % 101) for i, k, a, b, c in rows)
                    out.append(f"{name} {hit} of {len(rows)} right, {prog} as Progression")
                print(f"  {run}: " + "; ".join(out))

    # Does the answer follow the context? In the scaled models a midpoint and a Progression question row look the same;
    # the overlap of the question row's values with the complete rows (ov) is one input of the rule descriptor.
    print("\nunseen rows right in the scaled models, by whether the question row shares a value with a complete row")
    rows = [(i, k, rule, v, bool(set(v[6:8]) & set(v[:6]))) for i, k, rule, v in all_rows("midpoint", 6300, 1000)
            if v[6] != v[7]]
    for model in ("scaled", "scaledgate", "scaledsteps"):
        for n in (100, 600):
            for s in args.seeds:
                run = f"{model}_midpoint_n{n}_selexact_s{s}"
                path = os.path.join(args.dumps, run + ".json")
                if not os.path.exists(path):
                    continue
                gen = json.load(open(path))
                out = []
                for rule, name in (("Arithmetic", "midpoint"), ("Progression", "Progression")):
                    for share in (False, True):
                        x = [(i, k, v) for i, k, r, v, sh in rows if r == rule and sh == share]
                        hit = sum(int(gen[i][k] == v[8]) for i, k, v in x)
                        out.append(f"{name}, {'a shared value' if share else 'no shared value'}: {hit} of {len(x)}")
                print(f"  {run}: " + "; ".join(out))

    # How many training rows does an orbit need? The orbit of a question row is its first step under the shift, and the
    # pair (a, b) itself for weights per value.
    print("\nunseen midpoint rows right, by how many training question rows fall in the row's orbit (seeds pooled)")
    bins = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 9))
    unseen = test_rows("midpoint", 6300, 1000)
    for model, key in (("shared", lambda a, b: b - a), ("pervalue", lambda a, b: (a, b))):
        for n in (100, 600):
            seen = collections.Counter(key(a, b) for _, _, a, b, _ in test_rows("midpoint", 0, n))
            hit, tot = [0] * len(bins), [0] * len(bins)
            for s in args.seeds:
                path = os.path.join(args.dumps, f"{model}_midpoint_n{n}_selexact_s{s}.json")
                if not os.path.exists(path):
                    continue
                gen = json.load(open(path))
                for i, k, a, b, c in unseen:
                    j = next(q for q, (lo, hi) in enumerate(bins) if lo <= seen[key(a, b)] <= hi)
                    tot[j] += 1
                    hit[j] += int(gen[i][k] == c)
            label = lambda lo, hi: str(lo) if lo == hi else f"{lo}+" if hi > 10 ** 8 else f"{lo}-{hi}"
            print(f"  {model}, {n} puzzles: " + "; ".join(f"{label(*bins[j])} rows: {hit[j]} of {tot[j]}"
                                                           for j in range(len(bins)) if tot[j]))


if __name__ == "__main__":
    main()
