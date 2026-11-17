#!/usr/bin/env python3
"""Sum against midpoint: what a built-in symmetry does to each rule's table (docs/ARITHMETIC.md, "Sum and midpoint
compared"). No network: learners that know only a symmetry, fit by counting on the rule-twin puzzles
(data_prep/rule_twins.py, rows of 3, values 0..50).

A learner predicts the answer c of the question row (a, b, ?) from a and b. Its built-in symmetry is a group H of moves of
(a, b, c), and the learner must be equivariant: moving the input by h moves its answer by h. It therefore learns one
answer per orbit of inputs under H, and a training row teaches every row of its orbit.
  table     H = nothing: one answer per pair (a, b). This is memorising.
  diagonal  H = every panel +s, the convolution's symmetry: c = b + g(b - a), one answer per first step b - a.
  operand   H = a +s with c +s, and b +s with c +s, addition's own moves: c = a + b + k, one number k for the whole table.
  doubling  H = every panel +s, and every panel doubled: c = b + S g(m), where the first step is b - a = m S with m odd
            and S a power of two. One answer per odd part m ("The scaling kernel", value_axis.py --scale pow2).
  scaling   H = every panel +s, and every panel times any number: c = b + g (b - a), one number g for the whole table
            (value_axis.py --scale step).
The last two count on the ring of 101 values the model uses (value_axis.py --ring): 101 is prime, so every step has
an inverse and a half-step is a ring element. A row with a = b has one answer under either, c = a.
Each learner is fit on the question rows of the first N puzzles (those are the rows with a loss in every model here), by
majority vote per orbit, and scored on the unseen puzzles 6300..7299, attributes that follow the test rule. An input whose
orbit no training row reached counts as wrong. "ceiling" fits the learner on every row of the rule's table: the most an
equivariant learner can get with unlimited data.
    python3 analysis/rule_orbits.py [--rules sum midpoint mirror] [--complete]
"""
import argparse, collections, json

ATTRS = ("Type", "Size", "Color")
SIZES = (10, 30, 100, 300, 600, 1000, 6000)
RING = 101                                            # the model's ring of values, 2 * 50 + 1: a prime
inv = lambda x: pow(x % RING, -1, RING)
pow2 = lambda d: (d & -d) if d else 0                 # the largest power of two in a step (d & -d works for d < 0 too)
unit = {"doubling": pow2, "scaling": lambda d: d}     # the unit a row is read in
ORBIT = {"table": lambda a, b: (a, b), "diagonal": lambda a, b: b - a, "operand": lambda a, b: 0,
         "doubling": lambda a, b: (b - a) // pow2(b - a) if a != b else 0, "scaling": lambda a, b: a != b}
ANSWER = {"table": lambda a, b, c: c, "diagonal": lambda a, b, c: c - b, "operand": lambda a, b, c: c - a - b}
PREDICT = {"table": lambda a, b, r: r, "diagonal": lambda a, b, r: b + r, "operand": lambda a, b, r: a + b + r}
for _k, _u in unit.items():                           # the answer in units of the row: (c - b) / unit on the ring
    ANSWER[_k] = lambda a, b, c, u=_u: (c - b) * inv(u(b - a)) % RING if a != b else 0
    PREDICT[_k] = lambda a, b, r, u=_u: (b + r * u(b - a)) % RING if a != b else a


def rows(path, start, n, complete):
    """(a, b, c) for every attribute that follows the test rule, question row first, complete rows if asked."""
    d = json.load(open(path))
    out = []
    for i in range(start, start + n):
        p = d[str(i)]
        rpm, rule = p["rpm"], p["rules"][0]
        ans = rpm[8 + int(p["target"])][0]
        for att in ATTRS:
            if rule[att] != "Arithmetic":
                continue
            v = lambda k: int(rpm[k][0][att])
            out.append((v(6), v(7), int(ans[att])))
            if complete:
                out += [(v(0), v(1), v(2)), (v(3), v(4), v(5))]
    return out


def fit(train, kind):
    votes = collections.defaultdict(collections.Counter)
    for a, b, c in train:
        votes[ORBIT[kind](a, b)][ANSWER[kind](a, b, c)] += 1
    return {o: cnt.most_common(1)[0][0] for o, cnt in votes.items()}


def score(model, test, kind):
    hit = sum(1 for a, b, c in test if ORBIT[kind](a, b) in model and PREDICT[kind](a, b, model[ORBIT[kind](a, b)]) == c)
    return 100 * hit / len(test)


def table_rows(rule, m=50):
    r = range(m + 1)
    if rule == "sum":
        return [(a, b, a + b) for a in r for b in r if a >= 1 and b >= 1 and a + b <= m]
    if rule == "midpoint":
        return [(a, b, (a + b) // 2) for a in r for b in r if a != b and (a + b) % 2 == 0]
    return [(a, b, 2 * a - b) for a in r for b in r if a != b and 0 <= 2 * a - b <= m]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="+", default=["sum", "midpoint", "mirror"])
    ap.add_argument("--complete", action="store_true", help="also fit on the two complete rows of each training puzzle")
    args = ap.parse_args()
    for rule in args.rules:
        path = f"data/iravenx_twins/{rule}/center_single_shuffle_n_3_maxval_50.json"
        test = rows(path, 6300, 1000, False)
        full = table_rows(rule)
        orbits = {k: len({ORBIT[k](a, b) for a, b, _ in full}) for k in ORBIT}
        print(f"\n{rule}: {len(full)} rows in the table; orbits of inputs: "
              + ", ".join(f"{k} {v}" for k, v in orbits.items()) + f"; {len(test)} unseen test rows")
        print(f"{'puzzles':>8}  " + "  ".join(f"{k:>9}" for k in ORBIT) + "   rows   orbits reached (diagonal)")
        for n in SIZES:
            train = rows(path, 0, n, args.complete)
            fits = {k: fit(train, k) for k in ORBIT}
            print(f"{n:>8}  " + "  ".join(f"{score(fits[k], test, k):9.1f}" for k in ORBIT)
                  + f"   {len(train):>4}   {len(fits['diagonal'])}")
        print(f"{'ceiling':>8}  " + "  ".join(f"{score(fit(full, k), test, k):9.1f}" for k in ORBIT))


if __name__ == "__main__":
    main()
