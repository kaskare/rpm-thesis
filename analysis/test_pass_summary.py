#!/usr/bin/env python3
"""The test pass of Chapters 6 and 7 (jobs/pynl/submit_test_pass.sh, 2026-11-23): for every checkpoint in the lists
experiments/test_pass/*.txt, its score on the unseen puzzles (best.json, the epoch the selection puzzles picked) next to
its score on the test range (test.json, the same checkpoint, scored once). Runs that differ only in their seed (_s1,
_s2, _s3, and the unseeded first run) are grouped, with mean and sample sd.

Columns: matched and exact % (whole puzzle); for the rule twins also exact % on the twin rule (sum or midpoint, kept
under the key Arithmetic), the cell of Chapter 7. Unseen: I-RAVEN-X 6300..7299, I-RAVEN val 300..1299 per layout.
Test: I-RAVEN-X 8000..9999, I-RAVEN test 0..1999 per layout.

The unseen column is best.json as saved. Its exact is the old scorer for facet runs on I-RAVEN-X before 2026-11-08 and
value-axis runs before 2026-10-31 (RESULTS, notes on the main table), up to 4.4 points low; compare exact with the
cells of RESULTS §14, not with this column. The twins' 6000-puzzle runs saved the epoch picked by matched, not the one
their RESULTS cells read.

    python3 analysis/test_pass_summary.py                  # every list
    python3 analysis/test_pass_summary.py valaxis          # lists whose name contains this
"""
import json, os, re, statistics, sys

LISTS = "experiments/test_pass"


def unseen(d):
    b = json.load(open(os.path.join(d, "best.json")))
    u = b["unseen"]
    if "per_rule" in u:                                    # value axis: fractions
        return 100 * u["matched"], 100 * u["exact"], 100 * u["per_rule"].get("Arithmetic", float("nan"))
    n = sum(m for _, m in u["per_constellation"].values()) if u.get("per_constellation") else 1000
    return 100 * u["matched"] / n, 100 * u["exact"] / n, float("nan")


def test(d):
    f = os.path.join(d, "test.json")
    if not os.path.exists(f):
        return None
    t = json.load(open(f))
    ar = t.get("exact_value_per_rule", {}).get("Arithmetic")
    return t["matched"], t["exact"], (100 * ar[0] / ar[1] if ar and ar[1] else float("nan"))


def fmt(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return "–"
    if len(xs) == 1:
        return f"{xs[0]:.1f}"
    return f"{statistics.mean(xs):.1f} ± {statistics.stdev(xs):.1f}"


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    for name in sorted(os.listdir(LISTS)):
        if not name.endswith(".txt") or want not in name:
            continue
        runs = [l.strip() for l in open(os.path.join(LISTS, name)) if l.strip()]
        groups = {}
        for d in runs:
            groups.setdefault(re.sub(r"_s\d$", "", d), []).append(d)
        twins = "valaxis" in name or "twins" in name
        print(f"\n{name}: {len(runs)} checkpoints")
        head = "unseen m / e" + (" / rule" if twins else "")
        print(f"  {'run':<48} {'n':>2}  {head:<34} test m / e" + (" / rule" if twins else ""))
        missing = 0
        for g, ds in groups.items():
            us = [unseen(d) for d in ds]
            ts = [test(d) for d in ds]
            missing += sum(t is None for t in ts)
            ts = [t for t in ts if t is not None]
            k = 3 if twins else 2
            ucell = " / ".join(fmt([u[i] for u in us]) for i in range(k))
            tcell = " / ".join(fmt([t[i] for t in ts]) for i in range(k)) if ts else "not yet"
            print(f"  {os.path.basename(g):<48} {len(ds):>2}  {ucell:<34} {tcell}")
        if missing:
            print(f"  ({missing} checkpoints not scored yet)")


if __name__ == "__main__":
    main()
