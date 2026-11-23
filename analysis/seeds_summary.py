#!/usr/bin/env python3
"""Facet runs over seeds: for each run directory given, the first run (its seed was not recorded) and the seeded reruns
<dir>_s1, <dir>_s2, ... (bench_hub_grounding.py --seed; jobs/pynl/submit_symmetry_ablation.sh and submit_column_roles.sh
with SEED). Per run: matched % (exact %) on the unseen puzzles at the epoch picked on the selection puzzles, read from
best.json; then mean and sample sd over all runs found. Runs without a best.json yet are listed as missing, runs whose
history has not reached the last epoch as running (their best.json is not final).

    python3 analysis/seeds_summary.py experiments/facet_symabl_lrud_base_100 experiments/facet_symabl_lrud_sepm_100
"""
import json, os, statistics, sys


def score(d):
    f = os.path.join(d, "best.json")
    if not os.path.exists(f):
        return None
    b = json.load(open(f))
    h = os.path.join(d, "history.json")
    if not os.path.exists(h) or json.load(open(h))[-1]["epoch"] < b["args"]["train_epochs"]:
        return "running"
    u = b["unseen"]
    n = sum(m for _, m in u["per_constellation"].values()) if u.get("per_constellation") else 1000
    return 100 * u["matched"] / n, 100 * u["exact"] / n, b["epoch"]


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else float("nan")


if __name__ == "__main__":
    print(f"{'run':<44} {'first':>14} {'seed 1':>14} {'seed 2':>14} {'seed 3':>14}   mean ± sd matched (exact)")
    for base in sys.argv[1:]:
        base = base.rstrip("/")
        cells, ms, es = [], [], []
        for d in [base] + [f"{base}_s{k}" for k in (1, 2, 3)]:
            s = score(d) if os.path.isdir(d) else None
            if s is None or s == "running":
                cells.append("–" if not os.path.isdir(d) else (s or "missing"))
                continue
            cells.append(f"{s[0]:.1f} ({s[1]:.1f})")
            ms.append(s[0]); es.append(s[1])
        mean = (f"{statistics.mean(ms):.1f} ± {sd(ms):.1f} ({statistics.mean(es):.1f} ± {sd(es):.1f}), n={len(ms)}"
                if ms else "")
        print(f"{os.path.basename(base):<44} " + " ".join(f"{c:>14}" for c in cells) + f"   {mean}")
