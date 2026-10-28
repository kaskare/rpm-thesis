#!/usr/bin/env python3
"""Table of the torch baselines (baselines/train_baseline.py) next to the facet model, on the unseen puzzles
at the epoch picked on the selection split; mean +- standard deviation over seeds. Rows marked "Nx weights"
are the capacity sweep (--scale N); "reg_..." rows are regularized runs (submit_regularized.sh), and * marks
the run of each architecture and size whose selection accuracy is highest.
    python3 baselines/summarize_baselines.py [--per-rule]
"""
import argparse, glob, json, os, statistics

FACET = {"iraven": ["facet_all_w32-64-128_hidden"],
         "iravenx_n3_m10": ["facet_iravenx_n3_m10"],
         "iravenx_n10_m50": ["facet_iravenx_n10_m50", "facet_iravenx_n10_m50_copy"],
         "iravenx_n10_m100": ["facet_iravenx_n10_m100", "facet_iravenx_n10_m100_copy"],
         "iravenx_n10_m1000": ["facet_iravenx_n10_m1000", "facet_iravenx_n10_m1000_copy"]}


def pm(xs):
    xs = [100 * x for x in xs]
    return f"{statistics.mean(xs):5.1f} ± {statistics.stdev(xs):4.1f}" if len(xs) > 1 else f"{xs[0]:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="experiments/baselines/runs")
    ap.add_argument("--per-rule", action="store_true")
    args = ap.parse_args()
    runs = {}
    for f in sorted(glob.glob(os.path.join(args.runs, "*", "best.json"))):
        b = json.load(open(f))
        a = b["args"]
        runs.setdefault((a["data"], a["arch"], a.get("scale", 1.0), a.get("tag", "")), []).append(b)
    sel = {key: statistics.mean(r["best"]["sel"]["matched"] for r in rs) for key, rs in runs.items()}
    print(f"{'dataset':18s} {'model':42s} {'weights':>9s} {'seeds':>5s} {'epoch':>9s} {'sel %':>6s}  "
          f"{'matched %':>13s}  {'exact %':>13s}")
    for data, facets in FACET.items():
        for name in facets:
            path = f"experiments/{name}/best.json"
            if os.path.exists(path):
                u = json.load(open(path))["unseen"]
                n = sum(t for _, t in u["per_constellation"].values())
                label = "facet + copy links" if name.endswith("_copy") else "facet"
                print(f"{data:18s} {label:42s} {'':>9s} {1:5d} {'':>9s} {'':>6s}  {100 * u['matched'] / n:13.1f}  "
                      f"{100 * u['exact'] / n:13.1f}")
        keys = sorted((k for k in runs if k[0] == data), key=lambda k: (k[1] != "mlp", k[2], k[3]))
        for key in keys:
            _, arch, scale, tag = key
            rs = runs[key]
            ep = "/".join(str(r["best"]["epoch"]) for r in rs)
            rivals = [sel[k] for k in keys if k[1:3] == (arch, scale)]
            star = "*" if len(rivals) > 1 and sel[key] == max(rivals) else " "
            label = f"{star}{arch}" + (f", {scale:g}x weights" if scale != 1 else "") + (f", {tag}" if tag else "")
            print(f"{data:18s} {label:42s} {rs[0]['weights']:9d} {len(rs):5d} {ep:>9s} {100 * sel[key]:6.1f}  "
                  f"{pm([r['best']['unseen']['matched'] for r in rs]):>13s}  {pm([r['best']['unseen']['exact'] for r in rs]):>13s}")
            if args.per_rule and rs[0]["best"]["unseen"]["exact_per_rule"]:
                for rule in rs[0]["best"]["unseen"]["exact_per_rule"]:
                    xs = [r["best"]["unseen"]["exact_per_rule"][rule] for r in rs]
                    print(f"{'':18s}   exact, {rule:14s} {'':>53s} {pm([h / t for h, t in xs]):>29s}")
        print()


if __name__ == "__main__":
    main()
