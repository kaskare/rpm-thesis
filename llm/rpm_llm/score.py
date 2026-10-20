"""Scores result files from run.py. Unparseable answers count as wrong; API errors are
reported, not scored. Accuracy is given per run (mean ± sample std over runs), pooled with a
95% Wilson interval, and per configuration and variant.

  python -m rpm_llm.score results/*.jsonl [--lenient] [--md report.md]
"""
import argparse
import json
import math
import statistics as st
from collections import defaultdict

CHANCE = 1 / 8


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(paths):
    latest = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                key = (r["model"], r["prompt"], r.get("effort"), r["run"], r["id"])
                if r["status"] == "ok" or key not in latest or latest[key]["status"] != "ok":
                    latest[key] = r
    return list(latest.values())


def pct(x):
    return f"{100 * x:.1f}"


def summarise(recs, lenient=False):
    by_setting = defaultdict(list)
    for r in recs:
        by_setting[(r["model"], r["prompt"], r.get("effort"))].append(r)
    lines = []
    for (model, prompt, effort), rs in sorted(by_setting.items(), key=lambda kv: str(kv[0])):
        ok = [r for r in rs if r["status"] == "ok"]
        errors = len(rs) - len(ok)
        pred = (lambda r: r["lenient"]) if lenient else (lambda r: r["prediction"])
        hit = lambda r: pred(r) == r["target"]
        runs = defaultdict(list)
        for r in ok:
            runs[r["run"]].append(hit(r))
        per_run = [sum(v) / len(v) for _, v in sorted(runs.items())]
        k, n = sum(map(hit, ok)), len(ok)
        lo, hi = wilson(k, n)
        unparsed = sum(pred(r) is None for r in ok)
        targets = defaultdict(int)
        for r in ok:
            targets[r["target"]] += 1
        top, top_n = max(targets.items(), key=lambda kv: kv[1]) if targets else ("-", 0)
        sd = f" ± {pct(st.stdev(per_run))}" if len(per_run) > 1 else ""
        title = f"{model} | prompt {prompt}" + (f" | effort {effort}" if effort else "")
        lines += [f"## {title}", "",
                  f"- puzzles × runs: {n} ({len(runs)} runs), API errors not scored: {errors}",
                  f"- accuracy: {pct(k / n) if n else 'nan'}% (per-run mean {pct(st.mean(per_run)) if per_run else 'nan'}{sd}), "
                  f"95% CI {pct(lo)}–{pct(hi)}, chance {pct(CHANCE)}, always-{top} baseline {pct(top_n / n) if n else 'nan'}",
                  f"- unparseable (counted wrong): {unparsed} ({pct(unparsed / n) if n else 'nan'}%)", ""]
        for field in ("config", "variant"):
            groups = defaultdict(list)
            for r in ok:
                groups[r[field]].append(hit(r))
            if len(groups) > 1:
                lines += [f"| {field} | n | accuracy % |", "|---|---|---|"]
                lines += [f"| {g} | {len(v)} | {pct(sum(v) / len(v))} |" for g, v in sorted(groups.items())]
                lines.append("")
        letters = defaultdict(int)
        for r in ok:
            letters[pred(r) or "none"] += 1
        lines.append("- predicted letters: " + ", ".join(f"{l} {c}" for l, c in sorted(letters.items())))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--lenient", action="store_true", help="use the lenient parser")
    ap.add_argument("--md", help="also write the report to this file")
    a = ap.parse_args(argv)
    text = summarise(load(a.results), a.lenient)
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
