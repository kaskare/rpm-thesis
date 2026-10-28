#!/usr/bin/env python3
"""Per-rule evaluation of I-RAVEN-X facet checkpoints (bench_hub_grounding.py --save-dir) on the unseen
puzzles 6300..: matched and exact as in training (bench_hub_grounding.evaluate), matched with and without an
Arithmetic attribute, and the exact value per rule (the generated code's bits summed and rounded, as in notes
§20-21). Optionally every puzzle is transformed first: rows 1 <-> 2, or middle columns 2 <-> 3. The template
is rebuilt from the run's best.json, so any variant (copy links, gates, column roles, widths) loads.

Run in the pynl environment from the repository root, as a job (grounding 1000 puzzles takes ~48 GB):
    python3 pynl/facet_rule_eval.py experiments/facet_iravenx_n10_m50 experiments/facet_iravenx_n10_m50_roles
    python3 pynl/facet_rule_eval.py experiments/facet_iravenx_n10_m50_roles --transform none cols23
"""
import argparse, ast, collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_hub_grounding as B
from facet_symmetry_check import rows12, cols23

TRANSFORMS = {"none": lambda pz: pz, "rows12": rows12, "cols23": cols23}
RULES = ("Constant", "Progression", "Distribute_Three", "Arithmetic")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+", help="run directories with best.json and template_best.bin")
    ap.add_argument("--n", type=int, default=1000, help="unseen puzzles from 6300")
    ap.add_argument("--transform", nargs="*", choices=list(TRANSFORMS), default=["none"])
    ap.add_argument("--jvm-gb", type=int, default=48)
    args = ap.parse_args()

    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn import Adam, MSE

    for ckpt in args.ckpts:
        best = json.load(open(os.path.join(ckpt, "best.json")))
        a = best["args"]
        a = ast.literal_eval(a) if isinstance(a, str) else a
        path = a["iravenx"]
        first = B.load_iravenx(path, 0, 1)[0][0]
        B.configure(n_cols=(len(first) - 7) // 3, maxval=int(path.split("maxval_")[1].split(".")[0]))
        data = B.load_iravenx(path, 6300, args.n)
        raw = B._IRAVENX[path]
        rules = [(lambda r: ast.literal_eval(r) if isinstance(r, str) else r)(raw[str(i)]["rules"])[0]
                 for i in range(6300, 6300 + args.n)]
        copy, roles = bool(a.get("copy_links")), a.get("columns", "named") == "roles"
        widths = [int(w) for w in a["widths"].split(",")] if a.get("widths") else None
        model = B.build_facet_template(a["dim"], a["layers"], not a.get("per_attr_weights"), not a.get("no_rows"),
                                       widths, copy=copy, gates=a.get("copy_gates", "panel"),
                                       cols=a.get("columns", "named"))
        model.build(Settings(optimizer=Adam(lr=a["lr"]), error_function=MSE(), iso_value_compression=False))
        model.load(os.path.join(ckpt, "template_best.bin"))
        n_w = sum(np.asarray(v).size for v in model.state_dict()["weights"].values())
        print(f"\n{ckpt}: {n_w} weights, epoch {best['epoch']}, {args.n} unseen puzzles (6300..)", flush=True)

        for t in args.transform:
            samples = []
            for i, (pz, y, cfg) in enumerate(data):
                q, gid = TRANSFORMS[t](pz), f"{t}{i}"
                facts = (B.puzzle_facts(q, gid, with_feat=False, col_roles=roles) + B.facet_facts(q, "therm", cfg, copy)
                         + [R.qcomp(gid, "k0")])
                samples.append(Sample(R.out(gid)[B.facet_target(q[B.N_CTX + y]).tolist()], facts))
            outs = [np.asarray(o, dtype=float).flatten() for o in model.test(list(model.build_dataset(Dataset(samples))))]
            _, exact, matched, ties, _ = B.evaluate([{"k0": o} for o in outs], data, "therm")
            hits = collections.defaultdict(lambda: [0, 0])             # rule -> [exact values, values]
            by_ar = collections.defaultdict(lambda: [0, 0])            # has Arithmetic? -> [matched, puzzles]
            for o, (pz, y, _), r in zip(outs, data, rules):
                for j, attr in enumerate(B.FACET_ATTRS):
                    h = hits[r[attr.capitalize()]]
                    h[0] += int(round(o[j * B.THERM:(j + 1) * B.THERM].sum())) == pz[B.N_CTX + y][0][attr]
                    h[1] += 1
                dist = [np.sum((o - B.facet_target(pz[B.N_CTX + c])) ** 2) for c in range(8)]
                key = "with Arithmetic" if "Arithmetic" in (r["Type"], r["Size"], r["Color"]) else "without"
                by_ar[key][0] += int(np.argmin(dist) == y and sorted(dist)[1] > dist[y])
                by_ar[key][1] += 1
            n = len(data)
            print(f"  transform {t}: matched {100 * matched / n:.1f}%, exact {100 * exact['all'] / n:.1f}% (ties {ties}); "
                  + ", ".join(f"matched {k} {100 * h / m:.1f}% of {m}" for k, (h, m) in sorted(by_ar.items())))
            print("    exact value: " + " / ".join(f"{r} {100 * hits[r][0] / max(1, hits[r][1]):.1f} (of {hits[r][1]})"
                                                    for r in RULES), flush=True)


if __name__ == "__main__":
    main()
