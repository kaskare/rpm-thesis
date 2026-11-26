#!/usr/bin/env python3
"""Does the row descriptor of a trained value-axis model carry the row's rule (docs/ARITHMETIC.md, "The rule gate")?

On the rule twins a midpoint question row can start like a Progression row; only the two complete rows then say which
rule it is, and the model reads them through rd(Rw, A), the row's pairs pooled over the value axis. This script asks a
saved checkpoint for rd("r1", A) and for the rule descriptor ctx2(A) on unseen midpoint puzzles (any atom of the
template can be the query of a sample) and compares the attributes under midpoint with those under Progression.

    python3 analysis/twins_probe.py                         # the midpoint runs of the shared and scaled models, seeds 1-3
    python3 analysis/twins_probe.py --runs scaled_midpoint_n600_selexact_s2 --n 30

Per run: the exact % on the test rule (best.json), the largest difference per feature between the mean rd of a midpoint
row and of a Progression row, the largest spread of rd within the midpoint rows, and the same difference for ctx2.
With rows read in their own unit (scaled) every midpoint row has one rd and every Progression row another, so the
difference between the two is the whole of what rd says about the rule.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core", "pynl"))
import numpy as np

ROOT = "experiments/twins"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[f"{m}_midpoint_n{n}_selexact_s{s}" for m in
                    ("shared", "sharedgate", "scaled", "scaledgate") for n in (100, 600) for s in (1, 2, 3)])
    ap.add_argument("--n", type=int, default=30, help="unseen puzzles 6300..")
    ap.add_argument("--jvm-gb", type=int, default=16)
    args = ap.parse_args()

    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn import Adam, MSE
    import bench_hub_grounding as B
    import value_axis as VA

    print("largest difference per feature between midpoint rows and Progression rows; spread: within the midpoint rows")
    print(f"{'run':40s} {'exact %':>7s}  {'rd: difference':>14s} {'spread':>8s}   {'ctx2: difference':>16s} {'spread':>8s}")
    for run in args.runs:
        path = os.path.join(ROOT, run)
        if not os.path.exists(os.path.join(path, "best.json")):
            continue
        best = json.load(open(os.path.join(path, "best.json")))
        a = best["args"]
        B.load_iravenx(a["iravenx"], 0, 1)
        maxval = VA.maxval_of(a["iravenx"])
        B.configure(n_cols=3, maxval=maxval)
        top = 2 * maxval if a.get("ring") else maxval
        model = VA.build_value_template(a["dim"], top, plus=not a["no_plus"], per_value=a.get("per_value", False),
                                        offsets=tuple(range(-a["kernel"], a["kernel"] + 1)), ring=a.get("ring", False),
                                        pair=a.get("pair", False), scale=a.get("scale"), gate=a.get("gate", False),
                                        steps=a.get("steps", False))
        model.build(Settings(optimizer=Adam(lr=1e-3), error_function=MSE(), iso_value_compression=False,
                             chain_pruning=False))
        model.load(os.path.join(path, "template_best.bin"))
        data, rules = B.load_iravenx(a["iravenx"], 6300, args.n), VA.rules_of(a["iravenx"], 6300, args.n)
        atoms = {"rd": lambda A: R.rd("r1", A), "ctx2": lambda A: R.ctx2(A)}
        samples, keys = [], []
        for i, (pz, _, _) in enumerate(data):
            facts = VA.value_facts(pz, f"u{i}")
            for A in VA.ATTRS:
                rule = rules[i][A.capitalize()]
                if rule in ("Arithmetic", "Progression") and pz[0][0][A] != pz[1][0][A]:
                    for name, q in atoms.items():
                        samples.append(Sample(q(A)[[0.0] * a["dim"]], facts))
                        keys.append((name, rule))
        vec = {}
        for key, o in zip(keys, model.test(list(model.build_dataset(Dataset(samples))))):
            vec.setdefault(key, []).append(np.asarray(o, dtype=float).reshape(-1))
        cells = []
        for name in atoms:
            M, P = np.array(vec[(name, "Arithmetic")]), np.array(vec[(name, "Progression")])
            cells += [np.abs(M.mean(0) - P.mean(0)).max(), np.abs(M - M.mean(0)).max()]
        print(f"{run:40s} {100 * best['unseen']['per_rule']['Arithmetic']:7.1f}  {cells[0]:14.1e} {cells[1]:8.1e}   "
              f"{cells[2]:16.1e} {cells[3]:8.1e}", flush=True)


if __name__ == "__main__":
    main()
