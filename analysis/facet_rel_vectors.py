#!/usr/bin/env python3
"""Did the facet model rediscover the rules? It is never told them. This reads its learned row relation, the atom
rel(Rw, C, A), on unseen I-RAVEN puzzles: per component and attribute, the mean over the two complete rows, with the
rule the puzzle's XML gives that attribute. analysis/facet_rel_analysis.py then asks whether these vectors separate the
rules. Writes <run>/rel_vectors.npz, or with --untrained rel_vectors_untrained.npz: the same template at its random
initial weights, the control. Environment pynl, as a job, from the repository root:
    python3 analysis/facet_rel_vectors.py experiments/facet_all_w32-64-128_hidden --n 200
"""
import argparse, ast, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "pynl"))
import bench_hub_grounding as B
from facet_rule_eval import xml_rules


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=200, help="unseen puzzles per layout")
    ap.add_argument("--jvm-gb", type=int, default=64)
    ap.add_argument("--untrained", action="store_true", help="keep the initial weights: the control")
    args = ap.parse_args()

    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn import Adam, MSE

    best = json.load(open(os.path.join(args.ckpt, "best.json")))
    a = best["args"]
    a = ast.literal_eval(a) if isinstance(a, str) else a
    cfgs = list(B.POS_MAP) if a["constellation"] == "all" else a["constellation"].split(",")
    root = a.get("root", "data/I-RAVEN-10000")
    widths = [int(w) for w in a["widths"].split(",")]
    model = B.build_facet_template(a["dim"], a["layers"], True, True, widths)
    model.build(Settings(optimizer=Adam(lr=a["lr"]), error_function=MSE(), iso_value_compression=False))
    if not args.untrained:
        model.load(os.path.join(args.ckpt, "template_best.bin"))
    data = B.load(root, cfgs, "val", 300, args.n)
    rules = [xml_rules(os.path.join(root, c, f"RAVEN_{i}_val.xml")) for c in cfgs for i in range(300, 300 + args.n)]

    keys, samples = [], []
    for i, ((q, y, cfg), rr) in enumerate(zip(data, rules)):
        for k in B.comps(cfg):
            gid = f"u{i}{k}"
            facts = B.puzzle_facts(q, gid) + B.facet_facts(q, "therm", cfg, False) + [R.qcomp(gid, k)]
            for attr in B.FACET_ATTRS:
                keys.append((i, cfgs.index(cfg), int(k[1:]), B.FACET_ATTRS.index(attr), rr[k][attr.capitalize()]))
                for row in ("r1", "r2"):
                    samples.append(Sample(R.rel(row, k, attr)[[0.0] * widths[-1]], facts))
    print(f"{len(data)} puzzles, {len(keys)} attribute instances, {len(samples)} queries", flush=True)

    outs = []
    for c0 in range(0, len(samples), 1200):                                             # grounded in chunks, to bound memory
        outs += [np.asarray(o, dtype=np.float32).flatten()
                 for o in model.test(list(model.build_dataset(Dataset(samples[c0:c0 + 1200]))))]
        print(f"  {len(outs)} of {len(samples)}", flush=True)
    vecs = np.stack(outs).reshape(len(keys), 2, -1).mean(1)
    names = sorted({k[4] for k in keys})
    np.savez(os.path.join(args.ckpt, f"rel_vectors{'_untrained' if args.untrained else ''}.npz"), rel=vecs, puzzle=np.array([k[0] for k in keys]),
             layout=np.array([k[1] for k in keys]), component=np.array([k[2] for k in keys]),
             attr=np.array([k[3] for k in keys]), rule=np.array([names.index(k[4]) for k in keys]),
             rule_names=np.array(names), layouts=np.array(cfgs), attrs=np.array(B.FACET_ATTRS))


if __name__ == "__main__":
    main()
