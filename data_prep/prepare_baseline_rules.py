#!/usr/bin/env python3
"""The rules of every I-RAVEN puzzle in experiments/baselines/data/iraven.npz, in its order, from the XML files, for
train_baseline.py --hold-pair: an array [puzzles, 2 components, 5 attributes] of rule indices, -1 where there is
none. Number/Position (a Constant over both) counts for Number and Position. Environment nvsa, from the repository
root; a few minutes, so submit it rather than run it on the login node:
    python data_prep/prepare_baseline_rules.py      # -> experiments/baselines/data/iraven_rules.npz
"""
import argparse, json, os, sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "baselines"))
from rule_eval import RULES, xml_rules

SPLITS = {"train": ("train", 0, 6000), "sel": ("val", 0, 300), "unseen": ("val", 300, 1000)}   # prepare_baseline_data.py's
ATTRS = ("Type", "Size", "Color", "Number", "Position")


def table(path):
    t = np.full((2, 5), -1, np.int8)
    for k, r in xml_rules(path).items():
        for a, name in r.items():
            for at in a.split("/"):
                t[int(k[1:]), ATTRS.index(at)] = RULES.index(name)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/I-RAVEN-10000")
    ap.add_argument("--data-dir", default="experiments/baselines/data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    meta = json.loads(str(np.load(os.path.join(args.data_dir, "iraven.npz"))["meta"]))
    out = {"names": json.dumps({"attrs": ATTRS, "rules": RULES})}
    with Pool(args.workers) as pool:
        for s, (split, start, n) in SPLITS.items():
            paths = [os.path.join(args.root, cfg, f"RAVEN_{i}_{split}.xml")
                     for cfg in meta["constellations"] for i in range(start, start + n)]
            out[f"{s}_rules"] = np.stack(pool.map(table, paths, chunksize=64))
            print(s, out[f"{s}_rules"].shape, flush=True)
    np.savez(os.path.join(args.data_dir, "iraven_rules.npz"), **out)


if __name__ == "__main__":
    main()
