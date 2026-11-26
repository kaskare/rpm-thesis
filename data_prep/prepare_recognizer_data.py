#!/usr/bin/env python3
"""Panels and object labels for the ViT object recognizer (core/frontend/train_vit_recognizer.py), from I-RAVEN's
images and XML. Every panel of a puzzle (8 context, 8 candidates) is one sample: the 160x160 image, and per universal
slot (train_pynl.POS_MAP, 22 slots) whether it holds an object and that object's type, size and colour, read by the
parser the facet model uses (bench_hub_grounding.load). Angle is noise in I-RAVEN and is left out.

Splits follow the facet model's: train puzzles 0.. of the train split; selection val 0..299; unseen val 300..1299;
test 0..1999; all seven layouts, in POS_MAP's order. Written to OUT (default experiments/recognizer/data):
  <split>_img.npy   uint8 [puzzles, 16, 160, 160], loaded with mmap by the trainer
  <split>.npz       lab int8 [puzzles, 16, 22, 4] (occupied, type, size, colour; -1 where empty),
                    layout int8 [puzzles] (index in LAYOUTS), idx int32 (the puzzle's number), y int8 (answer index)
Environment pynl, from the repository root, as a job (about 30,000 XML files):
    python3 data_prep/prepare_recognizer_data.py --n-train 2000
"""
import argparse, functools, json, os, sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "pynl"))
import bench_hub_grounding as B

LAYOUTS = list(B.POS_MAP)
N_SLOTS = 22


def one(root, job):
    cfg, split, i = job
    stem = os.path.join(root, cfg, f"RAVEN_{i}_{split}")
    z = np.load(stem + ".npz")
    panels = B.parse(stem + ".xml", cfg)
    lab = np.full((16, N_SLOTS, 4), -1, np.int8)
    lab[:, :, 0] = 0
    for p, objs in enumerate(panels):
        for o in objs:
            lab[p, o["pos"]] = (1, o["type"], o["size"], o["color"])
    return z["image"], lab, int(z["target"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/I-RAVEN-10000")
    ap.add_argument("--out", default="experiments/recognizer/data")
    ap.add_argument("--n-train", type=int, default=2000, help="training puzzles per layout")
    ap.add_argument("--splits", default="train,sel,unseen,test")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    specs = {"train": ("train", 0, args.n_train), "sel": ("val", 0, 300), "unseen": ("val", 300, 1000), "test": ("test", 0, 2000)}
    os.makedirs(args.out, exist_ok=True)
    with Pool(args.workers) as pool:
        for s in args.splits.split(","):
            split, start, n = specs[s]
            jobs = [(cfg, split, i) for cfg in LAYOUTS for i in range(start, start + n)]
            img = np.lib.format.open_memmap(os.path.join(args.out, f"{s}_img.tmp.npy"), "w+", np.uint8, (len(jobs), 16, 160, 160))
            lab, y = np.zeros((len(jobs), 16, N_SLOTS, 4), np.int8), np.zeros(len(jobs), np.int8)
            for j, (im, lb, t) in enumerate(pool.imap(functools.partial(one, args.root), jobs, chunksize=32)):
                img[j], lab[j], y[j] = im, lb, t
            img.flush()
            del img
            os.replace(os.path.join(args.out, f"{s}_img.tmp.npy"), os.path.join(args.out, f"{s}_img.npy"))
            np.savez(os.path.join(args.out, f"{s}.npz"), lab=lab, y=y,
                     layout=np.array([LAYOUTS.index(c) for c, _, _ in jobs], np.int8),
                     idx=np.array([i for _, _, i in jobs], np.int32), split=split,
                     meta=json.dumps({"layouts": LAYOUTS, "slots": N_SLOTS, "spec": specs[s], "root": args.root}))
            occ = lab[..., 0].sum()
            print(f"{s}: {len(jobs)} puzzles, {16 * len(jobs)} panels, {occ} objects", flush=True)


if __name__ == "__main__":
    main()
