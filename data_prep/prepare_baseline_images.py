#!/usr/bin/env python3
"""Images for the end-to-end vision reference (core/baselines/train_baseline.py --arch vit): the eight context panels
of every I-RAVEN puzzle in experiments/baselines/data/iraven.npz, in its order, downsampled from 160x160 to
80x80 (2x2 means; most RPM image models read 80x80), as uint8; --size 160 keeps the original panels. The answer
index is stored as well, so the trainer can check that both files list the same puzzles. Environment nvsa, from
the repository root; a few minutes on CPU, so submit it rather than run it on the login node (jobs/baselines/submit_vit.sh
shows how):
    python data_prep/prepare_baseline_images.py      # -> experiments/baselines/data/iraven_images.npz
--test writes the test split alone (test 0..1999 per layout, data_prep/prepare_baseline_data.py --test's order) to
iraven_test_images.npz, for core/baselines/test_eval.py.
"""
import argparse, functools, json, os
from multiprocessing import Pool

import numpy as np

SPLITS = {"train": ("train", 0, 6000), "sel": ("val", 0, 300), "unseen": ("val", 300, 1000)}   # prepare_baseline_data.py's
TEST = {"test": ("test", 0, 2000)}


def panels(size, stem):
    z = np.load(stem + ".npz")
    im = z["image"][:8]
    if size == 80:
        im = np.rint(im.astype(np.float32).reshape(8, 80, 2, 80, 2).mean((2, 4))).astype(np.uint8)
    return im, int(z["target"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/I-RAVEN-10000")
    ap.add_argument("--data-dir", default="experiments/baselines/data")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, choices=(80, 160), default=80)
    ap.add_argument("--test", action="store_true", help="the test split only -> iraven_test_images.npz")
    args = ap.parse_args()
    meta = json.loads(str(np.load(os.path.join(args.data_dir, "iraven.npz"))["meta"]))
    out = {}
    with Pool(args.workers) as pool:
        for s, (split, start, n) in (TEST if args.test else SPLITS).items():
            stems = [os.path.join(args.root, cfg, f"RAVEN_{i}_{split}")
                     for cfg in meta["constellations"] for i in range(start, start + n)]
            res = pool.map(functools.partial(panels, args.size), stems, chunksize=64)
            out[f"{s}_img"] = np.stack([r[0] for r in res])
            out[f"{s}_y"] = np.array([r[1] for r in res], np.int16)
            print(s, out[f"{s}_img"].shape, flush=True)
    name = f"iraven{'_test' if args.test else ''}_images{args.size if args.size != 80 else ''}"
    tmp = os.path.join(args.data_dir, f"{name}.tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, os.path.join(args.data_dir, f"{name}.npz"))


if __name__ == "__main__":
    main()
