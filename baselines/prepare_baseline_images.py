#!/usr/bin/env python3
"""Images for the end-to-end vision reference (baselines/train_baseline.py --arch vit): the eight context panels
of every I-RAVEN puzzle in experiments/baselines/data/iraven.npz, in its order, downsampled from 160x160 to
80x80 (2x2 means; most RPM image models read 80x80), as uint8. The answer index is stored as well, so the
trainer can check that both files list the same puzzles. Environment nvsa, from the repository root; a few
minutes on CPU, so submit it rather than run it on the login node (baselines/submit_vit.sh shows how):
    python baselines/prepare_baseline_images.py      # -> experiments/baselines/data/iraven_images.npz
"""
import argparse, json, os
from multiprocessing import Pool

import numpy as np

SPLITS = {"train": ("train", 0, 6000), "sel": ("val", 0, 300), "unseen": ("val", 300, 1000)}   # prepare_baseline_data.py's


def panels(stem):
    z = np.load(stem + ".npz")
    im = z["image"][:8].astype(np.float32).reshape(8, 80, 2, 80, 2).mean((2, 4))
    return np.rint(im).astype(np.uint8), int(z["target"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/I-RAVEN-10000")
    ap.add_argument("--data-dir", default="experiments/baselines/data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    meta = json.loads(str(np.load(os.path.join(args.data_dir, "iraven.npz"))["meta"]))
    out = {}
    with Pool(args.workers) as pool:
        for s, (split, start, n) in SPLITS.items():
            stems = [os.path.join(args.root, cfg, f"RAVEN_{i}_{split}")
                     for cfg in meta["constellations"] for i in range(start, start + n)]
            res = pool.map(panels, stems, chunksize=64)
            out[f"{s}_img"] = np.stack([r[0] for r in res])
            out[f"{s}_y"] = np.array([r[1] for r in res], np.int16)
            print(s, out[f"{s}_img"].shape, flush=True)
    tmp = os.path.join(args.data_dir, "iraven_images.tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, os.path.join(args.data_dir, "iraven_images.npz"))


if __name__ == "__main__":
    main()
