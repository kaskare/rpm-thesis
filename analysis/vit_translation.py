#!/usr/bin/env python3
"""Does a trained image model keep its answers when the picture moves? A CNN's filters are equivariant to shifts by
construction; a ViT's patches are only shifted onto each other by whole patches, and its learned position embeddings
tie every patch to its place. This moves the image down and right on the unseen puzzles (the ViT: the matrix; the CNN
and vit-sym: every panel), filling with paper, and scores the run as trained. The training images are never shifted.
I-RAVEN has no such symmetry (a panel's place is a row and a column), so it asks what the model does, not what it
should do. Environment nvsa, one GPU, from the repository root:
    python analysis/vit_translation.py experiments/baselines/runs/iraven_vit_x16_wd0.05_do0.1_s0
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "baselines"))
from train_baseline import Data, build, evaluate, images, set_vit


def shifted(vit_input, d):
    def moved(idx, k, swap=None):
        x, k = vit_input(idx, k, swap)
        y = torch.zeros_like(x)
        y[..., d:, d:] = x[..., :x.shape[-2] - d, :x.shape[-1] - d]
        return y, k
    return moved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--shifts", default="0,1,2,4,8,16,32")
    ap.add_argument("--data-dir", default="experiments/baselines/data")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run_dir in args.runs:
        info = json.load(open(os.path.join(run_dir, "best.json")))
        a = info["args"]
        z = np.load(os.path.join(args.data_dir, f"{a['data']}.npz"))
        meta = json.loads(str(z["meta"]))
        set_vit(meta, a)
        data = Data(z, "unseen", meta, dev, images(args.data_dir, a["data"], a["arch"], z, a.get("image_size", 80)).get("unseen"))
        model = build(a["arch"], meta, info["width"], False, a.get("dropout", 0.0)).to(dev)
        model.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pt"), map_location=dev))
        attr = "vit_input" if a["arch"] == "vit" else "vit_panels"
        base, res = getattr(data, attr), {}
        print(run_dir, flush=True)
        for d in (int(x) for x in args.shifts.split(",")):
            setattr(data, attr, shifted(base, d))
            r = evaluate(model, a["arch"], data, meta)
            res[d] = {"matched": r["matched"], "exact": r["exact"], "per_constellation": r["per_constellation"]}
            print(f"  shift {d:3d} px  matched {100 * r['matched']:5.1f}  exact {100 * r['exact']:5.1f}", flush=True)
        json.dump(res, open(os.path.join(run_dir, "translation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
