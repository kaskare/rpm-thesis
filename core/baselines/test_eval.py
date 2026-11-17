#!/usr/bin/env python3
"""The torch baselines on the I-RAVEN test split (test 0..1999 of every layout, 14,000 puzzles: the facet model's test
pass, core/pynl/facet_e2e_eval.py), once per checkpoint, with train_baseline.py's own evaluate().

Per run directory:
  1. the checkpoint (model_best.pt, the epoch picked on the selection puzzles) is scored again on the unseen validation
     puzzles; it must give the numbers recorded in best.json, to 0.05 points (rounding on another GPU can move a puzzle
     or two). If it does not, the test split is not touched;
  2. it is scored on the test split, and the result is written to <run>/test.json.
A run that has a test.json is skipped: the test split is scored once per final checkpoint. --unseen-only stops after
step 1 (a dry run).

Data: data_prep/prepare_baseline_data.py --test and, for the image models, data_prep/prepare_baseline_images.py --test.
Environment nvsa, one GPU, from the repository root (jobs/baselines/submit_test_once.sh):
    python core/baselines/test_eval.py experiments/baselines/runs/iraven_attr-rel_reg_do0_wd0.5_s0 [more run directories]
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_baseline import IMAGES, Data, build, check, evaluate, set_vit

FILES = {"unseen": "iraven", "test": "iraven_test"}
TOL = 0.0005                                                                # 0.05 points


def split_data(data_dir, split, a, meta, dev):
    """One split on the device, its targets checked against bench_hub_grounding's own; the images for image models."""
    z = np.load(os.path.join(data_dir, FILES[split] + ".npz"))
    img = None
    if a["arch"] in IMAGES:
        size = a.get("image_size") or 80
        zi = np.load(os.path.join(data_dir, f"{FILES[split]}_images{size if size != 80 else ''}.npz"))
        assert (zi[f"{split}_y"] == z[f"{split}_y"]).all(), f"{split}: the images and the arrays list different puzzles"
        img = zi[f"{split}_img"]
    data = Data(z, split, meta, dev, img)
    check(data, meta)
    return data


def score(run_dir, args, dev):
    out = os.path.join(run_dir, "test.json")
    if os.path.exists(out) and not args.unseen_only:
        print(f"\n{run_dir}: already scored on the test split (test.json), skipped", flush=True)
        return json.load(open(out))
    info = json.load(open(os.path.join(run_dir, "best.json")))
    a = info["args"]
    if a["data"] != "iraven":
        raise SystemExit(f"{run_dir}: I-RAVEN only")
    meta = json.loads(str(np.load(os.path.join(args.data_dir, "iraven.npz"))["meta"]))
    set_vit(meta, a)
    model = build(a["arch"], meta, info["width"], a.get("norm", False), a.get("dropout") or 0.0).to(dev)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pt"), map_location=dev))

    rec = info["best"]["unseen"]
    again = evaluate(model, a["arch"], split_data(args.data_dir, "unseen", a, meta, dev), meta)
    same = all(abs(again[m] - rec[m]) <= TOL for m in ("matched", "exact"))
    print(f"\n{run_dir}: {info['weights']} weights, epoch {info['best']['epoch']}\n  unseen  matched "
          f"{100 * again['matched']:.2f} (exact {100 * again['exact']:.2f}); recorded {100 * rec['matched']:.2f} "
          f"({100 * rec['exact']:.2f}): {'reproduced' if same else 'NOT reproduced, the test split is left alone'}", flush=True)
    if not same or args.unseen_only:
        return None
    torch.cuda.empty_cache()
    data = split_data(args.data_dir, "test", a, meta, dev)
    res = evaluate(model, a["arch"], data, meta)
    res = {"run": run_dir, "arch": a["arch"], "epoch": info["best"]["epoch"], "weights": info["weights"],
           "split": "I-RAVEN test 0..1999 of every layout", "puzzles": int(data.keep.sum()),
           "unseen_again": {m: again[m] for m in ("matched", "exact")},
           "unseen_recorded": {m: rec[m] for m in ("matched", "exact")}, **res}
    json.dump(res, open(out, "w"), indent=1)
    print(f"  test    matched {100 * res['matched']:.2f} (exact {100 * res['exact']:.2f}) on {res['puzzles']} puzzles; "
          + "  ".join(f"{c[:12]} {100 * h / n:.1f}" for c, (h, n) in res["per_constellation"].items()), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories with best.json and model_best.pt")
    ap.add_argument("--data-dir", default="experiments/baselines/data")
    ap.add_argument("--unseen-only", action="store_true", help="only check that the unseen record is reproduced")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done = [(r, score(r, args, dev)) for r in args.runs]
    if not args.unseen_only:
        print("\nrun | weights | unseen matched (exact) | test matched (exact)")
        for r, t in done:
            name = os.path.basename(r.rstrip("/"))
            if t is None:
                print(f"{name} | not scored")
            else:
                print(f"{name} | {t['weights']} | {100 * t['unseen_recorded']['matched']:.2f} "
                      f"({100 * t['unseen_recorded']['exact']:.2f}) | {100 * t['matched']:.2f} ({100 * t['exact']:.2f})")


if __name__ == "__main__":
    main()
