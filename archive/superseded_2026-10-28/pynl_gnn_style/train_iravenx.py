#!/usr/bin/env python3
"""The best PyNeuraLogic graph model (train_all_constellations.py) on I-RAVEN-X, with XML-label-style
symbolic input. Only n = 3 fits the model's 16-panel graph, i.e. the 3x3 variant from IBM's generator:
    python src/datasets/generation/iravenx_task.py --n 3 --maxval 10 --nconf 0 --save_dir <dir>
Every attribute takes values 0..maxval-1, so the type and size vocabularies grow to maxval. Puzzles
0-5999 are train, 6000-7999 val, 8000-9999 test (SRAN's 6/2/2 split). All other arguments are those
of train_all_constellations.py; --dataset-root is the I-RAVEN-X .json(.gz).

    python3 pynl/train_iravenx.py --dataset-root data/iravenx/center_single_shuffle_n_3_maxval_10.json \
        --level 2 --n-train 6000 --n-val 300 --no-same-panel --epochs 30 --save-dir experiments/iravenx
"""
import argparse, gzip, json, math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_all_constellations as tac
from train_pynl import N_POS, N_PNL

SPLITS = {"train": (0, 6000), "val": (6000, 8000), "test": (8000, 10000)}


def read(path):
    with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
        return json.load(f)


def load(path, cfgs, split, n_per):
    d = read(path)
    lo, hi = SPLITS[split]
    out = []
    for k in range(lo, min(hi, lo + n_per) if n_per else hi):
        s = d[str(k)]
        if math.isqrt(len(s["rpm"]) - 7) != 3:
            raise SystemExit("only the 3x3 I-RAVEN-X (--n 3) fits the 16-panel graph")
        panels = [[{"pos": 0, "type": int(p[0]["Type"]), "size": int(p[0]["Size"]),
                    "color": int(p[0]["Color"])}] for p in s["rpm"]]
        a, m = tac.encode_puzzle(panels, "center_single")
        out.append(("center_single", a, m, int(s["target"])))
    return out


def setup(path):
    maxval = 1 + max(int(p[0][a]) for s in read(path).values() for p in s["rpm"]
                     for a in ("Type", "Size", "Color"))
    tac.ATTRS = [("pos", N_POS), ("col", maxval), ("shp", maxval), ("siz", maxval), ("pnl", N_PNL)]
    tac.CONSTELLATIONS = ["center_single"]
    tac.load = load


if __name__ == "__main__":
    argv = sys.argv[1:]
    root = argv[argv.index("--dataset-root") + 1]
    setup(root)
    if "--test" in argv:          # score model_best of an existing run on the test split, once
        argv.remove("--test")
        ap = argparse.ArgumentParser()
        ap.add_argument("--save-dir", default="experiments/all_constellations")
        ap.add_argument("--level", default="2")
        known, _ = ap.parse_known_args(argv)
        save = f"{known.save_dir}_L{known.level}"
        ck = torch.load(os.path.join(save, "model_best.pt"), map_location="cpu", weights_only=False)
        a = ck["args"]
        model = tac.AllConstellations(a["level"], a["embed_dim"], a["hidden_dim"], a["layers"],
                                      tac.Transformation.LEAKY_RELU, a["nl_lr"],
                                      agg={"avg": tac.Aggregation.AVG, "max": tac.Aggregation.MAX,
                                           "sum": tac.Aggregation.SUM}[a["agg"]],
                                      same_panel=not a["no_same_panel"], layout=a["layout"],
                                      per_type=a["per_type_weights"], absent_pos=a["absent_pos"])
        model.load_state_dict(ck["torch_state"])
        model.nl.model.load(os.path.join(save, "template_best.bin"))
        te = load(root, None, "test", 0)
        acc, _ = tac.evaluate(model, te)
        print(f"test {acc * 100:.2f}% ({len(te)} puzzles), best val epoch {ck['epoch']} "
              f"({ck['val_acc'] * 100:.2f}%)")
    else:
        tac.main()
