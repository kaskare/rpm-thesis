#!/usr/bin/env python3
"""Data for the torch baselines (baselines/train_baseline.py): the facet model's puzzles, splits and
weight counts, written as plain arrays, so that the GPU environment (nvsa: Python 3.7, no PyNeuraLogic)
can read them. The baselines compute the facet model's targets and metrics from these arrays; the
`check_*` arrays hold pynl/bench_hub_grounding.py's own targets and answer specs for the first puzzles
of every split, and the trainer refuses to run unless it reproduces them.

Run in the pynl environment, from the repository root (parsing all of I-RAVEN takes a few minutes, so
submit it as a job rather than running it on the login node):
    python3 baselines/prepare_baseline_data.py          # -> experiments/baselines/data/<dataset>.npz

Splits are the facet runs': I-RAVEN train 0-5999 per constellation, val 0-299 pick the epoch, val
300-1299 are unseen; I-RAVEN-X train 0-5999, 6000-6299 pick the epoch, 6300-7299 unseen.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pynl"))
import bench_hub_grounding as B

IRX = "data/iravenx/center_single_shuffle_n_{}_maxval_{}.json"
# dataset -> (I-RAVEN-X file or None, whose weight count the baselines match: the facet model we report)
DATASETS = {"iraven": (None, "plain"),
            "iravenx_n3_m10": (IRX.format(3, 10), "plain"),
            "iravenx_n10_m50": (IRX.format(10, 50), "copy"),
            "iravenx_n10_m100": (IRX.format(10, 100), "copy"),
            "iravenx_n10_m1000": (IRX.format(10, 1000), "copy")}
RULES = ["Constant", "Progression", "Distribute_Three", "Arithmetic"]
N_CHECK = 20


def facet_weights(copy):
    """Learned weights of the facet template (widths 32, 64, 128) in the current configuration."""
    from neuralogic.core import Settings
    from neuralogic.nn import Adam, MSE
    m = B.build_facet_template(32, 2, widths=[32, 64, 128], copy=copy)
    m.build(Settings(optimizer=Adam(lr=3e-4), error_function=MSE(), iso_value_compression=False))
    sd = m.state_dict()
    return sum(int(np.size(v)) for v in (sd["weights"] if "weights" in sd else sd).values())


def flat_spec(spec):
    """A decoded panel spec as one integer vector: occupancy bits, then the value counts per attribute."""
    occ = np.zeros(B.N_POS, np.int64)
    occ[list(spec["pos"])] = 1
    return np.concatenate([occ] + [np.asarray(spec[a], np.int64) for a in B.FACET_ATTRS])


def arrays(data, rules=None):
    maxobj = max(len(p) for panels, _, _ in data for p in panels)
    n, P = len(data), len(data[0][0])
    vals = np.full((n, P, maxobj, 4), -1, np.int16)            # pos, type, size, colour; pos -1 = no object
    comp = np.full((n, P, maxobj), -1, np.int8)                # 0 = k0, 1 = k1
    for i, (panels, _, cfg) in enumerate(data):
        for p, panel in enumerate(panels):
            for j, o in enumerate(panel):
                vals[i, p, j] = (o["pos"], o["type"], o["size"], o["color"])
                comp[i, p, j] = int(B.comp_of(cfg, o["pos"])[1])
    out = {"vals": vals, "comp": comp, "y": np.array([y for _, y, _ in data], np.int16),
           "ncomp": np.array([len(B.comps(cfg)) for _, _, cfg in data], np.int8),
           "cfg": np.array([list(B.POS_MAP).index(cfg) for _, _, cfg in data], np.int8)}
    if rules is not None:
        out["rules"] = np.array([[RULES.index(r[a]) for a in ("Type", "Size", "Color")] for r in rules], np.int8)
    ct = np.zeros((N_CHECK, 2, B.FACET_OUT))
    cs = np.full((N_CHECK, 2, B.N_POS + sum(B.NVALS.values())), -1, np.int64)
    for i, (panels, y, cfg) in enumerate(data[:N_CHECK]):
        for k in B.comps(cfg):
            objs = B.comp_objs(panels[B.N_CTX + y], cfg, k)
            ct[i, int(k[1])] = B.facet_target(objs)
            cs[i, int(k[1])] = flat_spec(B.decode(B.describe(objs)))
    out["check_target"], out["check_spec"] = ct, cs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(DATASETS))
    ap.add_argument("--out", default="experiments/baselines/data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import neuralogic
    neuralogic.set_max_memory_size(16)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    for name in args.only:
        path, match = DATASETS[name]
        if path is None:
            cfgs = list(B.POS_MAP)
            splits = {"train": B.load("data/I-RAVEN-10000", cfgs, "train", 0, 6000),
                      "sel": B.load("data/I-RAVEN-10000", cfgs, "val", 0, 300),
                      "unseen": B.load("data/I-RAVEN-10000", cfgs, "val", 300, 1000)}
            rules = {s: None for s in splits}
        else:
            splits = {s: B.load_iravenx(path, start, n)
                      for s, (start, n) in {"train": (0, 6000), "sel": (6000, 300), "unseen": (6300, 1000)}.items()}
            first = splits["train"][0][0]
            B.configure(n_cols=(len(first) - 7) // 3, maxval=int(path.split("maxval_")[1].split(".")[0]))
            rules = {s: [B._IRAVENX[path][str(start + i)]["rules"][0] for i in range(len(splits[s]))]
                     for s, start in (("train", 0), ("sel", 6000), ("unseen", 6300))}
        weights = {"plain": facet_weights(False), "copy": facet_weights(True)}
        meta = {"name": name, "source": path or "data/I-RAVEN-10000", "therm": B.THERM, "n_cols": len(B.COLS),
                "n_ctx": B.N_CTX, "n_pos": B.N_POS, "nvals": [B.NVALS[a] for a in B.FACET_ATTRS],
                "constellations": list(B.POS_MAP), "rules": RULES, "facet_weights": weights,
                "budget": weights[match], "budget_from": f"facet model, {'with' if match == 'copy' else 'without'} copy links",
                "slots": sorted({o["pos"] for panels, _, _ in splits["train"] for p in panels for o in p})}
        np.savez_compressed(os.path.join(args.out, f"{name}.npz"), meta=json.dumps(meta),
                            **{f"{s}_{k}": v for s, d in splits.items() for k, v in arrays(d, rules[s]).items()})
        print(name, {s: len(d) for s, d in splits.items()}, "therm", B.THERM, "budget", meta["budget"],
              "slots", len(meta["slots"]), flush=True)


if __name__ == "__main__":
    main()
