#!/usr/bin/env python3
"""The facet model end to end on I-RAVEN: its context panels come from the ViT slot recognizer
(core/frontend/train_vit_recognizer.py) instead of the XML labels, and so do the candidates it is matched against.

Three passes over the same puzzles, on one checkpoint (the template is rebuilt from the run's best.json):
  labels  objects from the labels (the recognizer data's lab array, which is the XML read by bench_hub_grounding's
          parser). On the full unseen split this must reproduce the checkpoint's recorded numbers; the script says so.
  hard    an object in every slot with P(occupied) > 0.5, each attribute the recognizer's argmax.
  soft    the same objects, each attribute's thermometer bit k set to the recognizer's P(value > k) (the code's own
          meaning) where that lies between 0.01 and 0.99, and to 0 or 1 otherwise (see snap); candidates' codes
          likewise. Occupancy stays hard: an object is a node or is not.

Reported per pass: matched against the perceived candidates (the end-to-end number), matched against the labels'
candidates (perceived context only), and exact: the generated panel equals the true answer. Per layout, and split by
whether the recognizer read all 16 panels of the puzzle right. Written to <ckpt>/e2e_<split>.json.

Splits are the recognizer data's (data_prep/prepare_recognizer_data.py): unseen = val 300..1299, test = test 0..1999.
Pick hard or soft on the unseen puzzles, then score test once. Environment pynl, from the repository root, as a job:
    python3 core/pynl/facet_e2e_eval.py experiments/facet_all_w32-64-128_hidden --split unseen
    python3 core/pynl/facet_e2e_eval.py experiments/facet_all_w32-64-128_hidden --split unseen --n 20   # a smoke test
"""
import argparse, ast, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_hub_grounding as B

ATTR_COLS = {"type": (1, 6), "size": (6, 12), "color": (12, 22)}     # columns of a slot's prediction vector


def label_panels(lab):
    """lab [16, 22, 4] (occupied, type, size, colour) -> 16 panels of objects, in slot order."""
    return [[{"pos": int(s), "type": int(p[s, 1]), "size": int(p[s, 2]), "color": int(p[s, 3])}
             for s in np.flatnonzero(p[:, 0] == 1)] for p in lab]


def snap(b, tol=0.01):
    """A soft bit, with a confident one made exactly 0 or 1. The exact zeros matter: value 0 is the all-zero code, the
    template has no biases, and its NORM step rescales any non-zero vector to full size, so a code of 1e-6 in place of
    0 reads as a large value (measured: outputs flip by 1.0). Only bits the recognizer is unsure of stay soft."""
    return 0.0 if b < tol else 1.0 if b > 1 - tol else round(b, 4)


def perceived_panels(pred, soft):
    """pred [16, 22, 22] (P(occupied), then type, size and colour probabilities) -> 16 panels of objects."""
    panels = []
    for p in pred.astype(np.float64):
        objs = []
        for s in np.flatnonzero(p[:, 0] > 0.5):
            o = {"pos": int(s), **{a: int(p[s, lo:hi].argmax()) for a, (lo, hi) in ATTR_COLS.items()}}
            if soft:                                                   # bit k = P(value > k)
                o["soft"] = {a: [snap(float(p[s, lo + k + 1:hi].sum())) for k in range(B.THERM)]
                             for a, (lo, hi) in ATTR_COLS.items()}
            objs.append(o)
        panels.append(objs)
    return panels


def strip(panels):
    return [sorted((o["pos"], o["type"], o["size"], o["color"]) for o in p) for p in panels]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="run directory with best.json and template_best.bin (an I-RAVEN facet run)")
    ap.add_argument("--preds", default="experiments/recognizer/vit64", help="the recognizer's run directory")
    ap.add_argument("--data", default="experiments/recognizer/data")
    ap.add_argument("--split", default="unseen", choices=["unseen", "test"])
    ap.add_argument("--modes", default="labels,hard,soft")
    ap.add_argument("--n", type=int, default=0, help="only the first N puzzles of every layout (smoke tests)")
    ap.add_argument("--jvm-gb", type=int, default=48)
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
    if a.get("iravenx"):
        raise SystemExit(f"{args.ckpt}: an I-RAVEN-X run; I-RAVEN-X has no images")
    z = np.load(os.path.join(args.data, f"{args.split}.npz"))
    meta = json.loads(str(z["meta"]))
    layouts, lab, layout, y = meta["layouts"], z["lab"], z["layout"], z["y"]
    preds = np.load(os.path.join(args.preds, f"preds_{args.split}.npy"), mmap_mode="r")
    assert layouts == list(B.POS_MAP) and len(preds) == len(lab), "recognizer data and predictions do not match"
    trained = list(B.POS_MAP) if a["constellation"] == "all" else a["constellation"].split(",")
    keep = [i for c in trained for i in np.flatnonzero(layout == layouts.index(c))[:args.n or None]]
    cfgs = [layouts[layout[i]] for i in keep]
    true = [(label_panels(lab[i]), int(y[i]), c) for i, c in zip(keep, cfgs)]
    print(f"{args.ckpt}, epoch {best['epoch']}; recognizer {args.preds}; {args.split} = {meta['spec']}, "
          f"{len(keep)} puzzles of {len(trained)} layouts", flush=True)

    cols = a.get("columns", "named")
    copy, roles, fold = bool(a.get("copy_links")), ("tie12" if cols == "tie12" else cols != "named"), cols == "fold"
    labels = tuple(k for k, f in (("components", "label_components"), ("attributes", "label_attributes")) if a.get(f))
    widths = [int(w) for w in a["widths"].split(",")] if a.get("widths") else None
    model = B.build_facet_template(a["dim"], a["layers"], not a.get("per_attr_weights"), not a.get("no_rows"),
                                   widths, copy=copy, gates=a.get("copy_gates", "panel"), cols=cols, labels=labels,
                                   shared_comps=not a.get("per_comp_weights"))
    model.build(Settings(optimizer=Adam(lr=a["lr"]), error_function=MSE(), iso_value_compression=False))
    model.load(os.path.join(args.ckpt, "template_best.bin"))

    hard = [(perceived_panels(preds[i], False), int(y[i]), c) for i, c in zip(keep, cfgs)]
    seen_right = np.array([strip(h[0]) == strip(t[0]) for h, t in zip(hard, true)])
    print(f"recognizer: all 16 panels right in {100 * seen_right.mean():.2f}% of these puzzles", flush=True)

    res = {"ckpt": args.ckpt, "epoch": best["epoch"], "preds": args.preds, "split": args.split, "spec": meta["spec"],
           "puzzles": len(keep), "all_panels_right": float(seen_right.mean()), "modes": {}}
    for mode in args.modes.split(","):
        data = {"labels": true, "hard": hard}.get(mode) or \
            [(perceived_panels(preds[i], True), int(y[i]), c) for i, c in zip(keep, cfgs)]
        units, samples = [], []
        for i, ((q, _, cfg), (t, ty, _)) in enumerate(zip(data, true)):
            for k in B.comps(cfg):
                gid = f"{mode}{i}{k}"
                facts = (B.puzzle_facts(q, gid, col_roles=roles, fold=fold)
                         + B.facet_facts(q, "therm", cfg, copy, labels=labels) + [R.qcomp(gid, k)])
                units.append((i, k))
                samples.append(Sample(R.out(gid)[B.facet_target(B.comp_objs(t[B.N_CTX + ty], cfg, k)).tolist()], facts))
        outs = []
        for c0 in range(0, len(samples), 2000):                        # grounded in chunks, to bound memory
            outs += [np.asarray(o, dtype=float).flatten()
                     for o in model.test(list(model.build_dataset(Dataset(samples[c0:c0 + 2000]))))]
        per_puzzle = [dict() for _ in data]
        for (i, k), o in zip(units, outs):
            per_puzzle[i][k] = o
        n = len(data)
        _, exact, m_true, _, _ = B.evaluate(per_puzzle, true, "therm")           # exact, and matched on the labels' candidates
        _, _, m_seen, ties, per_cfg = B.evaluate(per_puzzle, data, "therm")      # matched on this pass's candidates
        hit = np.array([B.evaluate([o], [d], "therm")[2] for o, d in zip(per_puzzle, data)], dtype=bool)
        r = {"matched": m_seen / n, "matched_label_candidates": m_true / n, "exact": exact["all"] / n, "ties": ties,
             "per_layout": {c: h / m for c, (h, m) in per_cfg.items()},
             "matched_all_panels_right": float(hit[seen_right].mean()) if seen_right.any() else None,
             "matched_some_panel_wrong": float(hit[~seen_right].mean()) if (~seen_right).any() else None,
             "n_some_panel_wrong": int((~seen_right).sum())}
        res["modes"][mode] = r
        print(f"  {mode:6s} matched {100 * r['matched']:.2f}% (ties {ties}), on the labels' candidates "
              f"{100 * r['matched_label_candidates']:.2f}%, exact {100 * r['exact']:.2f}%; "
              + "  ".join(f"{c[:12]} {100 * v:.1f}" for c, v in r["per_layout"].items()), flush=True)
        if mode != "labels" and (~seen_right).any():
            print(f"         matched where all 16 panels were read right {100 * r['matched_all_panels_right']:.2f}%, "
                  f"where some panel was not {100 * r['matched_some_panel_wrong']:.2f}% of {r['n_some_panel_wrong']}", flush=True)
        if mode == "labels" and args.split == "unseen" and not args.n and "unseen" in best:
            rec = best["unseen"]
            same = (m_true == rec["matched"] and exact["all"] == rec["exact"])
            res["labels_reproduce_record"] = bool(same)
            print(f"         the checkpoint's record: matched {rec['matched']}, exact {rec['exact']} of {n}; this pass "
                  f"{m_true}, {exact['all']}: {'reproduced' if same else 'NOT reproduced'}", flush=True)
    if not args.n:
        json.dump(res, open(os.path.join(args.ckpt, f"e2e_{args.split}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
