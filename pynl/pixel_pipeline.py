#!/usr/bin/env python3
"""I-RAVEN from pixels: the ResNet-18 frontend's view of every panel, for the end-to-end runs.

  perceive  (conda env nvsa, GPU)   Run the frontend(s) over every panel image of a split and
                                    store, per panel and universal position, the best attribute
                                    match and its cosine (experiments/frontend_preds/). Also scores
                                    perception against the ground truth, per constellation.

Decoding (`panels_from_preds`, used by pynl_symbolic_e2e.py): for each position the constellation
can use, take the best of its 300 attribute codebook rows; the object exists iff that cosine
exceeds tau (0.23, the frontend's training threshold). One object per position by construction.
The constellation is taken from the dataset folder -- the ground-truth pipeline uses it too.

Position IDs need no remapping: the frontend's universal codebook positions, train_pynl.POS_MAP
and gnn.config.CONFIG_POS_MAP are identical (checked). The second stage, which fed these
detections to the GNN-style PyNeuraLogic model, is archived with that model
(archive/superseded_2026-10-28/).

Usage:
    python pynl/pixel_pipeline.py perceive --splits val,test \
        --frontends log/universal_frontend_correct_proportional_mean1,log/universal_frontend_correct_proportional_mean3
"""
import argparse, glob, json, os, sys, time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
for _p in (ROOT, os.path.join(ROOT, "pynl"), os.path.join(ROOT, "torch_gnn"), os.path.join(ROOT, "frontend"),
           os.path.join(ROOT, "raven"), os.path.join(ROOT, "neuro-vector-symbolic-architectures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONSTELLATIONS = [
    "center_single", "distribute_four", "distribute_nine",
    "left_center_single_right_center_single", "up_center_single_down_center_single",
    "in_center_single_out_center_single", "in_distribute_four_out_center_single",
]
N_POS, DIM_TYPE, DIM_SIZE, DIM_COLOR = 22, 5, 6, 10
N_ATTR = DIM_COLOR * DIM_SIZE * DIM_TYPE          # 300 attribute rows per position
N_PER_POS = N_ATTR + 1                            # + 1 "object here" row
MAX_OBJ = 9


# ── shared decoding (pure numpy, py3.7-compatible) ─────────────────────────────
def decode_attr(k):
    """Row k (0..299) within a position block -> (type, size, color)."""
    return int(k % DIM_TYPE), int((k % (DIM_SIZE * DIM_TYPE)) // DIM_TYPE), int(k // (DIM_SIZE * DIM_TYPE))


def active_positions(cfg):
    from gnn.config import CONFIG_POS_MAP
    return sorted(set(CONFIG_POS_MAP[cfg].values()))


def panels_from_preds(maxcos, argidx, positions, tau):
    """(16,22) best-attribute cosine + index -> 16 lists of {pos,type,size,color} dicts."""
    panels = []
    for p in range(16):
        objs = []
        for pos in positions:
            if float(maxcos[p, pos]) > tau:
                t, s, c = decode_attr(int(argidx[p, pos]))
                objs.append({"pos": pos, "type": t, "size": s, "color": c})
        panels.append(objs)
    return panels


def gt_panels_from_array(gt):
    """(16,9,4) int8, -1 padded -> 16 lists of {pos,type,size,color} dicts."""
    return [[{"pos": int(o[0]), "type": int(o[1]), "size": int(o[2]), "color": int(o[3])}
             for o in gt[p] if o[0] >= 0] for p in range(16)]


def key(panel):
    return tuple(sorted((o["pos"], o["type"], o["size"], o["color"]) for o in panel))


def split_items(dataset, split, n=None):
    items = []
    for cfg in CONSTELLATIONS:
        for f in sorted(glob.glob(os.path.join(dataset, cfg, "*_%s.npz" % split)))[:n]:
            items.append((cfg, f))
    return items


# ── stage 1: perceive ──────────────────────────────────────────────────────────
def perceive(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from gnn_solver import load_frontend
    from train_frontend import decode_w_index
    from gnn.data import parse_ground_truth

    rng = np.random.default_rng(0)                  # decoder must agree with the frontend's own
    for _ in range(200):
        p, k = int(rng.integers(N_POS)), int(rng.integers(N_ATTR))
        assert decode_w_index(p * N_PER_POS + k) == (p,) + decode_attr(k)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fronts = {}
    for d in args.frontends.split(","):
        model, cos, _, _ = load_frontend(d, device=str(dev))
        ck = torch.load(os.path.join(d, "model_best.pth.tar"), map_location="cpu")
        fronts[os.path.basename(d.rstrip("/"))] = (model, F.normalize(cos.W.data.to(dev)),
                                                    {"epoch": ck.get("epoch"), "best_acc": ck.get("best_acc")})
    print("device:", dev, "| frontends:", list(fronts))

    class Split(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            cfg, f = self.items[i]
            d = np.load(f)
            gt = np.full((16, MAX_OBJ, 4), -1, dtype=np.int8)
            for p, objs in enumerate(parse_ground_truth(f.replace(".npz", ".xml"), cfg)):
                for j, o in enumerate(sorted(objs, key=lambda o: o["position"])):
                    gt[p, j] = (o["position"], o["type"], o["size"], o["color"])
            return torch.from_numpy(d["image"].astype(np.float32)).unsqueeze(1), int(d["target"]), i, gt

    def collate(b):
        return (torch.stack([x[0] for x in b]), [x[1] for x in b], [x[2] for x in b],
                np.stack([x[3] for x in b]))

    os.makedirs(args.out_dir, exist_ok=True)
    sum_path = os.path.join(args.out_dir, "summary.json")
    summary = json.load(open(sum_path)) if os.path.exists(sum_path) else {}

    for split in args.splits.split(","):
        items = split_items(args.dataset, split, args.n)
        P = len(items)
        loader = DataLoader(Split(items), batch_size=args.batch, num_workers=args.workers,
                            collate_fn=collate, pin_memory=True)
        store = {n: {"maxcos": np.zeros((P, 16, N_POS), np.float16),
                     "argidx": np.zeros((P, 16, N_POS), np.int16),
                     "detcos": np.zeros((P, 16, N_POS), np.float16)} for n in fronts}
        targets = np.zeros(P, np.int64)
        gts = np.zeros((P, 16, MAX_OBJ, 4), np.int8)
        t0 = time.time()
        with torch.no_grad():
            for bi, (imgs, tg, idx, gt) in enumerate(loader):
                x = imgs.view(-1, 1, 160, 160).to(dev, non_blocking=True)
                targets[idx] = tg
                gts[idx] = gt
                for n, (model, Wn, _) in fronts.items():
                    s = F.linear(F.normalize(model(x)), Wn).view(-1, 16, N_POS, N_PER_POS)
                    v, k = s[..., :N_ATTR].max(-1)
                    store[n]["maxcos"][idx] = v.half().cpu().numpy()
                    store[n]["argidx"][idx] = k.short().cpu().numpy()
                    store[n]["detcos"][idx] = s[..., N_ATTR].half().cpu().numpy()
                if bi % 100 == 0:
                    print("  %s: %d/%d puzzles, %.0fs" % (split, min((bi + 1) * args.batch, P), P, time.time() - t0),
                          flush=True)
        print("%s: %d puzzles perceived in %.0fs" % (split, P, time.time() - t0))

        rel = np.array([os.path.relpath(f, ROOT) for _, f in items])
        cfgs = np.array([c for c, _ in items])
        for n, (_, _, meta) in fronts.items():
            out = os.path.join(args.out_dir, "%s_%s.npz" % (n, split))
            np.savez(out, files=rel, cfgs=cfgs, targets=targets, gt=gts, tau=args.tau, **store[n])
            stats = perception_stats(store[n]["maxcos"], store[n]["argidx"], gts, cfgs, args.tau)
            summary.setdefault(n, {"meta": meta})[split] = stats
            print_stats(n, split, stats)
            print("  wrote", out)
    summary["tau"] = args.tau
    json.dump(summary, open(sum_path, "w"), indent=1)
    print("wrote", sum_path)


def perception_stats(maxcos, argidx, gts, cfgs, tau):
    acc = defaultdict(lambda: defaultdict(int))
    for i in range(len(cfgs)):
        cfg = str(cfgs[i])
        pos = active_positions(cfg)
        pred = panels_from_preds(maxcos[i], argidx[i], pos, tau)
        gt = gt_panels_from_array(gts[i])
        ok = [key(pred[p]) == key(gt[p]) for p in range(16)]
        inactive = [q for q in range(N_POS) if q not in pos]
        dropped = int((maxcos[i][:, inactive].astype(np.float32) > tau).sum()) if inactive else 0
        for bucket in (cfg, "OVERALL"):
            a = acc[bucket]
            a["puzzles"] += 1
            a["panels"] += 16
            a["panel_ok"] += sum(ok)
            a["cand_ok"] += sum(ok[8:])
            a["all16"] += all(ok)
            a["dropped"] += dropped
    return {b: {"panel_acc": 100.0 * a["panel_ok"] / a["panels"],
                "cand_panel_acc": 100.0 * a["cand_ok"] / (8 * a["puzzles"]),
                "puzzles_all16_ok": 100.0 * a["all16"] / a["puzzles"],
                "off_constellation_detections": a["dropped"], "puzzles": a["puzzles"]}
            for b, a in acc.items()}


def print_stats(name, split, stats):
    print("\nperception  %s  [%s]" % (name, split))
    print("%-40s %10s %11s %14s %9s" % ("constellation", "panel acc", "cand panels", "all 16 panels", "dropped"))
    for b in CONSTELLATIONS + ["OVERALL"]:
        if b in stats:
            s = stats[b]
            print("%-40s %9.2f%% %10.2f%% %13.2f%% %9d" % (b, s["panel_acc"], s["cand_panel_acc"],
                                                          s["puzzles_all16_ok"], s["off_constellation_detections"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("perceive")
    p.add_argument("--dataset", default=os.path.join(ROOT, "data/I-RAVEN-10000"))
    p.add_argument("--splits", default="val,test")
    p.add_argument("--frontends", default="log/universal_frontend_correct_proportional_mean1,"
                                          "log/universal_frontend_correct_proportional_mean3")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "experiments/frontend_preds"))
    p.add_argument("--tau", type=float, default=0.23)
    p.add_argument("--n", type=int, default=None, help="puzzles per constellation (default: all)")
    p.add_argument("--batch", type=int, default=16, help="puzzles per GPU batch (x16 panels)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    {"perceive": perceive}[args.stage](args)


if __name__ == "__main__":
    main()
