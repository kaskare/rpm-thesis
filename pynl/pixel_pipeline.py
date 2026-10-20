#!/usr/bin/env python3
"""End-to-end I-RAVEN from pixels: ResNet-18 perception -> PyNeuraLogic reasoning.

The PyNeuraLogic solver (train_all_constellations.py --level 2) was trained on ground-truth
object attributes read from the XML. This pipeline replaces the XML with what the ResNet-18
frontend (train_frontend.py) *sees* in the panel images, so the reported accuracy is from
pixels to answer.

Two stages, because they need different environments:

  perceive  (conda env nvsa, GPU)   Run the frontend(s) over every panel image of a split and
                                    store, per panel and universal position, the best attribute
                                    match and its cosine. Also scores perception against the
                                    ground truth, per constellation.
  solve     (conda env pynl, CPU)   Decode those detections into objects, feed them to the
                                    trained PyNeuraLogic model, and report pixel-input accuracy
                                    next to ground-truth-input accuracy on the same puzzles and
                                    the same checkpoint.

Decoding (identical in both stages, `panels_from_preds`): for each position the constellation
can use, take the best of its 300 attribute codebook rows; the object exists iff that cosine
exceeds tau (0.23, the frontend's training threshold). One object per position by construction.
The constellation is taken from the dataset folder -- the ground-truth pipeline uses it too.

Position IDs need no remapping: the frontend's universal codebook positions, train_pynl.POS_MAP
and gnn.config.CONFIG_POS_MAP are identical (checked).

Usage:
    python pynl/pixel_pipeline.py perceive --splits val,test \
        --frontends log/universal_frontend_correct_proportional_mean1,log/universal_frontend_correct_proportional_mean3
    python pynl/pixel_pipeline.py solve --run-dir experiments/max_data_L2 --frontend auto --split test
"""
import argparse, glob, json, os, shutil, sys, tempfile, time
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


# ── stage 2: solve ─────────────────────────────────────────────────────────────
def solve(args):
    import torch
    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from train_pynl import parse
    from train_all_constellations import encode_puzzle
    from eval_pynl_test import load_model

    summary = json.load(open(os.path.join(args.preds_dir, "summary.json")))
    front = args.frontend
    if front == "auto":                       # choose on VALIDATION perception, never on test
        cands = [n for n in summary if n != "tau" and "val" in summary[n]]
        if not cands:
            raise SystemExit("--frontend auto needs val perception stats; run perceive --splits val,...")
        front = max(cands, key=lambda n: summary[n]["val"]["OVERALL"]["panel_acc"])
        print("frontend (best val perception): %s  %.2f%%" % (front, summary[front]["val"]["OVERALL"]["panel_acc"]))
    d = np.load(os.path.join(args.preds_dir, "%s_%s.npz" % (front, args.split)))
    tau = float(d["tau"]) if args.tau is None else args.tau

    tmp = tempfile.mkdtemp()                  # run dir may be written by a live job
    for n in ("model_best.pt", "template_best.bin"):
        shutil.copy(os.path.join(args.run_dir, n), tmp)
    model, ck, a = load_model(tmp)
    print("reasoner: %s  epoch %s, val %.2f%%" % (args.run_dir, ck["epoch"], 100 * ck["val_acc"]))

    # the stored ground truth must be exactly what the reasoner was trained on
    seen = defaultdict(int)
    for i in range(len(d["cfgs"])):
        cfg = str(d["cfgs"][i])
        if seen[cfg] < 20:
            seen[cfg] += 1
            ref = parse(os.path.join(ROOT, str(d["files"][i])).replace(".npz", ".xml"), cfg)
            assert [key(p) for p in ref] == [key(p) for p in gt_panels_from_array(d["gt"][i])], d["files"][i]
    print("stored ground truth == train_pynl.parse on %d spot-checked puzzles" % sum(seen.values()))

    per = defaultdict(lambda: defaultdict(int))
    count = defaultdict(int)
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(d["cfgs"])):
            cfg = str(d["cfgs"][i])
            if args.n and count[cfg] >= args.n:
                continue
            count[cfg] += 1
            y = int(d["targets"][i])
            pix = panels_from_preds(d["maxcos"][i], d["argidx"][i], active_positions(cfg), tau)
            gt = gt_panels_from_array(d["gt"][i])
            perfect = all(key(pix[p]) == key(gt[p]) for p in range(16))
            hit_pix = int(model(*encode_puzzle(pix, cfg)).argmax()) == y
            hit_gt = int(model(*encode_puzzle(gt, cfg)).argmax()) == y
            for b in (cfg, "OVERALL"):
                r = per[b]
                r["n"] += 1
                r["pix"] += hit_pix
                r["gt"] += hit_gt
                r["perfect"] += perfect
                r["pix_perfect"] += hit_pix and perfect
                r["pix_imperfect"] += hit_pix and not perfect
            if sum(count.values()) % 2000 == 0:
                print("  %d puzzles, %.0fs" % (sum(count.values()), time.time() - t0), flush=True)

    print("\nI-RAVEN %s, %s frontend (tau %.2f) -> PyNeuraLogic, epoch %s\n" % (args.split, front, tau, ck["epoch"]))
    print("%-40s %7s %13s %12s %10s %8s %16s" % ("constellation", "n", "all 16 seen", "from pixels",
                                                 "from XML", "delta", "pix | imperfect"))
    res = {}
    for b in CONSTELLATIONS + ["OVERALL"]:
        r = per.get(b)
        if not r:
            continue
        n, imp = r["n"], r["n"] - r["perfect"]
        row = {"n": n, "all16_perceived": 100.0 * r["perfect"] / n, "pixels": 100.0 * r["pix"] / n,
               "xml": 100.0 * r["gt"] / n,
               "pixels_on_imperfect": (100.0 * r["pix_imperfect"] / imp) if imp else None}
        res[b] = row
        print("%-40s %7d %12.1f%% %11.2f%% %9.2f%% %+7.2f %15s" % (
            b, n, row["all16_perceived"], row["pixels"], row["xml"], row["pixels"] - row["xml"],
            ("%.1f%%" % row["pixels_on_imperfect"]) if imp else "-"))
    print("\nchance 12.5%. 'pix | imperfect' = pixel accuracy on puzzles with >=1 misperceived panel.")
    out = args.json_out or os.path.join(args.preds_dir, "solve_%s_%s.json" % (front, args.split))
    json.dump({"frontend": front, "tau": tau, "split": args.split, "run_dir": args.run_dir,
               "epoch": ck["epoch"], "val_acc": ck["val_acc"], "results": res}, open(out, "w"), indent=1)
    print("wrote", out)


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
    s = sub.add_parser("solve")
    s.add_argument("--run-dir", default=os.path.join(ROOT, "experiments/max_data_L2"))
    s.add_argument("--preds-dir", default=os.path.join(ROOT, "experiments/frontend_preds"))
    s.add_argument("--frontend", default="auto")
    s.add_argument("--split", default="test")
    s.add_argument("--tau", type=float, default=None, help="override the stored threshold")
    s.add_argument("--n", type=int, default=0, help="puzzles per constellation (0 = all)")
    s.add_argument("--jvm-gb", type=int, default=40)
    s.add_argument("--json-out", default="")
    args = ap.parse_args()
    {"perceive": perceive, "solve": solve}[args.stage](args)


if __name__ == "__main__":
    main()
