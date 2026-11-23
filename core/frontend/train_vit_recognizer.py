#!/usr/bin/env python3
"""A ViT object recognizer for I-RAVEN panels, the perception stage of the facet model end to end.

The black-box ViT baseline (core/baselines/train_baseline.py --arch vit) reads the whole matrix and answers. This is the
same architecture (its patch embedding, position embeddings and pre-norm blocks, vit_blocks) reading ONE panel and naming
its objects: one learned query token per universal slot (train_pynl.POS_MAP, 22 slots), and one head shared by every
slot that gives whether the slot holds an object and that object's type (5), size (6) and colour (10). Only the slots the
layout can use are scored, as in the ResNet-18 pipeline, which also takes the layout from the dataset folder. So the
same visual backbone appears twice in the thesis: as a black-box solver, and as perception for the facet model.

Loss per panel: binary cross-entropy on occupancy over the layout's slots, cross-entropy on the three attributes of
the occupied ones. The epoch is picked on the selection panels by "panel right": every slot of the layout has the right
occupancy and every object the right type, size and colour (RESULTS §8's measure for the ResNet-18 frontend).

    python core/frontend/train_vit_recognizer.py train --width 64 --out experiments/recognizer/vit64
    python core/frontend/train_vit_recognizer.py predict --out experiments/recognizer/vit64 --splits unseen,test

The learning curve end to end (RESULTS §8): --per-layout N trains on the panels of the first N puzzles of every layout,
with the epochs raised so the number of updates matches the run on 2000 (30 * 2000 / N), as the facet curve does.
N = 3000 needs training data prepared with --n-train 3000 (experiments/recognizer/data3000).

predict writes <out>/preds_<split>.npy, float16 [puzzles, 16, 22, 22]: per slot P(occupied) (0 outside the layout),
then the type, size and colour probabilities. Data: data_prep/prepare_recognizer_data.py. Environment nvsa, one GPU.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
sys.path.insert(0, os.path.join(ROOT, "core", "baselines"))
from train_baseline import vit_blocks                                                  # the baseline ViT's blocks

POS_MAP = {                                     # train_pynl.POS_MAP (that module needs PyNeuraLogic's environment)
    "center_single": {0: 0},
    "distribute_four": {0: 1, 1: 2, 2: 3, 3: 4},
    "distribute_nine": {i: 5 + i for i in range(9)},
    "left_center_single_right_center_single": {0: 14, 1: 15},
    "up_center_single_down_center_single": {0: 16, 1: 17},
    "in_center_single_out_center_single": {0: 0, 1: 9},
    "in_distribute_four_out_center_single": {0: 0, 1: 18, 2: 19, 3: 20, 4: 21},
}
N_SLOTS, SPLIT_ATTR = 22, (5, 6, 10)            # type, size, colour
N_OUT = 1 + sum(SPLIT_ATTR)


class SlotViT(nn.Module):
    """The baseline's ViT over one panel, with one query token per slot and one head for all slots."""

    def __init__(self, h=64, depth=6, size=160, patch=16, dropout=0.0):
        super().__init__()
        n = (size // patch) ** 2
        self.embed = nn.Conv2d(1, h, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, n, h))
        self.slot = nn.Parameter(torch.zeros(1, N_SLOTS, h))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.slot, std=0.02)
        self.blocks = vit_blocks(h, depth, dropout)
        self.norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, N_OUT)

    def forward(self, img):                                                            # [B, 1, s, s] in [0, 1]
        x = self.embed(img).flatten(2).transpose(1, 2) + self.pos
        x = torch.cat([x, self.slot.expand(len(x), -1, -1)], 1)
        return self.head(self.norm(self.blocks(x)[:, -N_SLOTS:]))                      # [B, 22, 22]


def layout_masks(layouts, dev):
    m = torch.zeros(len(layouts), N_SLOTS, dtype=torch.bool)
    for i, cfg in enumerate(layouts):
        m[i, list(POS_MAP[cfg].values())] = True
    return m.to(dev)


def split_logits(out):
    occ, rest = out[..., 0], out[..., 1:]
    return occ, torch.split(rest, SPLIT_ATTR, -1)


def loss_fn(out, lab, mask):
    """lab [B, 22, 4] (occupied, type, size, colour; -1 where empty), mask [B, 22] the layout's slots."""
    occ, attrs = split_logits(out)
    l = F.binary_cross_entropy_with_logits(occ[mask], lab[..., 0][mask].float())
    full = mask & (lab[..., 0] == 1)
    for k, a in enumerate(attrs):
        l = l + F.cross_entropy(a[full], lab[..., k + 1][full].long())
    return l


def right(out, lab, mask):
    """Per panel: every slot of the layout right (occupancy, and type, size, colour where occupied)."""
    occ, attrs = split_logits(out)
    ok = (occ > 0) == (lab[..., 0] == 1)
    for k, a in enumerate(attrs):
        ok &= (a.argmax(-1) == lab[..., k + 1].long()) | (lab[..., 0] != 1)
    return (ok | ~mask).all(-1)


class Panels:
    """One split's panels as (image, labels, layout) per panel. The images are read once and kept on the device as
    uint8 (224,000 panels take 5.7 GB): random reads from the network drive would starve the GPU."""

    def __init__(self, d, split, dev, per_layout=0):
        z = np.load(os.path.join(d, f"{split}.npz"))
        img = np.load(os.path.join(d, f"{split}_img.npy"), mmap_mode="r")
        keep = np.flatnonzero(z["idx"] < per_layout) if per_layout else slice(None)   # the first N puzzles per layout
        self.img = torch.as_tensor(np.ascontiguousarray(img[keep])).to(dev)
        self.lab = torch.as_tensor(z["lab"][keep]).to(dev)
        self.layout = torch.as_tensor(z["layout"][keep].astype(np.int64)).to(dev)
        self.layouts = json.loads(str(z["meta"]))["layouts"]
        self.n = self.img.shape[0] * 16

    def batch(self, idx, dev):
        idx = torch.as_tensor(idx, device=dev)
        p, q = torch.div(idx, 16, rounding_mode="floor"), idx % 16
        return self.img[p, q].float().div_(255).unsqueeze(1), self.lab[p, q], self.layout[p], None


def evaluate(model, data, masks, dev, bs=1024, probs=False):
    model.eval()
    ok, by_layout, out_all = [], np.zeros((len(data.layouts), 2)), []
    with torch.no_grad():
        for s in range(0, data.n, bs):
            idx = np.arange(s, min(s + bs, data.n))
            img, lab, lay, order = data.batch(idx, dev)
            out = model(img)
            r = right(out, lab, masks[lay]).cpu().numpy()
            ok.append(r)
            np.add.at(by_layout, lay.cpu().numpy(), np.stack([r, np.ones_like(r)], 1))
            if probs:
                occ, attrs = split_logits(out)
                pr = torch.cat([torch.sigmoid(occ)[..., None] * masks[lay][..., None]] + [a.softmax(-1) for a in attrs], -1)
                out_all.append(pr.cpu().numpy().astype(np.float16))                    # idx is sorted already
    model.train()
    r = np.concatenate(ok)
    res = {"panel_right": float(r.mean()), "puzzle_right": float(r.reshape(-1, 16).all(1).mean()),
           "by_layout": {c: float(a / b) for c, (a, b) in zip(data.layouts, by_layout)}}
    return (res, np.concatenate(out_all).reshape(-1, 16, N_SLOTS, N_OUT)) if probs else res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--data", default="experiments/recognizer/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--max-train", type=int, default=0, help="use only the first N training panels (smoke tests)")
    ap.add_argument("--per-layout", type=int, default=0,
                    help="train on the first N puzzles of every layout only, as the facet model's learning curve does")
    ap.add_argument("--splits", default="unseen,test", help="predict: the splits to write")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.mode == "predict":
        a = json.load(open(os.path.join(args.out, "best.json")))["args"]
        model = SlotViT(a["width"], a["depth"], patch=a["patch"]).to(dev)
        model.load_state_dict(torch.load(os.path.join(args.out, "model_best.pt"), map_location=dev))
        for s in args.splits.split(","):
            data = Panels(args.data, s, dev)
            res, pr = evaluate(model, data, layout_masks(data.layouts, dev), dev, probs=True)
            np.save(os.path.join(args.out, f"preds_{s}.npy"), pr)
            json.dump(res, open(os.path.join(args.out, f"perception_{s}.json"), "w"), indent=1)
            print(f"{s}: panels right {100 * res['panel_right']:.2f}%, puzzles with all 16 right {100 * res['puzzle_right']:.2f}%; "
                  + ", ".join(f"{c} {100 * v:.2f}" for c, v in res["by_layout"].items()), flush=True)
        return

    train, sel = Panels(args.data, "train", dev, args.per_layout), Panels(args.data, "sel", dev)
    masks = layout_masks(train.layouts, dev)
    model = SlotViT(args.width, args.depth, patch=args.patch).to(dev)
    n_w = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n = min(train.n, args.max_train) if args.max_train else train.n
    steps = args.epochs * math.ceil(n / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, t / 500) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    print(f"SlotViT width {args.width}, depth {args.depth}, {n_w} weights; {n} training panels, {sel.n} selection panels", flush=True)
    best, hist, t0 = -1.0, [], time.time()
    for ep in range(1, args.epochs + 1):
        perm, tot = np.random.permutation(n), 0.0
        for s in range(0, n, args.batch):
            img, lab, lay, _ = train.batch(perm[s:s + args.batch], dev)
            loss = loss_fn(model(img), lab, masks[lay])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(lab)
        res = evaluate(model, sel, masks, dev)
        hist.append({"epoch": ep, "loss": tot / n, **res})
        if res["panel_right"] > best:
            best = res["panel_right"]
            torch.save(model.state_dict(), os.path.join(args.out, "model_best.pt"))
            json.dump({"epoch": ep, "weights": n_w, "args": vars(args), **res}, open(os.path.join(args.out, "best.json"), "w"), indent=1)
        json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=1)
        print(f"epoch {ep:3d}  loss {tot / n:.4f}  sel panels right {100 * res['panel_right']:6.2f}%  puzzles {100 * res['puzzle_right']:6.2f}%  "
              + " ".join(f"{c.split('_')[0][:4]}{100 * v:6.2f}" for c, v in res["by_layout"].items()) + f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
