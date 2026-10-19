#!/usr/bin/env python3
"""Checks that the PyNeuraLogic I-RAVEN result is not an artefact.

Each check targets one specific way a high accuracy could be fake:

  splits    Test puzzles also appearing in the training set (exact duplicates of the
            full puzzle, or of its context), and the answer-index distribution.
  shortcut  Can the answer be picked from the 8 candidates ALONE? Original RAVEN has
            this flaw (distractors are one-attribute edits of the answer, so the answer
            is the candidates' "mode"); I-RAVEN was built to remove it. Runs the mode
            heuristic on I-RAVEN and, as a positive control, on RAVEN.
  images    Do the XML labels the model trains on describe the actual images, for
            candidate panels as well as context panels? Uses the per-object pixel masks
            stored in the XML: every non-white pixel must lie on some labelled object,
            and each object's fill must match its labelled Color.
  model     On the trained checkpoint: (a) permuting the candidates must permute the
            scores exactly (no candidate-index leakage, CONCAT order handled right);
            (b) giving a puzzle the CONTEXT of another puzzle must collapse accuracy to
            chance -- if it did not, the model would be answering without the context;
            (c) blanking the context entirely must do the same.

Usage:
    python sanity_checks.py splits
    python sanity_checks.py shortcut
    python sanity_checks.py images --n 200
    python sanity_checks.py images --n 200 --mismatch     # negative control: must FAIL
    python sanity_checks.py model --run-dir experiments/max_data_L2 --n 250
"""
import argparse, glob, os, random, shutil, sys, tempfile, xml.etree.ElementTree as ET
from collections import Counter
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from train_pynl import parse, POS_MAP                              # the parse the model uses

CONSTELLATIONS = list(POS_MAP)
IRAVEN, RAVEN = os.path.join(ROOT, "data/I-RAVEN-10000"), os.path.join(ROOT, "data/RAVEN-10000")
COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]     # RAVEN's rendering palette


def files(root, cfg, split, n=None):
    return sorted(glob.glob(os.path.join(root, cfg, f"*_{split}.npz")))[:n]


def panel_key(p):
    return tuple(sorted((o["pos"], o["type"], o["size"], o["color"]) for o in p))


# ── splits ──────────────────────────────────────────────────────────────────────
def _split_stats(cfg):
    enc = lambda f: [panel_key(p) for p in parse(f.replace(".npz", ".xml"), cfg)]
    train = [enc(f) for f in files(IRAVEN, cfg, "train")]
    test_f = files(IRAVEN, cfg, "test")
    test = [enc(f) for f in test_f]
    full_tr = {tuple(p) for p in train}
    ctx_tr = {tuple(p[:8]) for p in train}
    full = sum(tuple(p) in full_tr for p in test)
    ctx = sum(tuple(p[:8]) in ctx_tr for p in test)
    within = len(test) - len({tuple(p) for p in test})
    targets = Counter(int(np.load(f)["target"]) for f in test_f)
    return cfg, len(train), len(test), full, ctx, within, targets


def check_splits(args):
    print("Exact duplicates between the TEST split and the full 6000-puzzle TRAIN split\n")
    print(f"{'constellation':<40} {'train':>6} {'test':>5} {'full dup':>9} {'ctx dup':>8} {'dup in test':>11}   answer index 0..7")
    with Pool(len(CONSTELLATIONS)) as pool:
        for cfg, ntr, nte, full, ctx, within, t in pool.imap(_split_stats, CONSTELLATIONS):
            hist = " ".join(f"{t[k]:>3}" for k in range(8))
            print(f"{cfg:<40} {ntr:>6} {nte:>5} {full:>9} {ctx:>8} {within:>11}   {hist}")
    print("\nfull dup: whole puzzle (context+candidates) also in train. ctx dup: same 8 context panels.")


# ── shortcut ────────────────────────────────────────────────────────────────────
def mode_heuristic(cands):
    """Indices of the candidate(s) sharing the most attribute values with the other seven."""
    def views(p):
        return [tuple(o[0] for o in p), tuple(sorted(o[1] for o in p)),
                tuple(sorted(o[2] for o in p)), tuple(sorted(o[3] for o in p)), len(p)]
    V = [views(c) for c in cands]
    score = [sum(V[i][a] == V[j][a] for j in range(8) if j != i for a in range(5)) for i in range(8)]
    top = [i for i in range(8) if score[i] == max(score)]
    return top


def _shortcut(job):
    root, cfg = job
    hit = 0.0
    fs = files(root, cfg, "test")
    for f in fs:
        cands = [panel_key(p) for p in parse(f.replace(".npz", ".xml"), cfg)][8:]
        y = int(np.load(f)["target"])
        top = mode_heuristic(cands)
        hit += (1.0 / len(top)) if y in top else 0.0
    return root, cfg, 100 * hit / len(fs), len(fs)


def check_shortcut(args):
    print("Candidates-only 'mode' heuristic (never looks at the context). Chance = 12.5%.\n")
    jobs = [(r, c) for r in (IRAVEN, RAVEN) if os.path.isdir(r) for c in CONSTELLATIONS]
    res = {}
    with Pool(min(14, len(jobs))) as pool:
        for root, cfg, acc, n in pool.imap(_shortcut, jobs):
            res[(root, cfg)] = (acc, n)
    print(f"{'constellation':<40} {'I-RAVEN':>9} {'RAVEN (control)':>16}")
    for cfg in CONSTELLATIONS:
        a = res.get((IRAVEN, cfg), (float('nan'), 0))[0]
        b = res.get((RAVEN, cfg), (float('nan'), 0))[0]
        print(f"{cfg:<40} {a:>8.1f}% {b:>15.1f}%")


# ── images ──────────────────────────────────────────────────────────────────────
def _dilate(m, r=3):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def _xml_objects(xml):
    panels = []
    for panel in ET.parse(xml).getroot()[0]:
        ents = []
        for s in panel:
            for c in s:
                for l in c:
                    for e in l:
                        rle = eval(e.attrib["mask"])
                        m = np.zeros(160 * 160, bool)
                        for st, ln in zip(rle[0::2], rle[1::2]):
                            m[st:st + ln] = True
                        ents.append((m.reshape(160, 160), int(e.attrib["Color"])))
        panels.append(ents)
    return panels


# RAVEN draws a 1-pixel divider between the two halves of these layouts; it is
# scaffolding, not an object (verified: every uncovered pixel lies on it).
DIVIDERS = {"left_center_single_right_center_single": (slice(None), 80),
            "up_center_single_down_center_single": (80, slice(None))}


def _erode(m, r=2):
    return ~_dilate(~m, r)


def _images(job):
    cfg, n, mismatch = job
    stats = {"ctx": [0, 0, 0, 0, 100.0], "cand": [0, 0, 0, 0, 100.0]}  # panels, clean, objs, colour-ok, min cov
    fs = files(IRAVEN, cfg, "test", n)
    for i, f in enumerate(fs):
        img = np.load(f)["image"]
        # negative control: describe this image with ANOTHER puzzle's labels
        xml = fs[(i + 1) % len(fs)] if mismatch else f
        objs = _xml_objects(xml.replace(".npz", ".xml"))
        for p in range(16):
            s = stats["ctx" if p < 8 else "cand"]
            union = np.zeros((160, 160), bool)
            for m, _ in objs[p]:
                union |= m
            nonwhite = img[p] < 250
            if cfg in DIVIDERS:
                nonwhite[DIVIDERS[cfg]] = False
            cov = 100.0 * (nonwhite & _dilate(union)).sum() / max(nonwhite.sum(), 1)
            s[0] += 1
            s[1] += cov >= 99.0
            s[4] = min(s[4], cov)
            for k, (m, col) in enumerate(objs[p]):
                own = m.copy()
                for j, (m2, _) in enumerate(objs[p]):   # carve out objects nested INSIDE this one
                    if j != k and m2.sum() < m.sum() and (m & m2).sum() > 0.5 * m2.sum():
                        own &= ~m2
                inner = _erode(own)                      # interior only: outlines are black
                if inner.sum() < 5:
                    continue
                s[2] += 1
                s[3] += abs(int(np.median(img[p][inner])) - COLOR_VALUES[col]) <= 10
    return cfg, stats


def check_images(args):
    tag = "NEGATIVE CONTROL: each image paired with the NEXT puzzle's labels" if args.mismatch else "matched labels"
    print(f"XML labels vs rendered images, first {args.n} test puzzles per constellation -- {tag}\n")
    print("clean panel: >=99% of non-white pixels lie on a labelled object (masks dilated 3px for outlines)")
    print("colour ok  : median of the object's interior (mask eroded 2px) matches its labelled Color (+-10)\n")
    print(f"{'constellation':<40} {'':>6} {'panels':>7} {'clean':>8} {'worst cov':>10} {'objects':>8} {'colour ok':>10}")
    with Pool(len(CONSTELLATIONS)) as pool:
        for cfg, st in pool.imap(_images, [(c, args.n, args.mismatch) for c in CONSTELLATIONS]):
            for k in ("ctx", "cand"):
                np_, cl, no, co, mn = st[k]
                print(f"{cfg if k == 'ctx' else '':<40} {k:>6} {np_:>7} {100*cl/np_:>7.1f}% {mn:>9.1f}% "
                      f"{no:>8} {100*co/max(no,1):>9.1f}%")


# ── model ───────────────────────────────────────────────────────────────────────
def check_model(args):
    import torch
    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from train_all_constellations import encode_puzzle, SLOTS
    from eval_pynl_test import load_model

    # the run dir may be written by a live job: load from a copy
    tmp = tempfile.mkdtemp()
    for n in ("model_best.pt", "template_best.bin"):
        shutil.copy(os.path.join(args.run_dir, n), tmp)
    model, ck, a = load_model(tmp)
    print(f"checkpoint: {args.run_dir}  epoch {ck['epoch']}, val {100*ck['val_acc']:.2f}%\n")

    rng = random.Random(0)
    score = lambda panels, cfg: model(*encode_puzzle(panels, cfg))
    rows, worst_diff = [], 0.0
    tot = Counter()
    with torch.no_grad():
        for cfg in CONSTELLATIONS:
            fs = files(IRAVEN, cfg, "test", args.n)
            P = [parse(f.replace(".npz", ".xml"), cfg) for f in fs]
            Y = [int(np.load(f)["target"]) for f in fs]
            c = Counter()
            for i, (panels, y) in enumerate(zip(P, Y)):
                base = score(panels, cfg)
                c["base"] += int(base.argmax()) == y

                perm = list(range(8)); rng.shuffle(perm)            # (a) equivariance
                pp = panels[:8] + [panels[8 + perm[j]] for j in range(8)]
                out = score(pp, cfg)
                worst_diff = max(worst_diff, float((out - base[perm]).abs().max()))
                c["perm"] += int(out.argmax()) == perm.index(y)

                other = P[(i + 1) % len(P)]                          # (b) foreign context
                c["shuf"] += int(score(other[:8] + panels[8:], cfg).argmax()) == y

                at, ac = encode_puzzle(panels, cfg)                  # (c) no context
                ac = ac.clone(); ac[:8 * SLOTS] = False
                c["blank"] += int(model(at, ac).argmax()) == y
                c["n"] += 1
            rows.append((cfg, c)); tot += c

    print(f"{'constellation':<40} {'normal':>8} {'cands permuted':>15} {'foreign context':>16} {'no context':>11}")
    for cfg, c in rows + [("OVERALL", tot)]:
        n = c["n"]
        print(f"{cfg:<40} {100*c['base']/n:>7.1f}% {100*c['perm']/n:>14.1f}% "
              f"{100*c['shuf']/n:>15.1f}% {100*c['blank']/n:>10.1f}%")
    print(f"\nlargest |score change| under candidate permutation: {worst_diff:.2e}  (0 = exactly equivariant)")
    print("chance = 12.5%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["splits", "shortcut", "images", "model"])
    ap.add_argument("--n", type=int, default=200, help="test puzzles per constellation (images, model)")
    ap.add_argument("--run-dir", default="experiments/max_data_L2")
    ap.add_argument("--jvm-gb", type=int, default=30)
    ap.add_argument("--mismatch", action="store_true", help="images: negative control with wrong labels")
    args = ap.parse_args()
    {"splits": check_splits, "shortcut": check_shortcut,
     "images": check_images, "model": check_model}[args.check](args)


if __name__ == "__main__":
    main()
