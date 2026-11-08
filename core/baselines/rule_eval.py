#!/usr/bin/env python3
"""I-RAVEN baselines by rule and layout, on the unseen puzzles: what core/pynl/facet_rule_eval.py prints for facet
checkpoints, with the same definitions, so the two compare directly. Per component, an attribute is exact by the
rule the puzzle's XML gives it: the value histogram for type, size and colour, the object count for Number, the
positions for Position and Number/Position. Matched is split by whether any component has an Arithmetic rule.
Per layout: matched, exact, and the exact type, size and colour values by rule. Writes rule_eval.json next to the
run's best.json. Environment nvsa, one GPU, from the repository root:
    python core/baselines/rule_eval.py experiments/baselines/runs/iraven_vit_wd0.05_do0.1_s0 [more run directories]
"""
import argparse, collections, json, os, re, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_baseline import Data, build, decode, has_pair, images, run, set_vit, spec_parts

RULES = ("Constant", "Progression", "Distribute_Three", "Arithmetic")
ATTRS = {"type": "Type", "size": "Size", "color": "Color"}
NUMPOS = ("Number/Position", "Number", "Position")


def xml_rules(path):
    """facet_rule_eval.xml_rules (that module needs neuralogic): {"k0": {"Type": "Constant", ...}, ...}."""
    txt = open(path).read()
    return {f"k{g}": {at: name for name, at in re.findall(r'<Rule name="(\w+)" attr="([\w/]+)"', body)}
            for g, body in re.findall(r'<Rule_Group id="(\d+)">(.*?)</Rule_Group>', txt, re.S)}


def puzzle_stems(data, meta, root, start=300):
    """The file stem of every puzzle of the split (prepare_baseline_data.py lists them per constellation, in order);
    every 50th is checked against its answer index."""
    seen, stems = collections.Counter(), []
    for i, c in enumerate(data.cfg):
        cfg = meta["constellations"][c]
        stems.append(os.path.join(root, cfg, f"RAVEN_{start + seen[cfg]}_val"))
        seen[cfg] += 1
        if i % 50 == 0:
            assert int(np.load(stems[-1] + ".npz")["target"]) == data.y[i], f"{stems[-1]}: another puzzle"
    return stems


@torch.no_grad()
def score(run_dir, data_dir, root, dev):
    info = json.load(open(os.path.join(run_dir, "best.json")))
    a = info["args"]
    if a["data"] != "iraven":
        raise SystemExit(f"{run_dir}: I-RAVEN only; the I-RAVEN-X runs have exact_per_rule in best.json")
    z = np.load(os.path.join(data_dir, "iraven.npz"))
    meta = json.loads(str(z["meta"]))
    set_vit(meta, a)
    data = Data(z, "unseen", meta, dev, images(data_dir, a["data"], a["arch"], z, a.get("image_size", 80)).get("unseen"))
    model = build(a["arch"], meta, info["width"], a.get("norm", False), a.get("dropout", 0.0)).to(dev)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pt"), map_location=dev))
    model.eval()

    out = run(model, a["arch"], data, data.units, dev)
    k = torch.as_tensor(data.units[:, 1], device=dev)
    starts = range(0, len(k), 512)
    cand = torch.cat([data.candidate_targets(data.units[s:s + 512, 0], k[s:s + 512]) for s in starts])
    tgt = torch.cat([data.batch_target(data.units[s:s + 512, 0], k[s:s + 512]) for s in starts])
    d = ((out[:, None] - cand) ** 2).sum(-1).cpu().numpy()                             # [units, 8]
    out, tgt = out.cpu().numpy(), tgt.cpu().numpy()

    rules = [xml_rules(s + ".xml") for s in puzzle_stems(data, meta, root)]
    parts = spec_parts(meta)
    n = len(data.y)
    dist, ok = np.zeros((n, 8)), np.ones(n, bool)
    hits = collections.defaultdict(lambda: [0, 0])                                     # (part, rule) -> [exact, count]
    by_layout = collections.defaultdict(lambda: [0, 0])                                # (layout, rule) -> [exact, count]
    by_layout_occ = collections.defaultdict(lambda: [0, 0])                            # the same, occupancy given
    T4 = 4 * meta["therm"]
    for j, (i, kk) in enumerate(data.units):
        dist[i] += d[j]
        got, want = decode(out[j], meta), decode(tgt[j], meta)
        given = decode(np.concatenate([out[j][:T4], tgt[j][T4:]]), meta)               # an untrained slot hides no value
        r, cfg = rules[i][f"k{kk}"], meta["constellations"][data.cfg[i]]
        ok[i] &= bool((got == want).all())
        for attr, tag in ATTRS.items():
            hit = int((got[parts[attr]] == want[parts[attr]]).all())
            for h in (hits[(attr, r[tag])], by_layout[(cfg, r[tag])]):
                h[0] += hit
                h[1] += 1
            h = by_layout_occ[(cfg, r[tag])]
            h[0] += int((given[parts[attr]] == want[parts[attr]]).all())
            h[1] += 1
        pa = next(p for p in NUMPOS if p in r)
        g, w = got[parts["pos"]], want[parts["pos"]]
        h = hits[(pa, r[pa])]
        h[0] += int(g.sum() == w.sum()) if pa == "Number" else int((g == w).all())
        h[1] += 1
    best = dist <= dist.min(1, keepdims=True) + 1e-9
    matched = best[np.arange(n), data.y] & (best.sum(1) == 1)
    arith = np.array([any("Arithmetic" in c.values() for c in r.values()) for r in rules])
    pair = has_pair(data_dir, "unseen", "Progression:Size")

    res = {"epoch": info["best"]["epoch"], "weights": info["weights"], "matched": float(matched.mean()),
           "exact": float(ok.mean()),
           "with_arithmetic": [int(matched[arith].sum()), int(arith.sum())],
           "without_arithmetic": [int(matched[~arith].sum()), int((~arith).sum())],
           "with_progression_size": [int(matched[pair].sum()), int(pair.sum())],
           "by_part": {p: {r: hits[(p, r)] for r in RULES if (p, r) in hits} for p in list(ATTRS) + list(NUMPOS)},
           "by_layout": {}}
    for c, cfg in enumerate(meta["constellations"]):
        sel = data.cfg == c
        res["by_layout"][cfg] = {"matched": float(matched[sel].mean()), "exact": float(ok[sel].mean()),
                                 "rules": {r: by_layout[(cfg, r)] for r in RULES if (cfg, r) in by_layout},
                                 "rules_occupancy_given": {r: by_layout_occ[(cfg, r)] for r in RULES if (cfg, r) in by_layout_occ}}
    json.dump(res, open(os.path.join(run_dir, "rule_eval.json"), "w"), indent=1)

    pct = lambda h: f"{100 * h[0] / max(1, h[1]):.1f} (of {h[1]})"
    print(f"\n{run_dir}: {info['weights']} weights, epoch {res['epoch']}; matched {100 * res['matched']:.1f}, exact "
          f"{100 * res['exact']:.1f} (best.json: {100 * info['best']['unseen']['matched']:.1f}, "
          f"{100 * info['best']['unseen']['exact']:.1f})")
    print(f"  matched with Arithmetic {pct(res['with_arithmetic'])}, without {pct(res['without_arithmetic'])}, "
          f"with Progression on Size {pct(res['with_progression_size'])}")
    all3 = {r: [sum(hits[(p, r)][i] for p in ATTRS if (p, r) in hits) for i in (0, 1)] for r in RULES}
    print("  exact, type/size/colour: " + " / ".join(f"{r} {pct(all3[r])}" for r in RULES))
    for p, cells in res["by_part"].items():
        print(f"    {p}: " + " / ".join(f"{r} {pct(h)}" for r, h in cells.items()))
    for cfg, v in res["by_layout"].items():
        print(f"  {cfg}: matched {100 * v['matched']:.1f}, exact {100 * v['exact']:.1f}; "
              + " / ".join(f"{r} {pct(h)}" for r, h in v["rules"].items()), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories with best.json and model_best.pt")
    ap.add_argument("--data-dir", default="experiments/baselines/data")
    ap.add_argument("--root", default="data/I-RAVEN-10000")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for r in args.runs:
        score(r, args.data_dir, args.root, dev)


if __name__ == "__main__":
    main()
