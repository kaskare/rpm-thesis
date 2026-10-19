#!/usr/bin/env python3
"""Per-puzzle failure analysis for a trained all-constellations model.

Scores every test puzzle of the chosen constellations and breaks accuracy down by
the RAVEN rule governing each attribute, and by how many objects the answer
panel holds. Answers "which rules does the model fail on, in which constellation".
"""
import argparse, glob, os, sys, xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch
import neuralogic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pynl import parse
from train_all_constellations import encode_puzzle
from eval_pynl_test import load_model


def rules_of(xml_path):
    """{attribute: rule_name} from the XML <Rules> block.

    Two-component constellations (left/right, up/down, in/out, in_distribute_four) have
    one rule group per component; keying by attribute alone let the second group
    overwrite the first, so those attributes are prefixed with the component index.
    """
    out = {}
    for sec in ET.parse(xml_path).getroot():
        if sec.tag.lower().startswith("rule"):
            groups = list(sec)
            for gi, group in enumerate(groups):
                for r in group:
                    key = r.attrib["attr"] if len(groups) == 1 else f"c{gi}:{r.attrib['attr']}"
                    out[key] = r.attrib["name"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--constellations", default="distribute_four,distribute_nine")
    ap.add_argument("--dataset-root", default="data/I-RAVEN-10000")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--jvm-gb", type=int, default=60)
    args = ap.parse_args()

    neuralogic.set_max_memory_size(args.jvm_gb)
    model, ck, a = load_model(args.run_dir)
    print(f"model: {os.path.basename(args.run_dir)}  (level {a['level']}, best epoch {ck['epoch']})\n")

    for cfg in args.constellations.split(","):
        files = sorted(glob.glob(os.path.join(args.dataset_root, cfg, f"*_{args.split}.npz")))[:args.n]
        by_rule = defaultdict(lambda: [0, 0])      # (attr, rule) -> [correct, total]
        by_nobj = defaultdict(lambda: [0, 0])      # answer-panel object count
        by_margin = []
        hits = 0
        with torch.no_grad():
            for f in files:
                xml = f.replace(".npz", ".xml")
                panels = parse(xml, cfg)
                attrs, active = encode_puzzle(panels, cfg)
                y = int(np.load(f)["target"])
                logits = model(attrs, active)
                ok = int(logits.argmax()) == y
                hits += ok
                for attr, rule in rules_of(xml).items():
                    by_rule[(attr, rule)][0] += ok
                    by_rule[(attr, rule)][1] += 1
                n_ans = len(panels[8 + y])
                by_nobj[n_ans][0] += ok; by_nobj[n_ans][1] += 1
                srt = torch.sort(logits, descending=True).values
                by_margin.append(float(srt[0] - srt[1]))

        n = len(files)
        print(f"{'=' * 72}\n{cfg}   overall {100 * hits / n:.2f}%  ({hits}/{n})\n{'=' * 72}")
        print(f"\n  accuracy by rule (per attribute):")
        print(f"  {'attribute':<18} {'rule':<18} {'acc':>8} {'n':>6}")
        for (attr, rule), (c, t) in sorted(by_rule.items(), key=lambda kv: (kv[0][0], -kv[1][1])):
            print(f"  {attr:<18} {rule:<18} {100 * c / t:>7.1f}% {t:>6}")
        print(f"\n  accuracy by number of objects in the answer panel:")
        for k in sorted(by_nobj):
            c, t = by_nobj[k]
            print(f"     {k} objects   {100 * c / t:>6.1f}%   (n={t})")
        print(f"\n  median top-1 vs top-2 logit margin: {np.median(by_margin):.3f}\n")


if __name__ == "__main__":
    main()
