#!/usr/bin/env python3
"""Symbolic PyNeuraLogic models end to end: ResNet-18 perception -> symbolic facts -> trained template.

For one trained symbolic model (pynl_symbolic.py) this scores a split three ways per constellation:
  xml        the template on facts read from the XML labels (rpm_symbolic.read_puzzle)
  end2end    the template on facts built from what the ResNet-18 frontend perceived
  reference  the non-learned solver of the same vocabulary (rpm_reference_solvers), on both inputs

Perception is the cached frontend output (pixel_pipeline.py perceive), decoded exactly as in
pixel_pipeline.panels_from_preds. Every perceived object is placed into the per-component structure
read_puzzle builds from the XML (count, slot set, sorted attribute tuples, per-slot attributes), so the
template sees the same kind of facts. A self-check builds that structure from the XML objects the same
way and asserts it equals read_puzzle.

Ties share credit, as in pynl_symbolic.accuracy.

Usage:
    python pynl_symbolic_e2e.py --model rules --run-dir experiments/symb_rules_n300_lr0.01 --split test
"""
import argparse, json, os, sys, time
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from rpm_symbolic import read_puzzle, component_offsets, CONSTELLATIONS
from train_pynl import parse, POS_MAP, _canonical
from pixel_pipeline import panels_from_preds, key
from rpm_reference_solvers import solve_primitives, solve_rules
import pynl_symbolic as PS

DATA = os.path.join(ROOT, "data/I-RAVEN-10000")
REF = {"prims": solve_primitives, "rules": solve_rules}


def layout_of(cfg):
    """universal position id -> (component, slot within component); slots per component."""
    rs_canon, off = component_offsets(cfg, DATA)
    tp_canon = _canonical(cfg, DATA)
    where = {POS_MAP[cfg][tp_canon[cb]]: (cb[0], rs_canon[cb] - off[cb[0]]) for cb in tp_canon}
    return where, Counter(ci for (ci, _) in rs_canon)


def to_symbolic(panels, cfg, where, nslots):
    """16 lists of {pos, type, size, color} -> read_puzzle's per-component structure."""
    out = []
    for objs in panels:
        per = defaultdict(dict)
        for o in objs:
            ci, s = where[o["pos"]]
            per[ci][s] = (o["type"], o["size"], o["color"])
        comps = []
        for ci in range(len(nslots)):
            sl = per[ci]
            comps.append({"number": len(sl), "position": frozenset(sl),
                          "type": tuple(sorted(v[0] for v in sl.values())),
                          "size": tuple(sorted(v[1] for v in sl.values())),
                          "color": tuple(sorted(v[2] for v in sl.values())),
                          "slots": dict(sl), "nslots": nslots[ci], "uniformity": 0})
        out.append(comps)
    return out


FIELDS = ("number", "position", "type", "size", "color", "slots", "nslots")


def same_structure(a, b):
    return all(len(pa) == len(pb) and all(ca[f] == cb[f] for ca, cb in zip(pa, pb) for f in FIELDS)
               for pa, pb in zip(a, b))


def shared_accuracy(scores, ys):
    tot = 0.0
    for s, y in zip(scores, ys):
        top = np.flatnonzero(s == s.max())
        tot += (1.0 / len(top)) if y in top else 0.0
    return 100 * tot / len(ys)


def template_scores(model_name, tmpl_path, puzzles, settings_kw, R, Sample, Dataset, Settings, Adam, CrossEntropy):
    """Fresh build per chunk (build_dataset may be called once per built model), then load the weights."""
    model = PS.TEMPLATES[model_name]()
    model.build(Settings(optimizer=Adam(lr=0.0), error_function=CrossEntropy(with_logits=True), **settings_kw))
    samples = [PS.make_sample(p, 0, f"g{i}", list(range(8)), R, Sample) for i, p in enumerate(puzzles)]
    built = list(model.build_dataset(Dataset([PS.probe_sample(R, Sample)] + samples)))
    model.load(tmpl_path)
    order = PS.read_order(model, built[0])
    out = [PS.unpermute(np.asarray(o, dtype=float).flatten(), order) for o in model.test(built[1:])]
    assert len(out) == len(puzzles), f"{len(out)} outputs for {len(puzzles)} puzzles"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["prims", "rules"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--preds", default=os.path.join(ROOT, "experiments/frontend_preds/"
                                                    "universal_frontend_correct_proportional_mean3_{split}.npz"))
    ap.add_argument("--n", type=int, default=0, help="puzzles per constellation (0 = all; val uses 300+)")
    ap.add_argument("--self-check", type=int, default=50, help="puzzles per constellation to verify the converter on")
    ap.add_argument("--jvm-gb", type=int, default=40)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn import Adam, CrossEntropy

    tmpl = os.path.join(args.run_dir, "template_best.bin")
    best = json.load(open(os.path.join(args.run_dir, "best.json")))
    print(f"model {args.model}: {args.run_dir} (epoch {best['epoch']}, val {100 * best['val_acc']:.2f}%)", flush=True)
    z = np.load(args.preds.format(split=args.split))
    row_of = {f: i for i, f in enumerate(z["files"])}
    tau = float(z["tau"])

    results, t0 = {}, time.time()
    for cfg in CONSTELLATIONS:
        where, nslots = layout_of(cfg)
        positions = sorted(POS_MAP[cfg].values())
        rows = [i for i, c in enumerate(z["cfgs"]) if c == cfg]
        if args.split == "val":
            rows = rows[300:]
        if args.n:
            rows = rows[:args.n]
        xml_p, e2e_p, ys, exact = [], [], [], 0
        for k, r in enumerate(rows):
            f = os.path.join(ROOT, str(z["files"][r]))
            xml_path = f.replace(".npz", ".xml")
            truth, _ = read_puzzle(xml_path, cfg)
            if k < args.self_check:
                assert same_structure(to_symbolic(parse(xml_path, cfg), cfg, where, nslots), truth), \
                    f"converter differs from read_puzzle on {f}"
            seen = panels_from_preds(z["maxcos"][r], z["argidx"][r], positions, tau)
            exact += all(key(a) == key(b) for a, b in zip(seen, parse(xml_path, cfg)))
            xml_p.append(truth)
            e2e_p.append(to_symbolic(seen, cfg, where, nslots))
            ys.append(int(z["targets"][r]))
        kw = {"iso_value_compression": False}
        s_xml = template_scores(args.model, tmpl, xml_p, kw, R, Sample, Dataset, Settings, Adam, CrossEntropy)
        s_e2e = template_scores(args.model, tmpl, e2e_p, kw, R, Sample, Dataset, Settings, Adam, CrossEntropy)
        res = {"n": len(rows), "all16_perceived": 100 * exact / len(rows),
               "xml": shared_accuracy(s_xml, ys), "end2end": shared_accuracy(s_e2e, ys),
               "ref_xml": shared_accuracy([REF[args.model](p) for p in xml_p], ys),
               "ref_end2end": shared_accuracy([REF[args.model](p) for p in e2e_p], ys)}
        results[cfg] = res
        print(f"{cfg:<40} n {res['n']:>4}  perceived {res['all16_perceived']:6.2f}%  "
              f"template xml {res['xml']:6.2f}%  end2end {res['end2end']:6.2f}%  |  "
              f"reference xml {res['ref_xml']:6.2f}%  end2end {res['ref_end2end']:6.2f}%  ({time.time() - t0:.0f}s)",
              flush=True)

    n = sum(r["n"] for r in results.values())
    overall = {k: sum(r[k] * r["n"] for r in results.values()) / n
               for k in ("all16_perceived", "xml", "end2end", "ref_xml", "ref_end2end")}
    results["OVERALL"] = {"n": n, **overall}
    print("-" * 120)
    print(f"{'OVERALL':<40} n {n:>4}  perceived {overall['all16_perceived']:6.2f}%  "
          f"template xml {overall['xml']:6.2f}%  end2end {overall['end2end']:6.2f}%  |  "
          f"reference xml {overall['ref_xml']:6.2f}%  end2end {overall['ref_end2end']:6.2f}%")
    out = args.json_out or os.path.join(ROOT, f"experiments/symb_e2e_{args.model}_{args.split}.json")
    json.dump({"model": args.model, "run_dir": args.run_dir, "epoch": best["epoch"], "val_acc": best["val_acc"],
               "split": args.split, "tau": tau, "results": results}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
