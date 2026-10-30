#!/usr/bin/env python3
"""Evaluation of facet checkpoints (bench_hub_grounding.py --save-dir) on unseen puzzles, optionally after
transforming every puzzle. The template is rebuilt from the run's best.json, so any variant (copy links,
gates, column roles, row fold, labels, widths) loads.

I-RAVEN-X checkpoints: unseen puzzles 6300..; matched and exact as in training (bench_hub_grounding.evaluate),
matched with and without an Arithmetic attribute, and the exact value per rule (the generated code's bits
summed and rounded, as in notes §20-21). --iravenx scores another file with the same value range, for
row-length transfer; only templates that name no column (--columns roles or fold) can be built for another
row length.
I-RAVEN checkpoints: val puzzles 300.. of the constellations the run trained on (--n per constellation);
matched and exact per constellation, matched with and without an Arithmetic rule, and per component the
exact attribute (and Number/Position part) by the rule the puzzle's XML gives it (xml_rules).

Transforms (the answer and candidates are transformed too wherever the transformation reaches them):
  rows12     rows 1 <-> 2 of the context
  cols23     columns 2 <-> 3 of every row (two middle columns when a row has 4 or more)
  sizecolor  the size and colour values of every object exchanged (I-RAVEN-X: a data symmetry, §22)
  mirror     left-right and up-down layouts mirrored: slots 14 <-> 15 and 16 <-> 17, so the components swap

Run in the pynl environment from the repository root, as a job (grounding 1000 puzzles takes ~48 GB). One
dataset family per call (I-RAVEN and I-RAVEN-X configure the module differently):
    python3 pynl/facet_rule_eval.py experiments/facet_iravenx_n10_m50_roles --transform none cols23
    python3 pynl/facet_rule_eval.py experiments/facet_iravenx_n10_m50_roles \\
        --iravenx data/iravenx/center_single_shuffle_n_5_maxval_50.json     # rows of another length
    python3 pynl/facet_rule_eval.py experiments/facet_lrud_n100 --transform none mirror
"""
import argparse, ast, collections, json, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_hub_grounding as B
from facet_symmetry_check import rows12, cols23


def size_color(pz):
    return [[{**o, "size": o["color"], "color": o["size"]} for o in p] for p in pz]


def mirror(pz):
    sw = {14: 15, 15: 14, 16: 17, 17: 16}
    return [[{**o, "pos": sw.get(o["pos"], o["pos"])} for o in p] for p in pz]


TRANSFORMS = {"none": lambda pz: pz, "rows12": rows12, "cols23": cols23, "sizecolor": size_color, "mirror": mirror}
RULES = ("Constant", "Progression", "Distribute_Three", "Arithmetic")
NUMPOS = ("Number/Position", "Number", "Position")


def xml_rules(path):
    """I-RAVEN: a puzzle's rules per component, {"k0": {"Number/Position": "Constant", "Type": ..., ...}, ...};
    the XML's Rule_Group id is the component id, which is B.COMPONENTS' k index."""
    txt = open(path).read()
    return {f"k{g}": {at: name for name, at in re.findall(r'<Rule name="(\w+)" attr="([\w/]+)"', body)}
            for g, body in re.findall(r'<Rule_Group id="(\d+)">(.*?)</Rule_Group>', txt, re.S)}


def iraven_rule_split(per_puzzle, tdata, irules, t):
    """I-RAVEN: each component's attribute decoded exactly (its value histogram, as in B.evaluate), by the rule
    that governs it there; the Number/Position part likewise (Number: the object count, else the positions).
    Matched split by whether any component has an Arithmetic rule."""
    hits = collections.defaultdict(lambda: [0, 0])                  # (part, rule) -> [exact, count]
    by_ar = collections.defaultdict(lambda: [0, 0])                 # has Arithmetic? -> [matched, puzzles]
    for outs, (q, y, cfg), rr in zip(per_puzzle, tdata, irules):
        if t == "mirror" and cfg.startswith(("left", "up")):
            rr = {"k0": rr["k1"], "k1": rr["k0"]}
        if t == "sizecolor":
            rr = {k: {**r, "Size": r["Color"], "Color": r["Size"]} for k, r in rr.items()}
        dist = np.zeros(8)
        for k, o in outs.items():
            r = rr[k]
            got, want = B.facet_decode(o, "therm"), B.decode(B.describe(B.comp_objs(q[B.N_CTX + y], cfg, k)))
            for attr in B.FACET_ATTRS:
                h = hits[(attr, r[attr.capitalize()])]
                h[0] += got[attr] == want[attr]
                h[1] += 1
            pa = next(p for p in NUMPOS if p in r)
            h = hits[(pa, r[pa])]
            h[0] += len(got["pos"]) == len(want["pos"]) if pa == "Number" else got["pos"] == want["pos"]
            h[1] += 1
            dist += [np.sum((o - B.facet_target(B.comp_objs(q[B.N_CTX + c], cfg, k))) ** 2) for c in range(8)]
        best = np.flatnonzero(dist <= dist.min() + 1e-9)
        key = "with Arithmetic" if any("Arithmetic" in r.values() for r in rr.values()) else "without"
        by_ar[key][0] += int(y in best and len(best) == 1)
        by_ar[key][1] += 1
    pct = lambda h: f"{100 * h[0] / h[1]:.1f} (of {h[1]})"
    print("    " + ", ".join(f"matched {k} {pct(h)}" for k, h in sorted(by_ar.items())))
    tot = {r: [sum(hits.get((a, r), (0, 0))[i] for a in B.FACET_ATTRS) for i in (0, 1)] for r in RULES}
    print("    exact value, type/size/color: " + " / ".join(f"{r} {pct(tot[r])}" for r in RULES if tot[r][1]))
    for part in B.FACET_ATTRS + NUMPOS:
        cells = [f"{r} {pct(hits[(part, r)])}" for r in RULES if (part, r) in hits]
        if cells:
            print(f"      {part}: " + " / ".join(cells), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+", help="run directories with best.json and template_best.bin")
    ap.add_argument("--n", type=int, default=1000, help="I-RAVEN-X: unseen puzzles from 6300; I-RAVEN: per constellation")
    ap.add_argument("--transform", nargs="*", choices=list(TRANSFORMS), default=["none"])
    ap.add_argument("--jvm-gb", type=int, default=48)
    ap.add_argument("--iravenx", default=None, help="score on this I-RAVEN-X file instead of the training one")
    args = ap.parse_args()

    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn import Adam, MSE

    for ckpt in args.ckpts:
        best = json.load(open(os.path.join(ckpt, "best.json")))
        a = best["args"]
        a = ast.literal_eval(a) if isinstance(a, str) else a
        irx = args.iravenx or a.get("iravenx")
        if args.iravenx and a.get("columns", "named") == "named":
            raise SystemExit(f"{ckpt}: named columns fix the row length; only roles or fold templates transfer")
        if irx:
            first = B.load_iravenx(irx, 0, 1)[0][0]
            B.configure(n_cols=(len(first) - 7) // 3, maxval=int(irx.split("maxval_")[1].split(".")[0]))
            data = B.load_iravenx(irx, 6300, args.n)
            raw = B._IRAVENX[irx]
            rules = [(lambda r: ast.literal_eval(r) if isinstance(r, str) else r)(raw[str(i)]["rules"])[0]
                     for i in range(6300, 6300 + args.n)]
            where = f"{args.n} unseen puzzles (6300..) of {os.path.basename(irx)}, rows of {len(B.COLS)}"
            irules = None
        else:
            cfgs = list(B.POS_MAP) if a["constellation"] == "all" else a["constellation"].split(",")
            root = a.get("root", "data/I-RAVEN-10000")
            data = B.load(root, cfgs, "val", 300, args.n)
            rules = None
            irules = [xml_rules(os.path.join(root, c, f"RAVEN_{i}_val.xml")) for c in cfgs for i in range(300, 300 + args.n)]
            where = f"{args.n} val puzzles (300..) per constellation of {', '.join(c[:12] for c in cfgs)}"
        cols = a.get("columns", "named")
        copy, roles, fold = bool(a.get("copy_links")), cols != "named", cols == "fold"
        labels = tuple(k for k, f in (("components", "label_components"), ("attributes", "label_attributes")) if a.get(f))
        widths = [int(w) for w in a["widths"].split(",")] if a.get("widths") else None
        model = B.build_facet_template(a["dim"], a["layers"], not a.get("per_attr_weights"), not a.get("no_rows"),
                                       widths, copy=copy, gates=a.get("copy_gates", "panel"), cols=cols, labels=labels,
                                       shared_comps=not a.get("per_comp_weights"))
        model.build(Settings(optimizer=Adam(lr=a["lr"]), error_function=MSE(), iso_value_compression=False))
        model.load(os.path.join(ckpt, "template_best.bin"))
        n_w = sum(np.asarray(v).size for v in model.state_dict()["weights"].values())
        print(f"\n{ckpt}: {n_w} weights, epoch {best['epoch']}, {where}", flush=True)

        for t in args.transform:
            tdata = [(TRANSFORMS[t](pz), y, cfg) for pz, y, cfg in data]
            units, samples = [], []
            for i, (q, y, cfg) in enumerate(tdata):
                for k in B.comps(cfg):
                    gid = f"{t}{i}{k}"
                    facts = (B.puzzle_facts(q, gid, col_roles=roles, fold=fold)
                             + B.facet_facts(q, "therm", cfg, copy, labels=labels) + [R.qcomp(gid, k)])
                    units.append((i, k))
                    samples.append(Sample(R.out(gid)[B.facet_target(B.comp_objs(q[B.N_CTX + y], cfg, k)).tolist()], facts))
            outs = []
            for c0 in range(0, len(samples), 2000):                    # grounded in chunks, to bound memory
                outs += [np.asarray(o, dtype=float).flatten()
                         for o in model.test(list(model.build_dataset(Dataset(samples[c0:c0 + 2000]))))]
            per_puzzle = [dict() for _ in tdata]
            for (i, k), o in zip(units, outs):
                per_puzzle[i][k] = o
            _, exact, matched, ties, per_cfg = B.evaluate(per_puzzle, tdata, "therm")
            n = len(tdata)
            line = f"  transform {t}: matched {100 * matched / n:.1f}%, exact {100 * exact['all'] / n:.1f}% (ties {ties})"
            if rules is None:
                print(line + "; per constellation: " + "  ".join(f"{c[:12]} {100 * h / m:.1f}" for c, (h, m) in per_cfg.items()),
                      flush=True)
                iraven_rule_split(per_puzzle, tdata, irules, t)
                continue
            hits = collections.defaultdict(lambda: [0, 0])             # rule -> [exact values, values]
            by_ar = collections.defaultdict(lambda: [0, 0])            # has Arithmetic? -> [matched, puzzles]
            for o, (q, y, _), r in zip((p["k0"] for p in per_puzzle), tdata, rules):
                if t == "sizecolor":
                    r = {**r, "Size": r["Color"], "Color": r["Size"]}
                for j, attr in enumerate(B.FACET_ATTRS):
                    h = hits[r[attr.capitalize()]]
                    h[0] += int(round(o[j * B.THERM:(j + 1) * B.THERM].sum())) == q[B.N_CTX + y][0][attr]
                    h[1] += 1
                dist = [np.sum((o - B.facet_target(q[B.N_CTX + c])) ** 2) for c in range(8)]
                key = "with Arithmetic" if "Arithmetic" in (r["Type"], r["Size"], r["Color"]) else "without"
                by_ar[key][0] += int(np.argmin(dist) == y and sorted(dist)[1] > dist[y])
                by_ar[key][1] += 1
            print(line + "; " + ", ".join(f"matched {k} {100 * h / m:.1f}% of {m}" for k, (h, m) in sorted(by_ar.items())))
            print("    exact value: " + " / ".join(f"{r} {100 * hits[r][0] / max(1, hits[r][1]):.1f} (of {hits[r][1]})"
                                                    for r in RULES), flush=True)


if __name__ == "__main__":
    main()
