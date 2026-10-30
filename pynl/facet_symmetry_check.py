#!/usr/bin/env python3
"""Which symmetries the facet model (bench_hub_grounding.py --template facet) has by construction.

A transformation g of a puzzle is built in when the output on g(puzzle) equals the output on the puzzle
(invariance), or that output with its blocks permuted (equivariance), for ANY weights. So every check runs
on random weights (several draws) and, with --trained, on the trained checkpoints. Controls are
transformations a template is not built to respect: on random weights they change the output.
Why it holds (notes §21): the ground network depends on the constants of the facts only through the
constants the template itself names, so renaming any other constant (objects, panels, rows, components,
attributes) maps the network onto itself, with the same weight on every corresponding edge.

    python3 pynl/facet_symmetry_check.py              # random weights, four template variants (I-RAVEN)
    python3 pynl/facet_symmetry_check.py --trained    # + I-RAVEN and I-RAVEN-X 3x10 (values 0..49) checkpoints
The I-RAVEN-X table checks the column roles (--columns roles, notes §24): columns 2 and 3 exchanged in every
row, both middle columns for n = 10.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_hub_grounding as B

LR = "left_center_single_right_center_single"


# ---- transformations: panels (8 context + 8 candidates) -> panels ----
def objects(pz):                    # every panel lists its objects in reverse order
    return [list(reversed(p)) for p in pz]


def rows12(pz):                     # rows 1 and 2 exchanged
    n = len(B.COLS)
    return pz[n:2 * n] + pz[:n] + pz[2 * n:]


def cols12(pz):                     # columns 1 and 2 exchanged in every row (control)
    n, out = len(B.COLS), list(pz)
    for r in range(3):
        out[r * n], out[r * n + 1] = pz[r * n + 1], pz[r * n]
    return out


def cols23(pz):                     # columns 2 and 3 exchanged in every row: both middle columns when n >= 4
    n, out = len(B.COLS), list(pz)
    for r in range(3):
        out[r * n + 1], out[r * n + 2] = pz[r * n + 2], pz[r * n + 1]
    return out


def type_color(pz):                 # the values of type and colour exchanged in every object
    return [[{**o, "type": o["color"], "color": o["type"]} for o in p] for p in pz]


def mirror(pz):                     # left-right layout mirrored: the two components exchange
    sw = {14: 15, 15: 14}
    return [[{**o, "pos": sw.get(o["pos"], o["pos"])} for o in p] for p in pz]


def candidates(pz):                 # the eight candidates in reverse order
    return pz[:B.N_CTX] + pz[B.N_CTX:][::-1]


def swap_blocks(o, a, b):           # output with the code blocks of attributes a and b exchanged
    o, T = o.copy(), B.THERM
    o[a * T:(a + 1) * T], o[b * T:(b + 1) * T] = o[b * T:(b + 1) * T].copy(), o[a * T:(a + 1) * T].copy()
    return o


def swap_slots(o, a=14, b=15):      # output with occupancy slots a and b exchanged
    o, T = o.copy(), 4 * B.THERM
    o[T + a], o[T + b] = o[T + b], o[T + a]
    return o


# name -> (transformation, puzzles it applies to, expected output on g(puzzle) from the outputs on the
# puzzle, the part of the output compared)
T_ = lambda: 4 * B.THERM
other = {"k0": "k1", "k1": "k0"}
CHECKS = {
    "objects (order within a panel)": (objects, lambda cfg: True, lambda out, i, k: out[(i, k)], "all"),
    "rows 1 <-> 2": (rows12, lambda cfg: True, lambda out, i, k: out[(i, k)], "all"),
    "attributes type <-> colour": (type_color, lambda cfg: True,
                                   lambda out, i, k: swap_blocks(out[(i, k)], 0, 2), "all"),
    "components (mirror L-R), attributes": (mirror, lambda cfg: cfg == LR,
                                            lambda out, i, k: out[(i, other[k])], "attrs"),
    "components (mirror L-R), positions": (mirror, lambda cfg: cfg == LR,
                                           lambda out, i, k: swap_slots(out[(i, other[k])]), "pos"),
    "candidates (order)": (candidates, lambda cfg: True, lambda out, i, k: out[(i, k)], "all"),
    "middle columns 2 <-> 3 (n >= 4)": (cols23, lambda cfg: len(B.COLS) >= 4, lambda out, i, k: out[(i, k)], "all"),
    "control: columns 1 <-> 2": (cols12, lambda cfg: True, lambda out, i, k: out[(i, k)], "all"),
}


def predict(outs, i, pz, cfg):
    """The candidate the model picks: closest in the code space, summed over components."""
    return int(np.argmin([sum(np.sum((outs[(i, k)] - B.facet_target(B.comp_objs(pz[B.N_CTX + c], cfg, k))) ** 2)
                              for k in B.comps(cfg)) for c in range(8)]))


def run(model, puzzles, copy, draws, roles=False, fold=False, labels=()):
    """Per check: the max |output difference| over `draws` random weight settings, or (draws=None) on the
    model's current weights, where it also counts the puzzles whose chosen candidate changes and the
    puzzles answered right before and after the transformation."""
    from neuralogic.core import R
    from neuralogic.dataset import Dataset, Sample
    sets = {"orig": puzzles}
    for name, (g, applies, _, _) in CHECKS.items():
        sets[name] = [(g(pz), y, cfg) for pz, y, cfg in puzzles if applies(cfg)]
    units, samples = [], []
    for s, ps in sets.items():
        for i, (pz, y, cfg) in enumerate(ps):
            for k in B.comps(cfg):
                gid = f"g{len(units)}"
                facts = (B.puzzle_facts(pz, gid, col_roles=roles, fold=fold) + B.facet_facts(pz, "therm", cfg, copy, labels=labels)
                         + [R.qcomp(gid, k)])
                units.append((s, i, k))
                samples.append(Sample(R.out(gid)[B.facet_target(B.comp_objs(pz[B.N_CTX + y], cfg, k)).tolist()],
                                      facts))
    built = list(model.build_dataset(Dataset(samples)))
    assert len(built) == len(samples)
    sd = model.state_dict()
    rng = np.random.default_rng(0)
    res = {name: {"diff": 0.0, "n": len(sets[name]), "moved": 0, "right": [0, 0]} for name in CHECKS}
    for d in range(draws or 1):
        if draws:                    # every weight N(0, 1/fan-in): no saturation, every path carries signal
            w = {}
            for wid, v in sd["weights"].items():
                v = np.asarray(v, dtype=float)
                w[wid] = rng.normal(0, 1 / np.sqrt(v.shape[-1] if v.ndim == 2 else 1), v.shape).tolist()
            model.load_state_dict({"weights": w, "weight_names": sd["weight_names"]})
        outs = {}
        for u, o in zip(units, model.test(built)):
            outs.setdefault(u[0], {})[u[1:]] = np.asarray(o, dtype=float).flatten()
        for name, (g, applies, expect, part) in CHECKS.items():
            r = res[name]
            idx = [i for i, (_, _, cfg) in enumerate(puzzles) if applies(cfg)]
            sl = {"all": slice(None), "attrs": slice(0, T_()), "pos": slice(T_(), None)}[part]
            for j, i in enumerate(idx):
                for k in B.comps(puzzles[i][2]):
                    r["diff"] = max(r["diff"], np.max(np.abs(outs[name][(j, k)][sl] - expect(outs["orig"], i, k)[sl])))
                if not draws:
                    pz, y, cfg = sets[name][j]
                    before = predict(outs["orig"], i, puzzles[i][0], cfg)
                    after = predict(outs[name], j, pz, cfg)
                    flip = name.startswith("candidates")
                    r["moved"] += int(after != (7 - before if flip else before))
                    r["right"][0] += int(before == puzzles[i][1])
                    r["right"][1] += int(after == (7 - y if flip else y))
    if draws:
        model.load_state_dict(sd)
    return {name: r if r["n"] else None for name, r in res.items()}


def table(cols):
    names = list(CHECKS)
    w = max(map(len, names)) + 2
    cw = max(map(len, cols)) + 2
    print(f"{'max |output difference|':{w}s}" + "".join(f"{c:>{cw}s}" for c in cols))
    for n in names:
        print(f"{n:{w}s}" + "".join(f"{cols[c][n]:>{cw}s}" for c in cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-cfg", type=int, default=3, help="I-RAVEN val puzzles (300..) per constellation")
    ap.add_argument("--n-iravenx", type=int, default=40, help="I-RAVEN-X unseen puzzles (6300..)")
    ap.add_argument("--draws", type=int, default=3, help="random weight settings per template")
    ap.add_argument("--trained", action="store_true")
    ap.add_argument("--jvm-gb", type=int, default=8)
    args = ap.parse_args()

    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import Settings
    from neuralogic.nn import Adam, MSE

    def build(copy=False, rows=True, shared=True, gates="panel", cols="named", labels=(), shared_comps=True, ckpt=None):
        m = B.build_facet_template(32, 2, shared, rows, [32, 64, 128], copy=copy, gates=gates, cols=cols, labels=labels,
                                   shared_comps=shared_comps)
        m.build(Settings(optimizer=Adam(lr=3e-4), error_function=MSE(), iso_value_compression=False))
        if ckpt:
            m.load(os.path.join(ckpt, "template_best.bin"))
        n = sum(np.asarray(v).size for v in m.state_dict()["weights"].values())
        return m, n

    fmt_rand = lambda r: {k: f"{v['diff']:.1e}" if v else "n/a" for k, v in r.items()}
    fmt_tr = lambda r: {k: "n/a" if not v else f"{v['diff']:.1e}  moved {v['moved']:>2d}/{v['n']}  "
                        f"right {v['right'][0]}->{v['right'][1]}" for k, v in r.items()}

    puzzles = B.load("data/I-RAVEN-10000", list(B.POS_MAP), "val", 300, args.per_cfg)
    print(f"I-RAVEN: {len(puzzles)} val puzzles (300..), {args.per_cfg} per constellation; "
          f"random weights: {args.draws} draws, N(0, 1/fan-in)\n")
    cols = {}
    for label, kw in [("level 2 (default)", {}), ("copy links", {"copy": True}),
                      ("copy, content gates", {"copy": True, "gates": "content"}),
                      ("level 1 (--no-rows)", {"rows": False}), ("per-attribute weights", {"shared": False}),
                      ("components labelled", {"labels": ("components",)}),
                      ("attributes labelled", {"labels": ("attributes",)}),
                      ("per-component weights", {"shared_comps": False})]:
        m, n = build(**kw)
        cols[f"{label}, {n // 1000}k w"] = fmt_rand(run(m, puzzles, kw.get("copy", False), args.draws,
                                                        labels=kw.get("labels", ())))
    table(cols)
    if args.trained:
        print("\ntrained checkpoints (moved: puzzles whose chosen candidate changes; right: puzzles answered "
              "right before -> after the transformation)\n")
        m, _ = build(ckpt="experiments/facet_all_w32-64-128_hidden")
        table({"I-RAVEN, level 2": fmt_tr(run(m, puzzles, False, None))})

    path = "data/iravenx/center_single_shuffle_n_10_maxval_50.json"
    B.configure(n_cols=10, maxval=50)
    px = B.load_iravenx(path, 6300, args.n_iravenx)
    print(f"\nI-RAVEN-X 3x10, values 0..49: {len(px)} unseen puzzles (6300..; one object per panel, one "
          f"component); random weights\n")
    cols = {}
    for label, kw in [("named columns", {}), ("column roles", {"cols": "roles"}),
                      ("roles, content gates", {"cols": "roles", "copy": True, "gates": "content"}),
                      ("roles, panel gates", {"cols": "roles", "copy": True}), ("row fold", {"cols": "fold"}),
                      ("roles, per-attribute weights", {"cols": "roles", "shared": False})]:
        m, n = build(**kw)
        cols[f"{label}, {n // 1000}k w"] = fmt_rand(run(m, px, kw.get("copy", False), args.draws,
                                                        kw.get("cols") in ("roles", "fold"), kw.get("cols") == "fold"))
    table(cols)
    if not args.trained:
        return
    cols = {}
    for label, kw, ckpt in [("level 2", {}, "facet_iravenx_n10_m50"),
                            ("copy links", {"copy": True}, "facet_iravenx_n10_m50_copy"),
                            ("column roles", {"cols": "roles"}, "facet_iravenx_n10_m50_roles"),
                            ("roles, content gates", {"cols": "roles", "copy": True, "gates": "content"},
                             "facet_iravenx_n10_m50_roles_content")]:
        if os.path.exists(f"experiments/{ckpt}/template_best.bin"):
            m, n = build(ckpt=f"experiments/{ckpt}", **kw)
            cols[f"{label}, {n // 1000}k w"] = fmt_tr(run(m, px, kw.get("copy", False), None,
                                                          kw.get("cols") in ("roles", "fold"), kw.get("cols") == "fold"))
    print(f"\nI-RAVEN-X, trained checkpoints\n")
    table(cols)


if __name__ == "__main__":
    main()
