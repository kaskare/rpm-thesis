#!/usr/bin/env python3
"""The value-axis model (value_axis.py, notes §27) on I-RAVEN, all seven layouts. One sample per (puzzle, component),
as in the facet model, with the facet model's splits and metrics.

What changes from I-RAVEN-X:
  values       the same template over v0..v9. Sizes enter one up (v1..v6): the generator's Arithmetic on Size is
               s3 = s1 + s2 + 1 on the 0-based index, plain addition one up (checked on the data, like colours,
               c3 = c1 + c2, and counts, n3 = n1 + n2)
  grids        x(P, A, V) holds the share of the panel's objects with value V, so a panel is a distribution
  number       the object count is a fourth attribute on the same axis, with the same shared weights
  position     a set of occupied slots on a ring: the generator's Progression turns it by +-1 or +-2, cyclically
               in slot order (checked), so the slots are constants with ring(U, T, ok) facts, as off for values

The slot block answers each slot of a grid (own weights; a slot is occupied or not, so a sigmoid per slot):
  hs(P, T)     :- occ(P, U), ring(U, T, ok)      one weight per offset: a kernel along the ring
  prs(Rw, T)   :- how many panels of the row hold slot T
  ctxs         :- value-free descriptors: rc(Rw, ok), the overlap of neighbouring panels turned by k; tp(Rw, t), the
                  share of slots with each pattern t of occupied or empty across a complete row; tq(t), the same
                  for the question row's two panels. No weights; they come in as facts (slot_facts)
  os(T)        :- ctxs + hs(first and last question panel, T) + the counts at T, two tanh layers shared across T
The number head also reads ctxs and the two sets' union and difference sizes, uc(V) and dc(V): Arithmetic on Position
is set union or difference (checked), so the count of the answer follows from the sets, not from the count axis.
As with plus, these are given relations, not learned ones. --no-plus drops plus, uc and dc: no given arithmetic.

Evaluation: matched as in the facet model, the distributions turned into its thermometer codes; exact from the
model's own answer, the argmax per attribute and the n most likely slots for the argmax n. Exact per rule as
facet_rule_eval.py defines it.
    python3 core/pynl/value_axis_iraven.py --n 6000 --train-epochs 40 --save-dir experiments/valaxis_iraven --jvm-gb 150
"""
import argparse, collections, itertools, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_hub_grounding as B
import facet_rule_eval as E
import value_axis as VA

MAXVAL = 9
NV = MAXVAL + 1
ATTRS = ("type", "size", "color", "number")
SHIFT = {"type": 0, "size": 1, "color": 0}         # where value 0 of the attribute sits on the axis
PATTERNS = [t for t in itertools.product((0, 1), repeat=3) if any(t)]
QPATTERNS = [t for t in itertools.product((0, 1), repeat=2) if any(t)]
tname = lambda t: "t" + "".join(map(str, t))


def slots_of(cfg, k):
    return [p for p in sorted(B.POS_MAP[cfg].values()) if B.comp_of(cfg, p) == k]


def build_iraven_template(d, plus=True):
    from neuralogic.core import R, V, Transformation, Aggregation, Metadata, Combination
    m = VA.build_value_template(d, MAXVAL, plus=plus, attrs=ATTRS)
    S = R.hidden
    meta = lambda **kw: Metadata(**kw)
    tanh, ident = meta(transformation=Transformation.TANH), meta(transformation=Transformation.IDENTITY)
    avg, total = meta(aggregation=Aggregation.AVG), meta(aggregation=Aggregation.SUM)

    # the kernel along the ring, and the slots' counts per row
    for k in VA.OFFSETS:
        m += R.hs(V.P, V.T) <= (R.occ(V.P, V.U)[d, 1], S.ring(V.U, V.T, VA.oname(k)))
    m += (R.hs / 2) | ident
    m += (R.prs(V.Rw, V.T) <= (S.inrow(V.P, V.Rw), R.occ(V.P, V.T))) | total
    m += (R.prs / 2) | ident

    # which rule: value-free descriptors of the complete rows and of the question row
    for k in VA.OFFSETS:
        m += (R.ctxs <= (S.full(V.Rw), R.rc(V.Rw, VA.oname(k))[d, 1])) | avg
        m += R.ctxs <= R.rc("r3", VA.oname(k))[d, 1]
    for t in PATTERNS:
        m += (R.ctxs <= (S.full(V.Rw), R.tp(V.Rw, tname(t))[d, 1])) | avg
    for t in QPATTERNS:
        m += R.ctxs <= R.tq(tname(t))[d, 1]
    m += R.ctxs <= R.one[d, 1]
    m += (R.ctxs / 0) | tanh
    m += R.ctxs2 <= R.ctxs[d, d]
    m += R.ctxs2 <= R.one[d, 1]
    m += (R.ctxs2 / 0) | tanh

    # the answer at every slot: the same two layers at every T
    m += R.os(V.T) <= (S.slot(V.T), R.ctxs2[d, d])
    m += R.os(V.T) <= (S.qfirst(V.P), R.hs(V.P, V.T)[d, d])
    m += R.os(V.T) <= (S.qlast(V.P), R.hs(V.P, V.T)[d, d])
    m += (R.os(V.T) <= (S.full(V.Rw), R.prs(V.Rw, V.T)[d, 1])) | total
    m += R.os(V.T) <= R.prs("r3", V.T)[d, 1]
    m += (R.os / 1) | tanh
    m += R.os2(V.T) <= R.os(V.T)[d, d]
    m += R.os2(V.T) <= (S.slot(V.T), R.one[d, 1])
    m += (R.os2 / 1) | tanh
    m += R.so(V.T) <= R.os2(V.T)[1, d]
    m += R.so(V.T) <= (S.slot(V.T), R.one[1, 1])
    m += (R.so / 1) | meta(transformation=Transformation.SIGMOID)

    # the count under Arithmetic on Position
    m += R.ctx("number") <= R.ctxs2[d, d]
    if plus:
        m += R.o("number", V.V) <= R.uc(V.V)[d, 1]
        m += R.o("number", V.V) <= R.dc(V.V)[d, 1]

    for n in (4, 9):
        occ, out = getattr(R, f"occ{n}"), getattr(R, f"out{n}")
        m += (occ <= tuple(R.so(f"a{j}") for j in range(n))) | meta(combination=Combination.CONCAT,
                                                                    transformation=Transformation.IDENTITY)
        m += (occ / 0) | ident
        m += (out(V.G) <= (S.puzzle(V.G),) + tuple(R.dist(a) for a in ATTRS) + (occ,)) | meta(
            combination=Combination.CONCAT, transformation=Transformation.IDENTITY)
        m += (out / 1) | ident
    return m


def value_facts(panels, cfg, k, gid):
    """One component of one puzzle. puzzle_facts gets empty panels: it gives the row structure, and this model does
    not use its object atoms."""
    from neuralogic.core import R
    facts = B.puzzle_facts([[] for _ in panels], gid, col_roles=True, fold=True) + [R.one] + [R.attr(a) for a in ATTRS]
    facts.append(R.qfirst(f"p{B.ROWS['r3'][0]}"))
    for p in range(B.N_CTX):
        objs = B.comp_objs(panels[p], cfg, k)
        for a, s in SHIFT.items():
            for v, c in collections.Counter(o[a] + s for o in objs).items():
                facts.append(R.x(f"p{p}", a, f"v{v}")[c / len(objs)])
        facts.append(R.x(f"p{p}", "number", f"v{len(objs)}")[1.0])
    return facts


def slot_facts(panels, cfg, k):
    """The slot block's facts: slots, the ring, what each panel holds, and the descriptors, which carry no weights."""
    from neuralogic.core import R
    lay = slots_of(cfg, k)
    n, idx = len(lay), {p: j for j, p in enumerate(lay)}
    sets = [{idx[o["pos"]] for o in B.comp_objs(panels[p], cfg, k)} for p in range(B.N_CTX)]
    facts = [R.slot(f"a{j}") for j in range(n)]
    facts += [R.ring(f"a{j}", f"a{(j + o) % n}", VA.oname(o)) for j in range(n) for o in VA.OFFSETS]
    facts += [R.occ(f"p{p}", f"a{j}") for p in range(B.N_CTX) for j in sets[p]]
    for rw, ps in B.ROWS.items():
        pairs = list(zip(ps, ps[1:]))
        for o in VA.OFFSETS:
            ov = [len({(j + o) % n for j in sets[a]} & sets[b]) / max(len(sets[a]), len(sets[b])) for a, b in pairs]
            facts.append(R.rc(rw, VA.oname(o))[float(np.mean(ov))])
        if len(ps) == 3:
            for t in PATTERNS:
                c = sum(all((j in sets[p]) == bool(b) for p, b in zip(ps, t)) for j in range(n))
                facts.append(R.tp(rw, tname(t))[c / n])
    f, m = sets[B.ROWS["r3"][0]], sets[B.ROWS["r3"][1]]
    for t in QPATTERNS:
        c = sum((j in f) == bool(t[0]) and (j in m) == bool(t[1]) for j in range(n))
        facts.append(R.tq(tname(t))[c / n])
    return facts + [R.uc(f"v{len(f | m)}"), R.dc(f"v{len(f - m)}")]


def target(objs, cfg, k):
    t = np.zeros((len(ATTRS), NV))
    for a, s in SHIFT.items():
        for o in objs:
            t[ATTRS.index(a), o[a] + s] += 1 / len(objs)
    t[-1, len(objs)] = 1.0
    lay = slots_of(cfg, k)
    occ = [float(any(o["pos"] == p for o in objs)) for p in lay] if len(lay) > 1 else []
    return np.concatenate([t.flatten(), occ])


def to_codes(out, panels, cfg, k, hard=False):
    """The output -> the facet model's code space. hard=True: the model's own answer, for exact."""
    out = np.asarray(out, dtype=float).flatten()
    p = out[:len(ATTRS) * NV].reshape(len(ATTRS), NV)
    if hard:
        p = np.eye(NV)[p.argmax(1)]
    therm = [np.array([p[ATTRS.index(a), j + 1 + s:].sum() for j in range(B.THERM)]) for a, s in SHIFT.items()]
    therm.append(np.array([p[-1, j + 1:].sum() for j in range(B.THERM)]))
    lay, occ = slots_of(cfg, k), np.zeros(B.N_POS)
    if len(lay) == 1:
        occ[lay[0]] = 1
    else:
        s = out[len(ATTRS) * NV:]
        if hard:
            top = np.argsort(-s)[:int(p[-1].argmax())]
            s = np.isin(np.arange(len(lay)), top).astype(float)
        occ[lay] = s
    return np.concatenate(therm + [occ])


def score(outs, units, data, rules):
    """matched and exact per layout, and exact per rule: (part, rule) and (layout, rule), as in facet_rule_eval."""
    soft, hard = [dict() for _ in data], [dict() for _ in data]
    for (i, k), o in zip(units, outs):
        panels, _, cfg = data[i]
        soft[i][k], hard[i][k] = to_codes(o, panels, cfg, k), to_codes(o, panels, cfg, k, hard=True)
    _, _, _, ties, per = B.evaluate(soft, data, "therm")
    hits, by_layout = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
    exact = collections.defaultdict(lambda: [0, 0])
    for h, (panels, y, cfg), rr in zip(hard, data, rules):
        ok = True
        for k, o in h.items():
            r = rr[k]
            got, want = B.facet_decode(o, "therm"), B.decode(B.describe(B.comp_objs(panels[B.N_CTX + y], cfg, k)))
            for attr in B.FACET_ATTRS:
                for c in (hits[(attr, r[attr.capitalize()])], by_layout[(cfg, r[attr.capitalize()])]):
                    c[0] += got[attr] == want[attr]
                    c[1] += 1
            pa = next(p for p in E.NUMPOS if p in r)
            c = hits[(pa, r[pa])]
            c[0] += len(got["pos"]) == len(want["pos"]) if pa == "Number" else got["pos"] == want["pos"]
            c[1] += 1
            ok &= all(got[q] == want[q] for q in B.PARTS)
        exact[cfg][0] += ok
        exact[cfg][1] += 1
    n = len(data)
    return {"matched": sum(h for h, _ in per.values()) / n, "exact": sum(h for h, _ in exact.values()) / n, "ties": ties,
            "layouts": {c: {"matched": per[c][0] / per[c][1], "exact": exact[c][0] / exact[c][1]} for c in per},
            "by_part": {f"{p}:{r}": [int(a), int(b)] for (p, r), (a, b) in hits.items()},
            "by_layout": {f"{c}:{r}": [int(a), int(b)] for (c, r), (a, b) in by_layout.items()}}


def show(res):
    pct = lambda h: f"{100 * h[0] / max(1, h[1]):.1f} (of {h[1]})"
    tot = {r: [sum(res["by_part"].get(f"{a}:{r}", (0, 0))[i] for a in B.FACET_ATTRS) for i in (0, 1)] for r in E.RULES}
    print(f"  matched {100 * res['matched']:.1f}, exact {100 * res['exact']:.1f} (ties {res['ties']})")
    print("  exact, type/size/colour: " + " / ".join(f"{r} {pct(tot[r])}" for r in E.RULES if tot[r][1]))
    for part in B.FACET_ATTRS + E.NUMPOS:
        cells = [f"{r} {pct(res['by_part'][f'{part}:{r}'])}" for r in E.RULES if f"{part}:{r}" in res["by_part"]]
        if cells:
            print(f"    {part}: " + " / ".join(cells))
    for c, v in res["layouts"].items():
        cells = [f"{r} {pct(res['by_layout'][f'{c}:{r}'])}" for r in E.RULES if f"{c}:{r}" in res["by_layout"]]
        print(f"  {c}: matched {100 * v['matched']:.1f}, exact {100 * v['exact']:.1f}; " + " / ".join(cells), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--constellation", default="all", help="one, a comma list, or all")
    ap.add_argument("--root", default="data/I-RAVEN-10000")
    ap.add_argument("--n", type=int, default=6000, help="puzzles per layout")
    ap.add_argument("--n-sel", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--no-plus", action="store_true", help="without plus and the union and difference sizes")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--jvm-gb", type=int, default=16)
    args = ap.parse_args()

    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn import Adam, MSE

    model = build_iraven_template(args.dim, plus=not args.no_plus)
    model.build(Settings(optimizer=Adam(lr=args.lr), error_function=MSE(), iso_value_compression=False,
                         chain_pruning=False))
    n_w = sum(np.asarray(v).size for v in model.state_dict()["weights"].values())

    def samples_of(data, tag):
        units, samples = [], []
        for i, (pz, y, cfg) in enumerate(data):
            for k in B.comps(cfg):
                gid, lay = f"{tag}{i}{k}", slots_of(cfg, k)
                query = R.out if len(lay) == 1 else getattr(R, f"out{len(lay)}")
                facts = value_facts(pz, cfg, k, gid) + (slot_facts(pz, cfg, k) if len(lay) > 1 else [])
                samples.append(Sample(query(gid)[target(B.comp_objs(pz[B.N_CTX + y], cfg, k), cfg, k).tolist()], facts))
                units.append((i, k))
        return units, samples

    t = time.time()
    cfgs = list(B.POS_MAP) if args.constellation == "all" else args.constellation.split(",")
    spec = {"train": ("train", 0, args.n), "sel": ("val", 0, args.n_sel), "unseen": ("val", 300, args.n_val)}
    sets = {k: B.load(args.root, cfgs, *s) for k, s in spec.items()}
    rules = {k: [E.xml_rules(os.path.join(args.root, c, f"RAVEN_{s[1] + j}_{s[0]}.xml"))
                 for c in cfgs for j in range(s[2])] for k, s in spec.items()}
    units, built = {}, {}
    for k, s in sets.items():
        units[k], smp = samples_of(s, k)
        built[k] = list(model.build_dataset(Dataset(smp)))
        assert len(built[k]) == len(smp), f"{len(built[k])} built samples for {len(smp)} samples"
    print(f"value axis on I-RAVEN{', no plus' if args.no_plus else ''}: dim {args.dim}, {n_w} weights, {args.n} puzzles per layout of {len(cfgs)}, "
          f"{len(built['train'])} training samples; grounded in {time.time() - t:.0f}s", flush=True)

    best, hist = (-1, 0), []
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    t = time.time()
    for ep in range(args.eval_every, args.train_epochs + 1, args.eval_every):
        model.train(built["train"], epochs=args.eval_every)
        res = {k: score(model.test(built[k]), units[k], sets[k], rules[k]) for k in ("sel", "unseen")}
        hist.append({"epoch": ep, **res})
        if res["sel"]["matched"] > best[0]:
            best = (res["sel"]["matched"], ep)
            if args.save_dir:
                model.save(os.path.join(args.save_dir, "template_best.bin"))
                json.dump({"epoch": ep, "args": vars(args), "weights": n_w, **hist[-1]},
                          open(os.path.join(args.save_dir, "best.json"), "w"), indent=1)
        if args.save_dir:
            json.dump(hist, open(os.path.join(args.save_dir, "history.json"), "w"), indent=1)
        print(f"epoch {ep:>4}  sel matched {100 * res['sel']['matched']:5.1f}  |  unseen matched "
              f"{100 * res['unseen']['matched']:5.1f}  exact {100 * res['unseen']['exact']:5.1f}  "
              f"({time.time() - t:.0f}s)", flush=True)
    print(f"best on selection: epoch {best[1]}", flush=True)
    show(next(x for x in hist if x["epoch"] == best[1])["unseen"])


if __name__ == "__main__":
    main()
