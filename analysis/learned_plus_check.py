#!/usr/bin/env python3
"""What the value axis with plus used by learned rules (value_axis.py --learned-plus, notes §27) does on unseen
puzzles 6300.., for the Arithmetic attributes:
  1. no model: the values the row state st(P, A, V) reaches at the question row's last context panel (paths of
     + and - steps, and with --state-next also the offsets -K..K), and whether the answer is among them;
  2. the trained checkpoint: the state's vectors there, at the answer and at the other values reached
     (their length, and whether the answer's is the longest), and the answer's probability;
  3. one puzzle printed in full.
    python3 analysis/learned_plus_check.py experiments/valaxis_n3_m10_learnednext --n 300
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "pynl"))
import bench_hub_grounding as B
import value_axis as VA


def reach(values, M, K):
    """{value: the path that reaches it} for the state after the row's values: 'add' (every step adds, the plus
    form's answer), 'take' (every step takes away, the minus form's answer), 'mixed' (+ and - steps), 'next' (an
    offset step on the way; K > 0 adds the offsets -K..K)."""
    step = lambda S, w: {u + w for u in S if u + w <= M} | {u - w for u in S if u - w >= 0}
    add, take, p = values[0], values[0], {values[0]}
    s = set(p)
    for w in values[1:]:
        add, take = add + w, take - w
        p, s = step(p, w), step(s, w) | {u + k for u in s for k in range(-K, K + 1) if K and 0 <= u + k <= M}
    return {v: "add" if v == add else "take" if v == take else "mixed" if v in p else "next" for v in s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--jvm-gb", type=int, default=16)
    ap.add_argument("--random", action="store_true", help="control: the same template with untrained weights")
    args = ap.parse_args()
    a = json.load(open(os.path.join(args.run, "best.json")))["args"]
    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn import Adam, MSE

    path, M, d = a["iravenx"], VA.maxval_of(a["iravenx"]), a["dim"]
    first = B.load_iravenx(path, 0, 1)[0][0]
    B.configure(n_cols=(len(first) - 7) // 3, maxval=M)
    K = a["kernel"] if a.get("state_next") else 0
    model = VA.build_value_template(d, M, plus=True, learned_plus=True, state_next=a.get("state_next", False),
                                    offsets=tuple(range(-a["kernel"], a["kernel"] + 1)))
    model.build(Settings(optimizer=Adam(lr=1e-3), error_function=MSE(), iso_value_compression=False, chain_pruning=False))
    if not args.random:
        model.load(os.path.join(args.run, "template_best.bin"))

    data, rules = B.load_iravenx(path, 6300, args.n), VA.rules_of(path, 6300, args.n)
    ql = f"p{B.ROWS['r3'][-1]}"
    cases, samples = [], []
    for i, ((pz, y, _), r) in enumerate(zip(data, rules)):
        facts = VA.value_facts(pz, f"u{i}")
        for a_ in VA.ATTRS:
            if r[a_.capitalize()] != "Arithmetic":
                continue
            row1 = [pz[p][0][a_] for p in B.ROWS["r1"]]
            form = "plus" if row1[-1] == sum(row1[:-1]) else "minus"
            q = [pz[p][0][a_] for p in B.ROWS["r3"]]
            ans = pz[B.N_CTX + y][0][a_]
            got = reach(q, M, K)
            cases.append((i, a_, form, q, ans, got))
            samples += [Sample(R.st(ql, a_, f"v{v}")[[0.0] * d], facts) for v in sorted(got)]
        samples.append(Sample(R.out(f"u{i}")[VA.value_target([pz[B.N_CTX + y][0]], M).tolist()], facts))
    outs = iter(model.test(list(model.build_dataset(Dataset(samples)))))

    res, example, states = [], None, []
    for i in range(len(data)):
        mine = [c for c in cases if c[0] == i]
        vecs = {(c[1], v): np.asarray(next(outs), dtype=float) for c in mine for v in sorted(c[5])}
        dist = np.asarray(next(outs), dtype=float).reshape(len(VA.ATTRS), M + 1)
        for (_, a_, form, q, ans, got) in mine:
            states += [(got[v], vecs[(a_, v)] / max(1e-12, np.linalg.norm(vecs[(a_, v)]))) for v in got]
            n = {v: np.linalg.norm(vecs[(a_, v)]) for v in got}
            others = [n[v] for v in got if v != ans]
            p = dist[VA.ATTRS.index(a_)]
            res.append(dict(form=form, reached=len(got), ans_in=ans in got, n_ans=n.get(ans, 0.0),
                            n_other=max(others) if others else None, longest=ans in got and all(n[ans] > o for o in others),
                            p_ans=p[ans], right=int(p.argmax()) == ans))
            if example is None and form == "plus" and len(got) > 3:
                example = (i, a_, q, ans, got, n, p)
    # the direction of the state: one mean direction per path, and each state put with the nearest mean (cosine)
    labels = sorted({l for l, _ in states})
    mean = {l: np.mean([u for k, u in states if k == l], axis=0) for l in labels}
    mean = {l: m / np.linalg.norm(m) for l, m in mean.items()}
    near = lambda u: max(labels, key=lambda l: float(u @ mean[l]))

    print(f"{args.run}{' (untrained weights)' if args.random else ''}: {len(res)} Arithmetic attributes in {len(data)} unseen puzzles; state also along next: {bool(K)}")
    for form in ("plus", "minus", None):
        rs = [x for x in res if form is None or x["form"] == form]
        if not rs:
            continue
        oth = [x["n_other"] for x in rs if x["n_other"] is not None]
        print(f"  {form or 'all':5s} {len(rs):4d}: values reached {np.mean([x['reached'] for x in rs]):4.1f}, answer among "
              f"them {100 * np.mean([x['ans_in'] for x in rs]):5.1f}%; |state| at the answer {np.mean([x['n_ans'] for x in rs]):.2f}, "
              f"longest other {np.mean(oth) if oth else float('nan'):.2f}, answer's the longest {100 * np.mean([x['longest'] for x in rs]):5.1f}%; "
              f"P(answer) {np.mean([x['p_ans'] for x in rs]):.3f}, argmax right {100 * np.mean([x['right'] for x in rs]):5.1f}%")
    print("  the state's direction tells the path: states put with the nearest mean direction, per path (cosines between the means)")
    for l in labels:
        us = [u for k, u in states if k == l]
        print(f"    {l:5s} {len(us):5d} states, {100 * np.mean([near(u) == l for u in us]):5.1f}% nearest their own mean; "
              + ", ".join(f"cos to {k} {float(mean[l] @ mean[k]):+.2f}" for k in labels if k != l))
    if example:
        i, a_, q, ans, got, n, p = example
        print(f"  example: puzzle {6300 + i}, {a_}, question row {q} -> answer {ans}; state at the last context panel:")
        for v in sorted(got):
            print(f"    value {v:3d} ({got[v]:5s}): |state| {n[v]:.2f}   P {p[v]:.3f}{'   <- answer' if v == ans else ''}")


if __name__ == "__main__":
    main()
