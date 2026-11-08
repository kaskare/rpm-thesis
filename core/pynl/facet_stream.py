#!/usr/bin/env python3
"""Does the facet model grok Arithmetic? Column roles on 100,000 puzzles (data_prep/arith_big_data.py), with weight decay. Two things
bench_hub_grounding.py does not do:
  stream  a grounded puzzle takes up to 12 MB, so 100,000 do not fit a node: each chunk is grounded, trained on once and dropped.
  decay   native Adam has only coupled L2, so the weight matrices are multiplied here by 1 - lr * wd per update, as AdamW does,
          every --decay-every puzzles. Adam's moments sit on the weights and its step count on the trainer, so both survive.
  small   with --n at most --chunk the training set is grounded once and kept, so small sets (the grokking papers'
          setting: a set the model can memorise, many epochs) cost only the training.
  init    --init-scale multiplies every initial weight (Omnigrok, Liu et al. 2023: larger initial weights, then decay
          pulls the norm in; their MSE runs grok with it).
An update is one puzzle, as in every facet run. Every attribute is Arithmetic here, so "arith" is exact on type, size and colour
pooled; it picks the checkpoint (template_best.bin, best.json). A grokking run shows train arith high long before unseen.
template_last.bin and state.json allow --resume, which restarts Adam's moments.
    python3 -u core/pynl/facet_stream.py --save-dir experiments/facet_arith100k_wide
    python3 -u core/pynl/facet_stream.py --columns fold --widths 32,64,128 --n 1000 --epochs 400 --wd 1 \
        --decay-every 100 --eval-every 5000 --save-dir experiments/grok_fold_n1000_wd1
    python3 -u core/pynl/facet_rule_eval.py DIR --iravenx data/iravenx/center_single_shuffle_n_10_maxval_50.json   # the usual 1000 puzzles
"""
import argparse, json, os, time

import numpy as np

import bench_hub_grounding as B

DATA = "data/iravenx_arith/center_single_shuffle_n_10_maxval_50.json"
CFG = "center_single"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iravenx", default=DATA)
    ap.add_argument("--n", type=int, default=100000, help="training puzzles")
    ap.add_argument("--n-sel", type=int, default=300, help="selection puzzles, after the training ones")
    ap.add_argument("--n-val", type=int, default=1000, help="unseen puzzles, after the selection ones")
    ap.add_argument("--n-train-eval", type=int, default=300, help="training puzzles scored at every evaluation")
    ap.add_argument("--widths", default="64,144,288")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--columns", default="roles")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1, help="decoupled weight decay: weights shrink by lr * wd per update")
    ap.add_argument("--init-scale", type=float, default=1.0, help="multiply every initial weight by this")
    ap.add_argument("--chunk", type=int, default=2000, help="puzzles grounded at a time")
    ap.add_argument("--decay-every", type=int, default=500, help="puzzles between weight-decay steps")
    ap.add_argument("--eval-every", type=int, default=20000, help="puzzles between evaluations, a multiple of --chunk")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jvm-gb", type=int, default=64)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.chunk = min(args.chunk, args.n)
    assert args.eval_every % args.chunk == 0 and args.n % args.chunk == 0

    first = B.load_iravenx(args.iravenx, 0, 1)[0][0]
    B.configure(n_cols=(len(first) - 7) // 3, maxval=int(args.iravenx.split("maxval_")[1].split(".")[0]))
    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn import Adam, MSE

    train = B.load_iravenx(args.iravenx, 0, args.n)
    fixed = {"train": train[:args.n_train_eval], "sel": B.load_iravenx(args.iravenx, args.n, args.n_sel),
             "unseen": B.load_iravenx(args.iravenx, args.n + args.n_sel, args.n_val)}
    B._IRAVENX.clear()
    widths = [int(w) for w in args.widths.split(",")]
    model = B.build_facet_template(args.dim, args.layers, True, True, widths, cols=args.columns)
    model.build(Settings(optimizer=Adam(lr=args.lr), error_function=MSE(), iso_value_compression=False))
    k = B.comps(CFG)[0]

    def ground(puzzles, tag):
        samples = []
        for j, (panels, y, cfg) in enumerate(puzzles):
            gid = f"{tag}{j}"
            facts = (B.puzzle_facts(panels, gid, col_roles=True, fold=args.columns == "fold")
                     + B.facet_facts(panels, "therm", cfg) + [R.qcomp(gid, k)])
            samples.append(Sample(R.out(gid)[B.facet_target(B.comp_objs(panels[B.N_CTX + y], cfg, k)).tolist()], facts))
        return list(model.build_dataset(Dataset(samples)))

    def decay(factor):
        sd = model.state_dict()
        for i, v in sd["weights"].items():
            a = np.asarray(v, dtype=float)
            if a.ndim > 1:                                     # matrices only, as core/baselines/train_baseline.py
                sd["weights"][i] = (a * factor).tolist()
        model.load_state_dict(sd)

    def score(name):
        puzzles, built = grounded[name]
        mse, exact, matched, ties, _ = B.evaluate([{k: o} for o in model.test(built)], puzzles, "therm")
        n = len(puzzles)
        return {"mse": mse, "arith": sum(exact[a] for a in B.FACET_ATTRS) / (3 * n), "exact": exact["all"] / n, "matched": matched / n}

    if args.init_scale != 1:
        sd = model.state_dict()
        for i, v in sd["weights"].items():
            sd["weights"][i] = (np.asarray(v, dtype=float) * args.init_scale).tolist()
        model.load_state_dict(sd)
    grounded = {name: (p, ground(p, name)) for name, p in fixed.items()}
    kept = ground(train, "c") if args.n == args.chunk else None      # a small training set, grounded once
    os.makedirs(args.save_dir, exist_ok=True)
    state, history = os.path.join(args.save_dir, "state.json"), os.path.join(args.save_dir, "history.json")
    seen, best, hist = 0, -1.0, []
    if args.resume and os.path.exists(state):
        model.load(os.path.join(args.save_dir, "template_last.bin"))
        s = json.load(open(state))
        seen, best, hist = s["seen"], s["best"], json.load(open(history))
    print(f"{args.n} puzzles, columns {args.columns}, widths {args.widths}, lr {args.lr}, wd {args.wd}, "
          f"init x{args.init_scale}, from {seen}", flush=True)

    t = time.time()
    while seen < args.epochs * args.n:
        epoch, at = divmod(seen, args.n)
        order = np.random.default_rng([args.seed, epoch]).permutation(args.n)
        if kept is not None:
            built = [kept[i] for i in order]
        else:
            built = ground([train[i] for i in order[at:at + args.chunk]], "c")
        for s in range(0, len(built), args.decay_every):
            part = built[s:s + args.decay_every]
            model.train(part, epochs=1)
            if args.wd:
                decay((1 - args.lr * args.wd) ** len(part))
        seen += len(built)
        del built
        if seen % args.eval_every:
            continue
        r = {name: score(name) for name in fixed}
        all_w = np.concatenate([np.asarray(v, dtype=float).ravel() for v in model.state_dict()["weights"].values()])
        hist.append({"seen": seen, "epoch": seen / args.n, "rms": float(np.sqrt((all_w ** 2).mean())), **r})
        if r["sel"]["arith"] > best:
            best = r["sel"]["arith"]
            model.save(os.path.join(args.save_dir, "template_best.bin"))
            json.dump({"epoch": seen / args.n, "args": vars(args), **hist[-1]}, open(os.path.join(args.save_dir, "best.json"), "w"), indent=1)
        model.save(os.path.join(args.save_dir, "template_last.bin"))
        json.dump(hist, open(history, "w"))
        json.dump({"seen": seen, "best": best}, open(state, "w"))
        print(f"puzzles {seen:>9} epoch {seen / args.n:6.1f}  " + "  |  ".join(
            f"{name} arith {100 * v['arith']:5.1f} matched {100 * v['matched']:5.1f} mse {v['mse']:.4f}" for name, v in r.items())
            + f"  |  rms {hist[-1]['rms']:.3f} heap {B.jvm_used_mb():.0f} MB  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
