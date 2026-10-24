#!/usr/bin/env python3
"""Learn what the symbols mean from answer labels only, through the rules template (I-RAVEN-X 3x3).

Each panel shows every attribute as a SCRAMBLED symbol s = pi_A(v) (a fixed unknown permutation per
attribute). Perception is a learned table p(v | s) = softmax(W[A, s]); its 16 x 3 x 11 outputs are the
val(P, A, V) facts of pynl_soft's template, which scores the 8 candidates. The only loss is cross-entropy
on the answer; the template has no weights. Measured: answer accuracy and symbol accuracy
(argmax_v p(v | s) == true value), never trained on.

  python3 pynl/pynl_soft_learn.py --sem prob --n-train 2000 --epochs 3
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pynl_soft import load, build_template, rows_of, ATTRS, NV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem", default="prob", choices=["prob", "godel"])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.nn import Adam
    from neuralogic.nn.torch_function import NeuraLogic

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pi = np.stack([rng.permutation(NV) for _ in ATTRS])          # pi[a, v] = symbol shown for value v
    data = load(9000 + args.n_val)
    tr, va = data[:args.n_train], data[9000:9000 + args.n_val]
    sym = lambda vals: pi[np.arange(3)[None, :], vals]           # (16, 3) symbols

    keys = [(p, a, v) for p in range(16) for a in range(3) for v in range(NV)]
    structure = [R.row(r, *(f"p{p}" for p in idx)) for r, idx in rows_of().items()] + \
                [R.cand(f"k{k}") for k in range(8)]
    placeholders = [R.val(f"p{p}", ATTRS[a], f"v{v}")[0.1] for p, a, v in keys] + \
                   [R.cidx(f"k{k}")[0.0] for k in range(8)]

    def to_logic(probs, cidx):
        flat = probs.reshape(-1)
        return [R.val(f"p{p}", ATTRS[a], f"v{v}")[flat[i]] for i, (p, a, v) in enumerate(keys)] + \
               [R.cidx(f"k{k}")[cidx[k]] for k in range(8)]

    t0 = time.time()
    nl = NeuraLogic(build_template(args.sem, concat=True), placeholders + structure, R.out, to_logic,
                    settings=Settings(optimizer=Adam(lr=0.0)))
    with torch.no_grad():                                       # CONCAT order: candidate k scores 10 + k
        probe = nl(torch.zeros(16, 3, NV), torch.arange(10.0, 18.0)).flatten().tolist()
    order = [int(round(v)) - 10 for v in probe]
    assert sorted(order) == list(range(8)), probe
    inv = torch.tensor([order.index(k) for k in range(8)])
    print(f"grounded in {time.time() - t0:.0f}s, CONCAT order {order}", flush=True)

    W = torch.nn.Parameter(0.01 * torch.randn(3, NV, NV))
    tau = torch.nn.Parameter(torch.tensor(5.0))
    opt = torch.optim.Adam([W, tau], lr=args.lr)
    zeros = torch.zeros(8)

    def forward(vals):
        s = torch.as_tensor(sym(vals))
        probs = torch.softmax(W[torch.arange(3)[None, :], s], -1)     # (16, 3, NV)
        return tau * nl(probs, zeros).flatten()[inv]

    def symbol_acc():
        guess = W.argmax(-1).numpy()                                  # guess[a, s] = value
        true = np.argsort(pi, axis=1)                                 # true[a, s] = value with symbol s
        return (guess == true).mean(1)

    def evaluate():
        with torch.no_grad():
            ok = [float(forward(v).argmax() == y) for v, y in va]
        return float(np.mean(ok))

    hist = [{"step": 0, "val": evaluate(), "sym": symbol_acc().tolist()}]
    print(f"step 0  val {100 * hist[-1]['val']:.1f}%  symbols {np.round(100 * symbol_acc())}", flush=True)
    step = 0
    for ep in range(args.epochs):
        idx = rng.permutation(len(tr))
        for b in range(0, len(idx), args.batch):
            opt.zero_grad()
            loss = 0.0
            for i in idx[b:b + args.batch]:      # the wrapper holds ONE Java sample: backward before the next forward
                li = F.cross_entropy(forward(tr[i][0])[None], torch.tensor([tr[i][1]])) / args.batch
                li.backward()
                loss += float(li) * args.batch
            opt.step()
            step += 1
            if step % 25 == 0:
                hist.append({"step": step, "loss": float(loss) / args.batch, "val": evaluate(), "sym": symbol_acc().tolist()})
                print(f"ep {ep} step {step}  loss {hist[-1]['loss']:.3f}  val {100 * hist[-1]['val']:.1f}%  "
                      f"symbols {np.round(100 * symbol_acc())}  tau {float(tau):.1f}  {time.time() - t0:.0f}s", flush=True)
    if args.out:
        json.dump({"args": vars(args), "hist": hist, "W": W.tolist(), "pi": pi.tolist()}, open(args.out, "w"))


if __name__ == "__main__":
    main()
