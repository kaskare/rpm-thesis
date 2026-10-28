#!/usr/bin/env python3
"""Cost of a per-puzzle ("dynamic") graph in native PyNeuraLogic: grounding time, memory, forward
and training time per puzzle, for a minimal HUB template.

Nodes are the objects of the 8 context panels (no slot or position nodes, no padding); position is
an attribute like type, size and colour. Messages:
  pan_k(P)  :- obj(P, O), h_{k-1}(O)                 AVG over the panel's objects (+ count, layer 1)
  h_k(O)    :- obj(P, O), pan_k(P)                   own panel
  h_k(O)    :- obj(P, O), inrow(P, Rw), inrow(Q, Rw), P != Q, pan_k(Q)     the other panels of its row
  h_k(O)    :- samepos(O1, O), h_{k-1}(O1)           objects at the same position in the same row
  h_k(O)    :- h_{k-1}(O)                            residual, then NORM
The readout pools each row and decodes a DESCRIPTION of the missing panel (squared error):
type / size / colour histograms (mean of the objects' one-hots) and the 22-position occupancy.
The answer is generated, then matched: the candidate whose description is closest wins.

    python3 pynl/bench_hub_grounding.py --constellation distribute_nine --n 100 --dim 32
    python3 pynl/bench_hub_grounding.py --n 10 --overfit-epochs 2000      # can it memorise 10 puzzles?
"""
import argparse, json, os, resource, statistics, sys, time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pynl import parse, POS_MAP, N_POS, N_COL, N_SHP, N_SIZ

F = N_SHP + N_SIZ + N_COL + N_POS                       # 43-d fixed one-hot input per object
ROWS = {"r1": (0, 1, 2), "r2": (3, 4, 5), "r3": (6, 7)}
COLS = ("c1", "c2", "c3")
N_CTX = 8                                               # context panels; candidates follow


def encode(o):
    v = np.zeros(F)
    v[o["type"]] = 1
    v[N_SHP + o["size"]] = 1
    v[N_SHP + N_SIZ + o["color"]] = 1
    v[N_SHP + N_SIZ + N_COL + o["pos"]] = 1
    return v


def describe(panel):
    """What the model generates: attribute histograms (mean of one-hots) + position occupancy (0/1)."""
    if not panel:
        return np.zeros(F)
    v = np.mean([encode(o) for o in panel], axis=0)
    v[N_SHP + N_SIZ + N_COL:] = (v[N_SHP + N_SIZ + N_COL:] > 0).astype(float)
    return v


PARTS = {"type": slice(0, N_SHP), "size": slice(N_SHP, N_SHP + N_SIZ),
         "color": slice(N_SHP + N_SIZ, N_SHP + N_SIZ + N_COL), "pos": slice(N_SHP + N_SIZ + N_COL, F)}


def decode(v):
    """A description -> a discrete panel spec: occupied positions (> 0.5), and per attribute the
    number of objects with each value (histogram x object count, rounded). Rounding to whole objects
    is the tolerance: a generated description counts as exact if it is within half an object."""
    occ = v[PARTS["pos"]] > 0.5
    n = int(occ.sum())
    spec = {k: tuple(np.rint(v[s] * n).astype(int)) for k, s in PARTS.items() if k != "pos"}
    spec["pos"] = tuple(np.flatnonzero(occ))
    return spec


def evaluate(outs, data, values=None):
    """outs: one dict {component: output} per puzzle; component None = the whole panel (hub/ordered
    templates and the Python baselines, which output descriptions). With values set, outputs are facet
    codes: generation is checked through facet_to_description, matching in the code space.
    Generation: the decoded spec equals the answer's, per part and whole (every component).
    Matching: the candidate closest summed over components; a tie counts as wrong."""
    exact = {k: 0 for k in PARTS}
    exact["all"] = matched = ties = 0
    sq, per = [], defaultdict(lambda: [0, 0])
    for comp_out, (panels, y, cfg) in zip(outs, data):
        ok = {part: 1 for part in PARTS}
        dist = np.zeros(8)
        for k, out in comp_out.items():
            out = np.asarray(out, dtype=float).flatten()
            o = facet_to_description(out, values) if values else out
            t = describe(comp_objs(panels[N_CTX + y], cfg, k))
            sq.append(np.mean((o - t) ** 2))
            got, want = (facet_decode(out, values) if values else decode(o)), decode(t)
            for part in PARTS:
                ok[part] &= int(got[part] == want[part])
            if values:
                dist += [np.sum((out - facet_target(comp_objs(panels[N_CTX + c], cfg, k), values)) ** 2) for c in range(8)]
            else:
                dist += [np.sum((o - describe(comp_objs(panels[N_CTX + c], cfg, k))) ** 2) for c in range(8)]
        for part, v in ok.items():
            exact[part] += v
        exact["all"] += int(all(ok.values()))
        best = np.flatnonzero(dist <= dist.min() + 1e-9)
        hit = int(y in best and len(best) == 1)
        matched += hit
        ties += int(len(best) > 1)
        per[cfg][0] += hit
        per[cfg][1] += 1
    return float(np.mean(sq)), exact, matched, ties, dict(per)


def build_template(d, layers, derived_samepos=False, ordered=False):
    """ordered=False: the panels of a row are interchangeable (one weight for "another panel of my
    row"), and the readout pools whole rows. ordered=True: every message between panels has its own
    weight per (my column, its column), and the readout is a row-3 decoder,
        rel(Rw) :- row(Rw, P1, P2, P3), panF(P1)[A1], panF(P2)[A2], panF(P3)[A3]    rows 1 and 2
        ans     :- row3(P1, P2), panF(P1)[B1], panF(P2)[B2], ctx[B0]               ctx = AVG of rel(r1), rel(r2)
    i.e. the missing panel = g(the relation of the complete rows, the first two panels of row 3).
    NB: with PyNeuraLogic's default chain_pruning, atoms with a single incoming rule are removed together
    with their transformation, so the tanh of h0, pan2 (and in the ordered readout rel, ctx, ans) is not
    applied; the measured runs had that architecture."""
    from neuralogic.core import Model, R, V, Transformation, Aggregation, Metadata, Combination
    m = Model()
    crisp = Metadata(combination=Combination.MIN, transformation=Transformation.IDENTITY,
                     aggregation=Aggregation.MAX)
    avg = Metadata(aggregation=Aggregation.AVG)

    if derived_samepos:     # forward works, but training throws "scalar increment by vector"
        m += (R.samepos(V.O1, V.O2) <= (R.at(V.O1, V.S), R.at(V.O2, V.S), R.obj(V.P1, V.O1), R.obj(V.P2, V.O2),
                                        R.inrow(V.P1, V.Rw), R.inrow(V.P2, V.Rw), R.special.neq(V.P1, V.P2))) | crisp
        m += (R.samepos / 2) | Metadata(transformation=Transformation.IDENTITY)
    m += (R.cnt(V.P) <= R.obj(V.P, V.O)) | Metadata(transformation=Transformation.IDENTITY,
                                                      aggregation=Aggregation.SUM)
    m += (R.cnt / 1) | Metadata(transformation=Transformation.IDENTITY)

    m += (R.h0(V.O) <= R.feat(V.O)[d, F])
    m += (R.h0 / 1) | Metadata(transformation=Transformation.TANH)
    for k in range(1, layers + 1):
        prev, h, pan = f"h{k - 1}", f"h{k}", f"pan{k}"
        m += (R.get(pan)(V.P) <= (R.obj(V.P, V.O), R.get(prev)(V.O)[d, d])) | avg
        if k == 1:
            m += (R.get(pan)(V.P) <= R.cnt(V.P)[d, 1])
        m += (R.get(pan) / 1) | Metadata(transformation=Transformation.TANH)
        m += (R.get(h)(V.O) <= (R.obj(V.P, V.O), R.get(pan)(V.P)[d, d]))
        if ordered:
            for a in COLS:
                for b in COLS:
                    if a != b:
                        m += (R.get(h)(V.O) <= (R.obj(V.P, V.O), R.col(V.P, a), R.inrow(V.P, V.Rw),
                                                R.inrow(V.Q, V.Rw), R.col(V.Q, b), R.get(pan)(V.Q)[d, d])) | avg
                        m += (R.get(h)(V.O) <= (R.samepos(V.O1, V.O), R.obj(V.P, V.O), R.col(V.P, a),
                                                R.obj(V.Q, V.O1), R.col(V.Q, b), R.get(prev)(V.O1)[d, d])) | avg
        else:
            m += (R.get(h)(V.O) <= (R.obj(V.P, V.O), R.inrow(V.P, V.Rw), R.inrow(V.Q, V.Rw),
                                    R.special.neq(V.P, V.Q), R.get(pan)(V.Q)[d, d])) | avg
            m += (R.get(h)(V.O) <= (R.samepos(V.O1, V.O), R.get(prev)(V.O1)[d, d])) | avg
        m += (R.get(h)(V.O) <= R.get(prev)(V.O))
        m += (R.get(h) / 1) | Metadata(transformation=Transformation.NORM)

    last = f"h{layers}"
    if ordered:
        m += (R.panF(V.P) <= (R.obj(V.P, V.O), R.get(last)(V.O)[d, d])) | avg
        m += (R.panF(V.P) <= R.cnt(V.P)[d, 1])
        m += (R.panF / 1) | Metadata(transformation=Transformation.TANH)
        m += (R.rel(V.Rw) <= (R.row(V.Rw, V.P1, V.P2, V.P3),
                              R.panF(V.P1)[d, d], R.panF(V.P2)[d, d], R.panF(V.P3)[d, d]))
        m += (R.rel / 1) | Metadata(transformation=Transformation.TANH)
        m += (R.ctx <= (R.full(V.Rw), R.rel(V.Rw)[d, d])) | avg
        m += (R.ctx / 0) | Metadata(transformation=Transformation.TANH)
        m += (R.ans <= (R.row3(V.P1, V.P2), R.panF(V.P1)[d, d], R.panF(V.P2)[d, d], R.ctx[d, d]))
    else:
        m += (R.rowp(V.Rw) <= (R.inrow(V.P, V.Rw), R.obj(V.P, V.O), R.get(last)(V.O)[d, d])) | avg
        m += (R.rowp / 1) | Metadata(transformation=Transformation.TANH)
        m += (R.ans <= (R.rowp("r1")[d, d], R.rowp("r2")[d, d], R.rowp("r3")[d, d]))
    m += (R.ans / 0) | Metadata(transformation=Transformation.TANH)
    m += (R.out(V.G) <= (R.puzzle(V.G), R.ans[F, d]))          # unique query per puzzle
    m += (R.out / 1) | Metadata(transformation=Transformation.SIGMOID)
    return m


# ---- facet template: one node per (object, attribute), ordinal values, panel-attribute nodes ----
THERM = 10                                              # code length: values 0..9 and counts 1..9
FACET_ATTRS = ("type", "size", "color")
NUM_ATTRS = FACET_ATTRS + ("number",)
NVALS = {"type": N_SHP, "size": N_SIZ, "color": N_COL}
FACET_OUT = 4 * THERM + N_POS                           # type, size, colour, number, occupancy


def configure(n_cols=3, maxval=None):
    """Rows of n_cols panels (three rows: two complete, the question row one short) and, for I-RAVEN-X,
    attribute values 0..maxval-1 for type, size and colour alike, with thermometer codes of that length.
    The defaults are I-RAVEN's 3x3 matrices and value ranges."""
    global N_SHP, N_SIZ, N_COL, F, PARTS, ROWS, COLS, N_CTX, THERM, NVALS, FACET_OUT
    if maxval:
        N_SHP = N_SIZ = N_COL = maxval
        THERM = max(maxval, 10)                         # counts 1..9 also need room
    F = N_SHP + N_SIZ + N_COL + N_POS
    PARTS = {"type": slice(0, N_SHP), "size": slice(N_SHP, N_SHP + N_SIZ),
             "color": slice(N_SHP + N_SIZ, N_SHP + N_SIZ + N_COL), "pos": slice(N_SHP + N_SIZ + N_COL, F)}
    n = n_cols
    ROWS = {"r1": tuple(range(n)), "r2": tuple(range(n, 2 * n)), "r3": tuple(range(2 * n, 3 * n - 1))}
    COLS = tuple(f"c{j + 1}" for j in range(n))
    N_CTX = 3 * n - 1
    NVALS = {"type": N_SHP, "size": N_SIZ, "color": N_COL}
    FACET_OUT = 4 * THERM + N_POS
# RAVEN gives every COMPONENT of a two-part layout its own rules; the component of an object follows
# from its universal position. Single-component constellations have only k0.
COMPONENTS = {"left_center_single_right_center_single": {14: "k0", 15: "k1"},
              "up_center_single_down_center_single": {16: "k0", 17: "k1"},
              "in_center_single_out_center_single": {0: "k0", 9: "k1"},
              "in_distribute_four_out_center_single": {0: "k0", 18: "k1", 19: "k1", 20: "k1", 21: "k1"}}


def comp_of(cfg, pos):
    return COMPONENTS.get(cfg, {}).get(pos, "k0")


def comps(cfg):
    return sorted(set(COMPONENTS[cfg].values())) if cfg in COMPONENTS else ["k0"]


def comp_objs(panel, cfg, k):
    return panel if k is None else [o for o in panel if comp_of(cfg, o["pos"]) == k]


def code(v, values="therm"):
    """therm: value v -> the first v bits on (ordinal); onehot: bit v on (categorical)."""
    if values == "therm":
        return [1.0] * v + [0.0] * (THERM - v)
    return [1.0 if k == v else 0.0 for k in range(THERM)]


def facet_target(panel, values="therm"):
    """A panel (or one component of it) in the facet model's output space: per attribute the MEAN code
    of its objects (for thermometers the fraction of objects with value > k, for one-hots the value
    histogram; either determines the histogram exactly), the object count, and the position occupancy."""
    if not panel:
        return np.zeros(FACET_OUT)
    occ = np.zeros(N_POS)
    for o in panel:
        occ[o["pos"]] = 1
    return np.concatenate([np.mean([code(o[a], values) for o in panel], axis=0) for a in FACET_ATTRS]
                          + [code(len(panel), values), occ])


def facet_decode(out, values="therm"):
    """Facet output -> the discrete panel spec that decode() gives for a description. The object count
    n is the number of occupied positions; thermometer bit k holds the fraction of objects with value
    > k, so it is rounded to whole objects BEFORE differencing (differencing first, as
    facet_to_description does, needs very sharp outputs and decoded almost nothing exactly)."""
    occ = out[4 * THERM:] > 0.5
    n = int(occ.sum())
    spec = {"pos": tuple(np.flatnonzero(occ))}
    for i, a in enumerate(FACET_ATTRS):
        s = out[i * THERM:(i + 1) * THERM]
        if values == "therm":
            gt = np.minimum.accumulate(np.clip(np.rint(s * n), 0, n))      # objects with value > k
            counts = np.concatenate([[n], gt[:-1]]) - gt
        else:
            counts = np.rint(s * n)
        spec[a] = tuple(counts[:NVALS[a]].astype(int))
    return spec


def facet_to_description(out, values="therm"):
    """Facet output -> the 43-d description. Thermometer: P(value = j) = P(value > j-1) - P(value > j)."""
    desc = np.zeros(F)
    for i, a in enumerate(FACET_ATTRS):
        s = out[i * THERM:(i + 1) * THERM]
        if values == "therm":
            s = np.concatenate([[1.0], s[:-1]]) - s
        p = np.clip(s, 0, None)[:NVALS[a]]
        desc[PARTS[a]] = p / p.sum() if p.sum() > 0 else p
    desc[PARTS["pos"]] = out[4 * THERM:]
    return desc


def build_facet_template(d, layers, shared=True, rows=True, widths=None, hidden=True, copy=False, gates="panel"):
    """Objects are split into FACETS f(O, A), A in {type, size, color}; with shared=True every rule has
    the attribute as a variable, so one set of weights serves all attributes. Values are codes (code()).
    Everything above the objects is per COMPONENT C (ocomp(O, C)); components share all weights.
      f_k(O, A)     :- own panel-component pv_k(P, C, A); same-position objects in another column
                       (weights per column pair); residual; NORM
      vF(P, C, A)   :- the component's mean code pt(P, C, A) + AVG of its final facets
      rel(Rw, C, A) :- row(Rw, P1, P2, P3), vF(P1, C, A)[W1], vF(P2, C, A)[W2], vF(P3, C, A)[W3]
      ans(C, A)     :- row3(P1, P2), vF(P1, C, A)[B1], vF(P2, C, A)[B2], ctx(C, A)[B0]
    number is a fourth attribute (pt from the component's object count); positions are a separate
    channel with the same row structure on the component's occupancy. A sample asks for ONE component
    (qcomp(G, C)); its output is the sigmoid of the concatenated codes and occupancy. (With the default
    chain_pruning, single-rule atoms lose their tanh, as before.)
    widths = [w0, w1, ..., wL]: f0 has width w0 and facet layer k width wk (layers = L); everything
    above the facets has width wL. The position channel widens the same way: projection w0, row
    relation and context w1, answer wL. Where a layer is wider than the one below, its residual is a
    learned projection instead of the identity. Default: every width d.
    hidden=True declares the structural atoms (obj, ocomp, samepos, col, row, row3, full, qcomp) hidden:
    they ground the rule but add nothing to its sum. Visible, every one adds its value 1.0 to every
    component - the occupancy of a panel with objects at slots 1, 3, 4 arrived as [6, 7, 6, 7, 7, 6, ...]
    (+2 per object everywhere), which left the position channel seeing only the object count.
    Ablations: shared=False writes every attribute rule once per attribute, with its own weights;
    rows=False (no row prior, level 1) drops rel/ctx: the answer reads all eight panels, each through
    its own weight, so a panel's grid position is known only by which weight it gets.
    copy=True adds gated copy links: every context panel's code, as a +-COPY_SCALE fact pc(P, C, A), goes
    straight into the output logits, multiplied (ELPRODUCT) by a sigmoid gate computed from ans with
    its own weights per panel. A copied value skips the tanh layers, so it keeps full precision; the
    answer of Constant and Distribute-n is one of the context values.
    gates="content" (with copy=True): a gate reads the panel's own vF times ans, elementwise, with weights
    per ROLE (a complete row, the question row) instead of per panel. Naming a panel breaks the symmetry
    of rows 1 and 2 (notes §21); per-panel gates learned to copy Distribute-n by address, which fails once
    the rows are swapped."""
    from neuralogic.core import Model, R, V, Transformation, Aggregation, Metadata, Combination
    w = list(widths) if widths else [d] * (layers + 1)
    layers, W = len(w) - 1, w[-1]
    p0, p1, p2 = w[0], w[min(1, layers)], W                      # position channel widths
    m = Model()
    avg = Metadata(aggregation=Aggregation.AVG)
    tanh = Metadata(transformation=Transformation.TANH)
    ident = Metadata(transformation=Transformation.IDENTITY)
    attrs = lambda names: [V.A] if shared else list(names)
    S = R.hidden if hidden else R
    Ps = [getattr(V, f"P{i + 1}") for i in range(len(COLS))]       # the panels of a row, in order                                # structural atoms
    mine = (S.obj(V.P, V.O), S.ocomp(V.O, V.C))                   # object O of panel P, component C

    for A in attrs(FACET_ATTRS):
        m += (R.f0(V.O, A) <= R.fv(V.O, A)[w[0], THERM])
    m += (R.f0 / 2) | tanh
    for k in range(1, layers + 1):
        prev, f, pv = f"f{k - 1}", f"f{k}", f"pv{k}"
        for A in attrs(FACET_ATTRS):
            m += (R.get(pv)(V.P, V.C, A) <= mine + (R.get(prev)(V.O, A)[w[k], w[k - 1]],)) | avg
            m += (R.get(f)(V.O, A) <= mine + (R.get(pv)(V.P, V.C, A)[w[k], w[k]],))
            for a in COLS:
                for b in COLS:
                    if a != b:
                        m += (R.get(f)(V.O, A) <= (S.samepos(V.O1, V.O), S.obj(V.P, V.O), S.col(V.P, a),
                                                   S.obj(V.Q, V.O1), S.col(V.Q, b),
                                                   R.get(prev)(V.O1, A)[w[k], w[k - 1]])) | avg
        m += (R.get(pv) / 3) | tanh
        if w[k] == w[k - 1]:
            m += (R.get(f)(V.O, V.A) <= R.get(prev)(V.O, V.A))
        else:
            for A in attrs(FACET_ATTRS):
                m += (R.get(f)(V.O, A) <= R.get(prev)(V.O, A)[w[k], w[k - 1]])
        m += (R.get(f) / 2) | Metadata(transformation=Transformation.NORM)

    m += (R.pt(V.P, V.C, V.A) <= mine + (R.fv(V.O, V.A),)) | Metadata(aggregation=Aggregation.AVG,
                                                                     transformation=Transformation.IDENTITY)
    m += (R.pt(V.P, V.C, "number") <= R.nm(V.P, V.C)) | ident
    m += (R.pt / 3) | ident
    for A in attrs(NUM_ATTRS):
        m += (R.vF(V.P, V.C, A) <= R.pt(V.P, V.C, A)[W, THERM])
    for A in attrs(FACET_ATTRS):
        m += (R.vF(V.P, V.C, A) <= mine + (R.get(f"f{layers}")(V.O, A)[W, W],)) | avg
    m += (R.vF / 3) | tanh
    # rel and ans are each written as TWO rules (the same sum, split): chain_pruning removes the
    # transformation of single-rule atoms, and without these two tanh layers everything after vF is
    # one linear map, which cannot express "the relation of rows 1-2 decides how (3,1), (3,2) combine"
    for A in attrs(NUM_ATTRS):
        if rows:
            m += (R.rel(V.Rw, V.C, A) <= (S.row(V.Rw, *Ps),) + tuple(R.vF(P, V.C, A)[W, W] for P in Ps[:-1]))
            m += (R.rel(V.Rw, V.C, A) <= (S.row(V.Rw, *Ps), R.vF(Ps[-1], V.C, A)[W, W]))
            m += (R.ctx(V.C, A) <= (S.full(V.Rw), R.rel(V.Rw, V.C, A)[W, W])) | avg
            m += (R.ans(V.C, A) <= (S.row3(*Ps[:-1]),) + tuple(R.vF(P, V.C, A)[W, W] for P in Ps[:-1]))
            m += (R.ans(V.C, A) <= R.ctx(V.C, A)[W, W])
        else:
            for p in range(N_CTX):
                m += (R.hid(V.C, A) <= R.vF(f"p{p}", V.C, A)[W, W])
            m += (R.ans(V.C, A) <= R.hid(V.C, A)[W, W])
            m += (R.ans(V.C, A) <= R.vF(f"p{N_CTX - 1}", V.C, A)[W, W])
        # one decoder per weight set; the weighted puzzle(G) fact is a learned bias per code bit
        # ("value > k" needs its own threshold for every k; the template has no other bias)
        m += (R.oa(V.G, A) <= (R.puzzle(V.G)[THERM, 1], S.qcomp(V.G, V.C), R.ans(V.C, A)[THERM, W]))
        if copy:
            # two rules per gate so chain pruning keeps its sigmoid; the hidden pc atom only binds A. The
            # gate bias starts at GATE_BIAS0 (gates ~0.0025): open at 0.5, the 3n-1 copies swamped the logits
            # and memorising 200 3x10 puzzles went from 99% to 60% exact after 150 epochs
            if gates == "panel":
                for p in range(N_CTX):
                    m += (R.gate(f"p{p}", V.C, A) <= R.ans(V.C, A)[THERM, W])
                    m += (R.gate(f"p{p}", V.C, A) <= (R.puzzle(V.G)[[GATE_BIAS0] * THERM], R.hidden.qcomp(V.G, V.C),
                                                       R.hidden.pc(f"p{p}", V.C, A)))
            else:               # the role atoms are always hidden: in an ELPRODUCT a visible 1.0 would join the product
                for role in ((R.hidden.inrow(V.P, V.Rw), R.hidden.full(V.Rw)), (R.hidden.inrow(V.P, "r3"),)):
                    m += (R.gate(V.P, V.C, A) <= (R.ans(V.C, A)[THERM, W], R.vF(V.P, V.C, A)[THERM, W]) + role) | Metadata(
                        combination=Combination.ELPRODUCT)
                    m += (R.gate(V.P, V.C, A) <= (R.puzzle(V.G)[[GATE_BIAS0] * THERM], R.hidden.qcomp(V.G, V.C),
                                                  R.hidden.pc(V.P, V.C, A)) + role)
            m += (R.copy(V.C, A) <= (R.gate(V.P, V.C, A), R.pc(V.P, V.C, A))) | Metadata(
                combination=Combination.ELPRODUCT, aggregation=Aggregation.SUM)
            m += (R.oa(V.G, A) <= (S.qcomp(V.G, V.C), R.copy(V.C, A)))
    if rows:
        m += (R.rel / 3) | tanh
        m += (R.ctx / 2) | tanh
    else:
        m += (R.hid / 2) | tanh
    m += (R.ans / 2) | tanh
    m += (R.oa / 2) | ident
    if copy:
        m += (R.gate / 3) | Metadata(transformation=Transformation.SIGMOID)
        m += (R.copy / 2) | ident

    m += (R.occ(V.P, V.C) <= mine + (R.pos(V.O),)) | Metadata(aggregation=Aggregation.SUM,
                                                               transformation=Transformation.IDENTITY)
    m += (R.occ / 2) | ident
    m += (R.vP(V.P, V.C) <= R.occ(V.P, V.C)[p0, N_POS])
    m += (R.vP / 2) | tanh
    if rows:
        m += (R.relp(V.Rw, V.C) <= (S.row(V.Rw, *Ps),) + tuple(R.vP(P, V.C)[p1, p0] for P in Ps[:-1]))
        m += (R.relp(V.Rw, V.C) <= (S.row(V.Rw, *Ps), R.vP(Ps[-1], V.C)[p1, p0]))
        m += (R.relp / 2) | tanh
        m += (R.ctxp(V.C) <= (S.full(V.Rw), R.relp(V.Rw, V.C)[p1, p1])) | avg
        m += (R.ctxp / 1) | tanh
        m += (R.ansp(V.C) <= (S.row3(*Ps[:-1]),) + tuple(R.vP(P, V.C)[p2, p0] for P in Ps[:-1]))
        m += (R.ansp(V.C) <= R.ctxp(V.C)[p2, p1])
    else:
        for p in range(N_CTX):
            m += (R.hidp(V.C) <= R.vP(f"p{p}", V.C)[p1, p0])
        m += (R.hidp / 1) | tanh
        m += (R.ansp(V.C) <= R.hidp(V.C)[p2, p1])
        m += (R.ansp(V.C) <= R.vP(f"p{N_CTX - 1}", V.C)[p2, p0])
    m += (R.ansp / 1) | tanh
    m += (R.op(V.G) <= (R.puzzle(V.G)[N_POS, 1], S.qcomp(V.G, V.C), R.ansp(V.C)[N_POS, p2]))
    m += (R.op / 1) | ident

    m += (R.out(V.G) <= (R.oa(V.G, "type"), R.oa(V.G, "size"), R.oa(V.G, "color"), R.oa(V.G, "number"),
                         R.op(V.G))) | Metadata(combination=Combination.CONCAT, transformation=Transformation.IDENTITY)
    m += (R.out / 1) | Metadata(transformation=Transformation.SIGMOID)
    return m


COPY_SCALE = 4.0                                        # copied code bits enter the output logits as +-4
GATE_BIAS0 = -6.0                                       # initial bias of the copy gates


def facet_facts(panels, values, cfg, copy=False, source=None):
    """source: the puzzle the copied codes come from (default: this one; the shuffled control passes another)."""
    from neuralogic.core import R
    facts = []
    src = panels if source is None else source
    for p in range(N_CTX):
        for k in comps(cfg):
            objs = comp_objs(panels[p], cfg, k)
            if objs:
                facts.append(R.nm(f"p{p}", k)[code(len(objs), values)])
            if copy:                             # the component's codes as +-COPY_SCALE, for the copy links
                t = facet_target(comp_objs(src[p], cfg, k), values)
                facts += [R.pc(f"p{p}", k, a)[(COPY_SCALE * (2 * t[i * THERM:(i + 1) * THERM] - 1)).tolist()]
                          for i, a in enumerate(NUM_ATTRS)]
        for i, o in enumerate(panels[p]):
            oid = f"p{p}o{i}"
            facts.append(R.ocomp(oid, comp_of(cfg, o["pos"])))
            facts += [R.fv(oid, a)[code(o[a], values)] for a in FACET_ATTRS]
            facts.append(R.pos(oid)[[1.0 if s == o["pos"] else 0.0 for s in range(N_POS)]])
    return facts


def puzzle_facts(panels, gid, derived_samepos=False, with_feat=True):
    from neuralogic.core import R
    name = lambda ps: [f"p{p}" for p in ps]
    facts = ([R.puzzle(gid), R.row("r1", *name(ROWS["r1"])), R.row("r2", *name(ROWS["r2"])), R.row3(*name(ROWS["r3"])),
              R.full("r1"), R.full("r2")] + [R.col(f"p{p}", COLS[p % len(COLS)]) for p in range(N_CTX)])
    for rw, ps in ROWS.items():
        facts += [R.inrow(f"p{p}", rw) for p in ps]
        if not derived_samepos:                  # same position, same row, different panel
            for a in ps:
                for b in ps:
                    if a != b:
                        facts += [R.samepos(f"p{a}o{i}", f"p{b}o{j}") for i, oa in enumerate(panels[a])
                                  for j, ob in enumerate(panels[b]) if oa["pos"] == ob["pos"]]
    for p in range(N_CTX):
        for i, o in enumerate(panels[p]):
            oid = f"p{p}o{i}"
            facts += [R.obj(f"p{p}", oid), R.at(oid, f"s{o['pos']}")]
            if with_feat:                        # the hub/ordered input; the facet template has its own facts
                facts.append(R.feat(oid)[encode(o).tolist()])
    return facts


def edge_counts(panels):
    n = [len(panels[p]) for p in range(N_CTX)]
    same = 0
    for ps in ROWS.values():
        for a in ps:
            pa = {o["pos"] for o in panels[a]}
            same += sum(1 for b in ps if b != a for o in panels[b] if o["pos"] in pa)
    hub = sum(n) + sum(n[a] * (len(ps) - 1) for ps in ROWS.values() for a in ps) + sum(n)
    return sum(n), hub + same


def load(root, cfgs, split, start, n):
    """n puzzles per constellation, as (panels, answer index, constellation)."""
    data = []
    for cfg in cfgs:
        for i in range(start, start + n):
            stem = os.path.join(root, cfg, f"RAVEN_{i}_{split}")
            data.append((parse(stem + ".xml", cfg), int(np.load(stem + ".npz")["target"]), cfg))
    return data


def next_same_cfg(s):
    """index -> the next puzzle of the same constellation in s, cyclically (the shuffled-source control)."""
    by = defaultdict(list)
    for j, d in enumerate(s):
        by[d[2]].append(j)
    return {a: b for js in by.values() for a, b in zip(js, js[1:] + js[:1])}


def load_iravenx(path, start, n):
    """I-RAVEN-X (IBM's iravenx_task.py): center_single, an n x n matrix of which, as in the papers, the last
    three rows are shown; puzzles 0-5999 train, 6000-7999 val, 8000-9999 test."""
    import gzip
    if path not in _IRAVENX:
        with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
            _IRAVENX[path] = json.load(f)
    d = _IRAVENX[path]
    data = []
    for i in range(start, start + n):
        s = d[str(i)]
        size = int(round((len(s["rpm"]) - 7) ** 0.5))
        shown = s["rpm"][(size - 3) * size: size * size - 1] + s["rpm"][size * size - 1:]
        panels = [[{"pos": 0, "type": int(e["Type"]), "size": int(e["Size"]), "color": int(e["Color"]),
                    "cx": 0.5, "cy": 0.5} for e in panel] for panel in shown]
        data.append((panels, int(s["target"]), "center_single"))
    return data


_IRAVENX = {}


def fmt(res, n=None):
    mse, exact, matched = res[:3]
    n = n or sum(total for _, total in res[4].values())
    return f"mse {mse:.4f}  exact {100 * exact['all'] / n:5.1f}%  matched {100 * matched / n:5.1f}%"


def per_cfg(res):
    return "  ".join(f"{cfg[:12]} {100 * h / t:.1f}" for cfg, (h, t) in res[4].items())


def jvm_used_mb():
    import jpype
    rt = jpype.JClass("java.lang.Runtime").getRuntime()
    for _ in range(3):
        rt.gc()
    return (rt.totalMemory() - rt.freeMemory()) / 2 ** 20


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--constellation", default="distribute_nine", help="one, a comma list, or all")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=100, help="puzzles per constellation")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--widths", default=None,
                    help="facet: widths from input to top, e.g. 32,64,128 (overrides --dim/--layers)")
    ap.add_argument("--repeats", type=int, default=3, help="timed passes for forward and training")
    ap.add_argument("--jvm-gb", type=int, default=12)
    ap.add_argument("--root", default="data/I-RAVEN-10000")
    ap.add_argument("--derived-samepos", action="store_true",
                    help="derive samepos by a join in the template instead of Python facts (no training)")
    ap.add_argument("--overfit-epochs", type=int, default=0,
                    help="instead of timing, train this many epochs on the --n puzzles and report memorisation")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--train-epochs", type=int, default=0,
                    help="instead of timing, train on --n train puzzles and evaluate on unseen val puzzles")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--n-sel", type=int, default=300, help="val puzzles 0.. used to pick the epoch")
    ap.add_argument("--n-val", type=int, default=500, help="unseen val puzzles 300.. (reported only)")
    ap.add_argument("--template", choices=["hub", "ordered", "facet"], default="hub",
                    help="hub: panels of a row interchangeable; ordered: per-column weights + row-3 decoder; "
                         "facet: one node per (object, attribute), ordinal values, per-attribute row relation")
    ap.add_argument("--values", choices=["therm", "onehot"], default="therm", help="facet: value code")
    ap.add_argument("--per-attr-weights", action="store_true", help="facet: own weights for every attribute")
    ap.add_argument("--iravenx", default=None,
                    help="an I-RAVEN-X .json(.gz) from iravenx_task.py instead of I-RAVEN; rows of n panels, values 0..maxval-1")
    ap.add_argument("--save-dir", default=None, help="train/val: save the best-on-selection weights + history here")
    ap.add_argument("--n-train-eval", type=int, default=0,
                    help="train/val: report training accuracy on only this many puzzles per constellation")
    ap.add_argument("--visible-selectors", action="store_true",
                    help="facet: structural atoms visible (each adds 1.0 to every rule sum), as in runs before 2026-10-25")
    ap.add_argument("--copy-links", action="store_true",
                    help="facet: gated copy links from every context panel's code to the output (copy=True)")
    ap.add_argument("--copy-gates", choices=["panel", "content"], default="panel",
                    help="with --copy-links: a gate per panel from the answer (panel), or per row role from the "
                         "panel's own code times the answer (content; rows 1 and 2 stay interchangeable)")
    ap.add_argument("--copy-shuffled", action="store_true",
                    help="with --copy-links, a control: the copied codes come from the next puzzle of the same "
                         "constellation, so the links carry the same weights but no information")
    ap.add_argument("--no-rows", action="store_true",
                    help="facet: no row relation (level 1); the answer reads all eight panels directly")
    args = ap.parse_args()
    cfgs = list(POS_MAP) if args.constellation == "all" else args.constellation.split(",")
    if args.iravenx:
        import re
        first = load_iravenx(args.iravenx, 0, 1)[0][0]
        m = re.search(r"maxval_(\d+)", args.iravenx)
        configure(n_cols=(len(first) - 7) // 3, maxval=int(m.group(1)) if m else None)
        cfgs = ["center_single"]
    facet = args.template == "facet"
    assert facet or all(cfg not in COMPONENTS for cfg in cfgs), "two-component layouts need --template facet"

    import neuralogic
    neuralogic.set_max_memory_size(args.jvm_gb)
    from neuralogic.logging import clear_handlers
    clear_handlers()
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    from neuralogic.nn import Adam, MSE

    t = time.time()
    if args.iravenx:            # same protocol: val 6000.. pick the epoch, val 6300.. are unseen, test untouched
        assert args.n_sel <= 300 and args.n_val <= 1700
        sets = ({"train": load_iravenx(args.iravenx, 0, args.n), "sel": load_iravenx(args.iravenx, 6000, args.n_sel),
                 "unseen": load_iravenx(args.iravenx, 6300, args.n_val)} if args.train_epochs
                else {"main": load_iravenx(args.iravenx, 0, args.n)})
    elif args.train_epochs:     # val 0..n_sel-1 pick the epoch; val 300.. are unseen
        assert args.n_sel <= 300, "selection puzzles must stay below the unseen range (300+)"
        sets = {"train": load(args.root, cfgs, "train", 0, args.n),
                "sel": load(args.root, cfgs, "val", 0, args.n_sel),
                "unseen": load(args.root, cfgs, "val", 300, args.n_val)}
    else:
        sets = {"main": load(args.root, cfgs, args.split, 0, args.n)}
    data = [d for s in sets.values() for d in s]
    t_parse = time.time() - t

    values = args.values if facet else None
    widths = [int(x) for x in args.widths.split(',')] if args.widths else None
    model = (build_facet_template(args.dim, args.layers, not args.per_attr_weights, not args.no_rows, widths,
                                  hidden=not args.visible_selectors, copy=args.copy_links, gates=args.copy_gates) if facet
             else build_template(args.dim, args.layers, args.derived_samepos, args.template == "ordered"))
    model.build(Settings(optimizer=Adam(lr=args.lr), error_function=MSE(), iso_value_compression=False))
    heap0 = jvm_used_mb()

    # one sample per (puzzle, component): the facet model answers each component of the missing panel
    # separately; other templates answer the whole panel (component None)
    t = time.time()
    units = [(name, i, k) for name, s in sets.items() for i, (_, _, cfg) in enumerate(s)
             for k in (comps(cfg) if facet else [None])]
    samples = []
    nxt = {name: next_same_cfg(s) for name, s in sets.items()} if args.copy_shuffled else None
    for name, i, k in units:
        panels, y, cfg = sets[name][i]
        gid = f"{name}{i}{k or ''}"
        facts = puzzle_facts(panels, gid, args.derived_samepos, with_feat=not facet)
        if facet:
            target = facet_target(comp_objs(panels[N_CTX + y], cfg, k), values)
            src = sets[name][nxt[name][i]][0] if nxt else None
            facts += facet_facts(panels, values, cfg, args.copy_links, src) + [R.qcomp(gid, k)]
        else:
            target = describe(panels[N_CTX + y])
        samples.append(Sample(R.out(gid)[target.tolist()], facts))
    t_facts = time.time() - t

    t = time.time()
    built = list(model.build_dataset(Dataset(samples)))
    t_ground = time.time() - t
    assert len(built) == len(samples), f"{len(built)} built samples for {len(samples)} samples"
    heap1 = jvm_used_mb()

    groups = {name: [(u, b) for u, b in zip(units, built) if u[0] == name] for name in sets}

    def run_eval(name):
        # --n-train-eval: training accuracy on the first puzzles of every constellation only (sets are
        # loaded constellation by constellation, args.n each)
        sub = name == "train" and args.n_train_eval
        part = [(u, b) for u, b in groups[name] if not sub or u[1] % args.n < args.n_train_eval]
        outs = model.test([b for _, b in part])
        idx = sorted({u[1] for u, _ in part})
        row = {i: j for j, i in enumerate(idx)}
        per_puzzle = [dict() for _ in idx]
        for ((_, i, k), _), o in zip(part, outs):
            per_puzzle[row[i]][k] = o
        return evaluate(per_puzzle, [sets[name][i] for i in idx], values)

    what = (f"I-RAVEN-X {os.path.basename(args.iravenx)}, 3 x {len(COLS)}, values 0..{N_SHP - 1}" if args.iravenx
            else f"{'all' if len(cfgs) == len(POS_MAP) else ','.join(cfgs)}")
    if args.train_epochs:
        b_tr = [b for _, b in groups["train"]]
        print(f"train/val: {args.n} train, {args.n_sel} selection (val 0..{args.n_sel - 1}), "
              f"{args.n_val} unseen (val 300..{299 + args.n_val}) puzzles per constellation ({what}); "
              f"{len(b_tr)} training samples; template {args.template}"
              + (f" (values {args.values}, {'per-attribute' if args.per_attr_weights else 'shared'} weights, "
                 f"{'no row relation' if args.no_rows else 'row relation'}"
                 + (f", copy links ({args.copy_gates} gates{', shuffled sources' if args.copy_shuffled else ''})"
                    if args.copy_links else "") + ")" if facet else "")
              + (f", widths {args.widths}" if args.widths else f", dim {args.dim}, layers {args.layers}")
              + f", lr {args.lr}; "
              f"grounded in {t_ground:.0f}s", flush=True)
        un = sets["unseen"]
        for name, pick in [("ceiling: true description", lambda p, y: describe(p[N_CTX + y])),
                           ("baseline: copy the last panel", lambda p, y: describe(p[N_CTX - 1])),
                           ("baseline: copy the one before", lambda p, y: describe(p[N_CTX - 2]))]:
            print(f"{name:28s} unseen {fmt(evaluate([{None: pick(p, y)} for p, y, _ in un], un), len(un))}")
        best = (-1, 0, None)
        hist = []
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
        t = time.time()
        for ep in range(args.eval_every, args.train_epochs + 1, args.eval_every):
            model.train(b_tr, epochs=args.eval_every)
            r = {name: run_eval(name) for name in sets}
            hist.append({"epoch": ep, **{name: {"mse": v[0], "exact": v[1]["all"], "matched": v[2],
                                                "per_constellation": v[4]} for name, v in r.items()}})
            if r["sel"][2] > best[0]:
                best = (r["sel"][2], ep, r)
                if args.save_dir:            # weights only (Model.save); rebuild with the same args
                    model.save(os.path.join(args.save_dir, "template_best.bin"))
                    json.dump({"epoch": ep, "args": vars(args), **hist[-1]},
                              open(os.path.join(args.save_dir, "best.json"), "w"), indent=1)
            if args.save_dir:
                json.dump(hist, open(os.path.join(args.save_dir, "history.json"), "w"), indent=1)
            print(f"epoch {ep:>4}  " + "  |  ".join(f"{name} {fmt(r[name])}" for name in sets)
                  + f"  ({time.time() - t:.0f}s)", flush=True)
        n_un = len(un)
        for label, ep, rr in [("final", args.train_epochs, r), ("best on selection", best[1], best[2])]:
            mse, exact, matched, ties, _ = rr["unseen"]
            print(f"{label} (epoch {ep}): unseen exact {100 * exact['all'] / n_un:.1f}% "
                  f"(type {100 * exact['type'] / n_un:.0f}, size {100 * exact['size'] / n_un:.0f}, "
                  f"color {100 * exact['color'] / n_un:.0f}, pos {100 * exact['pos'] / n_un:.0f}), "
                  f"matched {100 * matched / n_un:.1f}%, ties {ties}")
            if len(cfgs) > 1:
                print(f"    per constellation: {per_cfg(rr['unseen'])}")
        return

    if args.overfit_epochs:
        n = len(sets["main"])
        b_all = [b for _, b in groups["main"]]
        def report(ep):
            mse, exact, matched, ties, _ = run_eval("main")
            print(f"epoch {ep:>5}  mse {mse:.5f}  exact " + " ".join(f"{k} {v}/{n}" for k, v in exact.items())
                  + f"  matched {matched}/{n}  ties {ties}", flush=True)
            return exact["all"], matched
        print(f"overfit: {n} puzzles ({what}), {len(b_all)} samples, dim {args.dim}, layers {args.layers}, lr {args.lr}")
        _, _, ceil, ceil_ties, _ = evaluate([{None: describe(p[N_CTX + y])} for p, y, _ in data], data)
        print(f"ceiling (the true description generated): matched {ceil}/{n}, ties {ceil_ties}")
        report(0)
        t = time.time()
        for ep in range(args.log_every, args.overfit_epochs + 1, args.log_every):
            model.train(b_all, epochs=args.log_every)
            exact_all, matched = report(ep)
        print(f"trained {args.overfit_epochs} epochs in {time.time() - t:.0f}s; final: "
              f"{exact_all}/{n} generated exactly, {matched}/{n} matched to the right candidate")
        return

    t_fwd = []
    for _ in range(args.repeats):
        t = time.time()
        out = model.test(built)
        t_fwd.append(time.time() - t)
    assert len(out) == len(built)
    t_train = []
    for _ in range(0 if args.derived_samepos else args.repeats):
        t = time.time()
        model.train(built, epochs=1)
        t_train.append(time.time() - t)

    sizes = [edge_counts(p) for p, *_ in data]
    objs, edges = [s[0] for s in sizes], [s[1] for s in sizes]
    n = len(built)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"constellation {what}  samples {n}  dim {args.dim}  layers {args.layers}")
    print(f"objects/puzzle   mean {statistics.mean(objs):.1f}  max {max(objs)}")
    print(f"hub edges/puzzle mean {statistics.mean(edges):.0f}  max {max(edges)}")
    print(f"parse XML        {t_parse:7.2f} s total")
    print(f"build facts      {t_facts:7.2f} s total   {1000 * t_facts / n:7.1f} ms/sample")
    print(f"grounding        {t_ground:7.2f} s total   {1000 * t_ground / n:7.1f} ms/sample")
    print(f"forward          {1000 * min(t_fwd) / n:7.1f} ms/sample   (best of {args.repeats}: "
          + ", ".join(f"{x:.2f}s" for x in t_fwd) + ")")
    if t_train:
        print(f"training epoch   {1000 * min(t_train) / n:7.1f} ms/sample   (best of {args.repeats}: "
              + ", ".join(f"{x:.2f}s" for x in t_train) + ")")
    print(f"JVM heap         {heap0:7.1f} MB after template, {heap1:7.1f} MB after grounding"
          f"  -> {(heap1 - heap0) / n * 1024:.0f} KB/sample")
    print(f"process max RSS  {rss_mb:7.1f} MB")


if __name__ == "__main__":
    main()
