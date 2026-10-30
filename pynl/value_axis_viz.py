"""Pictures of the value-axis model (pynl/value_axis.py, notebooks/value_axis.ipynb).

draw_value_graph() draws the network PyNeuraLogic grounds for one puzzle (facet_viz.ground_graph), one
attribute, with every ground atom placed by its own arguments: a panel's atoms on the panel's lane, at the
value they are about. Nothing is drawn that the grounding does not contain. Layers are merged where they sit
at the same place: a value's scoring layers o, o2, s are one output node; a row's counts pr and pb one node.
draw_kernel() draws the learned kernel, one row per offset.
"""
import numpy as np
from PIL import Image, ImageDraw

import bench_hub_grounding as B

INK, GREY, FAINT = (60, 64, 72), (150, 148, 140), (214, 212, 205)
COL = {"h": (83, 74, 183), "pv": (175, 169, 236), "desc": (29, 158, 117), "acc": (216, 90, 48),
       "accm": (153, 60, 29), "count": (55, 138, 221), "out": (29, 158, 117), "ans": (216, 90, 48)}


def _parse(atom):
    if "(" not in atom:
        return atom, []
    return atom[:atom.find("(")], atom[atom.find("(") + 1:-1].split(", ")


def draw_value_graph(panels, edges, attr, probs=None, answer=None, dx=15, lh=17, title=""):
    """panels: the puzzle (context panels first); edges: facet_viz.ground_graph(...) of the value template;
    attr: type, size or color; probs: the trained model's distribution over values for attr, drawn as bars
    on the output lane; answer: the true value, marked there."""
    maxval = B.THERM                                     # values 0..maxval (the generator's inclusive top)
    rows = [list(ps) for ps in B.ROWS.values()]
    L, top, gap = 96, 46, 14
    lane = {}
    y = top
    for r, ps in enumerate(rows):
        for p in ps:
            lane[f"p{p}"] = y
            y += lh
        lane[f"cnt{r}"] = y + 2                          # the row's value counts
        y += lh + gap
    y_sum, y_out = y + 4, y + 30
    H = y_out + 150
    X = lambda v: L + v * dx
    XR1 = X(maxval) + 34                                  # pair descriptors
    XR2 = XR1 + 74                                        # row descriptors, counts, checks
    XR3 = XR2 + 86                                        # context
    W = XR3 + 70
    rowname = {"r1": 0, "r2": 1, "r3": 2}
    row_mid = {r: (lane[f"p{ps[0]}"] + lane[f"p{ps[-1]}"]) / 2 for r, ps in enumerate(rows)}
    pos, kind = {}, {}

    def place(atom):
        pred, a = _parse(atom)
        if attr not in a:                                 # another attribute's copy of the graph
            return None
        v = lambda s: int(s[1:])
        if pred == "x":
            return (X(v(a[2])), lane[a[0]]), "x"
        if pred == "h":
            return (X(v(a[2])), lane[a[0]]), "h"
        if pred == "pv":
            return (X(v(a[3])), (lane[a[0]] + lane[a[1]]) / 2), "pv"
        if pred == "pd":
            return (XR1, (lane[a[0]] + lane[a[1]]) / 2), "desc"
        if pred in ("acc", "accm"):
            return (X(v(a[2])) + (3 if pred == "accm" else -3), lane[a[0]]), pred
        if pred in ("pr", "pb"):
            return (X(v(a[2])), lane[f"cnt{rowname[a[0]]}"]), "count"
        if pred in ("rd", "nd", "ov", "pchk", "mchk"):
            off = {"rd": -12, "nd": -4, "ov": 4, "pchk": 12, "mchk": 20}[pred]
            return (XR2, row_mid[rowname[a[0]]] + off), "desc"
        if pred in ("sa", "ma"):
            return (X(v(a[1])), y_sum + (0 if pred == "sa" else 8)), "acc" if pred == "sa" else "accm"
        if pred in ("o", "o2", "s"):
            return (X(v(a[1])), y_out), "out"
        if pred in ("ctx", "ctx2"):
            return (XR3, row_mid[2] + (0 if pred == "ctx" else 30)), "desc"
        if pred == "dist":
            return (XR3, y_out), "out"
        return None

    mine = set()
    for e in edges:
        for a in e:
            if a not in pos:
                pl = place(a)
                if pl:
                    pos[a], kind[a] = pl
            if a in pos:
                mine.add(a)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((8, 6), title, fill=INK)
    for vv in range(0, maxval + 1, 5):
        d.text((X(vv) - 4, top - 22), str(vv), fill=GREY)
        d.line([(X(vv), top - 10), (X(vv), y_out + 6)], fill=(242, 241, 236))
    for r, ps in enumerate(rows):
        for c, p in enumerate(ps):
            d.text((8, lane[f"p{p}"] - 6), f"({r + 1},{c + 1}) {panels[p][0][attr]:>2d}", fill=INK)
        d.text((8, lane[f"cnt{r}"] - 6), f"row {r + 1} counts", fill=COL["count"])
    d.text((8, y_sum - 2), "plus / minus", fill=COL["acc"])
    d.text((8, y_out - 6), "output", fill=COL["out"])
    for x, lab in ((XR1, "pd"), (XR2, "rd nd ov chk"), (XR3, "ctx")):
        d.text((x - 10, top - 22), lab, fill=GREY)
    # edges: faint for the many value-to-descriptor and count edges, coloured for the paths that carry the rules
    order = sorted(edges, key=lambda e: kind.get(e[1], "") in ("acc", "accm", "h", "pv"))
    for a, b in order:
        if a in pos and b in pos:
            ka, kb = kind[a], kind[b]
            col = FAINT
            if ka in ("acc", "accm") and kb in ("acc", "accm"):
                col = COL[ka]
            elif kb == "h" or (ka == "h" and kb == "pv"):
                col = COL["pv"]
            elif ka == "h" and kb == "out":
                col = COL["h"]
            elif ka == "desc" and kb == "out":                # the context reaches every value: lightest
                col = (238, 237, 232)
            d.line([pos[a], pos[b]], fill=col, width=2 if col in (COL["acc"], COL["accm"]) else 1)
    r = {"x": 4, "h": 2, "pv": 1, "desc": 4, "acc": 4, "accm": 4, "count": 2, "out": 2}
    for a, (x, y) in pos.items():
        k = kind[a]
        rr = r[k]
        if k == "x":
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=INK)
        elif k == "desc":
            d.rectangle([x - rr, y - rr, x + rr, y + rr], fill=COL["desc"])
        else:
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=COL[k])
    if probs is not None:
        for vv, pr in enumerate(probs):
            if pr > 0.005:
                d.rectangle([X(vv) - 5, y_out + 8, X(vv) + 5, y_out + 8 + int(70 * pr)], fill=COL["out"])
    if answer is not None:
        d.line([(X(answer), y_out + 84), (X(answer), y_out + 92)], fill=COL["ans"], width=2)
        d.text((X(answer) - 18, y_out + 95), f"answer {answer}", fill=COL["ans"])
    legend = ("black: panel values x;  purple: kernel h (offsets -2..2);  lilac: pair comparison pv;  green squares: "
              "value-free descriptors pd, rd, nd, ov, plus/minus checks, ctx")
    d.text((8, H - 30), legend, fill=GREY)
    d.text((8, H - 16), "orange: running sums acc (plus) and accm (minus);  blue: value counts per row;  "
           "green dots: output per value (bars: trained probabilities)", fill=GREY)
    return img, len(mine)


def draw_kernel(K, s=18):
    """K: offsets x channels. Coral positive, blue negative, opacity by size."""
    n, c = K.shape
    W, H = 60 + c * s, 30 + n * s
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    m = np.abs(K).max()
    for i in range(n):
        d.text((8, 22 + i * s), f"{i - n // 2:+d}", fill=INK)
        for j in range(c):
            t = abs(K[i, j]) / m
            base = (216, 90, 48) if K[i, j] > 0 else (55, 138, 221)
            col = tuple(int(255 - t * (255 - b)) for b in base)
            d.rectangle([50 + j * s, 18 + i * s, 50 + j * s + s - 2, 18 + i * s + s - 2], fill=col)
    d.text((50, 4), f"kernel: {n} offsets x {c} channels (coral positive, blue negative; largest {m:.2f})", fill=INK)
    return img
