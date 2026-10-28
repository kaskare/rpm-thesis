"""Pictures of the network PyNeuraLogic grounds for one facet-model sample (notebooks/facet_model.ipynb).

ground_graph() reads the grounded network itself, so the drawing shows the model's real nodes and
edges, not a sketch of them. draw_network() puts one attribute of one component onto the puzzle grid
and merges the layers: an object's f0, f1, f2 become one facet node, a panel's pt, pv1, pv2, vF one
panel node. gate_means() reads the trained copy gates of a sample after model.test().
"""
import numpy as np
from PIL import Image, ImageDraw

import bench_hub_grounding as B

_ATOMS = ("AtomNeuron", "FactNeuron", "WeightedAtomNeuron")
FACET = {"type": (83, 74, 183), "size": (29, 158, 117), "color": (216, 90, 48)}
EDGE = {"samepos": (175, 169, 236), "panel": (150, 148, 140), "row": (29, 158, 117), "answer": (216, 90, 48),
        "copy": (55, 138, 221)}
INK, GREY = (60, 64, 72), (180, 178, 169)


def _name(n):
    s = str(n.name)                     # a fact's name starts with its value: "[1.0, 0.0, ...] fv(p7o0, type)"
    return s[s.rfind(" ", 0, s.rfind("(")) + 1:]


def ground_graph(template, facts, gid):
    """Atom-level edges (X, Y) of the network grounded for one sample: atom X feeds a rule that computes
    atom Y. Built without chain pruning, so every ground atom keeps a neuron and a name of its own
    (training prunes chains of single-input neurons, which merges neurons but keeps these connections)."""
    from neuralogic.core import R, Settings
    from neuralogic.dataset import Dataset, Sample
    template.build(Settings(chain_pruning=False, iso_value_compression=False))
    sample = list(template.build_dataset(Dataset([Sample(R.out(gid)[[0.0] * B.FACET_OUT], facts)])))[0]

    def sources(n):                     # through rule and aggregation neurons down to the next atoms
        for i in n.getInputs():
            if str(i.getClass().getSimpleName()) in _ATOMS:
                yield _name(i)
            else:
                yield from sources(i)
    return {(x, _name(n)) for n in sample._java_sample.query.evidence.allNeuronsTopologic
            if str(n.getClass().getSimpleName()) in _ATOMS for x in sources(n)}


def gate_means(sample, attr, k="k0"):
    """{panel index: copy gate averaged over the code bits} for one attribute, after model.test([sample])."""
    from neuralogic.core.constructs.java_objects import ValueFactory
    return {int(p[1:]): float(np.mean(ValueFactory.from_java(n.getRawState().getValue())))
            for (p, c, a), n in sample.neurons["AtomNeuron"]["gate"].items() if c == k and a == attr}


def _role(atom):
    """A ground atom -> (drawing node, attribute, component); node None for the position channel and the
    bias. Facets carry no component (their object decides it); the output is shared by both."""
    pred, args = atom[:atom.find("(")], atom[atom.find("(") + 1:-1].split(", ")
    if pred in ("fv", "f0", "f1", "f2"):
        return ("facet", args[0]), args[1], None
    if pred in ("pt", "pv1", "pv2", "vF", "pc", "gate", "rel"):
        return (pred if pred in ("pc", "gate", "rel") else "panel", args[0]), args[2], args[1]
    if pred in ("ctx", "ans", "copy"):
        return (pred,), args[1], args[0]
    if pred == "oa":
        return ("out",), args[1], None
    return None, None, None


def _curve(d, a, b, fill, width, lift):
    """A quadratic curve from a to b, bowed upwards by lift (same-position edges along a row overlap
    when drawn straight)."""
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - lift
    t = np.linspace(0, 1, 24)[:, None]
    pts = (1 - t) ** 2 * np.array(a) + 2 * t * (1 - t) * np.array([mx, my]) + t ** 2 * np.array(b)
    d.line([tuple(p) for p in pts], fill=fill, width=width)


def draw_network(panels, edges, attr, cfg="center_single", k="k0", gates=None, s=150, title=""):
    """The grounded network of one attribute and component, on the puzzle grid. gates: {panel: gate}
    from gate_means(); copy links are drawn as panel -> output, darker and wider for larger gates."""
    n, g, top = len(B.COLS), max(12, s // 6), 26
    objs = {f"p{p}o{i}": (p, o) for p in range(B.N_CTX) for i, o in enumerate(panels[p])
            if B.comp_of(cfg, o["pos"]) == k}
    box = lambda p: (g + (p % n) * (s + g), top + (p // n) * (s + g))
    xr = g + n * (s + g) + s // 3                          # row nodes; the context node right of them
    yc = lambda r: top + r * (s + g) + s / 2
    pos = {("ctx",): (xr + s // 2, (yc(0) + yc(1)) / 2), ("ans",): (box(3 * n - 1)[0] + s / 2, yc(2)),
           ("out",): (xr + s // 2, yc(2)), ("rel", "r1"): (xr, yc(0)), ("rel", "r2"): (xr, yc(1))}
    for p in range(B.N_CTX):
        pos[("panel", f"p{p}")] = (box(p)[0] + s / 2, box(p)[1] + 0.88 * s)
    for oid, (p, o) in objs.items():
        pos[("facet", oid)] = (box(p)[0] + o.get("cx", 0.5) * s, box(p)[1] + (0.12 + 0.6 * o.get("cy", 0.5)) * s)

    keep = lambda r, a, c: (r in pos or r[0] in ("pc", "gate", "copy")) and a == attr and c in (None, k)
    kinds = {}
    for x, y in edges:
        (rx, ax, cx), (ry, ay, cy) = _role(x), _role(y)
        if rx is None or ry is None or rx == ry or not (keep(rx, ax, cx) and keep(ry, ay, cy)):
            continue
        if rx[0] == ry[0] == "facet":
            kinds[frozenset((rx, ry))] = "samepos"
        elif "facet" in (rx[0], ry[0]):
            kinds[frozenset((rx, ry))] = "panel"
        elif rx[0] == "panel" and ry[0] == "rel":
            kinds[(rx, ry)] = "row"
        elif ry[0] in ("ctx", "ans") or (rx[0], ry[0]) == ("ans", "out"):
            kinds[(rx, ry)] = "answer"
        elif rx[0] == "pc":
            kinds[(("panel", rx[1]), ("out",))] = "copy"

    W, H = int(xr + s + g), int(top + 3 * (s + g) + 50)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((g, 6), title, fill=INK)
    for p in range(3 * n):
        x0, y0 = box(p)
        d.rectangle([x0, y0, x0 + s, y0 + s], outline=GREY, width=1)
        d.text((x0 + 3, y0 + 2), "?" if p == 3 * n - 1 else f"({p // n + 1},{p % n + 1})", fill=(136, 135, 128))
    copies = sorted((e for e, kind in kinds.items() if kind == "copy"),
                    key=lambda e: (gates or {}).get(int(e[0][1][1:]), 0))
    for e in copies:                                       # weak gates first, so strong ones stay on top
        gv = (gates or {}).get(int(e[0][1][1:]), 0.5)
        fill = tuple(int(255 - gv * (255 - c)) for c in EDGE["copy"])
        d.line([pos[e[0]], pos[e[1]]], fill=fill, width=1 + int(6 * gv))
    for e, kind in kinds.items():
        if kind == "copy":
            continue
        a, b = (sorted(e) if isinstance(e, frozenset) else e)
        if kind == "samepos":
            _curve(d, pos[a], pos[b], EDGE[kind], 1, min(0.45 * s, 0.18 * abs(pos[b][0] - pos[a][0])))
        else:
            d.line([pos[a], pos[b]], fill=EDGE[kind], width=1 if kind == "panel" else 2)
    nodes = {r for e in kinds for r in e if r in pos}
    for r in nodes:
        x, y = pos[r]
        if r[0] == "facet":
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=FACET[attr], outline=(255, 255, 255))
        elif r[0] == "panel":
            d.rectangle([x - 6, y - 6, x + 6, y + 6], fill=INK)
        else:
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(250, 238, 218), outline=EDGE["answer"], width=2)
            d.text((x - 12, y + 11), " ".join(r) if r[0] == "rel" else r[0], fill=INK)
    lx, ly = g, H - 38
    d.ellipse([lx, ly - 5, lx + 10, ly + 5], fill=FACET[attr]); d.text((lx + 14, ly - 6), f"{attr} facet", fill=INK)
    lx += 90
    d.rectangle([lx, ly - 5, lx + 10, ly + 5], fill=INK); d.text((lx + 14, ly - 6), "panel node", fill=INK)
    lx += 95
    labels = {"samepos": "same position, another column", "panel": "object <-> its panel", "row": "panel -> row",
              "answer": "rows, (3,1), (3,2) -> answer", "copy": "copy link (width = gate)"}
    for kind in labels:
        if kind in kinds.values():
            if lx + 40 + 6 * len(labels[kind]) > W:       # wrap onto a second legend line
                lx, ly = g, ly + 18
            d.line([lx, ly, lx + 24, ly], fill=EDGE[kind], width=3)
            d.text((lx + 30, ly - 6), labels[kind], fill=INK)
            lx += 40 + 6 * len(labels[kind])
    return img
