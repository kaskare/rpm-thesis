"""Drawing helpers for notebooks/pynl_dynamic_graph.ipynb (puzzles, per-puzzle graphs, fixed slot graph)."""
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle, RegularPolygon

SIDES = {0: 3, 1: 4, 2: 5, 3: 6}                     # type 0-3: triangle .. hexagon, type 4: circle
SCALE = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]               # RAVEN size levels
EDGE_COL = {"row": "#1f77b4", "col": "#ff7f0e", "": "#9a9a9a"}
CELL, GAP = 1.0, 0.12


def panel_origin(p):
    """Canvas position of panel p: the 3x3 matrix on the left, candidates as 2x4 on the right."""
    if p < 8:
        r, c = divmod(p, 3)
        return c * (CELL + GAP), (2 - r) * (CELL + GAP)
    r, c = divmod(p - 8, 4)
    return 3.6 + c * (CELL + GAP), 1.6 - r * (CELL + GAP) + 0.55


def obj_xy(p, o):
    x0, y0 = panel_origin(p)
    return x0 + o["cx"] * CELL, y0 + (1 - o["cy"]) * CELL


def draw_shape(ax, x, y, o, scale=1.0, z=3):
    w = o.get("w", 0.4) * CELL * scale
    r = SCALE[o["size"]] * w / 2
    face = str(1 - o["color"] / 9)
    if o["type"] == 4:
        patch = Circle((x, y), r, fc=face, ec="k", lw=1, zorder=z)
    else:
        n = SIDES[o["type"]]
        patch = RegularPolygon((x, y), n, radius=r, orientation=0 if n % 2 else np.pi / n,
                               fc=face, ec="k", lw=1, zorder=z)
    ax.add_patch(patch)


def _frames(ax, pz, answer=True):
    for p in range(16):
        x0, y0 = panel_origin(p)
        good = answer and p - 8 == pz["target"]
        ax.add_patch(Rectangle((x0, y0), CELL, CELL, fc="white", ec="green" if good else "k",
                               lw=2.5 if good else 1, zorder=1))
        ax.text(x0 + 0.03, y0 + CELL - 0.03, str(p) if p < 8 else f"k{p - 8}", fontsize=7,
                va="top", color="0.4", zorder=5)
    x0, y0 = panel_origin(8)
    ax.text(2 * (CELL + GAP) + CELL / 2, CELL / 2, "?", ha="center", va="center", fontsize=28, color="0.6")
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 8.2)
    ax.set_ylim(-0.1, 3.4)
    ax.axis("off")


def draw_puzzle(pz, ax=None, title=None):
    ax = ax or plt.subplots(figsize=(12, 4.6))[1]
    _frames(ax, pz)
    for p, objs in enumerate(pz["panels"]):
        for o in objs:
            draw_shape(ax, *obj_xy(p, o), o)
    ax.set_title(title or f"{pz.get('grid', '')}  layout: {pz.get('layout_rule', '')}  "
                 f"attributes: {pz.get('attr_rules', '')}  (answer k{pz['target']} in green)", fontsize=9)
    return ax


def _edge_style(t):
    kind = "row" if t.endswith("_row") else "col" if t.endswith("_col") else ""
    return EDGE_COL[kind], ("-" if t.startswith("cc") else (0, (3, 2)))


def parse_facts(facts):
    """Facts -> {predicate: [argument tuples]} (weighted facts keep only their arguments)."""
    out = {}
    for f in facts:
        m = re.match(r"(?:\[[^\]]*\])?([a-z_0-9]+)\((.*?)\)", str(f).replace(" ", ""))
        out.setdefault(m.group(1), []).append(tuple(m.group(2).split(",")))
    return out


def _legend(ax, extra=()):
    h = [Line2D([], [], color=EDGE_COL["row"], label="same row"),
         Line2D([], [], color=EDGE_COL["col"], label="same column"),
         Line2D([], [], color=EDGE_COL[""], label="other"),
         Line2D([], [], color="k", ls=(0, (3, 2)), label="context <-> candidate")] + list(extra)
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=7, frameon=False, ncol=len(h))


def node_xy_dynamic(pz):
    xy = {}
    for p, objs in enumerate(pz["panels"]):
        for o in objs:
            xy[f"o{p}_{o['pos']}"] = obj_xy(p, o)
        x0, y0 = panel_origin(p)
        xy[f"q{p}"] = (x0 + CELL / 2, y0 + CELL / 2)
    return xy


def draw_object_graph(pz, facts, focus=None, ax=None, title=None):
    """Object nodes at their place in the puzzle; same-position edges. focus: draw only edges of that node."""
    ax = ax or plt.subplots(figsize=(12, 4.6))[1]
    _frames(ax, pz, answer=False)
    F, xy = parse_facts(facts), node_xy_dynamic(pz)
    for s, d, t in F.get("edge", []):
        if focus and focus not in (s, d):
            continue
        col, ls = _edge_style(t)
        (x1, y1), (x2, y2) = xy[s], xy[d]
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8,
                    ls=ls, alpha=0.9 if focus else 0.35, shrinkA=4, shrinkB=4,
                    connectionstyle="arc3,rad=0.08"), zorder=2)
    for p, objs in enumerate(pz["panels"]):
        for o in objs:
            n = f"o{p}_{o['pos']}"
            draw_shape(ax, *xy[n], o, scale=0.7, z=4)
            if focus == n:
                ax.add_patch(Circle(xy[n], 0.13, fc="none", ec="red", lw=2, zorder=6))
    n_e = len(F.get("edge", []))
    ax.set_title(title or f"object graph: {sum(map(len, pz['panels']))} object nodes, {n_e} same-position edges"
                 + (f" (showing those of {focus})" if focus else ""), fontsize=9)
    _legend(ax)
    return ax


def draw_panel_graph(pz, facts, ax=None, focus=None):
    """Panel nodes q0..q15, objects -> panel (in_panel), panel <-> panel edges labelled row / column / other."""
    ax = ax or plt.subplots(figsize=(12, 4.6))[1]
    _frames(ax, pz, answer=False)
    F, xy = parse_facts(facts), node_xy_dynamic(pz)
    for s, d, t in F.get("pedge", []):
        if focus and focus not in (s, d):
            continue
        col, ls = _edge_style(t)
        (x1, y1), (x2, y2) = xy[s], xy[d]
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=col, lw=1,
                    ls=ls, alpha=0.8, shrinkA=9, shrinkB=9, connectionstyle="arc3,rad=0.12"), zorder=2)
    for o, q in F.get("in_panel", []):
        (x1, y1), (x2, y2) = xy[o], xy[q]
        ax.plot([x1, x2], [y1, y2], color="green", lw=0.8, alpha=0.6, zorder=3)
        ax.plot(x1, y1, "o", ms=3, color="green", zorder=4)
    for p in range(16):
        ax.plot(*xy[f"q{p}"], "s", ms=9, mfc="gold" if p < 8 else "violet", mec="k", zorder=5)
    n_e = len(F.get("pedge", []))
    ax.set_title(f"panel level: 16 panel nodes, {len(F.get('in_panel', []))} in_panel links (green), "
                 f"{n_e} panel edges" + (f" (showing those of {focus})" if focus else ""), fontsize=9)
    _legend(ax, [Line2D([], [], color="green", label="object -> its panel (SUM)")])
    return ax


def draw_fixed_vs_dynamic(pz, slots, geometry, n_slots=9):
    """Left: the fixed graph's 16 x 9 slot nodes (grey circle = empty slot, grey x = slot beyond this
    layout). Right: the per-puzzle nodes. geometry(pos) -> (cx, cy, w)."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.4))
    for ax, mode in zip(axs, ["fixed", "dynamic"]):
        _frames(ax, pz, answer=False)
        n_real = n_pad = 0
        for p, objs in enumerate(pz["panels"]):
            at = {o["pos"]: o for o in objs}
            x0, y0 = panel_origin(p)
            for k in range(n_slots if mode == "fixed" else 0):
                if k < len(slots) and slots[k] in at:
                    continue
                if k < len(slots):
                    cx, cy, _ = geometry(slots[k])
                    ax.plot(x0 + cx * CELL, y0 + (1 - cy) * CELL, "o", ms=6, mfc="none", mec="0.55", zorder=3)
                else:
                    ax.plot(x0 + (k - len(slots) + 0.5) / (n_slots - len(slots)) * CELL, y0 + 0.07, "x",
                            ms=4, color="0.6", zorder=3)
                n_pad += 1
            for o in objs:
                draw_shape(ax, *obj_xy(p, o), o, scale=0.7, z=4)
                n_real += 1
        ax.set_title(f"fixed: {n_real + n_pad} nodes ({n_real} objects + {n_pad} absent slots)" if mode == "fixed"
                     else f"per puzzle: {n_real + 16} nodes ({n_real} objects + 16 panel nodes)", fontsize=9)
    return fig


def fact_counts(facts):
    return Counter({k: len(v) for k, v in parse_facts(facts).items()})
