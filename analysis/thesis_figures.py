#!/usr/bin/env python3
"""Thesis figures, drawn with matplotlib only: no JVM, no training, under a minute on the login node.

    python3 analysis/thesis_figures.py                       # every figure -> docs/figures/<name>.pdf and .png
    python3 analysis/thesis_figures.py data_factor grounding # some of them

  iraven_puzzle      an I-RAVEN puzzle drawn from its XML attributes (distribute_four, training puzzle 67)
  iravenx_puzzle     an I-RAVEN-X puzzle as its values (rows of ten, values 0-50, training puzzle 3)
  facet_graph        the facet model's graph of that I-RAVEN puzzle (build_facet_template in
                     core/pynl/bench_hub_grounding.py; notes §18.2)
  grounding          one template grounded for two puzzles: the same weights in both ground networks
  data_factor        matched % against training puzzles, with and without the symmetry; every number is read from
                     docs/RESULTS.md §7 when the script runs, and printed with its line
  iraven_transposed  the I-RAVEN puzzle and its transpose, panel (r, c) <- (c, r) (facet_rule_eval.transpose)

The twins figure (docs/figures/twins_sum_midpoint) is drawn by analysis/twins_figure.py and is not touched here.
Run from the repository root with any Python that has matplotlib (the base Anaconda on RCI; pynl has none). The XML
is read here directly, because core/pynl/train_pynl.parse imports neuralogic; the slots are the same (reading order).
"""
import ast, json, os, re, statistics, sys, textwrap, xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path

OUT = "docs/figures"
IRAVEN = "data/I-RAVEN-10000"
IRAVENX = "data/iravenx/center_single_shuffle_n_10_maxval_50.json"
RESULTS = "docs/RESULTS.md"

PUZZLE = ("distribute_four", 67, "train")    # constant positions, Progression on type, Arithmetic on size
PUZZLE_B = ("distribute_four", 140, "train")  # the second puzzle of the grounding sketch (2, 1 and 3 objects in row 1)
PUZZLE_X = 3                                  # I-RAVEN-X training puzzle: Distribute Three, Arithmetic, Progression

INK, INK2, MUTED, LIGHT, FAINT = "#111111", "#444444", "#777777", "#b4b4b4", "#e6e6e6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.labelcolor": INK2, "text.color": INK,
                     "mathtext.default": "regular"})

# RAVEN's rendering (Zhang et al. 2019, const.py): XML levels -> drawn values
TYPES = ("triangle", "square", "pentagon", "hexagon", "circle")          # XML Type 1..5 (model type 0..4)
SIZE_VALUES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
COLOR_VALUES = (255, 224, 196, 168, 140, 112, 84, 56, 28, 0)             # colour 0 white ... 9 black
ANGLE_VALUES = (-135, -90, -45, 0, 45, 90, 135, 180)                     # a noise attribute in RAVEN
DIVIDERS = {"left_center_single_right_center_single": "v", "up_center_single_down_center_single": "h"}
RULE_NAMES = {"Distribute_Three": "Distribute Three"}
ATTR_NAMES = {"Number/Position": "number and position", "Number": "number", "Position": "position",
              "Type": "type", "Size": "size", "Color": "colour"}
CAPTIONS = {}


# ---------------------------------------------------------------------------------------------------------- data
def read_iraven(cfg, idx, split="train"):
    """Panels (8 context + 8 candidates) as lists of objects, the rules per component, and the answer's index."""
    stem = os.path.join(IRAVEN, cfg, f"RAVEN_{idx}_{split}")
    root = ET.parse(stem + ".xml").getroot()
    panels = []
    for panel in root.find("Panels"):
        objs = []
        for struct in panel:
            for ci, comp in enumerate(struct):
                for layout in comp:
                    for e in layout:
                        b = [float(v) for v in ast.literal_eval(e.attrib["bbox"])]   # centre-based [y, x, h, w]
                        objs.append({"comp": ci, "bbox": b, "type": int(e.attrib["Type"]) - 1,
                                     "size": int(e.attrib["Size"]), "color": int(e.attrib["Color"]),
                                     "angle": int(e.attrib["Angle"]),
                                     "slot": 2 * (b[0] > 0.5) + (b[1] > 0.5) + 1})   # 2x2 grid, reading order
        panels.append(objs)
    rules = [[(r.attrib["name"], r.attrib["attr"]) for r in g] for g in root.find("Rules")]
    return panels, rules, int(np.load(stem + ".npz")["target"])


def rules_text(rules):
    return "; ".join(f"{ATTR_NAMES[a]}: {RULE_NAMES.get(n, n)}" for n, a in rules[0])


def transpose(panels):
    """facet_rule_eval.transpose: context panel (r, c) <- (c, r); (3, 3) is the answer's place; candidates unchanged."""
    return [panels[(i % 3) * 3 + i // 3] for i in range(8)] + panels[8:]


# ------------------------------------------------------------------------------------------------------ drawing
def canvas(w, h):
    """A figure whose data coordinates are inches."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=220, bbox_inches="tight", pad_inches=0.03,
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf and .png")


def draw_entity(ax, x0, y0, s, o, lw):
    """One RAVEN entity in the panel with lower-left corner (x0, y0) and side s, as RAVEN renders it: a regular
    polygon (or circle) of circumradius size * min(h, w) / 2 of its bbox, grey level by colour, black outline."""
    yc, xc, h, w = o["bbox"]
    cx, cy = x0 + xc * s, y0 + (1 - yc) * s
    r = SIZE_VALUES[o["size"]] * min(h, w) * s / 2
    grey = COLOR_VALUES[o["color"]] / 255
    kind = TYPES[o["type"]]
    if kind == "circle":
        patch = Circle((cx, cy), r)
    else:
        n = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6}[kind]
        a = np.pi / 2 + (np.pi / 4 if n == 4 else 0) + np.deg2rad(ANGLE_VALUES[o["angle"]]) + 2 * np.pi * np.arange(n) / n
        patch = Polygon(np.c_[cx + r * np.cos(a), cy + r * np.sin(a)], closed=True)
    patch.set(facecolor=(grey, grey, grey), edgecolor=INK, linewidth=lw, joinstyle="miter", zorder=3)
    ax.add_patch(patch)


def draw_panel(ax, x0, y0, s, objs, cfg, frame=0.8, question=False):
    ax.add_patch(Rectangle((x0, y0), s, s, fill=False, ec=INK, lw=frame, zorder=4))
    if question:
        ax.text(x0 + s / 2, y0 + s / 2, "?", ha="center", va="center", fontsize=s * 30, color=INK2)
        return
    if DIVIDERS.get(cfg) == "v":
        ax.plot([x0 + s / 2] * 2, [y0, y0 + s], color=INK, lw=0.6, zorder=2)
    elif DIVIDERS.get(cfg) == "h":
        ax.plot([x0, x0 + s], [y0 + s / 2] * 2, color=INK, lw=0.6, zorder=2)
    for o in objs:
        draw_entity(ax, x0, y0, s, o, lw=0.55 * s / 0.6)


def draw_matrix(ax, panels, cfg, x0, y_top, s, gap, answer=None, labels=None, label_gap=0.0):
    """The 3x3 context, row 1 at the top; (3, 3) holds '?' or, with answer=k, candidate k in a heavy frame.
    labels: a text under every panel (label_gap leaves room for it)."""
    for i in range(9):
        r, c = divmod(i, 3)
        x, y = x0 + c * (s + gap), y_top - (r + 1) * s - r * (gap + label_gap)
        if i < 8:
            draw_panel(ax, x, y, s, panels[i], cfg)
        elif answer is None:
            draw_panel(ax, x, y, s, [], cfg, question=True)
        else:
            draw_panel(ax, x, y, s, panels[8 + answer], cfg, frame=2.0)
        if labels:
            ax.text(x + s / 2, y - 0.03, labels[i], ha="center", va="top", fontsize=6.5, color=INK2)


def draw_candidates(ax, panels, cfg, answer, x0, y_top, s, gap):
    for k in range(8):
        r, c = divmod(k, 4)
        x, y = x0 + c * (s + gap), y_top - (r + 1) * s - r * 0.22
        draw_panel(ax, x, y, s, panels[8 + k], cfg, frame=2.0 if k == answer else 0.8)
        ax.text(x + s / 2, y - 0.03, f"{k + 1}" + ("  answer" if k == answer else ""), ha="center", va="top",
                fontsize=6.5, color=INK if k == answer else INK2, fontweight="bold" if k == answer else "normal")


# ---------------------------------------------------------------------------------------------- 1. the puzzles
def fig_iraven_puzzle():
    cfg, idx, split = PUZZLE
    panels, rules, answer = read_iraven(cfg, idx, split)
    fig, ax = canvas(6.3, 2.45)
    s, gap = 0.66, 0.06
    draw_matrix(ax, panels, cfg, 0.05, 2.38, s, gap)
    ax.text(0.05 + 1.5 * s + gap, 2.42, "context", ha="center", va="bottom", fontsize=7.5, color=INK2)
    cs, cgap = 0.6, 0.1
    cx0 = 2.6
    draw_candidates(ax, panels, cfg, answer, cx0, 2.3, cs, cgap)
    ax.text(cx0 + 2 * cs + 1.5 * cgap, 2.42, "candidates", ha="center", va="bottom", fontsize=7.5, color=INK2)
    parts = rules_text(rules).split("; ")
    ax.text(cx0, 0.62, f"Rules along the rows: {'; '.join(parts[:2])};\n{'; '.join(parts[2:])}.", fontsize=7,
            color=INK2, va="top", linespacing=1.4)
    save(fig, "iraven_puzzle")
    print(f"  I-RAVEN {cfg} {split} puzzle {idx}; answer = candidate {answer + 1}; {rules_text(rules)}")
    CAPTIONS["iraven_puzzle"] = (f"An I-RAVEN puzzle ({cfg.replace('_', ' ')}, training puzzle {idx}), drawn from "
                                 f"its XML attributes (type, size, colour, and angle, which is noise). Rules along "
                                 f"the rows: {rules_text(rules)}. The answer is candidate {answer + 1}.")


def fig_iravenx_puzzle():
    with open(IRAVENX) as f:
        s = json.load(f)[str(PUZZLE_X)]
    n = int(round((len(s["rpm"]) - 7) ** 0.5))
    shown = s["rpm"][(n - 3) * n: n * n - 1]          # the last three rows, as the models see them (load_iravenx)
    cands = s["rpm"][n * n - 1:]
    answer = int(s["target"])
    vals = lambda p: tuple(int(p[0][k]) for k in ("Type", "Size", "Color"))
    rows = [[vals(p) for p in shown[r * n:(r + 1) * n]] for r in range(3)]
    rules = s["rules"][0]
    attrs = (("Type", "type"), ("Size", "size"), ("Color", "colour"))

    cw, cg, ch, lh = 0.5, 0.035, 0.5, 0.135           # cell width, gap, cell height, line height
    x0, rg, cgap = 0.75, 0.1, 0.3                     # left labels, gap between rows, gap above the candidates
    H = 0.42 + 3 * ch + 2 * rg + cgap + ch + 0.45
    fig, ax = canvas(x0 + n * (cw + cg) + 0.05, H)
    top = H - 0.42                                    # rule line and column numbers above
    rule_line = "; ".join(f"{name}: {RULE_NAMES.get(rules[k], rules[k])}" for k, name in attrs)
    ax.text(x0, H - 0.08, f"Rules: {rule_line}.", fontsize=7, color=INK2, va="top")

    def cell(x, y, v, heavy=False, question=False):
        ax.add_patch(Rectangle((x, y), cw, ch, fill=False, ec=INK, lw=2.0 if heavy else 0.6))
        if question:
            ax.text(x + cw / 2, y + ch / 2, "?", ha="center", va="center", fontsize=13, color=INK2)
            return
        for j, val in enumerate(v):
            ax.text(x + cw / 2, y + ch - 0.09 - j * lh, str(val), ha="center", va="center", fontsize=7)

    for r in range(3):
        y = top - (r + 1) * ch - r * rg
        ax.text(0.02, y + ch / 2, f"row {r + 1}", ha="left", va="center", fontsize=7.5, color=INK)
        for j, (_, name) in enumerate(attrs):
            ax.text(x0 - 0.05, y + ch - 0.09 - j * lh, name, ha="right", va="center", fontsize=6, color=INK2)
        for c in range(n):
            x = x0 + c * (cw + cg)
            if r == 2 and c == n - 1:
                cell(x, y, None, question=True)
            else:
                cell(x, y, rows[r][c])
    for c in range(n):
        ax.text(x0 + c * (cw + cg) + cw / 2, top + 0.03, f"{c + 1}", ha="center", va="bottom", fontsize=6.5,
                color=MUTED)

    yc = top - 3 * ch - 2 * rg - cgap - ch
    ax.text(0.02, yc + ch / 2, "candidates", ha="left", va="center", fontsize=7.5)
    for k, p in enumerate(cands):
        x = x0 + k * (cw + cg)
        cell(x, yc, vals(p), heavy=(k == answer))
        ax.text(x + cw / 2, yc - 0.03, f"{k + 1}" + (" answer" if k == answer else ""), ha="center", va="top",
                fontsize=6.5, color=INK if k == answer else INK2, fontweight="bold" if k == answer else "normal")

    # the Arithmetic row, written out: I-RAVEN-X's plus (last = sum of the rest) or minus (first = sum of the rest)
    a = [v[1] for v in rows[0]]
    if a[0] == sum(a[1:]):
        eq = f"{a[0]} = " + " + ".join(map(str, a[1:])) + "  (the first panel is the sum of the others)"
    elif a[-1] == sum(a[:-1]):
        eq = " + ".join(map(str, a[:-1])) + f" = {a[-1]}  (the last panel is the sum of the others)"
    else:
        eq = None
    if rules["Size"] == "Arithmetic" and eq:
        ax.text(x0, yc - 0.22, f"Size in row 1: {eq}.", fontsize=7, color=INK2, va="top")
    save(fig, "iravenx_puzzle")
    print(f"  I-RAVEN-X {os.path.basename(IRAVENX)} puzzle {PUZZLE_X}; answer = candidate {answer + 1} "
          f"{vals(cands[answer])}; {rule_line}")
    CAPTIONS["iravenx_puzzle"] = (f"An I-RAVEN-X puzzle (rows of ten panels, values 0–50, training puzzle {PUZZLE_X}): "
                                  f"the last three rows of its 10×10 matrix, one object per panel, given as its type, "
                                  f"size and colour values. Rules: {rule_line}. The answer is candidate {answer + 1}.")


def fig_iraven_transposed():
    cfg, idx, split = PUZZLE
    panels, rules, answer = read_iraven(cfg, idx, split)
    tp = transpose(panels)
    fig, ax = canvas(6.3, 2.75)
    s, gap, lg = 0.6, 0.07, 0.14
    names = [f"({r + 1},{c + 1})" for r in range(3) for c in range(3)]
    src = [names[(i % 3) * 3 + i // 3] if i < 8 else names[8] for i in range(9)]   # where each panel comes from
    left, right, top = 0.35, 3.85, 2.42
    draw_matrix(ax, panels, cfg, left, top, s, gap, answer=answer, labels=names, label_gap=lg)
    draw_matrix(ax, tp, cfg, right, top, s, gap, answer=answer, labels=src, label_gap=lg)
    w = 3 * s + 2 * gap
    ax.text(left + w / 2, top + 0.2, "as given: the rules run along the rows", ha="center", va="bottom",
            fontsize=7.5)
    ax.text(right + w / 2, top + 0.2, "transposed: they run down the columns", ha="center", va="bottom",
            fontsize=7.5)
    ax.text(left + w / 2, top + 0.06, "label: the panel's place", ha="center", va="bottom", fontsize=6.5,
            color=INK2)
    ax.text(right + w / 2, top + 0.06, "label: where the panel comes from", ha="center", va="bottom", fontsize=6.5,
            color=INK2)
    ym = top - 1.5 * s - gap - lg
    ax.annotate("", xy=(right - 0.15, ym), xytext=(left + w + 0.15, ym),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK, mutation_scale=9))
    ax.text((left + w + right) / 2, ym + 0.08, "panel (r, c) takes\npanel (c, r)", ha="center", va="bottom",
            fontsize=7)
    ax.text((left + w + right) / 2, ym - 0.08, "the answer (heavy frame)\nstays at (3, 3); the\ncandidates "
            "are unchanged", ha="center", va="top", fontsize=6.5, color=INK2)
    save(fig, "iraven_transposed")
    CAPTIONS["iraven_transposed"] = ("The I-RAVEN puzzle of Figure (iraven_puzzle) as given and transposed: panel "
                                     "(r, c) takes panel (c, r), the answer keeps its place (3, 3), so the "
                                     "candidates are unchanged and the rules now run down the columns.")


# -------------------------------------------------------------------------------------------- 2. the facet graph
R_S, R_TC, DX = 0.052, 0.045, 0.1     # size node radius, type / colour node radius, centre spacing in an object
OBJ_W = 2 * DX + 2 * R_S + 0.02       # one object: type, size, colour side by side


def slot_glyph(ax, x, y, slot, side=0.07):
    """An object's position, drawn as its attribute: the 2x2 grid with the object's slot filled. (x, y): lower left."""
    h = side / 2
    for k in range(4):
        r, c = divmod(k, 2)
        ax.add_patch(Rectangle((x + c * h, y + (1 - r) * h), h, h, fc=INK if k == slot - 1 else "white", ec=INK2,
                               lw=0.35, zorder=3))


def facet_object(ax, x, y, o, show_tc=True, size_label=None):
    """Facet nodes of one object, centred on its size node at (x, y): type left, size middle, colour right."""
    if show_tc:
        for dx, v in ((-DX, o["type"]), (DX, o["color"])):
            ax.add_patch(Circle((x + dx, y), R_TC, fc="white", ec=LIGHT, lw=0.6, zorder=3))
            ax.text(x + dx, y, str(v), ha="center", va="center", fontsize=5, color=MUTED, zorder=4)
    ax.add_patch(Circle((x, y), R_S, fc="white", ec=INK, lw=0.9, zorder=3))
    ax.text(x, y, str(o["size"]) if size_label is None else size_label, ha="center", va="center", fontsize=5.5,
            zorder=4)
    slot_glyph(ax, x - DX - 0.035 if show_tc else x - R_S - 0.075, y + R_TC + 0.02, o["slot"])


def arc(ax, x1, x2, y, depth, lw=0.6, color=INK2, ls="-"):
    """A U-shaped edge below y from x1 to x2."""
    d = depth * 4 / 3
    path = Path([(x1, y), (x1, y - d), (x2, y - d), (x2, y)], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(path, fill=False, ec=color, lw=lw, ls=ls, zorder=2))


def edge(ax, p, q, lw=0.6, color=INK2, ls="-", head=True, both=False):
    style = "<|-|>" if both else "-|>" if head else "-"
    ax.annotate("", xy=q, xytext=p, zorder=2, arrowprops=dict(arrowstyle=style, lw=lw, color=color, ls=ls,
                                                              mutation_scale=5, shrinkA=0, shrinkB=0))


def stacked(ax, x, y, w, h, text="", fs=6, copies=3, d=0.022):
    """An aggregate node that stands for four, one per attribute (type, size, colour, number): the front one is
    size, the three behind it the others. (x, y): centre of the front node."""
    for k in range(copies, -1, -1):
        ax.add_patch(FancyBboxPatch((x - w / 2 + k * d, y - h / 2 + k * d), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.025", fc="white",
                                    ec=INK if k == 0 else LIGHT, lw=0.9 if k == 0 else 0.5, zorder=3 + 0.1 * (copies - k)))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=5)


def label_on(ax, x, y, text, fs=5.5, color=INK):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color, zorder=6,
            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))


def fig_facet_graph():
    """One node per (object, attribute); each facet reads its panel node and the same attribute of the objects at the
    same position in the other columns of its row (a weight per column pair); panel nodes vF(P, C, A) average the
    panel's facets (plus its mean code); rel(Rw, C, A) reads the panels of a complete row with a weight per column;
    ctx(C, A) averages rows 1-2; ans(C, A) reads the question row's two panels and ctx; the output is the generated
    panel. Positions are an attribute of each object (a one-hot fact pos(O)); the position channel sums them per panel
    into an occupancy and has the same row structure (not drawn)."""
    cfg, idx, split = PUZZLE
    panels, rules, answer = read_iraven(cfg, idx, split)
    rows = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    pad, gin, grow = 0.05, 0.05, 0.13                 # box padding, gap between boxes in a row, gap between rows
    y_obj, y_pan, y_row, y_top = 1.2, 1.93, 2.58, 3.2
    x = 0.62
    box, sx = {}, {}                                   # panel -> (x0, width); (panel, object) -> size-node x
    for r, ps in enumerate(rows):
        for p in ps:
            objs = sorted(panels[p], key=lambda o: o["slot"]) if p < 8 else [None]
            w = len(objs) * OBJ_W + 2 * pad
            box[p] = (x, w)
            for k, o in enumerate(objs):
                if o is not None:
                    sx[(p, o["slot"])] = x + pad + (k + 0.5) * OBJ_W
            x += w + gin
        x += grow - gin
    W, H = x + 0.02, 3.62
    fig, ax = canvas(W, H)

    # level labels
    for y, t in ((0.97, "same\nposition"), (y_obj, "facets"), (y_pan, "panels"), (y_row, "rows"),
                 (y_top, "rows 1–2,\nanswer")):
        ax.text(0.02, y, t, ha="left", va="center", fontsize=6.5, color=INK2, linespacing=1.15)

    # panels: boxes with their objects' facets
    by, bh = y_obj - R_S - 0.05, 2 * R_S + 0.16
    for p, (x0, w) in box.items():
        r, c = divmod(p, 3)
        q = p == 8
        ax.add_patch(FancyBboxPatch((x0, by), w, bh, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc="white", ec=INK2 if not q else MUTED, lw=0.6, ls="-" if not q else (0, (3, 2)),
                                    zorder=1))
        if q:
            ax.text(x0 + w / 2, y_obj - 0.01, "?", ha="center", va="center", fontsize=10, color=INK2)
            continue
        for o in sorted(panels[p], key=lambda o: o["slot"]):
            facet_object(ax, sx[(p, o["slot"])], y_obj, o)

    # same-position edges between size facets, within a row, between different columns (both ways, one weight
    # per ordered column pair U_ab); drawn once per pair
    arc_styles = (("-", INK2), ((0, (2.2, 1.4)), MUTED))
    for r, ps in enumerate(rows[:3]):
        ctx = [p for p in ps if p < 8]
        linked = sorted({o["slot"] for a in ctx for b in ctx if a != b for o in panels[a]
                         if any(ob["slot"] == o["slot"] for ob in panels[b])})
        for i, a in enumerate(ctx):
            for b in ctx[i + 1:]:
                for o in panels[a]:
                    if any(ob["slot"] == o["slot"] for ob in panels[b]):
                        x1, x2 = sx[(a, o["slot"])], sx[(b, o["slot"])]
                        fam = linked.index(o["slot"])
                        depth = 0.09 + 0.1 * (b - a - 1) + 0.05 * fam
                        ls, color = arc_styles[fam % 2]
                        arc(ax, x1, x2, y_obj - R_S, depth, ls=ls, color=color)
                        if r == 0:
                            label_on(ax, (x1 + x2) / 2, y_obj - R_S - depth,
                                     f"$U_{{{a % 3 + 1}{b % 3 + 1}}}$", fs=5.5)

    # panel nodes and the edges into them (size facets -> the panel's size node)
    pn = {}
    for p, (x0, w) in box.items():
        if p == 8:
            continue
        pn[p] = (x0 + w / 2, y_pan)
        stacked(ax, *pn[p], 0.27, 0.12, f"({p // 3 + 1},{p % 3 + 1})", fs=5.5)
        for o in panels[p]:
            edge(ax, (sx[(p, o["slot"])], y_obj + R_S), (pn[p][0], y_pan - 0.06), lw=0.6, both=True)

    # rows 1 and 2: a weight per column
    rn = {}
    for r in (0, 1):
        ps = rows[r]
        rn[r] = (pn[ps[1]][0], y_row)
        stacked(ax, *rn[r], 0.42, 0.15, f"row {r + 1}")
        for c, p in enumerate(ps):
            q = (rn[r][0] + (c - 1) * 0.13, y_row - 0.075)
            edge(ax, (pn[p][0], y_pan + 0.06), q)
            label_on(ax, pn[p][0] + 0.3 * (q[0] - pn[p][0]), y_pan + 0.06 + 0.3 * (q[1] - y_pan - 0.06),
                     f"$W_{c + 1}$")

    # context: the average over the complete rows; the answer reads (3,1), (3,2) and the context
    cx = ((rn[0][0] + rn[1][0]) / 2, y_top)
    stacked(ax, *cx, 0.88, 0.15, "mean of rows 1–2")
    for r in (0, 1):                  # one rule with the row as a variable: the same weight R for both rows
        q = (cx[0] + (r - 0.5) * 0.3, y_top - 0.075)
        edge(ax, (rn[r][0], y_row + 0.075), q)
        label_on(ax, rn[r][0] + 0.5 * (q[0] - rn[r][0]), y_row + 0.075 + 0.5 * (q[1] - y_row - 0.075), "$R$")
    an = ((pn[6][0] + pn[7][0]) / 2, y_top)
    stacked(ax, *an, 0.5, 0.15, "answer")
    for c, p in enumerate((6, 7)):
        q = (an[0] + (c - 0.5) * 0.2, y_top - 0.075)
        edge(ax, (pn[p][0], y_pan + 0.06), q)
        label_on(ax, pn[p][0] + 0.45 * (q[0] - pn[p][0]), y_pan + 0.06 + 0.45 * (q[1] - y_pan - 0.06), f"$B_{c + 1}$")
    edge(ax, (cx[0] + 0.44, y_top), (an[0] - 0.25 - 0.005, y_top))
    label_on(ax, (cx[0] + 0.44 + an[0] - 0.25) / 2, y_top, "$B_0$")

    # the output: the generated panel, compared with the candidates
    x0, w = box[8]
    on = (x0 + w / 2, y_row)
    ax.add_patch(FancyBboxPatch((on[0] - 0.3, y_row - 0.12), 0.6, 0.24, boxstyle="round,pad=0,rounding_size=0.025",
                                fc=FAINT, ec=INK, lw=0.9, zorder=3))
    ax.text(*on, "generated\npanel", ha="center", va="center", fontsize=6, zorder=5, linespacing=1.1)
    edge(ax, (an[0] + 0.25 + 0.06, y_top - 0.02), (on[0], y_row + 0.12))
    edge(ax, (on[0], y_row - 0.12), (on[0], by + bh), ls=(0, (2, 1.5)))

    # key
    kx, ky = 0.88, 0.5
    o = {"type": 1, "size": 3, "color": 2, "slot": 4}
    facet_object(ax, kx, ky, o)
    ax.text(kx - DX - R_TC - 0.02, ky, "type", ha="right", va="center", fontsize=5.5, color=INK2)
    ax.text(kx + DX + R_TC + 0.02, ky, "colour", ha="left", va="center", fontsize=5.5, color=INK2)
    ax.text(kx, ky - R_S - 0.02, "size", ha="center", va="top", fontsize=5.5, color=INK2)
    ax.text(kx - DX - 0.05, ky + R_TC + 0.055, "position", ha="right", va="center", fontsize=5.5, color=INK2)
    key = ("One object: its type, size and colour facets, with their values. Its position is an attribute (the filled "
           "slot of the 2×2 grid), never a node. Above the objects, edges are drawn for size; each stacked node "
           "stands for four, one per attribute (type, size, colour, number), and one set of weights serves all four. "
           "Number has no facets: its panel node reads the object count. A channel with the same row structure reads "
           "the positions, summed per panel (not drawn).")
    ax.text(kx + 0.45, ky, textwrap.fill(key, 112), ha="left", va="center", fontsize=5.8, color=INK2,
            linespacing=1.3)
    save(fig, "facet_graph")
    counts = [len(panels[p]) for p in range(8)]
    n_same = sum(1 for ps in rows for a in ps for b in ps if a != b and a < 8 and b < 8
                 for oa in panels[a] for ob in panels[b] if oa["slot"] == ob["slot"])
    print(f"  {cfg} {split} puzzle {idx}: objects per context panel {counts}, {sum(counts)} objects, "
          f"{3 * sum(counts)} facet nodes, {n_same} same-position edges per attribute (directed)")
    CAPTIONS["facet_graph"] = (
        "The facet model's graph of the puzzle in Figure (iraven_puzzle). One node per object and attribute; each "
        "facet reads its panel node and the same attribute of the objects at the same position in the other columns "
        "of its row, with a weight per column pair (U, labelled in row 1); the double arrows: a panel node averages "
        "its facets and, at every facet layer, sends the average back. A row node reads its three panels with a weight "
        "per column (W); both rows reach their mean through the same weight (R), since one rule with the row as a "
        "variable writes it; the answer reads panels (3,1) and (3,2) and that mean (B). Positions are an attribute "
        "of every object, never a node. The graph is built "
        "by grounding the template on this puzzle's facts, so another puzzle gives another graph.")


# ------------------------------------------------------------------------------------------- 3. the grounding sketch
def ground_row(ax, panels, row, x0, y_obj, y_pan, y_row, title):
    """Facets of attribute A (size) of one row, its panel nodes and its row node, every edge with its weight."""
    ps = row
    pad, gin = 0.06, 0.16
    obj_w = 0.3
    x = x0
    sx, box = {}, {}
    for p in ps:
        objs = sorted(panels[p], key=lambda o: o["slot"])
        w = len(objs) * obj_w + 2 * pad
        box[p] = (x, w)
        for k, o in enumerate(objs):
            sx[(p, o["slot"])] = x + pad + (k + 0.5) * obj_w
        x += w + gin
    x_end = x - gin
    by, bh = y_obj - 0.11, 0.25
    for c, p in enumerate(ps):
        bx, w = box[p]
        ax.add_patch(FancyBboxPatch((bx, by), w, bh, boxstyle="round,pad=0,rounding_size=0.03", fc="white",
                                    ec=INK2, lw=0.6, zorder=1))
        for o in panels[p]:
            facet_object(ax, sx[(p, o["slot"])], y_obj - 0.02, o, show_tc=False, size_label="")
    # same-position edges
    n_u = 0
    linked = sorted({o["slot"] for a in ps for b in ps if a != b for o in panels[a]
                     if any(ob["slot"] == o["slot"] for ob in panels[b])})
    for i, a in enumerate(ps):
        for j, b in enumerate(ps[i + 1:], start=i + 1):
            for o in panels[a]:
                if any(ob["slot"] == o["slot"] for ob in panels[b]):
                    x1, x2 = sx[(a, o["slot"])], sx[(b, o["slot"])]
                    depth = 0.09 + 0.08 * (j - i - 1) + 0.06 * linked.index(o["slot"])
                    arc(ax, x1, x2, y_obj - 0.02 - R_S, depth, ls=(0, (1.2, 1.2)), color=INK, lw=0.7)
                    label_on(ax, (x1 + x2) / 2, y_obj - 0.02 - R_S - depth, f"$U_{{{i + 1}{j + 1}}}$", fs=5.5)
                    n_u += 2
    # panel nodes, with V on every facet edge
    pn = {}
    for p in ps:
        bx, w = box[p]
        pn[p] = bx + w / 2
        ax.add_patch(FancyBboxPatch((pn[p] - 0.14, y_pan - 0.06), 0.28, 0.12, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc="white", ec=INK, lw=0.9, zorder=3))
        ax.text(pn[p], y_pan, f"(1,{ps.index(p) + 1})", ha="center", va="center", fontsize=5.5, zorder=5)
        for o in panels[p]:
            xs = sx[(p, o["slot"])]
            edge(ax, (xs, y_obj - 0.02 + R_S), (pn[p], y_pan - 0.06), lw=0.5, color=MUTED)
            label_on(ax, xs + 0.45 * (pn[p] - xs), y_obj - 0.02 + R_S + 0.45 * (y_pan - 0.06 - y_obj + 0.02 - R_S),
                     "V", fs=5.3, color=INK2)
    # the row node, W1 W2 W3
    xr = (x0 + x_end) / 2
    ax.add_patch(FancyBboxPatch((xr - 0.25, y_row - 0.075), 0.5, 0.15, boxstyle="round,pad=0,rounding_size=0.025",
                                fc="white", ec=INK, lw=0.9, zorder=3))
    ax.text(xr, y_row, "row 1", ha="center", va="center", fontsize=6, zorder=5)
    styles = ("-", (0, (4, 1.5)), (0, (4, 1.2, 1, 1.2)))
    for c, p in enumerate(ps):
        q = (xr + (c - 1) * 0.14, y_row - 0.075)
        edge(ax, (pn[p], y_pan + 0.06), q, lw=1.0, color=INK, ls=styles[c])
        label_on(ax, pn[p] + 0.45 * (q[0] - pn[p]), y_pan + 0.06 + 0.45 * (q[1] - y_pan - 0.06), f"$W_{c + 1}$",
                 fs=6)
    ax.text(xr, y_row + 0.17, title, ha="center", va="bottom", fontsize=7)
    n_obj = sum(len(panels[p]) for p in ps)
    return n_obj, n_u, x_end


def fig_grounding():
    (ca, ia, sa), (cb, ib, sb) = PUZZLE, PUZZLE_B
    pa, _, _ = read_iraven(ca, ia, sa)
    pb, _, _ = read_iraven(cb, ib, sb)
    W, H = 6.3, 3.55
    fig, ax = canvas(W, H)
    # the template (simplified: one component, one facet layer, no residual or width changes)
    tx, ty = 0.55, H - 0.08
    ax.add_patch(FancyBboxPatch((tx - 0.08, ty - 0.74), W - 2 * tx + 0.16, 0.74, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=FAINT, ec=INK2, lw=0.6))
    ax.text(tx, ty - 0.07, "template (simplified: one component, one facet layer)", fontsize=7, va="top",
            fontweight="bold")
    lines = ((r"$f(O, A)\ \leftarrow\ U_{ab}\cdot f_0(O_1, A),\ \ \mathrm{samepos}(O_1, O),\ O$ in column $a$, "
              r"$O_1$ in column $b$", "a weight per ordered pair of columns"),
             (r"$vF(P, A)\ \leftarrow\ V\cdot f(O, A),\ \ \mathrm{obj}(P, O)$", "averaged over the panel's objects"),
             (r"$rel(R, A)\ \leftarrow\ W_1\cdot vF(P_1, A),\ W_2\cdot vF(P_2, A),\ W_3\cdot vF(P_3, A),\ \ "
              r"\mathrm{row}(R, P_1, P_2, P_3)$", "a weight per column"))
    for k, (t, note) in enumerate(lines):
        ax.text(tx + 0.05, ty - 0.27 - k * 0.15, t, fontsize=6.3, va="center")
        ax.text(W - tx, ty - 0.27 - k * 0.15, note, fontsize=6.3, va="center", ha="right", color=INK2)
    # two ground networks, attribute A = size, row 1 of each puzzle
    y_obj, y_pan, y_row = 0.72, 1.48, 2.18
    na, ua, xa = ground_row(ax, pa, (0, 1, 2), 0.35, y_obj, y_pan, y_row,
                            f"puzzle {ia} ({ca.replace('_', ' ')}), row 1, A = size")
    nb, ub, xb = ground_row(ax, pb, (0, 1, 2), 3.3, y_obj, y_pan, y_row,
                            f"puzzle {ib} ({cb.replace('_', ' ')}), row 1, A = size")
    ax.plot([3.05, 3.05], [0.25, 2.5], color=LIGHT, lw=0.6)
    for x, n, u in (((0.35 + xa) / 2, na, ua), ((3.3 + xb) / 2, nb, ub)):
        ax.text(x, 0.12, f"{n} facet nodes; edges: {n} with V, {u} with a U, 3 with a W", fontsize=6.2,
                color=INK2, va="center", ha="center")
    ax.text(W / 2, -0.06, "Different nodes and edges, one set of weights: every edge in both networks carries a "
            "weight of the template, and so do\nthe networks for type, colour and number, for row 2, and for every "
            "other puzzle. An arc is two edges, one each way ($U_{13}$ and $U_{31}$).", ha="center", va="top",
            fontsize=6.8, linespacing=1.3)
    save(fig, "grounding")
    print(f"  puzzle A {ca} {ia}, row 1: {na} objects, {ua} same-position edges; puzzle B {cb} {ib}, row 1: {nb} "
          f"objects, {ub} same-position edges")
    CAPTIONS["grounding"] = (
        "One template grounded on two puzzles (attribute size, row 1 of each; simplified to one facet layer). The "
        "ground networks differ in their nodes and edges, but every edge carries one of the template's weights: V "
        "from a facet to its panel, W1–W3 from the panels to the row, U_ab between objects at the same position in "
        "columns a and b.")


# --------------------------------------------------------------------------------------------- 4. the data factor
def section(num):
    lines = open(RESULTS, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(f"## {num}."))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return [(i + 1, lines[i]) for i in range(start, end)]


def tables(sec):
    """Every markdown table of a section: (header cells, [(line number, cells), ...]); an empty first cell repeats
    the one above it."""
    out, cur = [], None
    for n, line in sec:
        s = line.strip()
        if not s.startswith("|"):
            cur = None
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cur is None:
            cur = (cells, [])
            out.append(cur)
        elif set(s) <= set("|-: "):
            continue
        else:
            if not cells[0] and cur[1]:
                cells[0] = cur[1][-1][1][0]
            cur[1].append((n, cells))
    return out


def cell_value(text):
    """'**88.0 ± 1.9** (42.4 ± 3.0)' or 'first 86.6, seeds 87.3 / 90.2: ...' -> (mean, sd or None, runs or None);
    '**99.3** (95.6)' -> (99.3, None, None). The runs are the numbers before a colon."""
    t = text.replace("**", "")
    m = re.search(r"(\d+\.\d)\s*±\s*(\d+\.\d)", t)
    if m:
        runs = [float(v) for v in re.findall(r"\d+\.\d", t.split(":")[0])] if ":" in t else None
        return float(m.group(1)), float(m.group(2)), runs
    return float(re.search(r"\d+\.\d", t).group(0)), None, None


def data_factor_points():
    """Every point of the figure, read from RESULTS §7: {(set, arm): [(puzzles, mean, sd, runs, line, note)]}."""
    sec = section(7)
    tabs = tables(sec)

    def table(first):
        return next(t for t in tabs if t[0][0] == first)

    def lookup(tab, row_pred, col):
        head, rows = tab
        j = head.index(col)
        n, cells = next((n, c) for n, c in rows if row_pred(c))
        return n, cells[j]

    main, seeds, factor = table("data"), table("data, puzzles"), table("separate weights, seeds 1–3")
    pts = {}

    def add(key, x, n, text, note=""):
        mean, sd, runs = cell_value(text)
        pts.setdefault(key, []).append((x, mean, sd, runs, n, text.replace("**", ""), note))

    lead = lambda c: int(re.match(r"\d+", c[1]).group(0))
    for kind, col, arm in (("components", "shared, symmetric (default)", "shared"),
                           ("components", "separate weights, same widths", "same"),
                           ("components", "separate weights, matched count", "matched"),
                           ("attributes", "separate weights, same widths", "same")):
        for x in ((1000, 6000) if kind == "components" and arm != "same" else
                  (100, 1000, 6000) if kind == "components" else (600, 6000)):
            n, text = lookup(main, lambda c, x=x: f"({kind})" in c[0] and lead(c) == x, col)
            add((kind, arm), x, n, text, "still rising at epoch 60" if "still rising" in text else "")
    for kind, x in (("components", 100), ("attributes", 600), ("attributes", 6000)):
        for col, arm in (("shared, symmetric", "shared"), ("separate, matched count", "matched")):
            n, text = lookup(seeds, lambda c, kind=kind, x=x: c[0] == f"{kind}, {x}", col)
            add((kind, arm), x, n, text)
    for x in (1200, 1800):
        n, text = lookup(factor, lambda c, x=x: c[0] == f"{x} puzzles", "matched")
        add(("attributes", "same"), x, n, text)
    n, line = next((n, l) for n, l in sec if "separate at 200, seeds 1–3:" in l)
    m = re.search(r"separate at 200, seeds 1–3: ([\d. /]+), \*\*(\d+\.\d) ± (\d+\.\d)\*\*", line)
    runs = [float(v) for v in m.group(1).split("/")]
    pts.setdefault(("components", "same"), []).append((200, float(m.group(2)), float(m.group(3)), runs, n,
                                                       m.group(0).replace("**", ""), ""))
    for k in pts:
        pts[k].sort()
    return pts


EXPECTED = {("components", "shared", 100): (88.0, 1.9), ("components", "same", 200): (88.1, 0.2),
            ("attributes", "shared", 600): (66.2, 2.7), ("attributes", "shared", 6000): (94.2, 0.2),
            ("attributes", "same", 1200): (62.6, 0.6), ("attributes", "same", 1800): (69.8, 0.9)}
ARMS = {"shared": ("shared weights: the symmetry built in", INK, "-", "o", INK),
        "same": ("separate weights, same widths", INK2, (0, (5, 2)), "s", "white"),
        "matched": ("separate weights, matched weight count", MUTED, (0, (1.2, 1.6)), "^", "white")}


def fig_data_factor():
    pts = data_factor_points()
    print(f"  numbers from {RESULTS} §7 (line: plotted value, from the cell):")
    for (kind, arm), ps in sorted(pts.items()):
        for x, mean, sd, runs, n, src, note in ps:
            check = ""
            if runs and sd is not None:       # the runs are printed rounded, so their mean may be off by 0.1
                ok = abs(statistics.mean(runs) - mean) < 0.101 and abs(statistics.stdev(runs) - sd) < 0.101
                check = f"; runs {' / '.join(map(str, runs))} give {statistics.mean(runs):.2f} ± " \
                        f"{statistics.stdev(runs):.2f} {'(agrees to rounding)' if ok else '(DISAGREES)'}"
            exp = EXPECTED.get((kind, arm, x))
            if exp and (abs(exp[0] - mean) > 1e-9 or abs(exp[1] - (sd or 0)) > 1e-9):
                check += f"; NOTE: the brief said {exp[0]} ± {exp[1]}"
            val = f"{mean} ± {sd}" if sd is not None else f"{mean} (one run)"
            print(f"    {kind:<10} {ARMS[arm][0]:<40} {x:>5}: {val:<16} line {n}: '{src}'{check}")

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    spec = {"components": ("Components: I-RAVEN left-right and up-down", "training puzzles per layout",
                           (70, 101.5), [100, 200, 1000, 6000]),
            "attributes": ("Attributes: I-RAVEN-X, rows of ten, values 0–50", "training puzzles",
                           (48, 97), [600, 1200, 1800, 6000])}
    for ax, kind in zip(axes, ("components", "attributes")):
        title, xlabel, ylim, ticks = spec[kind]
        ax.set_xscale("log")
        ax.set_xlim(ticks[0] / 1.35, ticks[-1] * 1.35)
        ax.set_ylim(*ylim)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.grid(axis="y", color=FAINT, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2.5, pad=3)
        ax.set_title(title, loc="left", fontsize=8, pad=6)
        ax.set_xlabel(xlabel)
        for arm in ("matched", "same", "shared"):
            ps = pts.get((kind, arm), [])
            if not ps:
                continue
            label, color, ls, marker, face = ARMS[arm]
            x = [p[0] for p in ps]
            y = [p[1] for p in ps]
            ax.plot(x, y, color=color, ls=ls, lw=1.1, zorder=3)
            for xi, yi, sd, *_ in ps:
                if sd is not None:
                    ax.errorbar([xi], [yi], yerr=[sd], fmt="none", ecolor=color, elinewidth=0.9, capsize=2.2,
                                capthick=0.9, zorder=4)
            ax.plot(x, y, ls="none", marker=marker, ms=4.6, mfc=face, mec=color, mew=1.0, zorder=5)
            for xi, yi, sd, runs, n, src, note in ps:
                if note:                       # a run stopped while its selection score was still rising
                    ax.text(xi * 1.1, yi, r"$\uparrow$", ha="left", va="center", fontsize=8, color=color)
    # the factor, as the runs count it
    sh = {p[0]: p[1] for p in pts[("components", "shared")]}
    axc = axes[0]
    axc.annotate("", xy=(200, sh[100]), xytext=(100, sh[100]),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=7, shrinkA=4, shrinkB=4))
    axc.text(141, sh[100] + 2.2, "twice the data", ha="center", va="bottom", fontsize=6.5)
    sa = {p[0]: p[1] for p in pts[("attributes", "shared")]}
    se = {p[0]: p[1] for p in pts[("attributes", "same")]}
    cross = 1200 + 600 * (sa[600] - se[1200]) / (se[1800] - se[1200])     # read linearly between 1200 and 1800
    print(f"    attributes: separate weights reach shared-at-600 ({sa[600]}) at about {cross:.0f} puzzles, "
          f"read linearly between 1200 and 1800 (factor {cross / 600:.1f})")
    axa = axes[1]
    axa.annotate("", xy=(cross, sa[600]), xytext=(600, sa[600]),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=7, shrinkA=4, shrinkB=0))
    axa.text(560, sa[600] + 4.5, "2 to 3 times the data", ha="left", va="bottom", fontsize=6.5)
    axes[0].set_ylabel("matched on unseen puzzles (%)")
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=1.1, marker=m, ms=4.6, mfc=f, mec=c, mew=1.0)
               for _, c, ls, m, f in ARMS.values()]
    labels = [v[0] for v in ARMS.values()]
    handles.append(plt.Line2D([], [], color=INK2, ls="none", marker="|", ms=7, mew=0.9))
    labels.append("bars: ± sd over three runs; no bar: one run")
    handles.append(plt.Line2D([], [], color=INK2, ls="none", marker=r"$\uparrow$", ms=6))
    labels.append("still rising when training stopped (epoch 60)")
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7, handlelength=2.8,
               columnspacing=1.6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=2.0)
    save(fig, "data_factor")
    CAPTIONS["data_factor"] = (
        "The data factor of a symmetry. Matched % on unseen puzzles against the number of training puzzles, for the "
        "facet model with the symmetry built in (one set of weights shared by the two components, or by the three "
        "attributes) and without it (separate weights, at the same widths or at a matched weight count). Separate "
        "weights need twice the data for the components (the order of the swap group) and two to three times for the "
        "attributes, as counted before the runs (RESULTS §7).")


# ------------------------------------------------------------------------------------------------------------ main
FIGURES = {"iraven_puzzle": fig_iraven_puzzle, "iravenx_puzzle": fig_iravenx_puzzle, "facet_graph": fig_facet_graph,
           "grounding": fig_grounding, "data_factor": fig_data_factor, "iraven_transposed": fig_iraven_transposed}


def main():
    names = sys.argv[1:] or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        sys.exit(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")
    for name in names:
        print(f"{name}:")
        FIGURES[name]()
    print("\nSuggested captions (the chapters may word them differently):")
    for name in names:
        if name in CAPTIONS:
            print(f"  {name}: {CAPTIONS[name]}")


if __name__ == "__main__":
    main()
