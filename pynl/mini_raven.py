"""Tiny I-RAVEN-like generator for prototyping (not the real dataset).
Grids 2x2 (positions 0-3) and 3x3 (positions 4-12); attributes uniform within a panel;
row rules on layout (constant, number progression, position progression, number distribute-three,
position union) and on type/size/colour (constant, progression, distribute-three);
8 candidates from an I-RAVEN-style 3-level modification tree."""
import random

GRIDS = {"2x2": list(range(0, 4)), "3x3": list(range(4, 13))}
RANGE = {"type": 5, "size": 6, "color": 10}
LAYOUT_RULES = ["const", "num_prog", "pos_prog", "num_d3", "pos_union"]
ATTR_RULES = ["const", "prog", "d3"]


def _layout_rows(rng, slots, rule):
    n = len(slots)
    rows = []
    if rule == "num_d3":
        counts = rng.sample(range(1, n + 1), 3)
    if rule == "num_prog":
        d = rng.choice([-1, 1])
        lo, hi = (1, n - 2) if d == 1 else (3, n)
        start = rng.randint(lo, hi)
    for r in range(3):
        if rule == "const":
            s = frozenset(rng.sample(slots, rng.randint(1, n)))
            row = [s, s, s]
        elif rule == "num_prog":
            row = [frozenset(rng.sample(slots, start + d * c)) for c in range(3)]
        elif rule == "pos_prog":
            k = rng.randint(1, n - 1)
            idx = rng.sample(range(n), k)
            row = [frozenset(slots[(i + c) % n] for i in idx) for c in range(3)]
        elif rule == "num_d3":
            cs = counts[r:] + counts[:r]
            row = [frozenset(rng.sample(slots, c)) for c in cs]
        else:  # pos_union
            a = frozenset(rng.sample(slots, rng.randint(1, n - 1)))
            b = frozenset(rng.sample(slots, rng.randint(1, n - 1)))
            row = [a, b, a | b]
        rows.append(row)
    return rows


def _attr_rows(rng, name, rule):
    m = RANGE[name]
    if rule == "d3":
        vals = rng.sample(range(m), 3)
        return [vals[r:] + vals[:r] for r in range(3)]
    rows = []
    for _ in range(3):
        if rule == "const":
            v = rng.randrange(m)
            rows.append([v, v, v])
        else:
            d = rng.choice([-1, 1])
            v = rng.randrange(2, m) if d == -1 else rng.randrange(0, m - 2)
            rows.append([v, v + d, v + 2 * d])
    return rows


def _geometry(pos):
    """(cx, cy, w) of a position in [0, 1] panel coordinates, y downwards (as RAVEN bboxes)."""
    for slots in GRIDS.values():
        if pos in slots:
            n = int(round(len(slots) ** 0.5))
            r, c = divmod(slots.index(pos), n)
            return (c + 0.5) / n, (r + 0.5) / n, 1 / n


def _panel(layout, attrs):
    return [dict(pos=p, type=attrs["type"], size=attrs["size"], color=attrs["color"],
                 **dict(zip(("cx", "cy", "w"), _geometry(p)))) for p in sorted(layout)]


def _modify(rng, cand, what, slots):
    lay, att = cand
    att = dict(att)
    if what == "layout":
        while True:
            new = frozenset(rng.sample(slots, rng.randint(1, len(slots))))
            if new != lay:
                return new, att
    att[what] = rng.choice([v for v in range(RANGE[what]) if v != att[what]])
    return lay, att


def puzzle(rng):
    grid = rng.choice(list(GRIDS))
    slots = GRIDS[grid]
    lrule = rng.choice(LAYOUT_RULES)
    lay = _layout_rows(rng, slots, lrule)
    arules = {a: rng.choice(ATTR_RULES) for a in RANGE}
    att = {a: _attr_rows(rng, a, r) for a, r in arules.items()}
    cells = [(lay[r][c], {a: att[a][r][c] for a in RANGE}) for r in range(3) for c in range(3)]
    answer = cells[8]
    cands = [answer]
    for what in rng.sample(["layout", "type", "size", "color"], 3):  # I-RAVEN tree: 1 -> 2 -> 4 -> 8
        cands += [_modify(rng, c, what, slots) for c in cands]
    order = list(range(8))
    rng.shuffle(order)
    cands = [cands[i] for i in order]
    y = order.index(0)
    panels = [_panel(*c) for c in cells[:8]] + [_panel(*c) for c in cands]
    return dict(grid=grid, layout_rule=lrule, attr_rules=arules, panels=panels, target=y)


def dataset(n, seed):
    rng = random.Random(seed)
    return [puzzle(rng) for _ in range(n)]
