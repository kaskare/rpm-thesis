"""Text input with XML labels: each panel's objects from the I-RAVEN .xml, written out as text.

`parse` and `POS_MAP` are copied from train_pynl.py (keep in sync); the only change is
ast.literal_eval instead of eval. The slot of an object comes from its bbox, not from
the XML `Position` list, which only lists occupied slots (the I-RAVEN position bug).
"""
import ast
import glob
import os
import xml.etree.ElementTree as ET

POS_MAP = {
    "center_single": {0: 0},
    "distribute_four": {0: 1, 1: 2, 2: 3, 3: 4},
    "distribute_nine": {i: 5 + i for i in range(9)},
    "left_center_single_right_center_single": {0: 14, 1: 15},
    "up_center_single_down_center_single": {0: 16, 1: 17},
    "in_center_single_out_center_single": {0: 0, 1: 9},
    "in_distribute_four_out_center_single": {0: 0, 1: 18, 2: 19, 3: 20, 4: 21},
}

_CANON = {}


def _canonical(config, root=None):
    # (component_index, bbox) -> canonical local slot, built from the union of bboxes
    # seen per component in the first 200 *_train.xml files of that constellation
    if config in _CANON:
        return _CANON[config]
    root = root or os.environ.get("IRAVEN_ROOT", "data/I-RAVEN-10000")
    seen = {}
    for f in sorted(glob.glob(os.path.join(root, config, "*_train.xml")))[:200]:
        for panel in ET.parse(f).getroot()[0]:
            for struct in panel:
                for ci, comp in enumerate(struct):
                    for layout in comp:
                        for b in ast.literal_eval(layout.attrib["Position"]):
                            seen.setdefault(ci, set()).add(tuple(b))
    out, off = {}, 0
    for ci in sorted(seen):
        for k, b in enumerate(sorted(seen[ci])):
            out[(ci, b)] = off + k
        off += len(seen[ci])
    if off != len(POS_MAP[config]):
        raise RuntimeError(f"{config}: canonical map found {off} slots, "
                           f"expected {len(POS_MAP[config])}")
    _CANON[config] = out
    return out


def parse(xml_path, config):
    pos_map = POS_MAP[config]
    canon = _canonical(config, os.path.dirname(os.path.dirname(xml_path)))
    panels = []
    for panel in ET.parse(xml_path).getroot()[0]:
        objs = []
        for struct in panel:
            for ci, comp in enumerate(struct):
                for layout in comp:
                    for e in layout:
                        b = ast.literal_eval(e.attrib["bbox"])
                        local = canon[(ci, tuple(b))]
                        objs.append({"pos": pos_map[local],
                                     "type": int(e.attrib["Type"]) - 1,
                                     "size": int(e.attrib["Size"]),
                                     "color": int(e.attrib["Color"]),
                                     # RAVEN bboxes are CENTRE-based, [y_centre, x_centre, ...]
                                     "cx": b[1], "cy": b[0]})
        panels.append(objs)
    return panels


SHAPES = ("triangle", "square", "pentagon", "hexagon", "circle")
_GRID4 = ("top-left", "top-right", "bottom-left", "bottom-right")
_GRID9 = tuple(f"{r}-{c}" for r in ("top", "middle", "bottom") for c in ("left", "centre", "right"))
SLOT_NAMES = {
    "center_single": {0: "centre"},
    "distribute_four": dict(zip(range(1, 5), _GRID4)),
    "distribute_nine": dict(zip(range(5, 14), _GRID9)),
    "left_center_single_right_center_single": {14: "left", 15: "right"},
    "up_center_single_down_center_single": {16: "top", 17: "bottom"},
    "in_center_single_out_center_single": {0: "outer", 9: "inner"},
    "in_distribute_four_out_center_single": {0: "outer", **dict(zip(range(18, 22), ("inner " + g for g in _GRID4)))},
}
LAYOUT_TEXT = {
    "center_single": "one object in the centre",
    "distribute_four": "a 2x2 grid of slots",
    "distribute_nine": "a 3x3 grid of slots",
    "left_center_single_right_center_single": "a left and a right object",
    "up_center_single_down_center_single": "a top and a bottom object",
    "in_center_single_out_center_single": "a large outer object containing a smaller inner object",
    "in_distribute_four_out_center_single": "a large outer object containing a 2x2 grid of inner slots",
}


def describe_panel(objs, config):
    names = SLOT_NAMES[config]
    if not objs:
        return "empty"
    return "; ".join(f"{names[o['pos']]}: {SHAPES[o['type']]}, size {o['size']}, colour {o['color']}"
                     for o in sorted(objs, key=lambda o: o["pos"]))


def describe(xml_path, config):
    panels = parse(xml_path, config)
    if len(panels) != 16:
        raise ValueError(f"{xml_path}: expected 16 panels, got {len(panels)}")
    lines = [f"Layout: {LAYOUT_TEXT[config]}. Empty slots are not listed.", "", "Matrix:"]
    for i in range(8):
        r, c = divmod(i, 3)
        lines.append(f"Row {r + 1}, column {c + 1}: {describe_panel(panels[i], config)}")
    lines.append("Row 3, column 3: ?")
    lines += ["", "Options:"]
    lines += [f"{'ABCDEFGH'[k]}: {describe_panel(panels[8 + k], config)}" for k in range(8)]
    return "\n".join(lines)
