#!/usr/bin/env python3
"""Symbolic view of I-RAVEN puzzles, and the semantics of RAVEN's rules.

A puzzle is turned into, per panel and per component:
    number    how many objects                       (int)
    position  which canonical slots are occupied     (frozenset of slot ids, local to the component)
    type/size/color  the objects' values, as a sorted tuple (one value if the panel is uniform)

Rows are (0,1,2), (3,4,5) and (6,7,X) where X is a candidate (panel 8+c).

This module only reads ground truth from the XML; it is the basis for the reference
solvers and for the symbolic PyNeuraLogic templates.
"""
import glob, os, sys, xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path.insert(0, ROOT)
from gnn.data import canonical_positions

CONSTELLATIONS = [
    "center_single", "distribute_four", "distribute_nine",
    "left_center_single_right_center_single", "up_center_single_down_center_single",
    "in_center_single_out_center_single", "in_distribute_four_out_center_single",
]
ENTITY_ATTRS = ("type", "size", "color")
LAYOUT_ATTRS = ("number", "position")


def component_offsets(cfg, dataset_root):
    """canonical slot ids are global across components; return each component's first id."""
    canon = canonical_positions(cfg, dataset_root)
    off = {}
    for (ci, _), k in canon.items():
        off[ci] = min(off.get(ci, k), k)
    return canon, off


def read_puzzle(xml_path, cfg, dataset_root=None):
    """-> (panels, rules)
    panels[p][ci] = {"number", "position", "type", "size", "color"}
    rules[ci]     = {attr: rule_name}   (attr in Number, Position, Number/Position, Type, Size, Color)
    """
    dataset_root = dataset_root or os.path.dirname(os.path.dirname(xml_path))
    canon, off = component_offsets(cfg, dataset_root)
    nslots = Counter(ci for (ci, _) in canon)
    root = ET.parse(xml_path).getroot()
    panels = []
    for panel in root.find("Panels"):
        comps = []
        for s in panel:
            for ci, comp in enumerate(s):
                for layout in comp:
                    ents = list(layout)
                    per_slot = {canon[(ci, tuple(eval(e.attrib["bbox"])))] - off[ci]:
                                (int(e.attrib["Type"]) - 1, int(e.attrib["Size"]), int(e.attrib["Color"]))
                                for e in ents}
                    comps.append({
                        "number": len(ents),
                        "position": frozenset(per_slot),
                        "type": tuple(sorted(v[0] for v in per_slot.values())),
                        "size": tuple(sorted(v[1] for v in per_slot.values())),
                        "color": tuple(sorted(v[2] for v in per_slot.values())),
                        "slots": per_slot,                 # slot -> (type, size, color)
                        "nslots": nslots[ci],
                        "uniformity": int(layout.attrib.get("Uniformity", 0)),
                    })
        panels.append(comps)
    rules = [{r.attrib["attr"]: r.attrib["name"] for r in g} for g in root.find("Rules")]
    return panels, rules


def rows_for(panels, answer_panel):
    return [(panels[0], panels[1], panels[2]), (panels[3], panels[4], panels[5]),
            (panels[6], panels[7], panels[answer_panel])]


def files(cfg, split, n=None, root="data/I-RAVEN-10000"):
    return sorted(glob.glob(os.path.join(ROOT, root, cfg, f"*_{split}.npz")))[:n]
