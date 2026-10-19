"""Datasets, parsing, and position centers for RAVEN GNN."""

import glob
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    CONFIG_POS_MAP,
    NUM_CANDIDATES,
    NUM_COLOR,
    NUM_CONTEXT,
    NUM_POSITIONS,
    NUM_SHAPE,
    NUM_SIZE,
)


def _parse_bbox(raw: str) -> Tuple[float, float, float, float]:
    vals = eval(raw)
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def _bbox_key(b: Tuple[float, float, float, float], decimals: int = 3) -> Tuple[float, float, float, float]:
    return tuple(round(v, decimals) for v in b)


def discover_xml_paths(dataset_root: str, split: str) -> List[str]:
    npz_files = sorted(glob.glob(os.path.join(dataset_root, "*", f"*_{split}.npz")))
    xml_files = [p.replace(".npz", ".xml") for p in npz_files]
    return [p for p in xml_files if os.path.isfile(p)]


def get_config_from_path(npz_path: str) -> str:
    """Extract constellation config from path, e.g. .../distribute_four/RAVEN_123_train.npz -> distribute_four."""
    return os.path.basename(os.path.dirname(npz_path))


_CANONICAL_POSITIONS: Dict[str, Dict[Tuple[int, Tuple[float, ...]], int]] = {}


def canonical_positions(config: str, dataset_root: str) -> Dict[Tuple[int, Tuple[float, ...]], int]:
    """(component_index, bbox) -> canonical local slot index for `config`.

    The XML `Position` attribute lists ONLY the slots a panel actually occupies,
    so `all_positions.index(bbox)` is an index into what happens to be present --
    its meaning shifts with the object count. For any constellation whose panels
    hold a varying number of objects (distribute_four, distribute_nine,
    in_distribute_four_out_center_single) that destroys position identity: a lone
    object at grid slot 3 and a lone object at grid slot 0 both come out as local
    index 0, so different candidate panels become indistinguishable. Measured on
    I-RAVEN: ~32% of puzzles in those three constellations had two or more
    identical candidate encodings, making them unanswerable.

    Fix: derive the slot from the bbox GEOMETRY, using the union of every bbox
    that appears in that component across the dataset. Verified to recover exactly
    len(CONFIG_POS_MAP[config]) slots for all seven constellations.
    """
    if config in _CANONICAL_POSITIONS:
        return _CANONICAL_POSITIONS[config]
    seen: Dict[int, set] = {}
    for path in sorted(glob.glob(os.path.join(dataset_root, config, "*_train.xml")))[:200]:
        for panel in ET.parse(path).getroot()[0]:
            for struct in panel:
                for ci, comp in enumerate(struct):
                    for layout in comp:
                        for bbox in eval(layout.attrib["Position"]):
                            seen.setdefault(ci, set()).add(tuple(bbox))
    mapping: Dict[Tuple[int, Tuple[float, ...]], int] = {}
    offset = 0
    for ci in sorted(seen):
        for k, bbox in enumerate(sorted(seen[ci])):
            mapping[(ci, bbox)] = offset + k
        offset += len(seen[ci])
    expected = len(CONFIG_POS_MAP[config])
    if offset != expected:
        raise RuntimeError(f"{config}: canonical map found {offset} slots, expected {expected}")
    _CANONICAL_POSITIONS[config] = mapping
    return mapping


def parse_ground_truth(xml_path: str, config: str) -> List[List[Dict[str, int]]]:
    """
    Parse RAVEN XML → per-panel list of ground truth objects.
    Positions are mapped to universal 0–21 via CONFIG_POS_MAP.
    Returns 16 lists, each [{position, type, size, color}, ...].
    """
    if config not in CONFIG_POS_MAP:
        raise KeyError(f"Unknown config '{config}'; known: {list(CONFIG_POS_MAP.keys())}")
    pos_map = CONFIG_POS_MAP[config]
    dataset_root = os.path.dirname(os.path.dirname(xml_path))
    canon = canonical_positions(config, dataset_root)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_panels = root[0]
    gt_panels = []
    for panel in xml_panels:
        panel_objects = []
        for struct in panel:
            for ci, comp in enumerate(struct):
                for layout in comp:
                    for entity in layout:
                        bbox = eval(entity.attrib["bbox"])
                        local_pos = canon[(ci, tuple(bbox))]
                        pos = pos_map[local_pos]
                        panel_objects.append({
                            "position": pos,
                            "type": int(entity.attrib["Type"]) - 1,  # XML 1-indexed → 0-indexed
                            "size": int(entity.attrib["Size"]),
                            "color": int(entity.attrib["Color"]),
                        })
        gt_panels.append(sorted(panel_objects, key=lambda o: o["position"]))
    return gt_panels


def gt_to_dense(gt_panels: List[List[Dict[str, int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert parsed GT (16 panels) to (obj_type, obj_size, obj_color, obj_exists) [16, 22] tensors."""
    obj_type = torch.zeros(NUM_CONTEXT + NUM_CANDIDATES, NUM_POSITIONS, dtype=torch.long)
    obj_size = torch.zeros(NUM_CONTEXT + NUM_CANDIDATES, NUM_POSITIONS, dtype=torch.long)
    obj_color = torch.zeros(NUM_CONTEXT + NUM_CANDIDATES, NUM_POSITIONS, dtype=torch.long)
    obj_exists = torch.zeros(NUM_CONTEXT + NUM_CANDIDATES, NUM_POSITIONS, dtype=torch.bool)
    for p, objs in enumerate(gt_panels):
        for o in objs:
            pos = o["position"]
            if 0 <= pos < NUM_POSITIONS:
                obj_type[p, pos] = min(o["type"], NUM_SHAPE - 1)
                obj_size[p, pos] = min(o["size"], NUM_SIZE - 1)
                obj_color[p, pos] = min(o["color"], NUM_COLOR - 1)
                obj_exists[p, pos] = True
    return obj_type, obj_size, obj_color, obj_exists


def build_position_centers(xml_paths: Sequence[str], max_positions: int = NUM_POSITIONS) -> Dict[int, Tuple[float, float]]:
    bboxes = set()
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        xml_panels = root[0]
        for panel in xml_panels:
            for struct in panel:
                for comp in struct:
                    for layout in comp:
                        for entity in layout:
                            bboxes.add(_bbox_key(_parse_bbox(entity.attrib["bbox"])))
    ordered = sorted(bboxes, key=lambda t: (t[1], t[0], t[2], t[3]))
    if not ordered:
        raise RuntimeError("No positions discovered from XML files.")
    if len(ordered) > max_positions:
        raise RuntimeError(f"Discovered {len(ordered)} positions > {max_positions}.")
    centers: Dict[int, Tuple[float, float]] = {}
    for i, (x, y, w, h) in enumerate(ordered):
        centers[i] = (x + 0.5 * w, y + 0.5 * h)
    return centers


def universal_position_centers(dataset_root: str) -> Dict[int, Tuple[float, float]]:
    """(x, y) centre of every UNIVERSAL position id (0-21), from bbox geometry.

    This is what gnn/graph.py::infer_relations needs: it looks centres up by the
    object's universal position id. `build_position_centers` above keys them by the
    RANK of each bbox in a sorted list instead -- a different numbering -- so every
    run that used it computed within-panel relations from the wrong objects' centres
    (measured on I-RAVEN: 33-100% of relations wrong in five of the six multi-object
    constellations; left/right happened to come out right). It also treated the
    centre-based bboxes as corner-based. Kept only so old checkpoints reproduce.

    RAVEN bboxes are centre-based [y_centre, x_centre, ...] (checked against the
    rendered pixel masks), so the centre is (bbox[1], bbox[0]).
    """
    centers: Dict[int, Tuple[float, float]] = {}
    for config, pos_map in CONFIG_POS_MAP.items():
        for (_, bbox), local in canonical_positions(config, dataset_root).items():
            xy = (float(bbox[1]), float(bbox[0]))
            u = pos_map[local]
            if centers.setdefault(u, xy) != xy:
                raise RuntimeError(f"universal position {u}: {centers[u]} vs {xy} ({config})")
    if sorted(centers) != list(range(NUM_POSITIONS)):
        raise RuntimeError(f"expected universal positions 0..{NUM_POSITIONS - 1}, got {sorted(centers)}")
    return centers


def build_position_centers_universal(xml_paths: Sequence[str], config: str) -> Dict[int, Tuple[float, float]]:
    """Like build_position_centers but keys by universal position (0-21) for infer_relations."""
    raw = build_position_centers(xml_paths)
    pos_map = CONFIG_POS_MAP.get(config)
    if not pos_map:
        return raw
    return {pos_map[i]: raw[i] for i in range(len(raw)) if i in pos_map}


def fallback_position_centers(max_positions: int = NUM_POSITIONS) -> Dict[int, Tuple[float, float]]:
    centers = {}
    cols = 6
    for i in range(max_positions):
        centers[i] = (float(i % cols), float(i // cols))
    return centers


class RavenImageDataset(Dataset):
    def __init__(self, dataset_root: str, split: str) -> None:
        self.files = sorted(glob.glob(os.path.join(dataset_root, "*", f"*_{split}.npz")))
        if not self.files:
            raise FileNotFoundError(f"No files for split={split} in {dataset_root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx])
        images = torch.tensor(data["image"], dtype=torch.float32)  # [16,H,W]
        target = torch.tensor(int(data["target"]), dtype=torch.long)
        return images, target


class RavenGTDataset(Dataset):
    """Dataset that yields ground truth objects from XML (no frontend). Much faster than image-based extraction."""

    def __init__(self, dataset_root: str, split: str) -> None:
        self.files = sorted(glob.glob(os.path.join(dataset_root, "*", f"*_{split}.npz")))
        if not self.files:
            raise FileNotFoundError(f"No files for split={split} in {dataset_root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        npz_path = self.files[idx]
        data = np.load(npz_path)
        target = torch.tensor(int(data["target"]), dtype=torch.long)
        xml_path = npz_path.replace(".npz", ".xml")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"XML not found: {xml_path}")
        config = get_config_from_path(npz_path)
        gt_panels = parse_ground_truth(xml_path, config)
        obj_type, obj_size, obj_color, obj_exists = gt_to_dense(gt_panels)
        return obj_type, obj_size, obj_color, obj_exists, target


def collate_images(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(targets, dim=0)


def collate_gt(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Collate GT batches into (extracted_batch, targets)."""
    extracted = [(o[0], o[1], o[2], o[3]) for o in batch]
    targets = torch.stack([o[4] for o in batch], dim=0)
    return extracted, targets
