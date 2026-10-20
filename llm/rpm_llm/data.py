"""Puzzle discovery and sampling from I-RAVEN style .npz files.

Accepted layouts (searched recursively):
  <root>/<configuration>/RAVEN_<i>_<split>.npz              (standard I-RAVEN)
  <root>/<variant>/<configuration>/RAVEN_<i>_<split>.npz    (Tehlar's 3_comp / 4_comp variants)
The answer is the npz key `target`. The key `predict` is the output of the generator's
built-in solver and must not be used as ground truth.
"""
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CONFIGS = (
    "center_single", "distribute_four", "distribute_nine",
    "left_center_single_right_center_single", "up_center_single_down_center_single",
    "in_center_single_out_center_single", "in_distribute_four_out_center_single",
)
LETTERS = "ABCDEFGH"
_NAME = re.compile(r"RAVEN_(\d+)_(train|val|test)\.npz$")


@dataclass(frozen=True)
class Puzzle:
    id: str
    variant: str
    config: str
    split: str
    path: str
    target: int

    @property
    def answer(self) -> str:
        return LETTERS[self.target]


def discover(root, splits=("test",)):
    root = Path(root)
    out = []
    for p in sorted(root.rglob("RAVEN_*.npz")):
        m = _NAME.search(p.name)
        if not m or m.group(2) not in splits:
            continue
        config = p.parent.name
        if config not in CONFIGS:
            continue
        rel = p.parent.parent.relative_to(root).parts
        variant = "/".join(x for x in rel if x != "input_data")
        pid = "/".join(x for x in (variant, config, p.stem) if x)
        out.append((pid, variant, config, m.group(2), p))
    return out


def load_puzzle(pid, variant, config, split, path):
    with np.load(path) as d:
        target = int(d["target"])
    return Puzzle(pid, variant, config, split, str(path), target)


def sample(root, splits=("test",), per_config=None, seed=0):
    """Deterministic sample of `per_config` puzzles per (variant, configuration); all if None."""
    groups = defaultdict(list)
    for item in discover(root, splits):
        groups[(item[1], item[2])].append(item)
    rng = random.Random(seed)
    chosen = []
    for key in sorted(groups):
        items = groups[key]
        if per_config is not None and per_config < len(items):
            items = sorted(rng.sample(items, per_config))
        chosen.extend(items)
    return [load_puzzle(*it) for it in chosen]


def save_manifest(puzzles, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in puzzles:
            f.write(json.dumps(asdict(p)) + "\n")


def load_manifest(path):
    with open(path) as f:
        return [Puzzle(**json.loads(line)) for line in f if line.strip()]
