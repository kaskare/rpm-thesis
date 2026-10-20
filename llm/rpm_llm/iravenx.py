"""I-RAVEN-X (Hersche, Camposampiero et al., IBM 2025): symbolic only, no images.

Input is the JSON written by IBM/raven-large-language-models src/datasets/generation/iravenx_task.py:
{"<i>": {"rules": [...], "rpm": n*n+7 panels of [{"Type", "Size", "Color", "Angle", ...}], "target": k}}
with an n x n matrix (n*n-1 context panels) followed by 8 candidates. Panels are written as
(type, size, colour) like IBM's prompts; Angle and confounders are left out. As in IBM's
iravenx.yml, only the last `nshow` rows are shown and the first 500 puzzles are the test set.
"""
import gzip
import json
import math
from functools import lru_cache
from pathlib import Path

from .data import Puzzle

ATTRS = ("Type", "Size", "Color")


@lru_cache(maxsize=4)
def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def grid_size(rpm):
    n = math.isqrt(len(rpm) - 7)
    if n * n + 7 != len(rpm):
        raise ValueError(f"{len(rpm)} panels is not n*n-1 context + 8 candidates")
    return n


def sample(path, first=500):
    data = load(str(path))
    stem = Path(path).name.split(".")[0]
    keys = sorted(data, key=int)[:first]
    return [Puzzle(f"iravenx/{stem}/{k}", stem, "center_single", "test", f"{path}#{k}",
                   int(data[k]["target"])) for k in keys]


def panel(p):
    obj = p[0]
    return "(" + ", ".join(str(int(obj[a])) for a in ATTRS) + ")"


def describe(ref, nshow=3):
    path, key = ref.rsplit("#", 1)
    rpm = load(path)[key]["rpm"]
    n = grid_size(rpm)
    nshow = min(nshow, n)
    ctx = [panel(p) for p in rpm[: n * n - 1]] + ["?"]
    lines = [f"Matrix ({n} columns; the last {nshow} of {n} rows are shown):"]
    for r in range(n - nshow, n):
        lines.append(f"Row {r - (n - nshow) + 1}: " + ", ".join(ctx[r * n:(r + 1) * n]))
    lines += ["", "Options:"]
    lines += [f"{'ABCDEFGH'[k]}: {panel(p)}" for k, p in enumerate(rpm[n * n - 1:])]
    return "\n".join(lines)
