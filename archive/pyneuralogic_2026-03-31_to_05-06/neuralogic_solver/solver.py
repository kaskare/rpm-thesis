"""
Logic-rule RPM solvers: hard (deterministic) and soft (differentiable).

The solvers implement *explicit relational reasoning* over the RPM grid:

1.  **Rule detection** — for each (position, attribute), determine which
    ternary rules are consistent with the fully-known context rows.
2.  **Specificity filtering** — prefer more constrained rules (e.g.
    progression over distribute_three) to eliminate false positives.
3.  **Candidate scoring** — for each of the 8 candidates, complete the
    answer row and verify the detected rules hold.

Key advantages over binary-GNN message passing:
  • k-ary constraints (ternary row triples) are first-class
  • variable binding (equality, ordering) is exact, not approximated
  • built-in constraint vocabulary (constant, progression, alldiff, …)
  • per-attribute / per-position reasoning is compositional & independent
"""

import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ANSWER_COL_PREFIX,
    ANSWER_ROW_PREFIX,
    ATTR_DIMS,
    ATTR_NAMES,
    CONTEXT_COLS,
    CONTEXT_ROWS,
    NUM_CANDIDATES,
    NUM_CONTEXT,
    NUM_POSITIONS,
    NUM_RULES,
    RULE_NAMES,
)
from .rules import check_rule, detect_rules, soft_all_rules

# ── Rule specificity tiers (lower = more specific / more constrained) ──
_RULE_PRIORITY = {
    "constant": 0,
    "progression_p1": 1, "progression_m1": 1,
    "progression_p2": 2, "progression_m2": 2,
    "arithmetic_plus": 3, "arithmetic_minus": 3,
    "arithmetic_plus_1": 3, "arithmetic_minus_1": 3,
    "distribute_three": 4,
}


def _filter_by_specificity(rules: Set[str]) -> Set[str]:
    """Keep only the most specific tier of detected rules."""
    if not rules:
        return rules
    best = min(_RULE_PRIORITY[r] for r in rules)
    return {r for r in rules if _RULE_PRIORITY[r] == best}


# ═══════════════════════════════════════════════════════════════════════
#  Hard Logic Solver  (deterministic, no learnable parameters)
# ═══════════════════════════════════════════════════════════════════════

class HardLogicSolver:
    """Deterministic RPM solver using hard (boolean) rule checking.

    With correct rule detection and GT attributes this should approach
    100 % accuracy.

    Parameters
    ----------
    use_cols : bool
        Also detect / check rules along columns (experimental; RAVEN rules
        are generated per-row, so column patterns are not guaranteed).
    use_count : bool
        Check object-count patterns across rows (helps distribute_four/nine
        where the Number rule determines how many objects appear per panel).
    """

    def __init__(self, use_cols: bool = False, use_count: bool = True):
        self.use_cols = use_cols
        self.use_count = use_count

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _occupied(obj_exists: torch.Tensor, panels: List[int], pos: int) -> bool:
        return all(bool(obj_exists[p, pos].item()) for p in panels)

    @staticmethod
    def _triple(attr: torch.Tensor, panels: List[int], pos: int) -> Tuple[int, int, int]:
        return tuple(int(attr[p, pos].item()) for p in panels)

    @staticmethod
    def _count(obj_exists: torch.Tensor, panel: int) -> int:
        return int(obj_exists[panel].sum().item())

    # ── rule detection ───────────────────────────────────────────────

    def _detect_attr_rules(
        self,
        attr: torch.Tensor,
        obj_exists: torch.Tensor,
        context_lines: List[List[int]],
        pos: int,
    ) -> Tuple[Set[str], Optional[FrozenSet[int]]]:
        """Detect attribute rules from context lines at a position.

        Returns (detected_rules, distribute_set) where distribute_set is
        the expected value set if distribute_three is detected, else None.
        """
        triples = []
        for line in context_lines:
            if not self._occupied(obj_exists, line, pos):
                return set(), None
            triples.append(self._triple(attr, line, pos))

        detected = detect_rules(triples)

        # Validate distribute_three: same set of 3 distinct values across rows
        dist_set: Optional[FrozenSet[int]] = None
        if "distribute_three" in detected:
            row_sets = [frozenset(t) for t in triples]
            if all(len(s) == 3 for s in row_sets) and len(set(row_sets)) == 1:
                dist_set = row_sets[0]
            else:
                detected.discard("distribute_three")

        detected = _filter_by_specificity(detected)

        # Preserve dist_set only if distribute_three survived filtering
        if "distribute_three" not in detected:
            dist_set = None

        return detected, dist_set

    def _detect_count_rules(
        self,
        obj_exists: torch.Tensor,
        context_lines: List[List[int]],
    ) -> Set[str]:
        """Detect rules on object counts across context rows."""
        triples = []
        for line in context_lines:
            triple = tuple(self._count(obj_exists, p) for p in line)
            triples.append(triple)
        return _filter_by_specificity(detect_rules(triples))

    # ── scoring ──────────────────────────────────────────────────────

    def _score_attr(
        self,
        attr: torch.Tensor,
        obj_exists: torch.Tensor,
        prefix: List[int],
        cand_panel: int,
        detected: Set[str],
        pos: int,
        dist_set: Optional[FrozenSet[int]],
    ) -> int:
        """Score answer line: 1 if ALL detected rules satisfied, else 0."""
        if not detected:
            return 0
        answer = prefix + [cand_panel]
        if not self._occupied(obj_exists, answer, pos):
            return 0

        v1, v2, v3 = self._triple(attr, answer, pos)

        for rule in detected:
            if rule == "distribute_three" and dist_set is not None:
                if frozenset([v1, v2, v3]) != dist_set:
                    return 0
            elif not check_rule(rule, v1, v2, v3):
                return 0
        return 1

    def _score_count(
        self,
        obj_exists: torch.Tensor,
        prefix: List[int],
        cand_panel: int,
        detected: Set[str],
    ) -> int:
        if not detected:
            return 0
        v1, v2, v3 = (self._count(obj_exists, p) for p in prefix + [cand_panel])
        for rule in detected:
            if not check_rule(rule, v1, v2, v3):
                return 0
        return 1

    # ── main entry points ────────────────────────────────────────────

    def score_candidates(
        self,
        obj_type: torch.Tensor,
        obj_size: torch.Tensor,
        obj_color: torch.Tensor,
        obj_exists: torch.Tensor,
    ) -> List[int]:
        """Score all 8 candidates for a single sample.

        Parameters
        ----------
        obj_type, obj_size, obj_color : Tensor [16, 22]
        obj_exists : Tensor [16, 22]  (bool)

        Returns
        -------
        list of 8 ints — higher is better.
        """
        attrs = {"type": obj_type, "size": obj_size, "color": obj_color}
        scores = [0] * NUM_CANDIDATES

        # ── Attribute-value rules (row direction) ────────────────────
        for attr_name in ATTR_NAMES:
            a = attrs[attr_name]
            for pos in range(NUM_POSITIONS):
                detected, dist_set = self._detect_attr_rules(
                    a, obj_exists, CONTEXT_ROWS, pos,
                )
                if not detected:
                    continue
                for ci in range(NUM_CANDIDATES):
                    scores[ci] += self._score_attr(
                        a, obj_exists,
                        ANSWER_ROW_PREFIX, NUM_CONTEXT + ci,
                        detected, pos, dist_set,
                    )

        # ── Attribute-value rules (column direction, optional) ───────
        if self.use_cols:
            for attr_name in ATTR_NAMES:
                a = attrs[attr_name]
                for pos in range(NUM_POSITIONS):
                    detected, dist_set = self._detect_attr_rules(
                        a, obj_exists, CONTEXT_COLS, pos,
                    )
                    if not detected:
                        continue
                    for ci in range(NUM_CANDIDATES):
                        scores[ci] += self._score_attr(
                            a, obj_exists,
                            ANSWER_COL_PREFIX, NUM_CONTEXT + ci,
                            detected, pos, dist_set,
                        )

        # ── Object-count rules (row direction) ───────────────────────
        if self.use_count:
            count_rules = self._detect_count_rules(obj_exists, CONTEXT_ROWS)
            if count_rules:
                for ci in range(NUM_CANDIDATES):
                    scores[ci] += self._score_count(
                        obj_exists, ANSWER_ROW_PREFIX,
                        NUM_CONTEXT + ci, count_rules,
                    )

        return scores

    def predict(
        self,
        obj_type: torch.Tensor,
        obj_size: torch.Tensor,
        obj_color: torch.Tensor,
        obj_exists: torch.Tensor,
    ) -> int:
        """Return the candidate index (0-7) with the highest score.

        Ties are broken randomly to avoid a systematic bias toward index 0.
        """
        scores = self.score_candidates(obj_type, obj_size, obj_color, obj_exists)
        best = max(scores)
        tied = [i for i in range(NUM_CANDIDATES) if scores[i] == best]
        return random.choice(tied)

    def predict_batch(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        preds = [self.predict(*sample) for sample in batch]
        return torch.tensor(preds, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════
#  Soft Logic Solver  (differentiable nn.Module)
# ═══════════════════════════════════════════════════════════════════════

class SoftLogicSolver(nn.Module):
    """Differentiable RPM solver on soft attribute distributions.

    Computes *satisfaction probabilities* for each rule then aggregates
    across positions, attributes, and rules to score each candidate.

    Learnable parameters (minimal — preserves logic-rule inductive bias):
      • ``rule_weight``   [NUM_RULES] — importance of each rule type
      • ``attr_weight``   [3]         — importance of each attribute
    """

    def __init__(self):
        super().__init__()
        self.rule_weight = nn.Parameter(torch.ones(NUM_RULES))
        self.attr_weight = nn.Parameter(torch.ones(len(ATTR_NAMES)))

    def _context_rule_scores(
        self,
        attr_dist: torch.Tensor,
        context_lines: List[List[int]],
        pos: int,
    ) -> torch.Tensor:
        """Geometric-mean rule scores over context lines → [NUM_RULES]."""
        line_scores = []
        for line in context_lines:
            p1, p2, p3 = (attr_dist[p, pos] for p in line)
            line_scores.append(soft_all_rules(p1, p2, p3))
        return torch.stack(line_scores, dim=0).prod(dim=0)

    def _answer_rule_scores(
        self,
        attr_dist: torch.Tensor,
        prefix: List[int],
        cand_panel: int,
        pos: int,
    ) -> torch.Tensor:
        p1, p2, p3 = attr_dist[prefix[0], pos], attr_dist[prefix[1], pos], attr_dist[cand_panel, pos]
        return soft_all_rules(p1, p2, p3)

    def forward(
        self,
        attr_dists: Dict[str, torch.Tensor],
        obj_exists: torch.Tensor,
    ) -> torch.Tensor:
        """Score 8 candidates.  Returns [8] logits."""
        rw = F.softmax(self.rule_weight, dim=0)
        aw = F.softmax(self.attr_weight, dim=0)

        scores = torch.zeros(NUM_CANDIDATES, device=rw.device)

        for ai, attr_name in enumerate(ATTR_NAMES):
            ad = attr_dists[attr_name]
            for pos in range(NUM_POSITIONS):
                if not all(obj_exists[p, pos] > 0.5 for line in CONTEXT_ROWS for p in line):
                    continue
                ctx = self._context_rule_scores(ad, CONTEXT_ROWS, pos)

                for ci in range(NUM_CANDIDATES):
                    cp = NUM_CONTEXT + ci
                    panels = ANSWER_ROW_PREFIX + [cp]
                    if not all(obj_exists[p, pos] > 0.5 for p in panels):
                        continue
                    ans = self._answer_rule_scores(ad, ANSWER_ROW_PREFIX, cp, pos)
                    scores[ci] = scores[ci] + aw[ai] * (ctx * ans * rw).sum()

        return scores

    @torch.no_grad()
    def predict(self, attr_dists, obj_exists) -> int:
        return int(self.forward(attr_dists, obj_exists).argmax().item())


# ═══════════════════════════════════════════════════════════════════════
#  Utility:  convert hard attributes to one-hot distributions
# ═══════════════════════════════════════════════════════════════════════

def attrs_to_dists(
    obj_type: torch.Tensor,
    obj_size: torch.Tensor,
    obj_color: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Integer attribute tensors [16, 22] → one-hot [16, 22, D]."""
    return {
        "type":  F.one_hot(obj_type.long(),  ATTR_DIMS["type"]).float(),
        "size":  F.one_hot(obj_size.long(),  ATTR_DIMS["size"]).float(),
        "color": F.one_hot(obj_color.long(), ATTR_DIMS["color"]).float(),
    }
