"""
RPM solver using PyNeuraLogic's relational logic programming engine.

Expresses RPM rule constraints (Constant, Progression, Arithmetic,
Distribute Three) as declarative logic rules with variable unification,
successor relations, addition tables, and ``alldiff`` constraints.  The
NeuraLogic inference engine then *grounds* these rules against per-sample
facts and returns all satisfying variable substitutions — no manual loops,
no approximations.

Two solvers are provided:

    ``PynlHardSolver``
        Uses the inference engine for pure constraint satisfaction.
        Equivalent to ``HardLogicSolver`` but the rule checking is
        delegated entirely to NeuraLogic's Java backend.

    ``PynlLearnableSolver``
        Wraps the same rules in a differentiable template with learnable
        rule-importance weights, trainable via NeuraLogic's evaluator or
        the ``NeuraLogic`` PyTorch module.
"""

from collections import defaultdict
import random
from typing import Dict, List, Set, Tuple

import torch

from neuralogic.core import R, V, Template, Settings, Transformation, Aggregation
from neuralogic.dataset import Dataset
from neuralogic.inference.inference_engine import InferenceEngine
from neuralogic.nn import get_evaluator
from neuralogic.optim import Adam

from .config import (
    ATTR_NAMES,
    NUM_CANDIDATES,
    NUM_CONTEXT,
    NUM_POSITIONS,
)

# ── Rule specificity tiers (lower = more constrained) ──────────────
_RULE_PRIORITY = {
    "constant": 0,
    "prog_p1": 1, "prog_m1": 1,
    "prog_p2": 2, "prog_m2": 2,
    "arith_plus": 3, "arith_minus": 3,
    "arith_plus1": 3, "arith_minus1": 3,
    "dist3": 4,
}


def _filter_by_specificity(rules: Set[str]) -> Set[str]:
    if not rules:
        return rules
    best = min(_RULE_PRIORITY[r] for r in rules)
    return {r for r in rules if _RULE_PRIORITY[r] == best}


# ═══════════════════════════════════════════════════════════════════
#  Template construction
# ═══════════════════════════════════════════════════════════════════

def _add_background(template: Template, max_val: int = 10) -> None:
    """Populate successor, addition, and inequality tables."""
    for v in range(max_val):
        if v + 1 < max_val:
            template += R.hidden.succ1(v, v + 1)
        if v + 2 < max_val:
            template += R.hidden.succ2(v, v + 2)

    for a in range(max_val):
        for b in range(max_val):
            if a + b < max_val:
                template += R.hidden.adds(a, b, a + b)
            if a - b >= 0:
                template += R.hidden.subs(a, b, a - b)
            if a + b + 1 < max_val:
                template += R.hidden.adds1(a, b, a + b + 1)
            if a - b - 1 >= 0:
                template += R.hidden.subs1(a, b, a - b - 1)


def _add_row_rule(template: Template, head_name: str, constraint_atoms: list) -> None:
    """Add a rule that checks *constraint_atoms* in all three rows.

    ``constraint_atoms`` is a callable ``(p1, p2, p3) -> list[atom]``
    that returns the body atoms expressing the constraint for one row
    given its three panel-ID variables.
    """
    body = (
        *constraint_atoms(0, 1, 2),
        *constraint_atoms(3, 4, 5),
        *constraint_atoms(6, 7, V.X),
        R.hidden.is_candidate(V.X),
    )
    template += R.get(head_name)(V.X, V.Pos, V.Attr) <= body


def build_rpm_template() -> Template:
    """Return a Template encoding all RPM rules as relational logic."""
    template = Template()
    _add_background(template)

    # Mark which panel IDs are candidates (8-15)
    for ci in range(NUM_CONTEXT, NUM_CONTEXT + NUM_CANDIDATES):
        template += R.hidden.is_candidate(ci)

    # Helper for distribute_three same-set constraint:
    # "value V appears in context row 0 at (Pos, Attr)"
    template += R.hidden.in_ctx_set(V.Pos, V.Attr, V.V) <= R.val(0, V.Pos, V.Attr, V.V)
    template += R.hidden.in_ctx_set(V.Pos, V.Attr, V.V) <= R.val(1, V.Pos, V.Attr, V.V)
    template += R.hidden.in_ctx_set(V.Pos, V.Attr, V.V) <= R.val(2, V.Pos, V.Attr, V.V)

    # ── CONSTANT: v1 == v2 == v3 within each row (per-row variable) ──
    def _const(p1, p2, p3):
        v = V.get(f"v{p1}")
        return [R.val(p1, V.Pos, V.Attr, v), R.val(p2, V.Pos, V.Attr, v), R.val(p3, V.Pos, V.Attr, v)]
    _add_row_rule(template, "rule_constant", _const)

    # ── PROGRESSION ±1 / ±2 ──
    def _make_prog(succ_rel):
        def _prog(p1, p2, p3):
            a, b, c = V.get(f"a{p1}"), V.get(f"b{p1}"), V.get(f"c{p1}")
            return [
                R.val(p1, V.Pos, V.Attr, a), R.val(p2, V.Pos, V.Attr, b), R.val(p3, V.Pos, V.Attr, c),
                succ_rel(a, b), succ_rel(b, c),
            ]
        return _prog

    _add_row_rule(template, "rule_prog_p1", _make_prog(R.hidden.succ1))
    _add_row_rule(template, "rule_prog_m1", _make_prog(lambda a, b: R.hidden.succ1(b, a)))
    _add_row_rule(template, "rule_prog_p2", _make_prog(R.hidden.succ2))
    _add_row_rule(template, "rule_prog_m2", _make_prog(lambda a, b: R.hidden.succ2(b, a)))

    # ── ARITHMETIC ±  and ± with offset 1 (Size) ──
    def _make_arith(table_rel):
        def _arith(p1, p2, p3):
            a, b, c = V.get(f"a{p1}"), V.get(f"b{p1}"), V.get(f"c{p1}")
            return [
                R.val(p1, V.Pos, V.Attr, a), R.val(p2, V.Pos, V.Attr, b), R.val(p3, V.Pos, V.Attr, c),
                table_rel(a, b, c),
            ]
        return _arith

    _add_row_rule(template, "rule_arith_plus",  _make_arith(R.hidden.adds))
    _add_row_rule(template, "rule_arith_minus", _make_arith(R.hidden.subs))
    _add_row_rule(template, "rule_arith_plus1", _make_arith(R.hidden.adds1))
    _add_row_rule(template, "rule_arith_minus1", _make_arith(R.hidden.subs1))

    # ── DISTRIBUTE THREE: alldiff + same-set across rows ──
    def _dist3(p1, p2, p3):
        a, b, c = V.get(f"a{p1}"), V.get(f"b{p1}"), V.get(f"c{p1}")
        return [
            R.val(p1, V.Pos, V.Attr, a), R.val(p2, V.Pos, V.Attr, b), R.val(p3, V.Pos, V.Attr, c),
            R.special.alldiff(a, b, c),
            R.hidden.in_ctx_set(V.Pos, V.Attr, a),
            R.hidden.in_ctx_set(V.Pos, V.Attr, b),
            R.hidden.in_ctx_set(V.Pos, V.Attr, c),
        ]
    _add_row_rule(template, "rule_dist3", _dist3)

    # ── COUNT (Number) rules: check object-count consistency across rows ──
    def _cnt_const(p1, p2, p3):
        n = V.get(f"n{p1}")
        return [R.cnt(p1, n), R.cnt(p2, n), R.cnt(p3, n)]

    def _make_cnt_prog(succ_rel):
        def _cnt_prog(p1, p2, p3):
            a, b, c = V.get(f"cn_a{p1}"), V.get(f"cn_b{p1}"), V.get(f"cn_c{p1}")
            return [R.cnt(p1, a), R.cnt(p2, b), R.cnt(p3, c), succ_rel(a, b), succ_rel(b, c)]
        return _cnt_prog

    def _make_cnt_arith(table_rel):
        def _cnt_arith(p1, p2, p3):
            a, b, c = V.get(f"cn_a{p1}"), V.get(f"cn_b{p1}"), V.get(f"cn_c{p1}")
            return [R.cnt(p1, a), R.cnt(p2, b), R.cnt(p3, c), table_rel(a, b, c)]
        return _cnt_arith

    # Count-constant
    cnt_body = (*_cnt_const(0, 1, 2), *_cnt_const(3, 4, 5), *_cnt_const(6, 7, V.X), R.hidden.is_candidate(V.X))
    template += R.count_ok(V.X) <= cnt_body

    # Count-progression ±1/±2
    for label, rel in [("cp1", R.hidden.succ1), ("cm1", lambda a, b: R.hidden.succ1(b, a)),
                        ("cp2", R.hidden.succ2), ("cm2", lambda a, b: R.hidden.succ2(b, a))]:
        fn = _make_cnt_prog(rel)
        body = (*fn(0, 1, 2), *fn(3, 4, 5), *fn(6, 7, V.X), R.hidden.is_candidate(V.X))
        template += R.count_ok(V.X) <= body

    # Count-arithmetic ±
    for label, rel in [("ca_p", R.hidden.adds), ("ca_m", R.hidden.subs),
                        ("ca_p1", R.hidden.adds1), ("ca_m1", R.hidden.subs1)]:
        fn = _make_cnt_arith(rel)
        body = (*fn(0, 1, 2), *fn(3, 4, 5), *fn(6, 7, V.X), R.hidden.is_candidate(V.X))
        template += R.count_ok(V.X) <= body

    # Count-distribute_three: same set of counts across rows
    template += R.hidden.in_cnt_set(V.N) <= R.cnt(0, V.N)
    template += R.hidden.in_cnt_set(V.N) <= R.cnt(1, V.N)
    template += R.hidden.in_cnt_set(V.N) <= R.cnt(2, V.N)

    def _cnt_dist3(p1, p2, p3):
        a, b, c = V.get(f"cd_a{p1}"), V.get(f"cd_b{p1}"), V.get(f"cd_c{p1}")
        return [
            R.cnt(p1, a), R.cnt(p2, b), R.cnt(p3, c),
            R.special.alldiff(a, b, c),
            R.hidden.in_cnt_set(a), R.hidden.in_cnt_set(b), R.hidden.in_cnt_set(c),
        ]

    cnt_d3_body = (*_cnt_dist3(0, 1, 2), *_cnt_dist3(3, 4, 5), *_cnt_dist3(6, 7, V.X), R.hidden.is_candidate(V.X))
    template += R.count_ok(V.X) <= cnt_d3_body

    # ── Meta-rule: tag each satisfied rule with its name ──
    for rname in _RULE_PRIORITY:
        template += R.satisfied(V.X, V.Pos, V.Attr, rname) <= R.get(f"rule_{rname}")(V.X, V.Pos, V.Attr)

    return template


# ═══════════════════════════════════════════════════════════════════
#  Data conversion
# ═══════════════════════════════════════════════════════════════════

def sample_to_facts(
    obj_type: torch.Tensor,
    obj_size: torch.Tensor,
    obj_color: torch.Tensor,
    obj_exists: torch.Tensor,
) -> list:
    """Convert one sample's [16, 22] tensors into a list of PyNeuraLogic facts."""
    facts = []
    attr_tensors = {"type": obj_type, "size": obj_size, "color": obj_color}
    for panel in range(16):
        count = 0
        for pos in range(NUM_POSITIONS):
            if not bool(obj_exists[panel, pos].item()):
                continue
            count += 1
            for attr_name, tensor in attr_tensors.items():
                val = int(tensor[panel, pos].item())
                facts.append(R.val(panel, pos, attr_name, val))
        facts.append(R.cnt(panel, count))
    return facts


# ═══════════════════════════════════════════════════════════════════
#  Hard solver via InferenceEngine
# ═══════════════════════════════════════════════════════════════════

class PynlHardSolver:
    """Constraint-satisfaction RPM solver powered by NeuraLogic inference.

    For each sample the engine grounds the RPM rule templates against
    the per-sample facts and returns all (candidate, position, attribute,
    rule) tuples that satisfy the constraints.  Specificity filtering
    then selects the most informative rule per (position, attribute) and
    the candidate with the highest score wins.
    """

    def __init__(self):
        self.template = build_rpm_template()
        self.engine = InferenceEngine(self.template)

    def _query_all(self, facts: list) -> List[dict]:
        return list(self.engine.q(
            R.satisfied(V.X, V.Pos, V.Attr, V.Rule), facts
        ))

    def _query_count_ok(self, facts: list) -> Set[int]:
        """Return set of candidate panel IDs that pass the count-consistency check."""
        subs = list(self.engine.q(R.count_ok(V.X), facts))
        return {int(s["X"]) for s in subs}

    def score_candidates(
        self,
        obj_type: torch.Tensor,
        obj_size: torch.Tensor,
        obj_color: torch.Tensor,
        obj_exists: torch.Tensor,
    ) -> List[int]:
        facts = sample_to_facts(obj_type, obj_size, obj_color, obj_exists)
        subs = self._query_all(facts)
        count_ok = self._query_count_ok(facts)

        # Group by (pos, attr) → {rule → set_of_candidates}
        pa_rules: Dict[Tuple[str, str], Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        for s in subs:
            cand = int(s["X"])
            pos = s["Pos"]
            attr = s["Attr"]
            rule = s["Rule"]
            pa_rules[(pos, attr)][rule].add(cand)

        scores = [0] * NUM_CANDIDATES
        for (pos, attr), rule_cands in pa_rules.items():
            active_rules = {r for r, cands in rule_cands.items() if cands}
            best_rules = _filter_by_specificity(active_rules)

            for ci in range(NUM_CANDIDATES):
                cand_id = NUM_CONTEXT + ci
                if all(cand_id in rule_cands.get(r, set()) for r in best_rules):
                    scores[ci] += 1

        # Count consistency bonus
        for ci in range(NUM_CANDIDATES):
            if (NUM_CONTEXT + ci) in count_ok:
                scores[ci] += 1

        return scores

    def predict(
        self,
        obj_type: torch.Tensor,
        obj_size: torch.Tensor,
        obj_color: torch.Tensor,
        obj_exists: torch.Tensor,
    ) -> int:
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


# ═══════════════════════════════════════════════════════════════════
#  Learnable solver via NeuraLogic evaluator
# ═══════════════════════════════════════════════════════════════════

def build_learnable_template() -> Template:
    """Template with learnable rule weights for differentiable scoring."""
    template = build_rpm_template()

    # Each rule type contributes to the score with a learnable weight.
    # Multiple derivations (different Pos/Attr) are summed.
    for rname in _RULE_PRIORITY:
        template += (
            R.score(V.X)[1,] <= R.get(f"rule_{rname}")(V.X, V.Pos, V.Attr)
        ) | [Transformation.IDENTITY, Aggregation.SUM]

    template += R.score / 1 | [Aggregation.SUM, Transformation.SIGMOID]
    return template


def build_dataset_from_samples(
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]],
) -> Dataset:
    """Convert a list of (obj_type, obj_size, obj_color, obj_exists, target) to a PyNeuraLogic Dataset."""
    dataset = Dataset()
    for obj_type, obj_size, obj_color, obj_exists, target in samples:
        facts = sample_to_facts(obj_type, obj_size, obj_color, obj_exists)
        dataset.add_example(facts)

        queries = []
        for ci in range(NUM_CANDIDATES):
            label = 1.0 if ci == target else 0.0
            queries.append(R.score(NUM_CONTEXT + ci)[label])
        dataset.add_queries(queries)

    return dataset
