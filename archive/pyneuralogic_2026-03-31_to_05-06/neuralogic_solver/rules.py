"""
Hard and soft (differentiable) rule checkers for Raven's Progressive Matrices.

Hard rules operate on integer attribute values and return booleans.
Soft rules operate on probability distributions over attribute domains
and return differentiable satisfaction probabilities in [0, 1].

Mapping to relational-logic concepts
-------------------------------------
Each rule is a *ternary constraint*  R(v1, v2, v3)  that would be expressed
in PyNeuraLogic as a logic rule with three body relations sharing variables:

    R.satisfied(Row, Attr) <= (
        R.val(P1, Pos, Attr, Var.V1),
        R.val(P2, Pos, Attr, Var.V2),
        R.val(P3, Pos, Attr, Var.V3),
        R.hidden.row(P1, P2, P3),
        <constraint on V1, V2, V3>
    )

The key advantage over binary-GNN message passing is that the constraint
binds three variables simultaneously — a k-ary relation that 1-WL GNNs
cannot express without multiple hops and no guarantee of correctness.
"""

from typing import List

import torch


# ═══════════════════════════════════════════════════════════════════════
#  Hard rule checkers  (integer values → bool)
# ═══════════════════════════════════════════════════════════════════════

def _hard_constant(v1: int, v2: int, v3: int) -> bool:
    return v1 == v2 == v3


def _hard_progression(v1: int, v2: int, v3: int, step: int) -> bool:
    return (v2 == v1 + step) and (v3 == v2 + step)


def _hard_arithmetic_plus(v1: int, v2: int, v3: int) -> bool:
    return v3 == v1 + v2


def _hard_arithmetic_minus(v1: int, v2: int, v3: int) -> bool:
    return v3 == v1 - v2


def _hard_distribute_three(v1: int, v2: int, v3: int) -> bool:
    return v1 != v2 and v2 != v3 and v1 != v3


def _hard_arithmetic_plus_1(v1: int, v2: int, v3: int) -> bool:
    return v3 == v1 + v2 + 1


def _hard_arithmetic_minus_1(v1: int, v2: int, v3: int) -> bool:
    return v3 == v1 - v2 - 1


_HARD_CHECKERS = {
    "constant":         lambda v1, v2, v3: _hard_constant(v1, v2, v3),
    "progression_p1":   lambda v1, v2, v3: _hard_progression(v1, v2, v3, 1),
    "progression_m1":   lambda v1, v2, v3: _hard_progression(v1, v2, v3, -1),
    "progression_p2":   lambda v1, v2, v3: _hard_progression(v1, v2, v3, 2),
    "progression_m2":   lambda v1, v2, v3: _hard_progression(v1, v2, v3, -2),
    "arithmetic_plus":  _hard_arithmetic_plus,
    "arithmetic_minus": _hard_arithmetic_minus,
    "arithmetic_plus_1":  _hard_arithmetic_plus_1,
    "arithmetic_minus_1": _hard_arithmetic_minus_1,
    "distribute_three": _hard_distribute_three,
}


def check_rule(rule_name: str, v1: int, v2: int, v3: int) -> bool:
    """Check whether integer triple (v1, v2, v3) satisfies *rule_name*."""
    return _HARD_CHECKERS[rule_name](v1, v2, v3)


def detect_rules(triples: List[tuple]) -> set:
    """Return rules satisfied by *every* triple in the list.

    ``triples`` is a list of (v1, v2, v3) integer tuples drawn from
    different context rows/columns for the same position and attribute.
    A rule is "detected" only if it matches all of them (intersection).
    """
    from .config import RULE_NAMES

    survivors = None
    for v1, v2, v3 in triples:
        matching = {r for r in RULE_NAMES if check_rule(r, v1, v2, v3)}
        survivors = matching if survivors is None else survivors & matching
    return survivors or set()


# ═══════════════════════════════════════════════════════════════════════
#  Soft rule checkers  (probability distributions → differentiable [0,1])
# ═══════════════════════════════════════════════════════════════════════
#
# Each p1, p2, p3 is a tensor of shape [..., D]  where D is the domain
# size of the attribute (5 for type, 6 for size, 10 for color).
# Values along the last dimension are probabilities summing to 1.
# Return shape: [...]  (batch dims).
#

def _soft_constant(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """P(v1 == v2 == v3) = Σ_v  p1[v] · p2[v] · p3[v]"""
    return (p1 * p2 * p3).sum(-1)


def _soft_progression(
    p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, step: int
) -> torch.Tensor:
    """P(v2 == v1+step, v3 == v1+2·step) = Σ_v  p1[v] · p2[v+s] · p3[v+2s]"""
    D = p1.shape[-1]
    if step >= 0:
        end = D - 2 * step
        if end <= 0:
            return torch.zeros(p1.shape[:-1], device=p1.device)
        return (p1[..., :end] * p2[..., step:end + step] * p3[..., 2 * step:end + 2 * step]).sum(-1)
    start = -2 * step
    if start >= D:
        return torch.zeros(p1.shape[:-1], device=p1.device)
    return (p1[..., start:] * p2[..., start + step:D + step] * p3[..., :D - start]).sum(-1)


def _soft_arithmetic_plus(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """P(v3 == v1 + v2) = Σ_{a+b<D}  p1[a] · p2[b] · p3[a+b]"""
    D = p1.shape[-1]
    result = torch.zeros(p1.shape[:-1], device=p1.device, dtype=p1.dtype)
    for a in range(D):
        for b in range(D - a):
            result = result + p1[..., a] * p2[..., b] * p3[..., a + b]
    return result


def _soft_arithmetic_minus(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """P(v3 == v1 - v2) = Σ_{a≥b}  p1[a] · p2[b] · p3[a-b]"""
    D = p1.shape[-1]
    result = torch.zeros(p1.shape[:-1], device=p1.device, dtype=p1.dtype)
    for a in range(D):
        for b in range(a + 1):
            result = result + p1[..., a] * p2[..., b] * p3[..., a - b]
    return result


def _soft_distribute_three(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """P(v1, v2, v3 all distinct)  via inclusion-exclusion:

    1  − P(v1=v2) − P(v2=v3) − P(v1=v3) + 2·P(v1=v2=v3)
    """
    p12 = (p1 * p2).sum(-1)
    p23 = (p2 * p3).sum(-1)
    p13 = (p1 * p3).sum(-1)
    p123 = (p1 * p2 * p3).sum(-1)
    return (1.0 - p12 - p23 - p13 + 2.0 * p123).clamp(min=0.0)


def _soft_arithmetic_plus_1(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """P(v3 == v1 + v2 + 1) — for 1-based-offset attributes (Size)."""
    D = p1.shape[-1]
    result = torch.zeros(p1.shape[:-1], device=p1.device, dtype=p1.dtype)
    for a in range(D):
        for b in range(D):
            c = a + b + 1
            if 0 <= c < D:
                result = result + p1[..., a] * p2[..., b] * p3[..., c]
    return result


def _soft_arithmetic_minus_1(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """P(v3 == v1 - v2 - 1) — for 1-based-offset attributes (Size)."""
    D = p1.shape[-1]
    result = torch.zeros(p1.shape[:-1], device=p1.device, dtype=p1.dtype)
    for a in range(D):
        for b in range(a + 1):
            c = a - b - 1
            if 0 <= c < D:
                result = result + p1[..., a] * p2[..., b] * p3[..., c]
    return result


_SOFT_CHECKERS = {
    "constant":          _soft_constant,
    "progression_p1":    lambda p1, p2, p3: _soft_progression(p1, p2, p3, 1),
    "progression_m1":    lambda p1, p2, p3: _soft_progression(p1, p2, p3, -1),
    "progression_p2":    lambda p1, p2, p3: _soft_progression(p1, p2, p3, 2),
    "progression_m2":    lambda p1, p2, p3: _soft_progression(p1, p2, p3, -2),
    "arithmetic_plus":   _soft_arithmetic_plus,
    "arithmetic_minus":  _soft_arithmetic_minus,
    "arithmetic_plus_1":  _soft_arithmetic_plus_1,
    "arithmetic_minus_1": _soft_arithmetic_minus_1,
    "distribute_three":  _soft_distribute_three,
}


def soft_rule_score(
    rule_name: str,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
) -> torch.Tensor:
    """Differentiable satisfaction probability for *rule_name* given soft
    attribute distributions ``p1, p2, p3``  (each shape ``[..., D]``)."""
    return _SOFT_CHECKERS[rule_name](p1, p2, p3)


def soft_all_rules(
    p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
) -> torch.Tensor:
    """Evaluate all rules and return stacked scores [..., NUM_RULES]."""
    from .config import RULE_NAMES
    scores = [_SOFT_CHECKERS[r](p1, p2, p3) for r in RULE_NAMES]
    return torch.stack(scores, dim=-1)
