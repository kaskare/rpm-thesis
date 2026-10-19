"""Logic-rule-based RPM solver: explicit relational reasoning without GNNs."""

from .config import ATTR_DIMS, ATTR_NAMES, NUM_RULES, RULE_NAMES
from .rules import check_rule, soft_rule_score
from .solver import HardLogicSolver, SoftLogicSolver

__all__ = [
    "ATTR_DIMS",
    "ATTR_NAMES",
    "NUM_RULES",
    "RULE_NAMES",
    "check_rule",
    "soft_rule_score",
    "HardLogicSolver",
    "SoftLogicSolver",
]
