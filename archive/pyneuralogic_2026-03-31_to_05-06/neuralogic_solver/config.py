"""Constants, rule vocabulary, and RPM grid layout for the logic solver."""

NUM_POSITIONS = 22
NUM_CONTEXT = 8
NUM_CANDIDATES = 8

ATTR_NAMES = ["type", "size", "color"]
ATTR_DIMS = {"type": 5, "size": 6, "color": 10}
MAX_ATTR_DIM = 10

# ── RPM 3×3 grid layout ────────────────────────────────────────────────
#   Panel indices:
#       Row 0:  P0  P1  P2
#       Row 1:  P3  P4  P5
#       Row 2:  P6  P7  P?   (P? = candidate)
#
#   Candidate panels are stored at indices 8–15 in the data tensors.

CONTEXT_ROWS = [
    [0, 1, 2],
    [3, 4, 5],
]
CONTEXT_COLS = [
    [0, 3, 6],
    [1, 4, 7],
]
ANSWER_ROW_PREFIX = [6, 7]   # answer row = [P6, P7, candidate]
ANSWER_COL_PREFIX = [2, 5]   # answer col = [P2, P5, candidate]

# ── Rule vocabulary ─────────────────────────────────────────────────────
# Each rule is a ternary constraint on a triple (v1, v2, v3) drawn from
# three successive panels in a row or column, for a given position+attribute.

RULE_NAMES = [
    "constant",           # v1 == v2 == v3
    "progression_p1",     # v2 == v1+1,  v3 == v2+1
    "progression_m1",     # v2 == v1-1,  v3 == v2-1
    "progression_p2",     # v2 == v1+2,  v3 == v2+2
    "progression_m2",     # v2 == v1-2,  v3 == v2-2
    "arithmetic_plus",    # v3 == v1 + v2            (Color)
    "arithmetic_minus",   # v3 == v1 - v2            (Color)
    "arithmetic_plus_1",  # v3 == v1 + v2 + 1        (Size — 1-based offset)
    "arithmetic_minus_1", # v3 == v1 - v2 - 1        (Size — 1-based offset)
    "distribute_three",   # v1, v2, v3 all distinct
]
NUM_RULES = len(RULE_NAMES)

ALL_CONFIGS = [
    "center_single",
    "distribute_four",
    "distribute_nine",
    "left_center_single_right_center_single",
    "up_center_single_down_center_single",
    "in_center_single_out_center_single",
    "in_distribute_four_out_center_single",
]
