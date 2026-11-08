"""Constants and configuration for RAVEN GNN training."""

NUM_CONTEXT = 8
NUM_CANDIDATES = 8
NUM_POSITIONS = 22
NUM_COLOR = 10
NUM_SHAPE = 5
NUM_SIZE = 6

# Codebook layout (must match gnn_solver / frontend)
NUM_PER_POS = NUM_COLOR * NUM_SIZE * NUM_SHAPE + 1  # 301
NUM_ATTR_COMBOS = NUM_COLOR * NUM_SIZE * NUM_SHAPE  # 300

RELATIONS = [
    "left_of",
    "right_of",
    "above",
    "below",
    "same_row",
    "same_col",
    "overlap",
    "obj_to_panel",
    "panel_to_obj",
    "panel_right_link",
    "panel_left_link",
    "panel_below_link",
    "panel_above_link",
    "same_pos_cross_panel",  # object-only: same universal position across panels
    "cand_to_adjacent_ctx",  # unified: candidate nodes -> context panels 5 and 7
    "attr_to_obj",
    "obj_to_attr",
]
REL_TO_ID = {name: i for i, name in enumerate(RELATIONS)}

# Node kinds: 0=object, 1=panel super-node, 2=attr_pos, 3=attr_color, 4=attr_type, 5=attr_size
NUM_NODE_KINDS = 6
NUM_PANEL_GRID = 9  # 3x3 grid positions for context + candidate

CONTEXT_ROWS = [[0, 1, 2], [3, 4, 5], [6, 7]]
CONTEXT_COLS = [[0, 3, 6], [1, 4, 7], [2, 5]]

# Maps constellation local position → universal position (0–21). Same as predict_frontend.
CONFIG_POS_MAP = {
    "center_single": {0: 0},
    "distribute_four": {0: 1, 1: 2, 2: 3, 3: 4},
    "distribute_nine": {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: 13},
    "left_center_single_right_center_single": {0: 14, 1: 15},
    "up_center_single_down_center_single": {0: 16, 1: 17},
    "in_center_single_out_center_single": {0: 0, 1: 9},
    "in_distribute_four_out_center_single": {0: 0, 1: 18, 2: 19, 3: 20, 4: 21},
}
