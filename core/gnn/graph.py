"""Graph construction for RAVEN GNN: build_candidate_graph, pack_graphs."""

from typing import Dict, List, Tuple

import torch

from .config import NUM_CANDIDATES, NUM_CONTEXT, NUM_POSITIONS, REL_TO_ID


def infer_relations(
    src_pos: int, dst_pos: int, centers: Dict[int, Tuple[float, float]], eps: float = 1e-4
) -> List[str]:
    sx, sy = centers[src_pos]
    dx, dy = centers[dst_pos]
    return _infer_relations_from_coords(sx, sy, dx, dy, eps)


def _infer_relations_from_coords(
    sx: float, sy: float, dx: float, dy: float, eps: float = 1e-4
) -> List[str]:
    """Infer spatial relations from (sx,sy) -> (dx,dy)."""
    rels: List[str] = []
    if sx < dx:
        rels.append("left_of")
    elif sx > dx:
        rels.append("right_of")
    if sy < dy:
        rels.append("above")
    elif sy > dy:
        rels.append("below")
    if abs(sy - dy) <= eps:
        rels.append("same_row")
    if abs(sx - dx) <= eps:
        rels.append("same_col")
    if abs(sx - dx) <= eps and abs(sy - dy) <= eps:
        rels.append("overlap")
    if not rels:
        rels.append("same_row")
    return rels


def _panel_grid_idx(panel_idx: int, candidate_idx: int) -> int:
    """Map panel index (0-7 context, 8-15 candidate) to 3x3 grid position 0-8."""
    return panel_idx if panel_idx < NUM_CONTEXT else 8  # candidate always at (2,2) = idx 8


def build_candidate_graph(
    obj_type: torch.Tensor,
    obj_size: torch.Tensor,
    obj_color: torch.Tensor,
    obj_exists: torch.Tensor,
    candidate_idx: int,
    centers: Dict[int, Tuple[float, float]],
    granular: bool = False,
    object_only: bool = False,
) -> Dict[str, torch.Tensor]:
    keep_panels = list(range(NUM_CONTEXT)) + [NUM_CONTEXT + candidate_idx]
    node_attrs: List[List[int]] = []
    node_kinds: List[int] = []  # 0=object, 1=panel, 2=attr_pos, 3=attr_color, 4=attr_type, 5=attr_size
    panel_roles: List[int] = []
    panel_nodes: Dict[int, List[int]] = {p: [] for p in keep_panels}
    panel_super_node: Dict[int, int] = {}
    obj_node_by_panel_pos: Dict[Tuple[int, int], int] = {}

    def _role(p: int) -> int:
        return 1 if p == (NUM_CONTEXT + candidate_idx) else 0

    for p in keep_panels:
        grid_idx = _panel_grid_idx(p, candidate_idx)
        for pos in range(NUM_POSITIONS):
            if not bool(obj_exists[p, pos].item()):
                continue
            pos_val = int(pos)
            col_val = int(obj_color[p, pos].item())
            typ_val = int(obj_type[p, pos].item())
            siz_val = int(obj_size[p, pos].item())

            if granular:
                for kind, attrs in [
                    (2, [pos_val, 0, 0, 0, grid_idx]),
                    (3, [0, col_val, 0, 0, grid_idx]),
                    (4, [0, 0, typ_val, 0, grid_idx]),
                    (5, [0, 0, 0, siz_val, grid_idx]),
                ]:
                    nid = len(node_attrs)
                    node_attrs.append(attrs)
                    node_kinds.append(kind)
                    panel_roles.append(_role(p))
                obj_nid = len(node_attrs)
                node_attrs.append([pos_val, col_val, typ_val, siz_val, grid_idx])
                node_kinds.append(0)
                panel_roles.append(_role(p))
                panel_nodes[p].append(obj_nid)
                obj_node_by_panel_pos[(p, pos)] = obj_nid
            else:
                nid = len(node_attrs)
                node_attrs.append([pos_val, col_val, typ_val, siz_val, grid_idx])
                node_kinds.append(0)
                panel_roles.append(_role(p))
                panel_nodes[p].append(nid)
                obj_node_by_panel_pos[(p, pos)] = nid

    # Add panel super-nodes only when not object_only
    if not object_only:
        for p in keep_panels:
            grid_idx = _panel_grid_idx(p, candidate_idx)
            nid = len(node_attrs)
            node_attrs.append([0, 0, 0, 0, grid_idx])
            node_kinds.append(1)
            panel_roles.append(_role(p))
            panel_super_node[p] = nid

    src, dst, et = [], [], []

    def add_edge(u: int, v: int, rel: str) -> None:
        src.append(u)
        dst.append(v)
        et.append(REL_TO_ID[rel])

    if granular:
        for p in keep_panels:
            for pos in range(NUM_POSITIONS):
                if not bool(obj_exists[p, pos].item()):
                    continue
                obj_nid = obj_node_by_panel_pos[(p, pos)]
                for k in range(4):
                    attr_nid = obj_nid - 1 - (3 - k)
                    add_edge(attr_nid, obj_nid, "attr_to_obj")
                    add_edge(obj_nid, attr_nid, "obj_to_attr")

    # Object-object spatial relations (between object nodes)
    for p in keep_panels:
        nodes = panel_nodes[p]
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                ni, nj = nodes[i], nodes[j]
                pi = node_attrs[ni][0]
                pj = node_attrs[nj][0]
                for rel in infer_relations(pi, pj, centers):
                    add_edge(ni, nj, rel)

    # Connect objects to their panel super-node (skip when object_only).
    if not object_only:
        for p in keep_panels:
            pnode = panel_super_node[p]
            for obj_node in panel_nodes[p]:
                add_edge(obj_node, pnode, "obj_to_panel")
                add_edge(pnode, obj_node, "panel_to_obj")

    # Cross-panel same-position edges (object-only): objects at same universal pos across panels
    if object_only:
        for pos in range(NUM_POSITIONS):
            nodes_at_pos = [
                (p, obj_node_by_panel_pos[(p, pos)])
                for p in keep_panels
                if (p, pos) in obj_node_by_panel_pos
            ]
            for i, (p, ni) in enumerate(nodes_at_pos):
                for j, (q, nj) in enumerate(nodes_at_pos):
                    if i < j:
                        add_edge(ni, nj, "same_pos_cross_panel")
                        add_edge(nj, ni, "same_pos_cross_panel")

    # Connect panel super-nodes to right and below neighbors in RPM 3x3 layout (skip when object_only).
    if not object_only:
        cand_panel = NUM_CONTEXT + candidate_idx
        panel_to_coord = {
            0: (0, 0), 1: (0, 1), 2: (0, 2),
            3: (1, 0), 4: (1, 1), 5: (1, 2),
            6: (2, 0), 7: (2, 1), cand_panel: (2, 2),
        }
        coord_to_panel = {coord: panel for panel, coord in panel_to_coord.items()}
        for p, (r, c) in panel_to_coord.items():
            if p not in panel_super_node:
                continue
            right_coord = (r, c + 1)
            below_coord = (r + 1, c)
            if right_coord in coord_to_panel:
                q = coord_to_panel[right_coord]
                if q in panel_super_node:
                    add_edge(panel_super_node[p], panel_super_node[q], "panel_right_link")
                    add_edge(panel_super_node[q], panel_super_node[p], "panel_left_link")
            if below_coord in coord_to_panel:
                q = coord_to_panel[below_coord]
                if q in panel_super_node:
                    add_edge(panel_super_node[p], panel_super_node[q], "panel_below_link")
                    add_edge(panel_super_node[q], panel_super_node[p], "panel_above_link")

    if not src:
        for i in range(len(node_attrs)):
            src.append(i)
            dst.append(i)
            et.append(REL_TO_ID["same_row"])

    return {
        "node_attrs": torch.tensor(node_attrs, dtype=torch.long),
        "node_kinds": torch.tensor(node_kinds, dtype=torch.long),
        "panel_roles": torch.tensor(panel_roles, dtype=torch.long),
        "edge_index": torch.tensor([src, dst], dtype=torch.long),
        "edge_type": torch.tensor(et, dtype=torch.long),
    }


# Context panels adjacent to answer slot (2,2): panel 5 (1,2) right, panel 7 (2,1) below
CONTEXT_ADJACENT_TO_ANSWER = [5, 7]

# 3x3 grid: panel idx -> (row, col). Candidates 8-15 at (2,2).
UNIFIED_PANEL_TO_COORD: Dict[int, Tuple[int, int]] = {
    0: (0, 0), 1: (0, 1), 2: (0, 2),
    3: (1, 0), 4: (1, 1), 5: (1, 2),
    6: (2, 0), 7: (2, 1),
}
for c in range(NUM_CANDIDATES):
    UNIFIED_PANEL_TO_COORD[NUM_CONTEXT + c] = (2, 2)

PANEL_SPACING = 10.0  # units between panel centers for global coord computation


def _global_pos(
    panel_idx: int, pos: int, centers: Dict[int, Tuple[float, float]]
) -> Tuple[float, float]:
    """Global (x,y) for object at panel_idx and universal position pos."""
    r, c = UNIFIED_PANEL_TO_COORD[panel_idx]
    lx, ly = centers.get(pos, (0.0, 0.0))
    return (c * PANEL_SPACING + lx, r * PANEL_SPACING + ly)


def build_unified_graph(
    obj_type: torch.Tensor,
    obj_size: torch.Tensor,
    obj_color: torch.Tensor,
    obj_exists: torch.Tensor,
    centers: Dict[int, Tuple[float, float]],
    granular: bool = False,
    object_only: bool = False,
) -> Dict[str, torch.Tensor]:
    """One graph with all 16 panels; candidates 8-15 connect only to context panels 5 and 7."""
    keep_panels = list(range(16))  # all 8 context + 8 candidates
    node_attrs: List[List[int]] = []
    node_kinds: List[int] = []
    panel_roles: List[int] = []  # 0=context, 1=candidate
    candidate_ids: List[int] = []  # 0-7 for candidate nodes, -1 for context
    panel_nodes: Dict[int, List[int]] = {p: [] for p in keep_panels}
    panel_super_node: Dict[int, int] = {}
    obj_node_by_panel_pos: Dict[Tuple[int, int], int] = {}

    def _role(p: int) -> int:
        return 1 if p >= NUM_CONTEXT else 0

    def _cand_id(p: int) -> int:
        return p - NUM_CONTEXT if p >= NUM_CONTEXT else -1

    def _grid_idx(p: int) -> int:
        return p if p < NUM_CONTEXT else 8  # candidate slot = (2,2) = idx 8

    for p in keep_panels:
        grid_idx = _grid_idx(p)
        for pos in range(NUM_POSITIONS):
            if not bool(obj_exists[p, pos].item()):
                continue
            pos_val = int(pos)
            col_val = int(obj_color[p, pos].item())
            typ_val = int(obj_type[p, pos].item())
            siz_val = int(obj_size[p, pos].item())

            if granular:
                for kind, attrs in [
                    (2, [pos_val, 0, 0, 0, grid_idx]),
                    (3, [0, col_val, 0, 0, grid_idx]),
                    (4, [0, 0, typ_val, 0, grid_idx]),
                    (5, [0, 0, 0, siz_val, grid_idx]),
                ]:
                    nid = len(node_attrs)
                    node_attrs.append(attrs)
                    node_kinds.append(kind)
                    panel_roles.append(_role(p))
                    candidate_ids.append(_cand_id(p))
                obj_nid = len(node_attrs)
                node_attrs.append([pos_val, col_val, typ_val, siz_val, grid_idx])
                node_kinds.append(0)
                panel_roles.append(_role(p))
                candidate_ids.append(_cand_id(p))
                panel_nodes[p].append(obj_nid)
                obj_node_by_panel_pos[(p, pos)] = obj_nid
            else:
                nid = len(node_attrs)
                node_attrs.append([pos_val, col_val, typ_val, siz_val, grid_idx])
                node_kinds.append(0)
                panel_roles.append(_role(p))
                candidate_ids.append(_cand_id(p))
                panel_nodes[p].append(nid)
                obj_node_by_panel_pos[(p, pos)] = nid

    if not object_only:
        for p in keep_panels:
            grid_idx = _grid_idx(p)
            nid = len(node_attrs)
            node_attrs.append([0, 0, 0, 0, grid_idx])
            node_kinds.append(1)
            panel_roles.append(_role(p))
            candidate_ids.append(_cand_id(p))
            panel_super_node[p] = nid

    src, dst, et = [], [], []

    def add_edge(u: int, v: int, rel: str) -> None:
        src.append(u)
        dst.append(v)
        et.append(REL_TO_ID[rel])

    if granular:
        for p in keep_panels:
            for pos in range(NUM_POSITIONS):
                if not bool(obj_exists[p, pos].item()):
                    continue
                obj_nid = obj_node_by_panel_pos[(p, pos)]
                for k in range(4):
                    attr_nid = obj_nid - 1 - (3 - k)
                    add_edge(attr_nid, obj_nid, "attr_to_obj")
                    add_edge(obj_nid, attr_nid, "obj_to_attr")

    # Intra-panel: object-object spatial
    for p in keep_panels:
        nodes = panel_nodes[p]
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                ni, nj = nodes[i], nodes[j]
                pi = node_attrs[ni][0]
                pj = node_attrs[nj][0]
                for rel in infer_relations(pi, pj, centers):
                    add_edge(ni, nj, rel)

    if not object_only:
        for p in keep_panels:
            pnode = panel_super_node[p]
            for obj_node in panel_nodes[p]:
                add_edge(obj_node, pnode, "obj_to_panel")
                add_edge(pnode, obj_node, "panel_to_obj")

        # Panel grid: context only (panels 0-7). Candidates don't connect to each other.
        panel_to_coord = {
            0: (0, 0), 1: (0, 1), 2: (0, 2),
            3: (1, 0), 4: (1, 1), 5: (1, 2),
            6: (2, 0), 7: (2, 1),
        }
        coord_to_panel = {coord: panel for panel, coord in panel_to_coord.items()}
        for p, (r, c) in panel_to_coord.items():
            if p not in panel_super_node:
                continue
            right_coord = (r, c + 1)
            below_coord = (r + 1, c)
            if right_coord in coord_to_panel:
                qp = coord_to_panel[right_coord]
                if qp in panel_super_node:
                    add_edge(panel_super_node[p], panel_super_node[qp], "panel_right_link")
                    add_edge(panel_super_node[qp], panel_super_node[p], "panel_left_link")
            if below_coord in coord_to_panel:
                qp = coord_to_panel[below_coord]
                if qp in panel_super_node:
                    add_edge(panel_super_node[p], panel_super_node[qp], "panel_below_link")
                    add_edge(panel_super_node[qp], panel_super_node[p], "panel_above_link")

        # Unified: candidate panels 8-15 connect only to context panels 5 and 7
        for c in range(NUM_CANDIDATES):
            cand_panel = NUM_CONTEXT + c
            if cand_panel not in panel_super_node:
                continue
            for ctx_p in CONTEXT_ADJACENT_TO_ANSWER:
                if ctx_p in panel_super_node:
                    add_edge(panel_super_node[cand_panel], panel_super_node[ctx_p], "cand_to_adjacent_ctx")
                    add_edge(panel_super_node[ctx_p], panel_super_node[cand_panel], "cand_to_adjacent_ctx")
    else:
        # Object-only unified: connect all objects via spatial edges (like panel grid but object-level)
        # 1) same_pos_cross_panel: objects at same universal position across panels
        for pos in range(NUM_POSITIONS):
            nodes_at_pos = [
                (p, obj_node_by_panel_pos[(p, pos)])
                for p in keep_panels
                if (p, pos) in obj_node_by_panel_pos
            ]
            for i, (_, ni) in enumerate(nodes_at_pos):
                for j, (_, nj) in enumerate(nodes_at_pos):
                    if i < j:
                        add_edge(ni, nj, "same_pos_cross_panel")
                        add_edge(nj, ni, "same_pos_cross_panel")

        # 2) Cross-panel spatial edges: objects in adjacent panels (right, below) get left_of/right_of/above/below
        coord_to_panels: Dict[Tuple[int, int], List[int]] = {}
        for p, (r, c) in UNIFIED_PANEL_TO_COORD.items():
            coord_to_panels.setdefault((r, c), []).append(p)
        # Adjacent pairs: (r,c) -> (r, c+1) right, (r+1, c) below
        for (r, c), panels_at in coord_to_panels.items():
            right_coord = (r, c + 1)
            below_coord = (r + 1, c)
            for right_co in [right_coord, below_coord]:
                neighbor_panels = coord_to_panels.get(right_co, [])
                for p in panels_at:
                    for q in neighbor_panels:
                        for ni in panel_nodes.get(p, []):
                            pi = node_attrs[ni][0]
                            gx_i, gy_i = _global_pos(p, pi, centers)
                            for nj in panel_nodes.get(q, []):
                                pj = node_attrs[nj][0]
                                gx_j, gy_j = _global_pos(q, pj, centers)
                                for rel in _infer_relations_from_coords(gx_i, gy_i, gx_j, gy_j):
                                    add_edge(ni, nj, rel)
                                for rel in _infer_relations_from_coords(gx_j, gy_j, gx_i, gy_i):
                                    add_edge(nj, ni, rel)

    if not src:
        for i in range(len(node_attrs)):
            src.append(i)
            dst.append(i)
            et.append(REL_TO_ID["same_row"])

    return {
        "node_attrs": torch.tensor(node_attrs, dtype=torch.long),
        "node_kinds": torch.tensor(node_kinds, dtype=torch.long),
        "panel_roles": torch.tensor(panel_roles, dtype=torch.long),
        "candidate_id": torch.tensor(candidate_ids, dtype=torch.long),
        "edge_index": torch.tensor([src, dst], dtype=torch.long),
        "edge_type": torch.tensor(et, dtype=torch.long),
    }


def pack_graphs(graphs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    x, kinds, roles, gidx, eidx, etype = [], [], [], [], [], []
    off = 0
    for gi, g in enumerate(graphs):
        n = g["node_attrs"].size(0)
        x.append(g["node_attrs"])
        kinds.append(g["node_kinds"])
        roles.append(g["panel_roles"])
        gidx.append(torch.full((n,), gi, dtype=torch.long))
        eidx.append(g["edge_index"] + off)
        etype.append(g["edge_type"])
        off += n
    return {
        "node_attrs": torch.cat(x, dim=0),
        "node_kinds": torch.cat(kinds, dim=0),
        "panel_roles": torch.cat(roles, dim=0),
        "graph_idx": torch.cat(gidx, dim=0),
        "edge_index": torch.cat(eidx, dim=1),
        "edge_type": torch.cat(etype, dim=0),
        "num_graphs": len(graphs),
    }
