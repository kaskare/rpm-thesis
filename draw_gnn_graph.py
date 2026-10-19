"""
Draw the GNN graph representation for a RAVEN sample.

Supports all graph modes (same as train_object_vsa_gnn_frontend.py):

Graph representation flags:
  --granular         Attribute nodes (pos, col, typ, siz) per object
  --object-only      No panel super-nodes; objects connect via spatial + same-pos cross-panel
  --unified-graph    One graph with all 8 candidates; candidates connect to context panels 5 and 7

Unified object-only: context objects at 3x3 grid; candidates in ring around center.
  Cross-panel spatial edges (left_of, right_of, above, below) connect objects in adjacent
  panels (like panel_right_link/panel_below_link but at object level). same_pos_cross_panel
  connects objects at the same universal position across all 16 panels.

Usage:
    # Default: panel nodes, 8 context + 1 candidate
    python draw_gnn_graph.py --sample data/RAVEN-10000/center_single/RAVEN_0_train.npz --use-gt -o graph.png

    # Object-only (no panel nodes)
    python draw_gnn_graph.py --sample ... --use-gt --object-only -o graph_obj.png

    # Unified (all 8 candidates in one graph)
    python draw_gnn_graph.py --sample ... --use-gt --unified-graph -o graph_unified.png

    # Granular (attr nodes per object)
    python draw_gnn_graph.py --sample ... --use-gt --granular -o graph_gran.png

    # Combined: unified + object-only
    python draw_gnn_graph.py --sample ... --use-gt --unified-graph --object-only -o graph_unified_obj.png

    # Choose candidate (default: target). Ignored when --unified-graph
    python draw_gnn_graph.py --sample ... --use-gt --candidate 3 -o graph.png
"""

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "raven"))
sys.path.insert(0, os.path.join(ROOT, "neuro-vector-symbolic-architectures"))

import numpy as np
import torch

from gnn_solver import (
    load_frontend,
    extract_cosine_sims,
    extract_discrete_objects,
    NUM_POS,
    NUM_PANELS,
)
from gnn import (
    build_candidate_graph,
    build_position_centers_universal,
    build_unified_graph,
    fallback_position_centers,
    get_config_from_path,
    gt_to_dense,
    parse_ground_truth,
    RELATIONS,
)

TYPE_NAMES = {0: "tri", 1: "sq", 2: "pent", 3: "hex", 4: "circ"}
SIZE_NAMES = {i: f"s{i}" for i in range(6)}
COLOR_NAMES = {i: f"c{i}" for i in range(10)}
KIND_NAMES = ["obj", "panel", "pos", "col", "typ", "siz"]

# Universal position 0-21 → human-readable names (matches CONFIG_POS_MAP layouts)
POSITION_NAMES = {
    0: "center",
    1: "tl", 2: "tr", 3: "bl", 4: "br",  # distribute_four
    5: "tl", 6: "tc", 7: "tr", 8: "ml", 9: "c", 10: "mr", 11: "bl", 12: "bc", 13: "br",  # distribute_nine
    14: "left", 15: "right",
    16: "up", 17: "down",
    18: "tl", 19: "tr", 20: "bl", 21: "br",  # in_distribute_four
}


def _panel_layout(node_attrs, node_kinds, edge_index, edge_type, candidate_id=None):
    """Position nodes by panel: 9 clusters (3x3 grid), objects around panel centers.
    For unified graphs, candidate nodes (candidate_id>=0) are spread in a ring around center."""
    import math
    n = node_attrs.size(0)
    pos = {}
    grid_spacing = 5.0
    obj_radius = 1.2
    attr_radius = 0.4
    cand_ring_radius = 3.0  # for unified: candidates around center

    def grid_to_xy(g):
        r, c = g // 3, g % 3
        return (c * grid_spacing, -r * grid_spacing)

    def cand_xy(cid):
        """Position for candidate cid in ring around center (2,2)."""
        angle = 2 * math.pi * cid / 8 - math.pi / 2  # start from top
        cx, cy = grid_to_xy(8)
        return (cx + cand_ring_radius * math.cos(angle), cy + cand_ring_radius * math.sin(angle))

    attr_to_id = RELATIONS.index("attr_to_obj")
    attr_to_obj = {}
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if int(edge_type[i]) == attr_to_id:
            attr_to_obj[u] = v

    is_unified = candidate_id is not None
    cid = candidate_id.tolist() if is_unified and hasattr(candidate_id, "tolist") else ([0] * n if not is_unified else list(candidate_id))

    for i in range(n):
        if int(node_kinds[i].item()) == 1:
            if is_unified and cid[i] >= 0:
                pos[i] = cand_xy(cid[i])
            else:
                g = int(node_attrs[i, 4].item()) if node_attrs.size(1) > 4 else 0
                pos[i] = grid_to_xy(g)

    # Group objects by (grid_idx, candidate_id) for unified; by grid_idx otherwise
    panels_to_objs = {}
    for i in range(n):
        if int(node_kinds[i].item()) == 0:
            g = int(node_attrs[i, 4].item()) if node_attrs.size(1) > 4 else 0
            if is_unified and cid[i] >= 0:
                key = ("cand", cid[i])
            else:
                key = g
            panels_to_objs.setdefault(key, []).append(i)

    for key, objs in panels_to_objs.items():
        if isinstance(key, tuple) and key[0] == "cand":
            cx, cy = cand_xy(key[1])
        else:
            cx, cy = grid_to_xy(key)
        for k, idx in enumerate(objs):
            angle = 2 * math.pi * k / max(len(objs), 1)
            pos[idx] = (cx + obj_radius * math.cos(angle), cy + obj_radius * math.sin(angle))

    for i in range(n):
        if int(node_kinds[i].item()) >= 2:
            obj = attr_to_obj.get(i)
            if obj is not None and obj in pos:
                ox, oy = pos[obj]
                off = (i % 4) * 0.5
                pos[i] = (ox + attr_radius * math.cos(off), oy + attr_radius * math.sin(off))
            else:
                if is_unified and cid[i] >= 0:
                    cx, cy = cand_xy(cid[i])
                else:
                    g = int(node_attrs[i, 4].item()) if node_attrs.size(1) > 4 else 0
                    cx, cy = grid_to_xy(g)
                pos[i] = (cx + attr_radius, cy + attr_radius)
    return pos


def _save_problem_image(images: np.ndarray, target: int, out_path: str) -> None:
    """Save the RPM problem as a 3x3 context grid + 8 candidates to problem.png."""
    import matplotlib.pyplot as plt
    # images: (16, H, W), panels 0-7 context, 8-15 candidates
    H, W = images.shape[1], images.shape[2]
    if images.dtype == np.float32 or images.dtype == np.float64:
        images = (np.clip(images, 0, 1) * 255).astype(np.uint8)
    pad = 4
    cell = max(H, W) + pad
    ctx_w = 3 * cell - pad
    cand_w = 8 * cell - pad
    total_w = max(ctx_w, cand_w)
    total_h = 4 * cell - pad
    fig, ax = plt.subplots(1, 1, figsize=(total_w / 50, total_h / 50))
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    # 3x3 context (8 panels + ? for answer slot)
    ctx_x0 = (total_w - ctx_w) / 2
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            x = ctx_x0 + col * cell
            y = row * cell
            if idx < 8:
                img = images[idx]
            else:
                img = np.full((H, W), 200, dtype=np.uint8)
            ax.imshow(img, extent=[x, x + W, y + H, y], cmap="gray", aspect="auto")
    # 8 candidates
    cand_x0 = (total_w - cand_w) / 2
    for i in range(8):
        x = cand_x0 + i * cell
        y = 3 * cell
        img = images[8 + i]
        ax.imshow(img, extent=[x, x + W, y + H, y], cmap="gray", aspect="auto")
        if i == target:
            rect = plt.Rectangle((x - 1, y - 1), W + 2, H + 2, fill=False, edgecolor="green", linewidth=3)
            ax.add_patch(rect)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=80, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def _node_label(attrs, kind):
    """Format node label from attrs and kind."""
    if kind == 0:  # object
        pos_idx, col, typ, siz = int(attrs[0]), int(attrs[1]), int(attrs[2]), int(attrs[3])
        pos_name = POSITION_NAMES.get(pos_idx, f"p{pos_idx}")
        t = TYPE_NAMES.get(typ, f"t{typ}")
        s = SIZE_NAMES.get(siz, f"s{siz}")
        c = COLOR_NAMES.get(col, f"c{col}")
        return f"obj\n{pos_name}\n{t},{s},{c}"
    if kind == 1:  # panel
        return "panel"
    if kind == 2:
        pos_idx = int(attrs[0])
        return POSITION_NAMES.get(pos_idx, f"pos{pos_idx}")
    if kind == 3:
        return f"col{int(attrs[1])}"
    if kind == 4:
        return TYPE_NAMES.get(int(attrs[2]), f"t{attrs[2]}")
    if kind == 5:
        return SIZE_NAMES.get(int(attrs[3]), f"s{attrs[3]}")
    return "?"


def main():
    parser = argparse.ArgumentParser(description="Draw GNN graph (train_object_vsa_gnn_frontend style)")
    parser.add_argument("--sample", type=str, required=True, help="Path to .npz sample")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use-gt", action="store_true", help="Use ground truth from XML")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--candidate", type=int, default=None,
                        help="Candidate index 0-7 (default: target/correct answer)")
    parser.add_argument("--granular", action="store_true", help="Include attribute nodes")
    parser.add_argument("--object-only", action="store_true",
                        help="Object-only graph (no panel nodes, same-pos cross-panel edges)")
    parser.add_argument("--unified-graph", action="store_true",
                        help="Unified graph: all 8 candidates connected to context panels 5 and 7")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path. Default: <config>/<example_num>/graph_<mode>.png (config from sample path)")
    parser.add_argument("--threshold", type=float, default=0.23)
    parser.add_argument("--resize", type=int, default=160)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--figsize", type=int, default=16)
    parser.add_argument("--font-size", type=float, default=5.0)
    args = parser.parse_args()

    if not os.path.isfile(args.sample):
        parser.error(f"Sample not found: {args.sample}")

    # Extract example number from path (e.g. RAVEN_0_train.npz -> 0)
    sample_basename = os.path.basename(args.sample)
    example_match = re.search(r"RAVEN_(\d+)_", sample_basename)
    example_num = int(example_match.group(1)) if example_match else 0

    data = np.load(args.sample)
    target = int(data["target"])
    candidate_idx = args.candidate if args.candidate is not None else target

    if args.use_gt:
        config = args.config or get_config_from_path(args.sample)
        if not config:
            parser.error("Cannot infer config; pass --config")
        xml_path = args.sample.replace(".npz", ".xml")
        if not os.path.isfile(xml_path):
            parser.error(f"XML not found: {xml_path}")
        gt_panels = parse_ground_truth(xml_path, config)
        obj_type, obj_size, obj_color, obj_exists = gt_to_dense(gt_panels)
        data_source = "GT"
    else:
        if not args.checkpoint or not os.path.isdir(args.checkpoint):
            parser.error("--checkpoint required without --use-gt")
        device = torch.device(args.device)
        model, cos_layer, _, _ = load_frontend(args.checkpoint, device=device)
        images = data["image"]
        if args.resize and images.shape[1] != args.resize:
            from torch.nn.functional import interpolate
            imgs_t = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
            imgs_t = interpolate(imgs_t, size=(args.resize, args.resize), mode="bilinear", align_corners=False)
            images = imgs_t.squeeze(1).numpy()
        with torch.no_grad():
            cosine_sim = extract_cosine_sims(model, cos_layer, images, device)
        obj_type, obj_size, obj_color, obj_exists, _ = extract_discrete_objects(cosine_sim, args.threshold)
        data_source = "frontend"

    # Position centers for spatial relations (must be keyed by universal position 0-21)
    config = args.config or (get_config_from_path(args.sample) if args.sample else None)
    xml_path = args.sample.replace(".npz", ".xml")
    if os.path.isfile(xml_path) and config:
        try:
            centers = build_position_centers_universal([xml_path], config)
            # Merge with fallback for any universal positions not in this config
            fallback = fallback_position_centers()
            centers = {**fallback, **centers}
        except Exception:
            centers = fallback_position_centers()
    else:
        centers = fallback_position_centers()

    # Build graph (same as train_object_vsa_gnn_frontend)
    if args.unified_graph:
        g = build_unified_graph(
            obj_type, obj_size, obj_color, obj_exists,
            centers, granular=args.granular, object_only=args.object_only
        )
    else:
        g = build_candidate_graph(
            obj_type, obj_size, obj_color, obj_exists,
            candidate_idx, centers, granular=args.granular, object_only=args.object_only
        )

    node_attrs = g["node_attrs"]
    node_kinds = g["node_kinds"]
    edge_index = g["edge_index"]
    edge_type = g["edge_type"]
    candidate_id = g.get("candidate_id")

    # Output path: <config>/<example_num>/graph_<mode>.png (config = distribute_four, distribute_nine, etc.)
    if args.output is None:
        config = get_config_from_path(args.sample)
        mode_suffix = []
        if args.granular:
            mode_suffix.append("gran")
        if args.object_only:
            mode_suffix.append("obj")
        if args.unified_graph:
            mode_suffix.append("unified")
        mode_str_file = "_" + "_".join(mode_suffix) if mode_suffix else ""
        out_dir = os.path.join(config, str(example_num))
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f"graph{mode_str_file}.png")
        # Save the RPM problem image in the same directory
        _save_problem_image(data["image"], target, os.path.join(out_dir, "problem.png"))

    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install networkx and matplotlib: pip install networkx matplotlib")
        sys.exit(1)

    G = nx.DiGraph()
    n = node_attrs.size(0)
    for i in range(n):
        G.add_node(i)
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        et = int(edge_type[i])
        G.add_edge(u, v, rel=RELATIONS[et])

    labels = {}
    for i in range(n):
        attrs = node_attrs[i].tolist()
        kind = int(node_kinds[i].item())
        labels[i] = _node_label(attrs, kind)

    # Node colors by kind
    kind_colors = ["#92c5de", "#f4a582", "#fee090", "#d9f0a3", "#c7eae5", "#e0d4f7"]
    node_colors = [kind_colors[min(k, 5)] for k in node_kinds.tolist()]

    # Edge colors by relation
    rel_colors = plt.cm.tab20(np.linspace(0, 1, len(RELATIONS)))
    edge_colors = [rel_colors[et] for et in edge_type.tolist()]

    pos = _panel_layout(node_attrs, node_kinds, edge_index, edge_type, candidate_id=candidate_id)

    fig, ax = plt.subplots(figsize=(args.figsize, args.figsize), facecolor="white")
    ax.set_facecolor("white")

    # Draw edges first (behind nodes) - curved, with clear arrows
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, alpha=0.5, ax=ax,
        arrows=True, arrowsize=12, arrowstyle="-|>", min_source_margin=12, min_target_margin=12,
        connectionstyle="arc3,rad=0.1", width=1.2
    )

    # Node shapes: circles for objects/attrs, squares for panels
    node_sizes = []
    node_shapes = []
    for k in node_kinds.tolist():
        kind = min(int(k), 5)
        node_sizes.append(600 if kind == 0 else (500 if kind == 1 else 280))
        node_shapes.append("s" if kind == 1 else "o")  # square for panel, circle for rest

    # Draw nodes by shape (networkx doesn't support mixed shapes directly, so draw in two passes)
    circ_nodes = [i for i in range(n) if node_shapes[i] == "o"]
    sq_nodes = [i for i in range(n) if node_shapes[i] == "s"]
    circ_sizes = [node_sizes[i] for i in circ_nodes]
    sq_sizes = [node_sizes[i] for i in sq_nodes]
    circ_colors = [node_colors[i] for i in circ_nodes]
    sq_colors = [node_colors[i] for i in sq_nodes]
    if circ_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=circ_nodes, node_color=circ_colors, node_size=circ_sizes,
            node_shape="o", ax=ax, alpha=0.95, edgecolors="gray", linewidths=1.5
        )
    if sq_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=sq_nodes, node_color=sq_colors, node_size=sq_sizes,
            node_shape="s", ax=ax, alpha=0.95, edgecolors="gray", linewidths=1.5
        )

    nx.draw_networkx_labels(
        G, pos, labels, font_size=args.font_size, ax=ax,
        font_weight="normal", font_color="black"
    )

    mode_parts = []
    if args.granular:
        mode_parts.append("granular")
    if args.object_only:
        mode_parts.append("object-only")
    if args.unified_graph:
        mode_parts.append("unified")
    mode_str = " | " + " | ".join(mode_parts) if mode_parts else ""

    ax.set_title(
        f"RAVEN GNN | {os.path.basename(args.sample)} | "
        f"target={target}" + ("" if args.unified_graph else f" candidate={candidate_idx}") + f" | {data_source}"
        + mode_str
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved graph to {args.output}")


if __name__ == "__main__":
    main()
