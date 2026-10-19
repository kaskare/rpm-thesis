#!/usr/bin/env python3
"""
CLI runner for the PyNeuraLogic GNN solver on RAVEN / I-RAVEN.

Trains one model on ALL constellations: per-example graph = context panels +
one candidate; label = that candidate correct or not. Test = argmax over 8
candidate scores.

PyTorch parity (train_object_vsa_gnn_frontend.py / gnn/model.py CandidateVsaGNN):
  - Graph: object-only spatial edges (+ same_pos) analogous to graph.py object_only.
  - Readout: PyTorch pools context and candidate SEPARATELY then
    score_head(concat(ctx_emb, cand_emb)). Here we approximate that by four
    terms summing pooled h0/h3 over is_ctx nodes and over is_cand nodes
    (sum of projections, not concat+MLP — same information flow, weaker interaction).
Still different from PyTorch unless noted below:
  - Backbone: relational sum-GNN vs GAT(+optional Transformer), hidden sizes, dropout.
  - Loss: NeuraLogic trains one binary CE per subgraph; PyTorch uses softmax CE on
    [batch, 8] logits jointly. Joint softmax prefers one winner among eight in one step.
  - Position centers for spatial edges: grid fallback POS_CENTERS; PyTorch GT path uses
    bbox-derived centers when available (see gnn/data.py build_position_centers).
"""

import argparse
import glob
import json
import os
import random
import time
import xml.etree.ElementTree as ET

import numpy as np

import neuralogic
neuralogic.set_max_memory_size(64)

from neuralogic.core import R, V, Template, Settings
from neuralogic.dataset import Dataset
from neuralogic.nn import get_evaluator
from neuralogic.optim import Adam
from neuralogic.nn.loss import CrossEntropy

# ── Constants ───────────────────────────────────────────────────────

NUM_CONTEXT = 8
NUM_CANDIDATES = 8
NUM_POSITIONS = 22
NUM_SHAPE = 5
NUM_SIZE = 6
NUM_COLOR = 10

CONFIG_POS_MAP = {
    "center_single": {0: 0},
    "distribute_four": {0: 1, 1: 2, 2: 3, 3: 4},
    "distribute_nine": {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: 13},
    "left_center_single_right_center_single": {0: 14, 1: 15},
    "up_center_single_down_center_single": {0: 16, 1: 17},
    "in_center_single_out_center_single": {0: 0, 1: 9},
    "in_distribute_four_out_center_single": {0: 0, 1: 18, 2: 19, 3: 20, 4: 21},
}
ALL_CONFIGS = list(CONFIG_POS_MAP.keys())

CONTEXT_ROWS = [[0, 1, 2], [3, 4, 5]]
CONTEXT_COLS = [[0, 3, 6], [1, 4, 7]]

# Spatial layout: fallback grid coordinates for universal positions 0-21
# (same as gnn/data.py fallback_position_centers)
POS_CENTERS = {i: (float(i % 6), float(i // 6)) for i in range(NUM_POSITIONS)}

# Panel -> (row, col) in the 3x3 RPM grid; all candidates share cell (2,2)
PANEL_TO_COORD = {
    0: (0, 0), 1: (0, 1), 2: (0, 2),
    3: (1, 0), 4: (1, 1), 5: (1, 2),
    6: (2, 0), 7: (2, 1),
}
for _c in range(NUM_CANDIDATES):
    PANEL_TO_COORD[NUM_CONTEXT + _c] = (2, 2)

PANEL_SPACING = 10.0

SPATIAL_RELS = ["left_of", "right_of", "above", "below", "same_row", "same_col"]


def _infer_spatial_rels(sx, sy, dx, dy, eps=1e-4):
    """Infer spatial relations from source (sx,sy) to dest (dx,dy)."""
    rels = []
    if sx < dx - eps:
        rels.append("left_of")
    elif sx > dx + eps:
        rels.append("right_of")
    if sy < dy - eps:
        rels.append("above")
    elif sy > dy + eps:
        rels.append("below")
    if abs(sy - dy) <= eps:
        rels.append("same_row")
    if abs(sx - dx) <= eps:
        rels.append("same_col")
    return rels


def _global_pos(panel_idx, pos):
    """Global (x, y) for an object at panel_idx and universal position pos."""
    r, c = PANEL_TO_COORD[panel_idx]
    lx, ly = POS_CENTERS.get(pos, (0.0, 0.0))
    return (c * PANEL_SPACING + lx, r * PANEL_SPACING + ly)


# ── Data loading ────────────────────────────────────────────────────

def parse_ground_truth(xml_path, config):
    pos_map = CONFIG_POS_MAP[config]
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_panels = []
    for panel in root[0]:
        panel_objects = []
        global_pos_offset = 0
        for struct in panel:
            for comp in struct:
                for layout in comp:
                    all_positions = eval(layout.attrib["Position"])
                    for entity in layout:
                        bbox = eval(entity.attrib["bbox"])
                        local_pos = all_positions.index(bbox) + global_pos_offset
                        panel_objects.append({
                            "position": pos_map[local_pos],
                            "type": int(entity.attrib["Type"]) - 1,
                            "size": int(entity.attrib["Size"]),
                            "color": int(entity.attrib["Color"]),
                        })
                    global_pos_offset += len(all_positions)
        gt_panels.append(sorted(panel_objects, key=lambda o: o["position"]))
    return gt_panels


def gt_to_dense(gt_panels):
    obj_type = np.zeros((16, NUM_POSITIONS), dtype=np.int64)
    obj_size = np.zeros((16, NUM_POSITIONS), dtype=np.int64)
    obj_color = np.zeros((16, NUM_POSITIONS), dtype=np.int64)
    obj_exists = np.zeros((16, NUM_POSITIONS), dtype=bool)
    for p, objs in enumerate(gt_panels):
        for o in objs:
            pos = o["position"]
            if 0 <= pos < NUM_POSITIONS:
                obj_type[p, pos] = min(o["type"], NUM_SHAPE - 1)
                obj_size[p, pos] = min(o["size"], NUM_SIZE - 1)
                obj_color[p, pos] = min(o["color"], NUM_COLOR - 1)
                obj_exists[p, pos] = True
    return obj_type, obj_size, obj_color, obj_exists


def load_samples(dataset_root, config, split, max_samples=None):
    config_dir = os.path.join(dataset_root, config)
    npz_files = sorted(glob.glob(os.path.join(config_dir, f"*_{split}.npz")))
    if max_samples:
        npz_files = npz_files[:max_samples]
    samples = []
    for npz_path in npz_files:
        data = np.load(npz_path)
        target = int(data["target"])
        xml_path = npz_path.replace(".npz", ".xml")
        if not os.path.isfile(xml_path):
            continue
        gt_panels = parse_ground_truth(xml_path, config)
        obj_type, obj_size, obj_color, obj_exists = gt_to_dense(gt_panels)
        samples.append((obj_type, obj_size, obj_color, obj_exists, target))
    return samples


# ── Per-candidate sub-graph fact construction ───────────────────────

def _nid(panel, pos):
    return f"p{panel}_o{pos}"


def rpm_candidate_subgraph_facts(obj_type, obj_size, obj_color, obj_exists, cand_idx):
    """
    Build object-only sub-graph facts: 8 context panels + 1 candidate panel.

    Node attributes: type, size, color, position (4 facts per object).
    Edges (object-only, matching gnn/graph.py):
      - Intra-panel: spatial relations (left_of/right_of/above/below/same_row/same_col)
        between all pairs of objects within the same panel.
      - Cross-panel: same_pos_cross_panel between objects at the same universal
        position in different panels.
      - Cross-panel spatial: objects in grid-adjacent panels get spatial edges
        based on their global (panel + local) coordinates.
    """
    facts = []
    cand_panel = NUM_CONTEXT + cand_idx
    active_panels = list(range(NUM_CONTEXT)) + [cand_panel]

    # Collect which nodes exist per panel for edge construction
    panel_nodes = {}  # panel -> [(pos, node_id), ...]
    for panel in active_panels:
        pnodes = []
        for pos in range(NUM_POSITIONS):
            if not obj_exists[panel, pos]:
                continue
            n = _nid(panel, pos)
            # 4 attribute facts: type, size, color, position
            facts.append(R.get(f"t{int(obj_type[panel, pos])}")(n))
            facts.append(R.get(f"s{int(obj_size[panel, pos])}")(n))
            facts.append(R.get(f"c{int(obj_color[panel, pos])}")(n))
            facts.append(R.get(f"pos{pos}")(n))
            pnodes.append((pos, n))
        panel_nodes[panel] = pnodes

    # --- Intra-panel spatial edges ---
    for panel in active_panels:
        nodes = panel_nodes[panel]
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                pi, ni = nodes[i]
                pj, nj = nodes[j]
                sx, sy = POS_CENTERS[pi]
                dx, dy = POS_CENTERS[pj]
                for rel_name in _infer_spatial_rels(sx, sy, dx, dy):
                    facts.append(R.get(rel_name)(ni, nj))

    # --- Cross-panel: same_pos_cross_panel ---
    for pos in range(NUM_POSITIONS):
        nodes_at_pos = []
        for panel in active_panels:
            for p, n in panel_nodes[panel]:
                if p == pos:
                    nodes_at_pos.append(n)
        for i in range(len(nodes_at_pos)):
            for j in range(i + 1, len(nodes_at_pos)):
                facts.append(R.same_pos(nodes_at_pos[i], nodes_at_pos[j]))
                facts.append(R.same_pos(nodes_at_pos[j], nodes_at_pos[i]))

    # --- Cross-panel spatial edges (grid-adjacent panels) ---
    coord_to_panels = {}
    for p in active_panels:
        rc = PANEL_TO_COORD[p]
        coord_to_panels.setdefault(rc, []).append(p)

    for (r, c), panels_at in coord_to_panels.items():
        for neighbor_rc in [(r, c + 1), (r + 1, c)]:
            neighbor_panels = coord_to_panels.get(neighbor_rc, [])
            for p in panels_at:
                for q in neighbor_panels:
                    for pi, ni in panel_nodes[p]:
                        gx_i, gy_i = _global_pos(p, pi)
                        for pj, nj in panel_nodes[q]:
                            gx_j, gy_j = _global_pos(q, pj)
                            for rel_name in _infer_spatial_rels(gx_i, gy_i, gx_j, gy_j):
                                facts.append(R.get(rel_name)(ni, nj))
                            for rel_name in _infer_spatial_rels(gx_j, gy_j, gx_i, gy_i):
                                facts.append(R.get(rel_name)(nj, ni))

    # --- Mask context vs candidate (align with gnn/model.py CandidateVsaGNN.forward_one) ---
    # PyTorch: ctx_emb = scatter_mean(context nodes); cand_emb = scatter_mean(candidate nodes);
    #          logits = score_head(cat([ctx_emb, cand_emb])). We approximate concat+MLP by
    #          summing two learned projections: pool(h|is_ctx) + pool(h|is_cand).
    for p in range(NUM_CONTEXT):
        for _, n in panel_nodes.get(p, []):
            facts.append(R.is_ctx(n))
    for pos, n in panel_nodes[cand_panel]:
        facts.append(R.is_cand(n))

    return facts


# ── Template ────────────────────────────────────────────────────────

def _add_gnn_layer(template, h_out, h_in, D, edge_rels):
    """Add one GNN layer: message passing along each edge type + self-connection."""
    for rel in edge_rels:
        template += R.get(h_out)(V.X) <= (R.get(h_in)(V.X)[D, D], R.get(h_in)(V.Y)[D, D], R.get(rel)(V.X, V.Y))
    template += R.get(h_out)(V.X) <= R.get(h_in)(V.X)[D, D]


def build_rpm_gnn_template(embed_dim=32):
    D = embed_dim
    template = Template()

    # Attribute embeddings (type, size, color, position)
    template.add_rules([R.type_emb(V.X)[D, ] <= R.get(f"t{t}")(V.X) for t in range(NUM_SHAPE)])
    template.add_rules([R.size_emb(V.X)[D, ] <= R.get(f"s{s}")(V.X) for s in range(NUM_SIZE)])
    template.add_rules([R.color_emb(V.X)[D, ] <= R.get(f"c{c}")(V.X) for c in range(NUM_COLOR)])
    template.add_rules([R.pos_emb(V.X)[D, ] <= R.get(f"pos{p}")(V.X) for p in range(NUM_POSITIONS)])

    # Initial node embedding: sum of 4 attribute embeddings
    template += R.h0(V.X) <= (R.type_emb(V.X), R.size_emb(V.X), R.color_emb(V.X), R.pos_emb(V.X))

    # Edge types: 6 spatial relations + same_pos_cross_panel
    edge_rels = SPATIAL_RELS + ["same_pos"]

    # GNN Layers 1-3
    _add_gnn_layer(template, "h1", "h0", D, edge_rels)
    _add_gnn_layer(template, "h2", "h1", D, edge_rels)
    _add_gnn_layer(template, "h3", "h2", D, edge_rels)

    # Readout mirrors PyTorch score_head(cat(ctx_pool, cand_pool)) roughly:
    # four terms = deep/shallow × context/candidate pooled reprs -> summed into scalar predict.
    template += R.predict[1, D] <= (R.h3(V.X), R.is_ctx(V.X))
    template += R.predict[1, D] <= (R.h3(V.X), R.is_cand(V.X))
    template += R.predict[1, D] <= (R.h0(V.X), R.is_ctx(V.X))
    template += R.predict[1, D] <= (R.h0(V.X), R.is_cand(V.X))

    return template


# ── Dataset construction ────────────────────────────────────────────

def build_pynl_dataset(samples, oversample_pos=1):
    """
    Build per-candidate sub-graph dataset.
    oversample_pos: repeat correct-candidate examples this many times (1=no oversampling).
    """
    dataset = Dataset()
    for obj_type, obj_size, obj_color, obj_exists, target in samples:
        for ci in range(NUM_CANDIDATES):
            facts = rpm_candidate_subgraph_facts(obj_type, obj_size, obj_color, obj_exists, ci)
            label = 1.0 if ci == target else 0.0
            repeats = oversample_pos if ci == target else 1
            for _ in range(repeats):
                dataset.add_example(facts)
                dataset.add_query(R.predict[label])
    return dataset


# ── Evaluation ──────────────────────────────────────────────────────

def evaluate_accuracy(evaluator, built_dataset, samples):
    """Score 8 sub-graphs per RPM, pick argmax."""
    preds = list(evaluator.test(built_dataset))
    n = len(samples)
    correct = 0
    for i in range(n):
        scores = preds[i * 8 : (i + 1) * 8]
        if int(np.argmax(scores)) == samples[i][4]:
            correct += 1
    return correct, n


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PyNeuraLogic GNN solver for RAVEN — per-candidate sub-graph approach",
    )
    parser.add_argument("--dataset-root", type=str, default="data/I-RAVEN-10000")
    parser.add_argument("--n-train", type=int, default=500,
                        help="Max training samples PER constellation")
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--oversample-pos", type=int, default=3,
                        help="Repeat correct-candidate examples N times during training")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    configs = ALL_CONFIGS

    print("PyNeuraLogic GNN solver (per-candidate sub-graph)")
    print(f"  dataset:        {args.dataset_root}")
    print(f"  configs:        {configs}")
    print(f"  embed_dim:      {args.embed_dim}")
    print(f"  epochs:         {args.epochs}")
    print(f"  lr:             {args.lr}")
    print(f"  oversample_pos: {args.oversample_pos}x")
    print(f"  n_train:        {args.n_train}/config  n_val: {args.n_val}  n_test: {args.n_test}")
    print(f"  seed:           {args.seed}")

    # ── Load data ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Loading data from all constellations...")
    print(f"{'=' * 60}")

    train_samples_all = []
    val_per_config = {}
    test_per_config = {}

    for cfg in configs:
        t = load_samples(args.dataset_root, cfg, "train", max_samples=args.n_train)
        v = load_samples(args.dataset_root, cfg, "val", max_samples=args.n_val)
        ts = load_samples(args.dataset_root, cfg, "test", max_samples=args.n_test)
        train_samples_all.extend(t)
        val_per_config[cfg] = v
        test_per_config[cfg] = ts
        print(f"  {cfg:50s}  train={len(t):4d}  val={len(v):4d}  test={len(ts):4d}")

    random.shuffle(train_samples_all)
    total_train = len(train_samples_all)
    n_train_examples = total_train * (7 + args.oversample_pos)
    print(f"\n  Total train RPMs: {total_train}")
    print(f"  Train sub-graphs: {n_train_examples} ({args.oversample_pos} pos + 7 neg per RPM)")

    if not train_samples_all:
        print("ERROR: no training data found.")
        return

    # ── Build datasets ──────────────────────────────────────────────
    print(f"\nBuilding training dataset...")
    t0 = time.time()
    train_ds = build_pynl_dataset(train_samples_all, oversample_pos=args.oversample_pos)
    print(f"  Built in {time.time() - t0:.1f}s")

    # ── Build model ─────────────────────────────────────────────────
    template = build_rpm_gnn_template(args.embed_dim)
    settings = Settings(optimizer=Adam(lr=args.lr), epochs=args.epochs, error_function=CrossEntropy())
    evaluator = get_evaluator(template, settings)

    print(f"\nGrounding training data ({n_train_examples} sub-graphs)...")
    t0 = time.time()
    built_train = evaluator.build_dataset(train_ds)
    print(f"  Grounded in {time.time() - t0:.1f}s")

    # ── Train ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Training — {args.epochs} epochs")
    print(f"{'=' * 60}")
    t0 = time.time()
    log_every = max(1, args.epochs // 10)
    for epoch_i, (loss, n) in enumerate(evaluator.train(built_train), 1):
        if epoch_i % log_every == 0 or epoch_i == args.epochs:
            print(f"  epoch {epoch_i:4d}/{args.epochs}  loss={loss / n:.4f}")
    print(f"  Training done in {time.time() - t0:.1f}s")

    # ── Evaluate per-constellation (no oversampling) ────────────────
    print(f"\n{'=' * 60}")
    print("Per-constellation evaluation")
    print(f"{'=' * 60}")

    all_results = {}

    for cfg in configs:
        ts = test_per_config[cfg]
        if not ts:
            print(f"  {cfg}: no test data")
            continue

        test_ds = build_pynl_dataset(ts, oversample_pos=1)
        built_test = evaluator.build_dataset(test_ds)
        tc, tn = evaluate_accuracy(evaluator, built_test, ts)
        acc = tc / tn if tn else 0.0

        v = val_per_config[cfg]
        val_acc = None
        if v:
            val_ds = build_pynl_dataset(v, oversample_pos=1)
            built_val = evaluator.build_dataset(val_ds)
            vc, vn = evaluate_accuracy(evaluator, built_val, v)
            val_acc = vc / vn if vn else 0.0

        all_results[cfg] = {"test_acc": acc, "test_correct": tc, "test_total": tn}
        if val_acc is not None:
            all_results[cfg]["val_acc"] = val_acc

        val_str = f"  val={val_acc * 100:5.1f}%" if val_acc is not None else ""
        print(f"  {cfg:50s}  test={acc * 100:6.2f}% ({tc}/{tn}){val_str}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for cfg, res in all_results.items():
        print(f"  {cfg:50s}  {res['test_acc'] * 100:6.2f}%")
    if all_results:
        mean_acc = np.mean([r["test_acc"] for r in all_results.values()])
        print(f"  {'MEAN':50s}  {mean_acc * 100:6.2f}%")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
