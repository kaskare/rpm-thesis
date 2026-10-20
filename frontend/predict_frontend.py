"""
Universal NVSA Frontend – Predict & Inspect

Loads a trained universal frontend checkpoint and runs inference on a single
RAVEN sample, showing predictions vs ground truth side by side.

Usage:
    # Basic prediction (prints to console)
    python frontend/predict_frontend.py \
        --checkpoint log/universal_frontend_test4 \
        --sample data/RAVEN-10000/center_single/RAVEN_0_train.npz

    # With visualization (saves annotated PNG)
    python frontend/predict_frontend.py \
        --checkpoint log/universal_frontend_test4 \
        --sample data/RAVEN-10000/distribute_four/RAVEN_42_train.npz \
        --visualize

    # Custom threshold and device
    python frontend/predict_frontend.py \
        --checkpoint log/universal_frontend_test4 \
        --sample data/RAVEN-10000/distribute_nine/RAVEN_0_train.npz \
        --threshold 0.2 --device cuda:0

    # Batch evaluation over a whole constellation/split
    python frontend/predict_frontend.py \
        --checkpoint log/universal_frontend_test4 \
        --dataset data/RAVEN-10000 \
        --config center_single --split val --max-samples 100
"""

import argparse, os, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'raven'))
sys.path.insert(0, os.path.join(ROOT, 'neuro-vector-symbolic-architectures'))

import numpy as np
import torch
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

from nvsa.perception.resnet import resnet18
from nvsa.perception.metrics import fixCos
from const import DIM_TYPE, DIM_SIZE, DIM_COLOR
from gnn.data import canonical_positions


# ── Constants ────────────────────────────────────────────────────────────────

TYPE_NAMES  = {0: "triangle", 1: "square", 2: "pentagon", 3: "hexagon", 4: "circle"}
SIZE_NAMES  = {i: f"size_{i}" for i in range(DIM_SIZE)}
COLOR_NAMES = {i: f"color_{i}" for i in range(DIM_COLOR)}

ALL_CONFIGS = [
    "center_single",
    "distribute_four",
    "distribute_nine",
    "left_center_single_right_center_single",
    "up_center_single_down_center_single",
    "in_center_single_out_center_single",
    "in_distribute_four_out_center_single",
]

# ── Universal position mapping ───────────────────────────────────────────────
#
# Maps each constellation's LOCAL position index (as computed from the XML via
# local_pos + global_pos_offset) to a UNIVERSAL position ID in the codebook.
#
# Shared positions (same physical bbox on the image):
#   • Universal 0  [0.5, 0.5, 1, 1]      = center_single(0) = in_out outer(0) = in_d4_out outer(0)
#   • Universal 9  [0.5, 0.5, 0.33, 0.33] = distribute_nine center(4) = in_out inner(1)
#
# Layout of universal IDs:
#    0          center / outer large shape
#    1– 4       distribute_four quadrants
#    5–13       distribute_nine 3×3 grid  (9 = center cell)
#   14–15       left_right  (left, right)
#   16–17       up_down     (up, down)
#   18–21       in_distribute_four inner 4 positions
#   Total: 22
#
CONFIG_POS_MAP = {
    "center_single":                              {0: 0},
    "distribute_four":                            {0: 1, 1: 2, 2: 3, 3: 4},
    "distribute_nine":                            {0: 5, 1: 6, 2: 7, 3: 8, 4: 9,
                                                   5: 10, 6: 11, 7: 12, 8: 13},
    "left_center_single_right_center_single":     {0: 14, 1: 15},
    "up_center_single_down_center_single":        {0: 16, 1: 17},
    "in_center_single_out_center_single":         {0: 0, 1: 9},       # out=center, in=d9 center
    "in_distribute_four_out_center_single":       {0: 0, 1: 18, 2: 19, 3: 20, 4: 21},  # out=center
}

# Which universal position IDs each constellation touches (for evaluation)
CONFIG_ACTIVE_POSITIONS = {
    config: sorted(mapping.values())
    for config, mapping in CONFIG_POS_MAP.items()
}


# ── Index helpers (same as train_frontend.py) ────────────────────────────────

def compute_num_per_pos():
    return DIM_COLOR * DIM_SIZE * DIM_TYPE + 1   # +1 for detection vector


def compute_w_index(pos, typ, siz, col, num_per_pos):
    return pos * num_per_pos + col * DIM_SIZE * DIM_TYPE + siz * DIM_TYPE + typ


def decode_w_index(idx, num_per_pos):
    """Decode a W row index → (pos, type, size, color). Returns None for detection vectors."""
    pos = idx // num_per_pos
    remainder = idx % num_per_pos
    if remainder == num_per_pos - 1:
        return None   # object detection vector
    col = remainder // (DIM_SIZE * DIM_TYPE)
    remainder2 = remainder % (DIM_SIZE * DIM_TYPE)
    siz = remainder2 // DIM_TYPE
    typ = remainder2 % DIM_TYPE
    return (pos, typ, siz, col)


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_dir, device="cpu", use_best=True):
    """Load universal frontend model from a checkpoint directory."""
    device = torch.device(device)

    # Load codebook
    cb_path = os.path.join(checkpoint_dir, "codebooks.pt")
    if not os.path.isfile(cb_path):
        raise FileNotFoundError(f"Codebook not found at {cb_path}")
    cb_data = torch.load(cb_path, map_location="cpu")
    perception_cb = cb_data["perception_cb"]
    num_pos = cb_data.get("num_pos", 22)

    n_out = perception_cb.shape[0]
    num_per_pos = compute_num_per_pos()
    d = perception_cb.shape[1]

    print(f"Codebook: W shape={perception_cb.shape}, num_pos={num_pos}, "
          f"num_per_pos={num_per_pos}, d={d}")

    # Load model checkpoint
    if use_best:
        ckpt_path = os.path.join(checkpoint_dir, "model_best.pth.tar")
    else:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pth.tar")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_d = ckpt.get("d", d)
    threshold = ckpt.get("threshold", 0.23)
    s = ckpt.get("s", 7.0)
    epoch = ckpt.get("epoch", "?")
    best_acc = ckpt.get("best_acc", "?")

    print(f"Checkpoint: epoch={epoch}, best_acc={best_acc}%, s={s:.3f}, "
          f"threshold={threshold}")

    # Build model
    model = resnet18(pretrained=True, progress=False, num_classes=ckpt_d, no_maxpool=True)
    cos_layer = fixCos(num_features=ckpt_d, num_classes=n_out, mat=perception_cb,
                       fixed_weights=True, s=s, trainable_s=True)

    model.load_state_dict(ckpt["state_dict_model"])
    cos_layer.load_state_dict(ckpt["state_dict_cos"])

    model.to(device).eval()
    cos_layer.to(device).eval()

    return model, cos_layer, num_pos, num_per_pos, n_out, device


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_panels(model, cos_layer, images, device, threshold=0.23):
    """
    Run universal frontend on panel images and return raw cosine similarities
    plus detected objects per panel.

    Parameters
    ----------
    images : np.ndarray (16, 160, 160)
    threshold : float — detection threshold τ

    Returns
    -------
    cosine_sim : Tensor (16, N_OUT) — raw cosine similarities
    """
    imgs = torch.tensor(images, dtype=torch.float).unsqueeze(1).to(device)  # (16,1,160,160)
    q = model(imgs)              # (16, d)
    W_norm = F.normalize(cos_layer.W.data)
    q_norm = F.normalize(q)
    cosine_sim = F.linear(q_norm, W_norm)   # (16, N_OUT)
    return cosine_sim.cpu()


def decode_predictions(cosine_sim, num_per_pos, num_pos, threshold=0.23):
    """
    Decode cosine similarities → per-panel list of detected objects.

    Returns list of 16 lists, each containing dicts:
        [{position, type, size, color, score}, ...]
    """
    all_panels = []
    for panel_idx in range(cosine_sim.shape[0]):
        sims = cosine_sim[panel_idx]
        detected_mask = sims > threshold
        detected_indices = detected_mask.nonzero(as_tuple=True)[0].tolist()

        objects = {}  # pos → best object (highest score)
        for w_idx in detected_indices:
            decoded = decode_w_index(w_idx, num_per_pos)
            if decoded is None:
                continue
            pos, typ, siz, col = decoded
            score = sims[w_idx].item()
            # Keep the highest-scoring detection per position
            if pos not in objects or score > objects[pos]["score"]:
                objects[pos] = {
                    "position": pos,
                    "type": typ,
                    "size": siz,
                    "color": col,
                    "score": score,
                }

        panel_objects = sorted(objects.values(), key=lambda o: o["position"])
        all_panels.append(panel_objects)
    return all_panels


# ── Ground truth ─────────────────────────────────────────────────────────────

def parse_ground_truth(xml_path, config):
    """
    Parse RAVEN XML → per-panel list of ground truth objects.

    Parameters
    ----------
    xml_path : str
    config   : str – constellation name, used to map local positions to
                     universal position IDs via CONFIG_POS_MAP.

    Returns list of 16 lists, each containing dicts:
        [{position, type, size, color}, ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_panels = root[0]

    pos_map = CONFIG_POS_MAP[config]
    # slot from bbox geometry: I-RAVEN's Position attribute lists only occupied
    # slots, so Position.index(bbox) is wrong there (see train_frontend.py)
    canon = canonical_positions(config, os.path.dirname(os.path.dirname(xml_path)))

    gt_panels = []
    for panel in xml_panels:
        panel_objects = []
        for struct in panel:
            for ci, comp in enumerate(struct):
                for layout in comp:
                    for entity in layout:
                        bbox = eval(entity.attrib["bbox"])
                        pos = pos_map[canon[(ci, tuple(bbox))]]
                        panel_objects.append({
                            "position": pos,
                            "type":     int(entity.attrib["Type"]) - 1,  # XML 1-indexed → 0-indexed
                            "size":     int(entity.attrib["Size"]),
                            "color":    int(entity.attrib["Color"]),
                        })
        gt_panels.append(sorted(panel_objects, key=lambda o: o["position"]))
    return gt_panels


# ── Display ──────────────────────────────────────────────────────────────────

def panel_label(idx):
    if idx < 8:
        return f"context {idx}"
    return f"answer candidate {idx - 8}"


def obj_str(obj, include_score=False):
    """Format one object as a readable string."""
    tname = TYPE_NAMES.get(obj["type"], f"type_{obj['type']}")
    s = f"pos={obj['position']}  {tname:<10s}  size={obj['size']}  color={obj['color']}"
    if include_score and "score" in obj:
        s += f"  (sim={obj['score']:.3f})"
    return s


def objects_match(pred, gt):
    """Check if a predicted object matches a ground truth object."""
    return (pred["position"] == gt["position"] and
            pred["type"] == gt["type"] and
            pred["size"] == gt["size"] and
            pred["color"] == gt["color"])


def panel_matches(pred_objects, gt_objects):
    """Check if predicted set exactly matches ground truth set."""
    if len(pred_objects) != len(gt_objects):
        return False
    pred_sorted = sorted(pred_objects, key=lambda o: o["position"])
    gt_sorted = sorted(gt_objects, key=lambda o: o["position"])
    for p, g in zip(pred_sorted, gt_sorted):
        if not objects_match(p, g):
            return False
    return True


def print_panel(panel_idx, pred_objects, gt_objects):
    """Pretty-print one panel: predictions vs ground truth."""
    label = panel_label(panel_idx)
    match = panel_matches(pred_objects, gt_objects)
    status = "\033[92m MATCH \033[0m" if match else "\033[91m MISMATCH \033[0m"

    print(f"\n  Panel {panel_idx:2d} ({label})  [{status}]")

    # Build per-position comparison
    all_positions = set()
    for o in pred_objects:
        all_positions.add(o["position"])
    for o in gt_objects:
        all_positions.add(o["position"])

    pred_by_pos = {o["position"]: o for o in pred_objects}
    gt_by_pos   = {o["position"]: o for o in gt_objects}

    for pos in sorted(all_positions):
        has_pred = pos in pred_by_pos
        has_gt   = pos in gt_by_pos

        if has_pred and has_gt:
            p = pred_by_pos[pos]
            g = gt_by_pos[pos]
            attrs_match = objects_match(p, g)
            mark = "\033[92m✓\033[0m" if attrs_match else "\033[91m✗\033[0m"
            tname_p = TYPE_NAMES.get(p["type"], f"t{p['type']}")
            tname_g = TYPE_NAMES.get(g["type"], f"t{g['type']}")

            # Show attribute-level comparison
            t_mark = "=" if p["type"] == g["type"] else "\033[91m≠\033[0m"
            s_mark = "=" if p["size"] == g["size"] else "\033[91m≠\033[0m"
            c_mark = "=" if p["color"] == g["color"] else "\033[91m≠\033[0m"

            print(f"    {mark} pos={pos}  "
                  f"type: {tname_p:<10s} {t_mark} {tname_g:<10s}  "
                  f"size: {p['size']} {s_mark} {g['size']}  "
                  f"color: {p['color']} {c_mark} {g['color']}  "
                  f"(sim={p.get('score', 0):.3f})")

        elif has_pred and not has_gt:
            p = pred_by_pos[pos]
            tname_p = TYPE_NAMES.get(p["type"], f"t{p['type']}")
            print(f"    \033[91m✗\033[0m pos={pos}  "
                  f"PRED: {tname_p} s={p['size']} c={p['color']} (sim={p.get('score',0):.3f})  "
                  f"\033[91m← FALSE POSITIVE\033[0m")

        elif has_gt and not has_pred:
            g = gt_by_pos[pos]
            tname_g = TYPE_NAMES.get(g["type"], f"t{g['type']}")
            print(f"    \033[91m✗\033[0m pos={pos}  "
                  f"GT:   {tname_g} s={g['size']} c={g['color']}  "
                  f"\033[91m← MISSED\033[0m")


# ── Visualization ────────────────────────────────────────────────────────────

def _get_font(size=12):
    for name in ["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf",
                 "UbuntuMono-R.ttf", "cour.ttf", "consola.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def visualize_sample(images, target, pred_panels, gt_panels, save_path, config=None):
    """Render 16 RAVEN panels as an annotated image with predictions & GT."""
    H, W = images.shape[1], images.shape[2]
    pad = 6
    ann_h = 60
    border = 3

    font = _get_font(11)
    font_sm = _get_font(10)

    cell_w = W + pad
    cell_h = H + ann_h + pad
    ctx_cols, ctx_rows = 3, 3
    ctx_w = ctx_cols * cell_w - pad
    ctx_h = ctx_rows * cell_h - pad

    ans_cols = 8
    ans_w = ans_cols * cell_w - pad
    ans_h = cell_h

    total_w = max(ctx_w, ans_w) + 2 * pad
    gap_between = 16
    title_h = 22
    total_h = title_h + ctx_h + gap_between + ans_h + pad

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    cfg_label = config or "unknown"
    draw.text((pad, 2), f"Config: {cfg_label}   |   Correct answer: candidate {target}",
              fill=(60, 60, 60), font=font)

    def _paste_panel(panel_idx, x0, y0, highlight=False):
        panel_img = Image.fromarray(images[panel_idx], mode="L").convert("RGB")
        if highlight:
            bordered = Image.new("RGB", (W + 2*border, H + 2*border), (0, 180, 0))
            bordered.paste(panel_img, (border, border))
            canvas.paste(bordered, (x0 - border, y0 - border))
        else:
            canvas.paste(panel_img, (x0, y0))

        pred = pred_panels[panel_idx]
        gt = gt_panels[panel_idx] if gt_panels else None
        match = panel_matches(pred, gt) if gt else None

        # Annotation lines
        lines = []
        for obj in pred:
            tname = TYPE_NAMES.get(obj["type"], f"t{obj['type']}")
            lines.append(f"p{obj['position']} {tname} s{obj['size']} c{obj['color']}")
        if not lines:
            lines = ["[empty]"]

        if match is not None:
            color = (0, 140, 0) if match else (200, 0, 0)
            prefix = "OK " if match else "X  "
        else:
            color = (60, 60, 60)
            prefix = ""

        ty = y0 + H + 2
        for line in lines[:4]:
            draw.text((x0, ty), prefix + line, fill=color, font=font_sm)
            ty += 14

    # Context grid
    ctx_x0 = (total_w - ctx_w) // 2
    ctx_y0 = title_h
    for row in range(ctx_rows):
        for col in range(ctx_cols):
            idx = row * 3 + col
            x = ctx_x0 + col * cell_w
            y = ctx_y0 + row * cell_h
            if idx < 8:
                _paste_panel(idx, x, y)
            else:
                qimg = Image.new("RGB", (W, H), (180, 180, 180))
                qdraw = ImageDraw.Draw(qimg)
                qdraw.text((W//2 - 6, H//2 - 8), "?", fill=(80, 80, 80), font=_get_font(24))
                canvas.paste(qimg, (x, y))

    # Answer candidates
    ans_x0 = (total_w - ans_w) // 2
    ans_y0 = ctx_y0 + ctx_h + gap_between
    for i in range(8):
        x = ans_x0 + i * cell_w
        y = ans_y0
        _paste_panel(8 + i, x, y, highlight=(i == target))

    canvas.save(save_path)
    print(f"\nVisualization saved to: {save_path}")


# ── Single sample ────────────────────────────────────────────────────────────

def predict_single(model, cos_layer, num_pos, num_per_pos, n_out, device,
                   npz_path, threshold=0.23, visualize=False, save_dir=None, output_path=None):
    """Run inference on a single .npz sample."""
    data = np.load(npz_path)
    images = data["image"]     # (16, 160, 160)
    target = int(data["target"])

    # Detect config from path
    config = None
    for c in ALL_CONFIGS:
        if c in npz_path:
            config = c
            break

    print(f"\n{'='*70}")
    print(f"Sample: {npz_path}")
    print(f"Config: {config or 'unknown'}   |   Correct answer: candidate {target}")
    print(f"Threshold: τ = {threshold}")
    print(f"{'='*70}")

    # Run model
    cosine_sim = predict_panels(model, cos_layer, images, device, threshold)
    pred_panels = decode_predictions(cosine_sim, num_per_pos, num_pos, threshold)

    # Load ground truth
    xml_path = npz_path.replace(".npz", ".xml")
    gt_panels = None
    if os.path.isfile(xml_path) and config is not None:
        gt_panels = parse_ground_truth(xml_path, config)

    # Print per-panel comparison
    panel_correct = 0
    for panel_idx in range(16):
        gt = gt_panels[panel_idx] if gt_panels else []
        print_panel(panel_idx, pred_panels[panel_idx], gt)
        if gt_panels and panel_matches(pred_panels[panel_idx], gt):
            panel_correct += 1

    # Summary
    if gt_panels:
        print(f"\n{'─'*70}")
        print(f"  Panel accuracy: {panel_correct}/16 ({100*panel_correct/16:.1f}%)")

        # Per-attribute summary
        attr_stats = {"exist": [0, 0], "type": [0, 0], "size": [0, 0], "color": [0, 0]}
        active_positions = CONFIG_ACTIVE_POSITIONS[config] if config else list(range(num_pos))
        for pi in range(16):
            gt = gt_panels[pi]
            pred = pred_panels[pi]
            gt_by_pos = {o["position"]: o for o in gt}
            pred_by_pos = {o["position"]: o for o in pred}

            for pos in active_positions:
                gt_exists = pos in gt_by_pos
                pred_exists = pos in pred_by_pos
                attr_stats["exist"][1] += 1
                if gt_exists == pred_exists:
                    attr_stats["exist"][0] += 1
                if gt_exists and pred_exists:
                    for attr in ["type", "size", "color"]:
                        attr_stats[attr][1] += 1
                        if pred_by_pos[pos][attr] == gt_by_pos[pos][attr]:
                            attr_stats[attr][0] += 1

        for attr in ["exist", "type", "size", "color"]:
            c, t = attr_stats[attr]
            pct = 100 * c / t if t > 0 else 0
            print(f"  {attr:6s}: {c}/{t} ({pct:.1f}%)")
        print(f"{'─'*70}")

    # Top cosine similarity entries per panel (diagnostic)
    print(f"\n  Top-5 cosine similarities per panel:")
    for panel_idx in range(16):
        sims = cosine_sim[panel_idx]
        topk_vals, topk_idxs = torch.topk(sims, k=min(5, sims.shape[0]))
        parts = []
        for val, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
            decoded = decode_w_index(idx, num_per_pos)
            if decoded:
                pos, typ, siz, col = decoded
                tname = TYPE_NAMES.get(typ, f"t{typ}")
                parts.append(f"{val:.3f}(p{pos} {tname} s{siz} c{col})")
            else:
                parts.append(f"{val:.3f}(det)")
        print(f"    Panel {panel_idx:2d}: {', '.join(parts)}")

    # Visualization
    if visualize:
        if output_path:
            save_path = os.path.abspath(output_path)
        else:
            basename = os.path.splitext(os.path.basename(npz_path))[0]
            out_dir = save_dir or os.path.dirname(npz_path)
            save_path = os.path.join(os.path.abspath(out_dir), f"{basename}_predict.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        visualize_sample(images, target, pred_panels, gt_panels, save_path, config)

    return pred_panels, gt_panels


# ── Batch evaluation ─────────────────────────────────────────────────────────

def evaluate_batch(model, cos_layer, num_pos, num_per_pos, n_out, device,
                   dataset_path, config, split, threshold=0.23, max_samples=None):
    """Evaluate on a whole constellation/split."""
    pattern = os.path.join(dataset_path, config, f"*_{split}.npz")
    npz_files = sorted(glob.glob(pattern))
    if not npz_files:
        print(f"No files found matching {pattern}")
        return

    if max_samples:
        npz_files = npz_files[:max_samples]

    print(f"\nEvaluating {len(npz_files)} samples from {config}/{split} ...")

    panel_correct = 0
    panel_total = 0
    attr_stats = {a: [0, 0] for a in ["exist", "type", "size", "color"]}

    for npz_path in npz_files:
        data = np.load(npz_path)
        images = data["image"]

        xml_path = npz_path.replace(".npz", ".xml")
        if not os.path.isfile(xml_path):
            continue

        cosine_sim = predict_panels(model, cos_layer, images, device, threshold)
        pred_panels = decode_predictions(cosine_sim, num_per_pos, num_pos, threshold)
        gt_panels = parse_ground_truth(xml_path, config)

        for pi in range(16):
            pred = pred_panels[pi]
            gt = gt_panels[pi]

            if panel_matches(pred, gt):
                panel_correct += 1
            panel_total += 1

            gt_by_pos = {o["position"]: o for o in gt}
            pred_by_pos = {o["position"]: o for o in pred}

            # Only check positions relevant to this constellation
            active_positions = CONFIG_ACTIVE_POSITIONS[config]
            for pos in active_positions:
                gt_exists = pos in gt_by_pos
                pred_exists = pos in pred_by_pos
                attr_stats["exist"][1] += 1
                if gt_exists == pred_exists:
                    attr_stats["exist"][0] += 1
                if gt_exists and pred_exists:
                    for attr in ["type", "size", "color"]:
                        attr_stats[attr][1] += 1
                        if pred_by_pos[pos][attr] == gt_by_pos[pos][attr]:
                            attr_stats[attr][0] += 1

    print(f"\n{'='*60}")
    print(f"Config: {config}  |  Split: {split}  |  Samples: {len(npz_files)}")
    print(f"{'='*60}")
    print(f"  Panel accuracy: {panel_correct}/{panel_total} "
          f"({100*panel_correct/panel_total:.2f}%)")
    print(f"\n  Per-attribute accuracy:")
    for attr in ["exist", "type", "size", "color"]:
        c, t = attr_stats[attr]
        pct = 100 * c / t if t > 0 else 0
        print(f"    {attr:6s}: {c}/{t} ({pct:.2f}%)")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Universal NVSA Frontend – predict & inspect single samples"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory "
                             "(contains model_best.pth.tar + codebooks.pt)")
    parser.add_argument("--use-latest", action="store_true",
                        help="Use checkpoint.pth.tar instead of model_best.pth.tar")

    # Single sample mode
    parser.add_argument("--sample", type=str, default=None,
                        help="Path to a single .npz file")

    # Batch mode
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to RAVEN-10000 root (for batch evaluation)")
    parser.add_argument("--config", type=str, default=None,
                        choices=ALL_CONFIGS,
                        help="Constellation for batch mode")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)

    # Inference
    parser.add_argument("--threshold", type=float, default=0.23,
                        help="Detection threshold τ")
    parser.add_argument("--device", type=str, default="cpu")

    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Save annotated PNG (single sample mode)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory for visualization output")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Explicit output path for PNG (overrides --save-dir)")

    args = parser.parse_args()

    if args.sample is None and args.dataset is None:
        parser.error("Provide either --sample or --dataset + --config")
    if args.dataset and not args.config:
        parser.error("--dataset requires --config")

    # Load model
    model, cos_layer, num_pos, num_per_pos, n_out, device = load_model(
        args.checkpoint, device=args.device, use_best=not args.use_latest
    )

    if args.sample:
        predict_single(model, cos_layer, num_pos, num_per_pos, n_out, device,
                        args.sample, threshold=args.threshold,
                        visualize=args.visualize, save_dir=args.save_dir,
                        output_path=args.output)

    if args.dataset:
        evaluate_batch(model, cos_layer, num_pos, num_per_pos, n_out, device,
                       args.dataset, args.config, args.split,
                       threshold=args.threshold, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
