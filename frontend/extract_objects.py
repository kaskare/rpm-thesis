"""
NVSA Frontend Object Extraction Inference Script

Extracts per-panel object positions and attributes (type, size, color) from
RAVEN dataset samples using a trained NVSA frontend (ResNet-18 + VSA codebook).

Usage:
    # Single sample
    python frontend/extract_objects.py \
        --config center_single \
        --checkpoint log/nvsa_test01/center_single_.../0/ckpt \
        --sample data/RAVEN-10000/center_single/RAVEN_0_train.npz

    # Single sample with visualization (saves annotated PNG)
    python frontend/extract_objects.py \
        --config center_single \
        --checkpoint log/nvsa_test01/center_single_.../0/ckpt \
        --sample data/RAVEN-10000/center_single/RAVEN_0_train.npz \
        --visualize

    # Batch evaluation over the test set (accuracy report)
    python frontend/extract_objects.py \
        --config center_single \
        --checkpoint log/nvsa_test01/center_single_.../0/ckpt \
        --dataset data/RAVEN-10000 \
        --split test --max-samples 200

    # Use --verbose for per-panel printouts
"""

import argparse, os, sys, glob

# Add paths so we can import nvsa (from neuro-vector-symbolic-architectures/) and raven/
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
from nvsa.perception.metrics import hd_mult_frontend
from const import NUMPOS, DIM_COLOR, DIM_SIZE, DIM_TYPE
from gnn.data import canonical_positions


# ── Attribute label names (model 0-indexed → human-readable) ─────────────────
# RAVEN XML uses 1-indexed types (1-5); the model codebook uses 0-indexed (0-4)
TYPE_NAMES  = {0: "triangle", 1: "square", 2: "pentagon", 3: "hexagon", 4: "circle"}
COLOR_NAMES = {i: f"color_{i}" for i in range(DIM_COLOR)}
SIZE_NAMES  = {i: f"size_{i}" for i in range(DIM_SIZE)}


def load_model(checkpoint_dir, config, d=512, pretrained=True, device="cpu"):
    """Load the trained NVSA frontend (ResNet-18 + codebook + marginalization)."""
    device = torch.device(device)
    num_pos = NUMPOS[config]

    # Load codebooks (fixed dictionary matrix W)
    cb_path = os.path.join(checkpoint_dir, "codebooks.pt")
    cb_data = torch.load(cb_path, map_location="cpu")
    perception_cb = cb_data["perception_cb"]

    # Build model — pretrained flag must match training (controls in_channels: 3 vs 1)
    model = resnet18(pretrained=pretrained, progress=False, num_classes=d, no_maxpool=True)
    metric_fc = hd_mult_frontend(
        num_features=d, num_classes=5, mat=perception_cb,
        marg_m=0., marg_in_act="ReLU", num_pos=num_pos,
    )

    # Load trained weights
    ckpt_path = os.path.join(checkpoint_dir, "model_best.pth.tar")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict_model"])
    metric_fc.load_state_dict(ckpt["state_dict_metric_fc"])
    print(f"Loaded checkpoint from {ckpt_path}  (best_acc={ckpt.get('best_acc', '?')})")

    model.to(device).eval()
    metric_fc.to(device).eval()
    return model, metric_fc, num_pos


def run_frontend(model, metric_fc, images, device):
    """
    Run the NVSA frontend on a batch of 16 panel images.

    Parameters
    ----------
    images : np.ndarray of shape (16, 160, 160)

    Returns
    -------
    exist_prob  : (16, num_pos, 2)       – P(not exist), P(exist)
    type_prob   : (16, num_pos, DIM_TYPE)
    size_prob   : (16, num_pos, DIM_SIZE)
    color_prob  : (16, num_pos, DIM_COLOR)
    """
    imgs = torch.tensor(np.expand_dims(images, 1), dtype=torch.float).to(device)
    B, N = 1, imgs.shape[0]  # 1 sample, 16 panels

    with torch.no_grad():
        query = model(imgs)                        # (16, d)
        marg_logprob = metric_fc(query, B, N)      # list of 4 tensors

    # Convert log-probabilities → probabilities and squeeze batch dim
    exist_prob = torch.exp(marg_logprob[0]).squeeze(0)   # (16, num_pos, 2)
    type_prob  = torch.exp(marg_logprob[1]).squeeze(0)   # (16, num_pos, DIM_TYPE)
    size_prob  = torch.exp(marg_logprob[2]).squeeze(0)   # (16, num_pos, DIM_SIZE)
    color_prob = torch.exp(marg_logprob[3]).squeeze(0)   # (16, num_pos, DIM_COLOR)

    return exist_prob, type_prob, size_prob, color_prob


def decode_panel(exist_prob, type_prob, size_prob, color_prob, panel_idx,
                 exist_threshold=0.5):
    """
    Decode detected objects for a single panel.

    Returns list of dicts: [{pos, type, size, color, exist_p}, ...]
    """
    num_pos = exist_prob.shape[1]
    objects = []
    for pos in range(num_pos):
        p_exist = exist_prob[panel_idx, pos, 1].item()
        if p_exist > exist_threshold:
            objects.append({
                "position":  pos,
                "type":      torch.argmax(type_prob[panel_idx, pos]).item(),
                "size":      torch.argmax(size_prob[panel_idx, pos]).item(),
                "color":     torch.argmax(color_prob[panel_idx, pos]).item(),
                "exist_p":   p_exist,
            })
    return objects


def parse_ground_truth(xml_path, config):
    """
    Parse the XML annotation file and return per-panel ground truth.

    Returns list of 16 lists, each containing dicts:
        [{position, type, size, color}, ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_panels = root[0]
    # local slot (component offset included) from bbox geometry: I-RAVEN's Position
    # attribute lists only occupied slots, so Position.index(bbox) is wrong there
    canon = canonical_positions(config, os.path.dirname(os.path.dirname(xml_path)))

    gt_panels = []
    for panel in xml_panels:
        panel_objects = []
        for struct in panel:
            for ci, comp in enumerate(struct):
                for layout in comp:
                    for entity in layout:
                        bbox = eval(entity.attrib["bbox"])
                        panel_objects.append({
                            "position": canon[(ci, tuple(bbox))],
                            "type":     int(entity.attrib["Type"]) - 1,  # XML is 1-indexed, model is 0-indexed
                            "size":     int(entity.attrib["Size"]),
                            "color":    int(entity.attrib["Color"]),
                        })
        gt_panels.append(panel_objects)
    return gt_panels


def compare_panel(pred_objects, gt_objects):
    """
    Compare predicted objects against ground truth for one panel.
    Returns True if all objects match exactly (existence + all attributes).
    """
    if len(pred_objects) != len(gt_objects):
        return False

    # Sort both by position for consistent comparison
    pred_sorted = sorted(pred_objects, key=lambda o: o["position"])
    gt_sorted   = sorted(gt_objects,   key=lambda o: o["position"])

    for pred, gt in zip(pred_sorted, gt_sorted):
        if (pred["position"] != gt["position"] or
            pred["type"]     != gt["type"] or
            pred["size"]     != gt["size"] or
            pred["color"]    != gt["color"]):
            return False
    return True


def print_panel_results(panel_idx, pred_objects, gt_objects=None):
    """Pretty-print the predicted (and optionally ground truth) objects."""
    label = "context" if panel_idx < 8 else f"answer candidate {panel_idx - 8}"
    print(f"\n  Panel {panel_idx:2d} ({label}):")

    if not pred_objects:
        print("    [no objects detected]")
    for obj in pred_objects:
        type_name = TYPE_NAMES.get(obj["type"], f"type_{obj['type']}")
        print(f"    pos={obj['position']}  type={type_name:<10s}  "
              f"size={obj['size']}  color={obj['color']}  "
              f"(exist_p={obj['exist_p']:.3f})")

    if gt_objects is not None:
        match = compare_panel(pred_objects, gt_objects)
        status = "MATCH" if match else "MISMATCH"
        if not match:
            print(f"    GT: {gt_objects}")
        print(f"    --> {status}")


# ── Visualization ────────────────────────────────────────────────────────────

def _get_font(size=12):
    """Try to load a monospace font; fall back to default."""
    for name in ["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf",
                 "UbuntuMono-R.ttf", "cour.ttf", "consola.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def visualize_sample(images, target, all_predictions, gt_panels, save_path,
                     config):
    """
    Render the 16 RAVEN panels as an annotated image and save it to disk.

    Layout
    ------
    Row 1-3 : 3×3 context grid (panel 0-7, bottom-right = "?")
    Row 4   : 8 answer candidates (panel 8-15), correct one highlighted
    Each panel has a text annotation below it showing predicted attributes.
    """
    H, W = images.shape[1], images.shape[2]   # 160×160
    pad = 6           # gap between panels
    ann_h = 48        # space for annotation text below each panel
    border = 3        # highlight border width

    font = _get_font(11)
    font_sm = _get_font(10)

    # ── Build context grid (3 rows × 3 cols) ──
    cell_w = W + pad
    cell_h = H + ann_h + pad
    ctx_cols, ctx_rows = 3, 3
    ctx_w = ctx_cols * cell_w - pad
    ctx_h = ctx_rows * cell_h - pad

    # ── Build answer row (1 row × 8 cols) ──
    ans_cols = 8
    ans_w = ans_cols * cell_w - pad
    ans_h = cell_h

    # Canvas
    total_w = max(ctx_w, ans_w) + 2 * pad
    gap_between = 16
    title_h = 22
    total_h = title_h + ctx_h + gap_between + ans_h + pad

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Title
    draw.text((pad, 2), f"Config: {config}   |   Correct answer: candidate {target}",
              fill=(60, 60, 60), font=font)

    def _paste_panel(panel_idx, x0, y0, highlight=False):
        """Paste one panel image + annotation at (x0, y0)."""
        panel_img = Image.fromarray(images[panel_idx], mode="L").convert("RGB")

        # Green border for correct answer
        if highlight:
            bordered = Image.new("RGB", (W + 2*border, H + 2*border), (0, 180, 0))
            bordered.paste(panel_img, (border, border))
            canvas.paste(bordered, (x0 - border, y0 - border))
        else:
            canvas.paste(panel_img, (x0, y0))

        # Annotation text
        pred = all_predictions[panel_idx]
        lines = []
        for obj in pred:
            tname = TYPE_NAMES.get(obj["type"], f"t{obj['type']}")
            lines.append(f"p{obj['position']} {tname} s{obj['size']} c{obj['color']}")
        if not lines:
            lines = ["[empty]"]

        # Draw match/mismatch indicator if GT available
        if gt_panels is not None:
            match = compare_panel(pred, gt_panels[panel_idx])
            color = (0, 140, 0) if match else (200, 0, 0)
            symbol = "OK" if match else "X "
            prefix = symbol + " "
        else:
            color = (60, 60, 60)
            prefix = ""

        ty = y0 + H + 2
        for line in lines[:3]:  # max 3 lines
            draw.text((x0, ty), prefix + line, fill=color, font=font_sm)
            ty += 14

    # ── Draw context grid ──
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
                # "?" panel
                qimg = Image.new("RGB", (W, H), (180, 180, 180))
                qdraw = ImageDraw.Draw(qimg)
                qdraw.text((W//2 - 6, H//2 - 8), "?", fill=(80, 80, 80),
                           font=_get_font(24))
                canvas.paste(qimg, (x, y))

    # ── Draw answer candidates ──
    ans_x0 = (total_w - ans_w) // 2
    ans_y0 = ctx_y0 + ctx_h + gap_between
    for i in range(8):
        x = ans_x0 + i * cell_w
        y = ans_y0
        is_correct = (i == target)
        _paste_panel(8 + i, x, y, highlight=is_correct)

    canvas.save(save_path)
    print(f"\nVisualization saved to: {save_path}")


# ── Single-sample inference ──────────────────────────────────────────────────

def extract_single(model, metric_fc, npz_path, config, device,
                    verbose=True, visualize=False, save_dir=None):
    """Run inference on a single .npz sample. Returns per-panel predictions."""
    data = np.load(npz_path)
    images = data["image"]  # (16, 160, 160)
    target = int(data["target"])

    exist_prob, type_prob, size_prob, color_prob = run_frontend(
        model, metric_fc, images, device
    )

    # Try to load ground truth XML
    xml_path = npz_path.replace(".npz", ".xml")
    gt_panels = None
    if os.path.isfile(xml_path):
        gt_panels = parse_ground_truth(xml_path, config)

    all_predictions = []
    panel_correct = 0
    total_panels = 16

    if verbose:
        print(f"\nSample: {os.path.basename(npz_path)}  (answer target = {target})")

    for panel_idx in range(16):
        pred = decode_panel(exist_prob, type_prob, size_prob, color_prob, panel_idx)
        all_predictions.append(pred)

        gt = gt_panels[panel_idx] if gt_panels else None
        if verbose:
            print_panel_results(panel_idx, pred, gt)
        if gt is not None and compare_panel(pred, gt):
            panel_correct += 1

    if gt_panels is not None:
        print(f"\n  Panel accuracy: {panel_correct}/{total_panels} "
              f"({100 * panel_correct / total_panels:.1f}%)")

    # Save annotated visualization
    if visualize:
        basename = os.path.splitext(os.path.basename(npz_path))[0]
        out_dir = save_dir or os.path.dirname(npz_path)
        save_path = os.path.join(out_dir, f"{basename}_viz.png")
        visualize_sample(images, target, all_predictions, gt_panels,
                         save_path, config)

    return all_predictions, gt_panels


# ── Batch evaluation ─────────────────────────────────────────────────────────

def evaluate_batch(model, metric_fc, dataset_path, config, split, device,
                   max_samples=None, verbose=False):
    """
    Evaluate perception accuracy over a dataset split.
    A sample is correct only if ALL 16 panels are predicted correctly.
    """
    pattern = os.path.join(dataset_path, config, f"*_{split}.npz")
    npz_files = sorted(glob.glob(pattern))
    if not npz_files:
        print(f"No files found matching {pattern}")
        return

    if max_samples is not None:
        npz_files = npz_files[:max_samples]

    print(f"Evaluating {len(npz_files)} samples from {config}/{split} ...")

    sample_correct = 0
    total_panels_correct = 0
    total_panels = 0
    attr_correct = {"type": 0, "size": 0, "color": 0, "exist": 0}
    attr_total = {"type": 0, "size": 0, "color": 0, "exist": 0}

    for npz_path in npz_files:
        data = np.load(npz_path)
        images = data["image"]

        exist_prob, type_prob, size_prob, color_prob = run_frontend(
            model, metric_fc, images, device
        )

        xml_path = npz_path.replace(".npz", ".xml")
        if not os.path.isfile(xml_path):
            continue
        gt_panels = parse_ground_truth(xml_path, config)

        all_panels_correct = True
        for panel_idx in range(16):
            pred = decode_panel(exist_prob, type_prob, size_prob, color_prob, panel_idx)
            gt = gt_panels[panel_idx]

            if verbose:
                print_panel_results(panel_idx, pred, gt)

            panel_match = compare_panel(pred, gt)
            if panel_match:
                total_panels_correct += 1
            else:
                all_panels_correct = False
            total_panels += 1

            # Per-attribute accuracy (for positions that exist in GT)
            gt_by_pos = {o["position"]: o for o in gt}
            pred_by_pos = {o["position"]: o for o in pred}
            num_pos = exist_prob.shape[1]
            for pos in range(num_pos):
                gt_exists = pos in gt_by_pos
                pred_exists = pos in pred_by_pos
                attr_total["exist"] += 1
                if gt_exists == pred_exists:
                    attr_correct["exist"] += 1

                if gt_exists and pred_exists:
                    for attr in ["type", "size", "color"]:
                        attr_total[attr] += 1
                        if pred_by_pos[pos][attr] == gt_by_pos[pos][attr]:
                            attr_correct[attr] += 1

        if all_panels_correct:
            sample_correct += 1

    # ── Report ──
    n = len(npz_files)
    print(f"\n{'='*60}")
    print(f"Config: {config}  |  Split: {split}  |  Samples: {n}")
    print(f"{'='*60}")
    print(f"  Sample accuracy (all 16 panels correct): "
          f"{sample_correct}/{n} ({100*sample_correct/n:.2f}%)")
    print(f"  Panel  accuracy: "
          f"{total_panels_correct}/{total_panels} "
          f"({100*total_panels_correct/total_panels:.2f}%)")
    print(f"\n  Per-attribute accuracy (on occupied positions):")
    for attr in ["exist", "type", "size", "color"]:
        if attr_total[attr] > 0:
            pct = 100 * attr_correct[attr] / attr_total[attr]
            print(f"    {attr:6s}: {attr_correct[attr]}/{attr_total[attr]} ({pct:.2f}%)")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NVSA Frontend – extract object positions & properties from RAVEN panels"
    )
    parser.add_argument("--config", type=str, required=True,
                        choices=list(NUMPOS.keys()),
                        help="RAVEN constellation name")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to ckpt/ directory (contains model_best.pth.tar + codebooks.pt)")
    parser.add_argument("--d", type=int, default=512,
                        help="Frontend VSA dimension (ResNet-18 output dim)")
    parser.add_argument("--pretrained", type=int, default=1,
                        help="Must match training: 1 if trained with --pretrained 1 (3ch input)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: 'cpu' or 'cuda:0'")
    parser.add_argument("--exist-threshold", type=float, default=0.5,
                        help="Threshold on P(exist) for object detection")

    # Single sample mode
    parser.add_argument("--sample", type=str, default=None,
                        help="Path to a single .npz file for inspection")

    # Batch evaluation mode
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to RAVEN-10000 dataset root (for batch eval)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split for batch evaluation")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples in batch evaluation")

    parser.add_argument("--verbose", action="store_true",
                        help="Print per-panel details in batch mode")
    parser.add_argument("--visualize", action="store_true",
                        help="Save annotated PNG visualization (works with --sample)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory for visualization output (default: same as input)")

    args = parser.parse_args()

    if args.sample is None and args.dataset is None:
        parser.error("Provide either --sample (single file) or --dataset (batch eval)")

    # Load model once
    model, metric_fc, num_pos = load_model(
        args.checkpoint, args.config, d=args.d,
        pretrained=bool(args.pretrained), device=args.device
    )
    print(f"Config: {args.config}  |  num_pos: {num_pos}  |  device: {args.device}")

    if args.sample:
        extract_single(model, metric_fc, args.sample, args.config, args.device,
                        verbose=True, visualize=args.visualize,
                        save_dir=args.save_dir)

    if args.dataset:
        evaluate_batch(model, metric_fc, args.dataset, args.config, args.split,
                       args.device, max_samples=args.max_samples, verbose=args.verbose)


if __name__ == "__main__":
    main()


# python frontend/extract_objects.py \
#   --config center_single \
#   --checkpoint log/nvsa_test01/center_single_pret_1_bs_8_lr_9.5e-05_wd_0.0005_s_7.0_ts_True_m_0.0_clip_10_main_loss_1.0_aux_1.0_d_1024_k_4_exe_s_1_exe_cos2pmf_Identity_det_actthreshold_det_m0.000/0/ckpt \
#   --sample data/RAVEN-10000/center_single/RAVEN_0_train.npz \
#   --visualize

# python frontend/extract_objects.py \
#   --config distribute_four \
#   --checkpoint log/nvsa_test01/distribute_four_pret_1_bs_8_lr_9.5e-05_wd_0.0005_s_6.0_ts_True_m_0.0_clip_10_main_loss_1.0_aux_1.0_d_1024_k_4_exe_s_1_exe_cos2pmf_Identity_det_actthreshold_det_m0.000/0/ckpt \
#   --sample data/RAVEN-10000/distribute_four/RAVEN_0_train.npz \
#   --visualize