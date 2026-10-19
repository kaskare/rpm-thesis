"""
Universal NVSA Frontend Training with Additive Cross-Entropy Loss

Trains a single ResNet-18 frontend on ALL RAVEN constellations simultaneously
using the supervised additive cross-entropy loss (Eq. 3 from the paper).

Key differences from the per-constellation end-to-end training (main_nvsa_marg.py):
  - Supervised: uses ground-truth attribute labels from XML (no backend/rule reasoning)
  - Universal: trains on all 7 constellations jointly with a shared codebook (num_pos=25)
  - Additive XE loss: optimizes sum of target similarities vs. individual similarities
  - SGD optimizer with momentum=0.9 and weight_decay=1e-4 (per paper)

Usage:
    python train_frontend.py \
        --dataset data/RAVEN-10000 \
        --save-dir log/universal_frontend \
        --epochs 50 --batch-size 32 --lr 0.01

    # Evaluate a trained model
    python train_frontend.py \
        --dataset data/RAVEN-10000 \
        --save-dir log/universal_frontend \
        --mode test
"""

import argparse, os, sys, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'raven'))
sys.path.insert(0, os.path.join(ROOT, 'neuro-vector-symbolic-architectures'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from tqdm import tqdm

from nvsa.perception.resnet import resnet18
from nvsa.perception.metrics import fixCos, generate_IM
from const import DIM_TYPE, DIM_SIZE, DIM_COLOR
from gnn.data import canonical_positions


class AddXentropyloss(nn.Module):
    """
    Additive cross-entropy loss for multi-label classification (Eq. 3).
    Fixed version that works on both CPU and GPU (original uses get_device() which fails on CPU).

    Modes (controlled by `per_object` and `reduction`):
      per_object=False (default / paper):
          Sums all target similarities into ONE logit per panel.
      per_object=True, reduction="sum":
          Each target object gets its own loss term, summed.
          Multi-object panels contribute proportionally more.
      per_object=True, reduction="mean":
          Each target object gets its own loss term, averaged.
          Every object weighted equally regardless of panel.
    """
    def __init__(self, per_object=False, reduction="sum"):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.per_object = per_object
        self.reduction = reduction

    def forward(self, x, target_onehot):
        # x: (B, m) — scaled cosine similarities
        # target_onehot: (B, m) — binary mask with 1s at target indices
        B = x.shape[0]

        if not self.per_object:
            # Original paper formulation: sum of target sims vs all individual sims
            target_logit = torch.sum(x * target_onehot, dim=-1, keepdim=True)  # (B, 1)
            logit = torch.cat((target_logit, x), dim=1)                         # (B, 1+m)
            new_target = torch.zeros(B, dtype=torch.long, device=x.device)
            return self.criterion(logit, new_target)

        # Per-object: each target index gets its own loss term
        losses = []
        for i in range(B):
            target_indices = target_onehot[i].nonzero(as_tuple=True)[0]
            if len(target_indices) == 0:
                continue
            for idx in target_indices:
                obj_logit = x[i, idx].view(1, 1)              # (1, 1) — this object's sim
                logit = torch.cat((obj_logit, x[i].unsqueeze(0)), dim=1)  # (1, 1+m)
                target_label = torch.zeros(1, dtype=torch.long, device=x.device)
                losses.append(self.criterion(logit, target_label))
        if losses:
            stacked = torch.stack(losses)
            return stacked.sum() if self.reduction == "sum" else stacked.mean()
        return torch.tensor(0.0, device=x.device, requires_grad=True)


# ── Constants ────────────────────────────────────────────────────────────────

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

UNIVERSAL_NUM_POS = 22   # total unique positions across ALL constellations
NUM_PER_POS = DIM_COLOR * DIM_SIZE * DIM_TYPE + 1 # 301
N_OUT = UNIVERSAL_NUM_POS * NUM_PER_POS              # 22 * 301 = 6622


def get_config_active_positions(gt_positions):
    """
    Given a set of GT position indices, determine which constellation they belong
    to and return the list of active universal position IDs for that constellation.
    Falls back to full range if no GT objects.
    """
    if not gt_positions:
        return list(range(UNIVERSAL_NUM_POS))
    for config in ALL_CONFIGS:
        active = CONFIG_ACTIVE_POSITIONS[config]
        if gt_positions.issubset(set(active)):
            return active
    return list(range(UNIVERSAL_NUM_POS))


# ── Index helpers ────────────────────────────────────────────────────────────

def compute_w_index(pos, typ, siz, col):
    """Compute the row index in W for a (position, type, size, color) tuple."""
    return pos * NUM_PER_POS + col * DIM_SIZE * DIM_TYPE + siz * DIM_TYPE + typ


def decode_w_index(idx):
    """Decode a W row index → (pos, type, size, color).  Returns None for detection vectors."""
    pos = idx // NUM_PER_POS
    remainder = idx % NUM_PER_POS
    if remainder == NUM_PER_POS - 1:
        return None   # object detection vector, not an attribute entry
    col = remainder // (DIM_SIZE * DIM_TYPE)
    remainder2 = remainder % (DIM_SIZE * DIM_TYPE)
    siz = remainder2 // DIM_TYPE
    typ = remainder2 % DIM_TYPE
    return (pos, typ, siz, col)


# ── XML parsing ──────────────────────────────────────────────────────────────

def parse_xml_panel_targets(xml_path, config):
    """
    Parse a RAVEN XML file and return target W-indices for each of the 16 panels.

    Parameters
    ----------
    xml_path : str
    config   : str – constellation name, used to map local positions to
                     universal position IDs via CONFIG_POS_MAP.

    Returns
    -------
    list of 16 lists of ints   (target indices into the universal W matrix)

    Slots come from bbox geometry (gnn.data.canonical_positions), NOT from
    `layout.attrib["Position"].index(bbox)`: I-RAVEN's Position lists only the
    OCCUPIED slots, so that index silently becomes "rank among present objects".
    On RAVEN, whose Position lists every slot, both give identical labels.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_panels = root[0]

    pos_map = CONFIG_POS_MAP[config]
    canon = canonical_positions(config, os.path.dirname(os.path.dirname(xml_path)))

    all_panel_targets = []
    for panel in xml_panels:
        targets = []
        for struct in panel:
            for ci, comp in enumerate(struct):
                for layout in comp:
                    for entity in layout:
                        bbox = eval(entity.attrib["bbox"])
                        pos = pos_map[canon[(ci, tuple(bbox))]]
                        typ = int(entity.attrib["Type"]) - 1   # XML is 1-indexed
                        siz = int(entity.attrib["Size"])
                        col = int(entity.attrib["Color"])
                        targets.append(compute_w_index(pos, typ, siz, col))
        all_panel_targets.append(targets)
    return all_panel_targets


# ── Dataset ──────────────────────────────────────────────────────────────────

class UniversalFrontendDataset(Dataset):
    """
    Dataset that returns full RAVEN samples (16 panels) with ground-truth
    target one-hot vectors for supervised frontend training.

    Each __getitem__ returns:
        images        : (16, 1, 160, 160)   float tensor
        target_onehots: (16, N_OUT)          binary float tensor
    """

    def __init__(self, dataset_path, split, configs=None):
        if configs is None:
            configs = ALL_CONFIGS
        self.samples = []   # list of (npz_path, [16 × list-of-target-indices])

        print(f"Loading {split} data from {len(configs)} constellations...")
        for config in configs:
            pattern = os.path.join(dataset_path, config, f"*_{split}.npz")
            npz_files = sorted(glob.glob(pattern))

            count = 0
            for npz_path in npz_files:
                xml_path = npz_path.replace(".npz", ".xml")
                if not os.path.isfile(xml_path):
                    continue
                panel_targets = parse_xml_panel_targets(xml_path, config)
                self.samples.append((npz_path, panel_targets))
                count += 1

            print(f"  {config}: {count} samples ({count * 16} panels)")
        print(f"Total: {len(self.samples)} samples ({len(self.samples) * 16} panels)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, panel_targets = self.samples[idx]

        data = np.load(npz_path)
        images = torch.tensor(data["image"], dtype=torch.float).unsqueeze(1)  # (16,1,160,160)

        target_onehots = torch.zeros(16, N_OUT, dtype=torch.float)
        for p in range(16):
            for w_idx in panel_targets[p]:
                target_onehots[p, w_idx] = 1.0

        return images, target_onehots


# ── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, cos_layer, criterion, optimizer, loader, device,
                clip_norm=10.0):
    model.train()
    cos_layer.train()
    running_loss = 0.0
    n_samples = 0

    for images, target_onehots in tqdm(loader, desc="  train"):
        B = images.shape[0]
        images = images.view(B * 16, 1, 160, 160).to(device)
        target_onehots = target_onehots.view(B * 16, N_OUT).to(device)

        q = model(images)                               # (B*16, d)
        z = cos_layer(q)                                 # (B*16, N_OUT)

        loss = criterion(z, target_onehots)

        optimizer.zero_grad()
        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        running_loss += loss.item() * B
        n_samples += B

    return running_loss / n_samples


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, cos_layer, loader, device, threshold=0.23):
    """
    Evaluate panel accuracy using threshold-based detection (τ = 0.23).

    A panel is correct only if the EXACT set of detected (pos, type, size, color)
    tuples matches the ground truth.
    """
    model.eval()
    cos_layer.eval()

    W_norm = F.normalize(cos_layer.W.data)

    panel_correct = 0
    panel_total = 0
    # Per-attribute tracking
    exist_correct = 0
    exist_total = 0
    attr_correct = {"type": 0, "size": 0, "color": 0}
    attr_total = {"type": 0, "size": 0, "color": 0}

    for images, target_onehots in tqdm(loader, desc="  eval "):
        B = images.shape[0]
        images_flat = images.view(B * 16, 1, 160, 160).to(device)
        target_onehots_flat = target_onehots.view(B * 16, N_OUT)

        q = model(images_flat)
        q_norm = F.normalize(q)
        cosine_sim = F.linear(q_norm, W_norm)          # (B*16, N_OUT)  raw cosines

        for i in range(B * 16):
            # ── Predictions via thresholding ──
            detected_mask = cosine_sim[i].cpu() > threshold
            detected_indices = detected_mask.nonzero(as_tuple=True)[0].tolist()
            pred_objects = set()
            for w_idx in detected_indices:
                decoded = decode_w_index(w_idx)
                if decoded is not None:
                    pred_objects.add(decoded)            # (pos, typ, siz, col)

            # ── Ground truth ──
            gt_indices = target_onehots_flat[i].nonzero(as_tuple=True)[0].tolist()
            gt_objects = set()
            for w_idx in gt_indices:
                decoded = decode_w_index(w_idx)
                if decoded is not None:
                    gt_objects.add(decoded)

            # ── Panel accuracy ──
            if pred_objects == gt_objects:
                panel_correct += 1
            panel_total += 1

            # ── Per-attribute accuracy ──
            gt_by_pos = {o[0]: o for o in gt_objects}
            pred_by_pos = {o[0]: o for o in pred_objects}

            # Only check positions relevant to this sample's constellation
            active_positions = get_config_active_positions(set(gt_by_pos.keys()))
            for pos in active_positions:
                gt_exists = pos in gt_by_pos
                pred_exists = pos in pred_by_pos
                exist_total += 1
                if gt_exists == pred_exists:
                    exist_correct += 1

                if gt_exists and pred_exists:
                    g = gt_by_pos[pos]      # (pos, typ, siz, col)
                    p = pred_by_pos[pos]
                    for j, attr in enumerate(["type", "size", "color"]):
                        attr_total[attr] += 1
                        if g[j + 1] == p[j + 1]:
                            attr_correct[attr] += 1

    panel_acc = 100.0 * panel_correct / panel_total if panel_total > 0 else 0
    return {
        "panel_acc": panel_acc,
        "panel_correct": panel_correct,
        "panel_total": panel_total,
        "exist_acc": 100.0 * exist_correct / exist_total if exist_total else 0,
        "type_acc": 100.0 * attr_correct["type"] / attr_total["type"] if attr_total["type"] else 0,
        "size_acc": 100.0 * attr_correct["size"] / attr_total["size"] if attr_total["size"] else 0,
        "color_acc": 100.0 * attr_correct["color"] / attr_total["color"] if attr_total["color"] else 0,
    }


@torch.no_grad()
def evaluate_per_config(model, cos_layer, dataset_path, split, device,
                        threshold=0.23, batch_size=32, num_workers=4):
    """Evaluate panel accuracy separately for each constellation."""
    model.eval()
    cos_layer.eval()

    results = {}
    for config in ALL_CONFIGS:
        ds = UniversalFrontendDataset(dataset_path, split, configs=[config])
        if len(ds) == 0:
            print(f"  {config}: no data")
            continue
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True)
        r = evaluate(model, cos_layer, loader, device, threshold)
        results[config] = r
        print(f"  {config:50s} panel_acc={r['panel_acc']:6.2f}%  "
              f"(exist={r['exist_acc']:.1f}% type={r['type_acc']:.1f}% "
              f"size={r['size_acc']:.1f}% color={r['color_acc']:.1f}%)")

    # Overall weighted average
    total_correct = sum(r["panel_correct"] for r in results.values())
    total_panels = sum(r["panel_total"] for r in results.values())
    overall = 100.0 * total_correct / total_panels if total_panels else 0
    print(f"  {'OVERALL':50s} panel_acc={overall:6.2f}%  ({total_correct}/{total_panels})")
    return results, overall


# ── Save / Load ──────────────────────────────────────────────────────────────

def save_checkpoint(state, save_dir, is_best=False):
    torch.save(state, os.path.join(save_dir, "checkpoint.pth.tar"))
    if is_best:
        torch.save(state, os.path.join(save_dir, "model_best.pth.tar"))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Universal NVSA Frontend – supervised training with additive XE loss"
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to RAVEN-10000 dataset root")
    parser.add_argument("--save-dir", type=str, default="log/universal_frontend",
                        help="Directory to save new checkpoints and codebooks")
    parser.add_argument("--load-from", type=str, default="",
                        help="Load model/codebook from this directory instead of --save-dir "
                             "(useful to continue training into a new folder)")

    # Training hyperparameters (per paper: SGD, wd=1e-4, momentum=0.9)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size in RAVEN samples (each has 16 panels)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for SGD")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=10.0, help="Gradient clip (L2 norm)")
    parser.add_argument("--per-object-loss", action="store_true",
                        help="Compute loss per object instead of per panel "
                             "(helps multi-object constellations)")
    parser.add_argument("--loss-reduction", type=str, default="sum",
                        choices=["sum", "mean"],
                        help="Reduction for per-object loss: "
                             "'sum' = multi-object panels contribute more, "
                             "'mean' = every object weighted equally")

    # Model
    parser.add_argument("--d", type=int, default=512, help="VSA / ResNet output dimension")
    parser.add_argument("--s", type=float, default=7.0,
                        help="Initial inverse softmax temperature s_l")
    parser.add_argument("--pretrained", type=int, default=1,
                        help="Use ImageNet-pretrained ResNet-18 (sets 3-ch input)")

    # Inference
    parser.add_argument("--threshold", type=float, default=0.23,
                        help="Detection threshold τ for evaluation")

    # Misc
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="Resume model weights but reset optimizer/scheduler/epoch "
                             "(use fresh lr, schedule, and epoch counter)")

    args = parser.parse_args()

    # ── Setup ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Universal codebook ──
    rng = np.random.default_rng(seed=args.seed)
    load_dir = args.load_from if args.load_from else args.save_dir
    codebook_path = os.path.join(load_dir, "codebooks.pt")

    if os.path.isfile(codebook_path):
        print(f"Loading existing codebook from {codebook_path}")
        cb_data = torch.load(codebook_path, map_location="cpu")
        perception_cb = cb_data["perception_cb"]
    else:
        print(f"Generating new universal codebook (num_pos={UNIVERSAL_NUM_POS})")
        perception_cb, perception_imdict = generate_IM(
            args.d, (DIM_TYPE, DIM_SIZE, DIM_COLOR, UNIVERSAL_NUM_POS), rng
        )
        cb_data = {
            "perception_cb": perception_cb,
            "perception_imdict": perception_imdict,
            "num_pos": UNIVERSAL_NUM_POS,
        }

    # Copy codebook into save-dir so the output folder is self-contained
    save_cb_path = os.path.join(args.save_dir, "codebooks.pt")
    if not os.path.isfile(save_cb_path):
        torch.save(cb_data, save_cb_path)
        print(f"Copied codebook to {save_cb_path}")

    print(f"W shape: {perception_cb.shape}  (d={args.d}, num_pos={UNIVERSAL_NUM_POS})")

    # ── Model ──
    model = resnet18(pretrained=bool(args.pretrained), progress=True,
                     num_classes=args.d, no_maxpool=True)
    model.to(device)

    cos_layer = fixCos(
        num_features=args.d, num_classes=N_OUT, mat=perception_cb,
        fixed_weights=True, s=args.s, trainable_s=True
    )
    cos_layer.to(device)

    criterion = AddXentropyloss(per_object=args.per_object_loss,
                                reduction=args.loss_reduction)
    if args.per_object_loss:
        print(f"Using PER-OBJECT loss (reduction={args.loss_reduction})")

    # ── Load checkpoint if resuming ──
    ckpt_path = os.path.join(load_dir, "checkpoint.pth.tar")
    start_epoch = 0
    best_acc = 0.0

    if os.path.isfile(ckpt_path) and args.mode == "train":
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict_model"])
        cos_layer.load_state_dict(ckpt["state_dict_cos"])
        if args.reset_optimizer:
            print("  --reset-optimizer: keeping model weights, resetting optimizer/scheduler/epoch")
            start_epoch = 0
            best_acc = 0.0
        else:
            start_epoch = ckpt.get("epoch", 0)
            best_acc = ckpt.get("best_acc", 0.0)
        print(f"  epoch={start_epoch}, best_acc={best_acc:.2f}%")

    if args.mode == "test":
        best_path = os.path.join(load_dir, "model_best.pth.tar")
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict_model"])
            cos_layer.load_state_dict(ckpt["state_dict_cos"])
            print(f"Loaded best model (acc={ckpt.get('best_acc', '?')}%)")
        else:
            print(f"WARNING: no best model found at {best_path}")

        model.to(device)
        cos_layer.to(device)

        print("\n=== Per-constellation evaluation (test) ===")
        evaluate_per_config(model, cos_layer, args.dataset, "test", device,
                            args.threshold, args.batch_size, args.num_workers)
        return

    # ── Data ──
    print("\n--- Loading training data ---")
    train_ds = UniversalFrontendDataset(args.dataset, "train")
    print("\n--- Loading validation data ---")
    val_ds = UniversalFrontendDataset(args.dataset, "val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ── Optimizer: SGD with momentum (per paper) ──
    # Only train ResNet-18 parameters + the trainable s scalar; W is frozen
    train_params = list(model.parameters()) + [cos_layer.s]
    optimizer = optim.SGD(train_params, lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    # Cosine annealing LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if start_epoch > 0 and "optimizer" in ckpt and not args.reset_optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
        # Advance scheduler to current epoch
        for _ in range(start_epoch):
            scheduler.step()

    # ── Training loop ──
    print(f"\nTraining for {args.epochs - start_epoch} epochs "
          f"(batch_size={args.batch_size} samples = {args.batch_size * 16} panels)")
    print(f"Optimizer: SGD(lr={args.lr}, momentum={args.momentum}, wd={args.weight_decay})")
    print(f"Device: {device}\n")

    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs - 1}  (lr={lr:.6f}, s={cos_layer.get_s():.3f})")

        loss = train_epoch(model, cos_layer, criterion, optimizer,
                           train_loader, device, args.clip)

        r = evaluate(model, cos_layer, val_loader, device, args.threshold)
        acc = r["panel_acc"]

        print(f"  loss={loss:.4f}  val_panel_acc={acc:.2f}%  "
              f"(exist={r['exist_acc']:.1f}% type={r['type_acc']:.1f}% "
              f"size={r['size_acc']:.1f}% color={r['color_acc']:.1f}%)")

        scheduler.step()

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict_model": model.state_dict(),
            "state_dict_cos": cos_layer.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
            "s": cos_layer.get_s(),
            "threshold": args.threshold,
            "num_pos": UNIVERSAL_NUM_POS,
            "d": args.d,
        }, args.save_dir, is_best=is_best)

        if is_best:
            print(f"  *** New best: {best_acc:.2f}% ***")

    print(f"\nTraining complete. Best validation panel accuracy: {best_acc:.2f}%")

    # Final per-constellation evaluation on test set
    print("\n=== Per-constellation evaluation (test) ===")
    best_ckpt = torch.load(os.path.join(args.save_dir, "model_best.pth.tar"),
                           map_location="cpu")
    model.load_state_dict(best_ckpt["state_dict_model"])
    cos_layer.load_state_dict(best_ckpt["state_dict_cos"])
    model.to(device)
    cos_layer.to(device)

    evaluate_per_config(model, cos_layer, args.dataset, "test", device,
                        args.threshold, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()



# python predict_frontend.py \
#     --checkpoint log/universal_frontend_test4 \
#     --sample data/RAVEN-10000/distribute_nine/RAVEN_0_train.npz \
#     --threshold 0.15