#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def render_raven_example(npz_path: Path, out_path: Path) -> int:
    data = np.load(npz_path)
    images = data["image"]  # shape: (16, H, W)
    target = int(data["target"])

    if images.ndim != 3 or images.shape[0] != 16:
        raise ValueError(f"Expected image shape (16, H, W), got {images.shape}")

    panel_h, panel_w = images.shape[1], images.shape[2]
    gap = 12
    margin = 20
    separator = 36

    grid_w = 3 * panel_w + 2 * gap
    choices_w = 8 * panel_w + 7 * gap
    canvas_w = margin * 2 + max(grid_w, choices_w)
    canvas_h = margin * 2 + (3 * panel_h + 2 * gap) + separator + panel_h

    canvas = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(canvas)

    # Place first 8 panels as the 3x3 problem matrix (last one is missing).
    grid_x0 = (canvas_w - grid_w) // 2
    grid_y0 = margin
    for i in range(8):
        row, col = divmod(i, 3)
        x = grid_x0 + col * (panel_w + gap)
        y = grid_y0 + row * (panel_h + gap)
        canvas.paste(Image.fromarray(images[i]), (x, y))
        draw.rectangle([x, y, x + panel_w - 1, y + panel_h - 1], outline=0, width=2)

    # Draw missing panel slot (bottom-right in the 3x3 matrix).
    miss_x = grid_x0 + 2 * (panel_w + gap)
    miss_y = grid_y0 + 2 * (panel_h + gap)
    draw.rectangle([miss_x, miss_y, miss_x + panel_w - 1, miss_y + panel_h - 1], outline=0, width=3)
    pad_x = max(20, panel_w // 6)
    pad_y = max(20, panel_h // 6)
    draw.line((miss_x + pad_x, miss_y + pad_y, miss_x + panel_w - pad_x, miss_y + panel_h - pad_y), fill=0, width=3)
    draw.line((miss_x + panel_w - pad_x, miss_y + pad_y, miss_x + pad_x, miss_y + panel_h - pad_y), fill=0, width=3)

    # Place last 8 panels as answer choices.
    choices_x0 = (canvas_w - choices_w) // 2
    choices_y = grid_y0 + (3 * panel_h + 2 * gap) + separator
    for i in range(8):
        x = choices_x0 + i * (panel_w + gap)
        canvas.paste(Image.fromarray(images[8 + i]), (x, choices_y))
        draw.rectangle([x, choices_y, x + panel_w - 1, choices_y + panel_h - 1], outline=0, width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return target


def build_paths(dataset_root: Path, config: str, split: str, index: int, output: str | None):
    npz_name = f"RAVEN_{index}_{split}.npz"
    npz_path = dataset_root / config / npz_name
    if output is None:
        output = str(dataset_root / config / f"RAVEN_{index}_{split}_visualized.png")
    return npz_path, Path(output)


def main():
    parser = argparse.ArgumentParser(description="Visualize an I-RAVEN NPZ example in RPM layout.")
    parser.add_argument("--dataset-root", default="data/I-RAVEN-10000", help="Path to I-RAVEN dataset root.")
    parser.add_argument("--config", required=True, help="I-RAVEN configuration folder name.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--index", type=int, default=0, help="Example index N in RAVEN_N_<split>.npz.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    npz_path, out_path = build_paths(dataset_root, args.config, args.split, args.index, args.output)

    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing file: {npz_path}")

    target = render_raven_example(npz_path, out_path)
    print(f"Saved visualization: {out_path}")
    print(f"Target index: {target}")


if __name__ == "__main__":
    main()
