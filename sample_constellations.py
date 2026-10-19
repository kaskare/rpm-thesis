#!/usr/bin/env python3
"""Sample RAVEN/I-RAVEN files per constellation and export preview images."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required. Install it with `pip install pillow`."
    ) from exc


VALID_IMAGE_FORMATS = {"png", "jpg", "jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample N main RAVEN/I-RAVEN samples from each constellation directory, "
            "copy the paired files, and export each sampled matrix as an image."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the dataset root, e.g. data/I-RAVEN-10000",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("sampled_constellations"),
        help="Directory where sampled files and exported images will be written.",
    )
    parser.add_argument(
        "--samples-per-constellation",
        type=int,
        default=3,
        help="How many main .npz files to sample from each constellation directory.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Restrict sampling to one split. Default: all.",
    )
    parser.add_argument(
        "--constellations",
        nargs="*",
        default=None,
        help=(
            "Optional list of constellation directory names to sample. "
            "Default: every subdirectory under dataset_root."
        ),
    )
    parser.add_argument(
        "--image-format",
        choices=tuple(sorted(VALID_IMAGE_FORMATS)),
        default="png",
        help="Image format for exported previews.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--no-copy-files",
        action="store_true",
        help="Do not copy sampled .npz/.xml/_rule_comp*.npz files. Previews are still exported.",
    )
    return parser.parse_args()


def discover_constellations(dataset_root: Path, selected: list[str] | None) -> list[Path]:
    available = {path.name: path for path in dataset_root.iterdir() if path.is_dir()}
    if selected is None:
        return sorted(available.values(), key=lambda path: path.name)

    missing = [name for name in selected if name not in available]
    if missing:
        raise SystemExit(f"Unknown constellation directories: {', '.join(missing)}")
    return [available[name] for name in selected]


def list_main_npz_files(constellation_dir: Path, split: str) -> list[Path]:
    candidates = sorted(constellation_dir.glob("*.npz"))
    main_files = [
        path
        for path in candidates
        if "_rule_comp" not in path.name and (split == "all" or f"_{split}." in path.name)
    ]
    return main_files


def paired_files(main_npz: Path) -> Iterable[Path]:
    stem = main_npz.stem
    yield main_npz

    xml_path = main_npz.with_suffix(".xml")
    if xml_path.exists():
        yield xml_path

    for rule_path in sorted(main_npz.parent.glob(f"{stem}_rule_comp*.npz")):
        yield rule_path


def normalize_panel(panel: np.ndarray) -> np.ndarray:
    panel = np.asarray(panel)
    if panel.dtype == np.uint8:
        return panel

    panel = panel.astype(np.float32)
    min_value = float(panel.min())
    max_value = float(panel.max())
    if max_value <= min_value:
        return np.zeros(panel.shape, dtype=np.uint8)
    scaled = (panel - min_value) / (max_value - min_value)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def to_pil_image(panel: np.ndarray) -> Image.Image:
    panel = normalize_panel(panel)
    if panel.ndim == 2:
        return Image.fromarray(panel, mode="L")
    if panel.ndim == 3 and panel.shape[-1] in (1, 3, 4):
        if panel.shape[-1] == 1:
            return Image.fromarray(panel[..., 0], mode="L")
        if panel.shape[-1] == 3:
            return Image.fromarray(panel, mode="RGB")
        return Image.fromarray(panel, mode="RGBA")
    raise ValueError(f"Unsupported panel shape: {panel.shape}")


def create_matrix_preview(image_array: np.ndarray, tile_gap: int = 4) -> Image.Image:
    panels = np.asarray(image_array)
    if panels.ndim < 3:
        raise ValueError(f"Expected image array with multiple panels, got shape {panels.shape}")

    if panels.shape[0] != 16:
        rows = int(np.ceil(np.sqrt(panels.shape[0])))
        cols = int(np.ceil(panels.shape[0] / rows))
    else:
        rows, cols = 4, 4

    pil_panels = [to_pil_image(panel) for panel in panels]
    widths = [img.width for img in pil_panels]
    heights = [img.height for img in pil_panels]
    tile_width = max(widths)
    tile_height = max(heights)

    mode = "RGB" if any(img.mode != "L" for img in pil_panels) else "L"
    background = (255, 255, 255) if mode == "RGB" else 255
    canvas = Image.new(
        mode,
        (
            cols * tile_width + (cols - 1) * tile_gap,
            rows * tile_height + (rows - 1) * tile_gap,
        ),
        color=background,
    )

    for index, img in enumerate(pil_panels):
        if img.mode != mode:
            img = img.convert(mode)
        row = index // cols
        col = index % cols
        x = col * (tile_width + tile_gap)
        y = row * (tile_height + tile_gap)
        canvas.paste(img, (x, y))

    return canvas


def save_preview(main_npz: Path, image_output_path: Path, image_format: str) -> None:
    with np.load(main_npz) as data:
        if "image" not in data:
            raise KeyError(f"{main_npz} does not contain an 'image' array.")
        preview = create_matrix_preview(data["image"])

    if image_format in {"jpg", "jpeg"} and preview.mode == "RGBA":
        preview = preview.convert("RGB")
    preview.save(image_output_path, format="JPEG" if image_format in {"jpg", "jpeg"} else "PNG")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root is not a directory: {dataset_root}")
    if args.samples_per_constellation <= 0:
        raise SystemExit("--samples-per-constellation must be greater than 0")

    rng = random.Random(args.seed)
    constellation_dirs = discover_constellations(dataset_root, args.constellations)
    output_root.mkdir(parents=True, exist_ok=True)

    for constellation_dir in constellation_dirs:
        main_files = list_main_npz_files(constellation_dir, args.split)
        if not main_files:
            print(f"[skip] {constellation_dir.name}: no matching samples for split={args.split}")
            continue

        sample_count = min(args.samples_per_constellation, len(main_files))
        chosen = sorted(rng.sample(main_files, sample_count), key=lambda path: path.name)
        constellation_output = output_root / constellation_dir.name
        image_output = constellation_output / "images"
        image_output.mkdir(parents=True, exist_ok=True)

        if not args.no_copy_files:
            files_output = constellation_output / "files"
            files_output.mkdir(parents=True, exist_ok=True)
        else:
            files_output = None

        for main_npz in chosen:
            image_name = f"{main_npz.stem}.{args.image_format}"
            save_preview(main_npz, image_output / image_name, args.image_format)

            if files_output is not None:
                for path in paired_files(main_npz):
                    shutil.copy2(path, files_output / path.name)

        print(
            f"[ok] {constellation_dir.name}: exported {len(chosen)} previews"
            + (" and copied paired files" if files_output is not None else "")
        )


if __name__ == "__main__":
    main()
