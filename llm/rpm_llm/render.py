"""Renders a puzzle as one image: 3x3 context grid with a cross in the missing cell,
a thick separator, then the 8 candidates labelled A-H. Same layout as Tehlar's processor.py,
except that candidate separators sit on the panel borders instead of 10 px inside the panels."""
import io
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import LETTERS


@lru_cache(maxsize=4)
def _font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def render_panels(panels: np.ndarray) -> Image.Image:
    if panels.ndim != 3 or panels.shape[0] != 16:
        raise ValueError(f"expected (16, H, W) panels, got {panels.shape}")
    _, h, w = panels.shape
    text_h = int(h * 0.2)
    canvas = Image.new("L", (8 * w, 4 * h + text_h), 255)
    draw = ImageDraw.Draw(canvas)
    for i in range(8):
        r, c = divmod(i, 3)
        canvas.paste(Image.fromarray(panels[i]), (c * w, r * h))
    for i in (1, 2):
        draw.line([(i * w, 0), (i * w, 3 * h)], fill=0, width=2)
        draw.line([(0, i * h), (3 * w, i * h)], fill=0, width=2)
    cx, cy = 2 * w + w // 2, 2 * h + h // 2
    draw.line([(cx - 15, cy - 15), (cx + 15, cy + 15)], fill=0, width=1)
    draw.line([(cx - 15, cy + 15), (cx + 15, cy - 15)], fill=0, width=1)
    draw.line([(0, 3 * h - 3), (8 * w, 3 * h - 3)], fill=0, width=5)
    font = _font(int(text_h * 0.85))
    for k in range(8):
        canvas.paste(Image.fromarray(panels[8 + k]), (k * w, 3 * h))
        draw.text((k * w + w // 2, 4 * h + text_h // 2), LETTERS[k], fill=0, font=font, anchor="mm")
        if k:
            draw.line([(k * w, 3 * h), (k * w, 4 * h + text_h)], fill=0, width=4)
    return canvas


def render_png(npz_path) -> bytes:
    with np.load(npz_path) as d:
        img = render_panels(d["image"])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
