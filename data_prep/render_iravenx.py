#!/usr/bin/env python3
"""I-RAVEN-X has no images: its puzzles are attribute values. This draws them, so that the vision baselines can read
them. Each panel holds one arrowhead on white, like RAVEN's center layout, with the three attributes as three visual
channels: size is its radius, colour its grey level, and type its direction, because RAVEN's polygons cannot be told
apart past a dozen sides. A value step is about 1 px of radius and 7 degrees at values 0-50, and 2.6 px and 33 degrees
at 0-10. Ranges 0-100 and above are not drawn: a step would fall under a pixel.

Draws the context panels of experiments/baselines/data/<name>.npz, split by split, in its order, as uint8 [puzzles,
panels, side, side]. Environment nvsa, from the repository root; a few minutes, so submit it:
    python data_prep/render_iravenx.py iravenx_n3_m10 --size 80     # -> <name>_images.npz
    python data_prep/render_iravenx.py iravenx_n10_m50 --size 160   # -> <name>_images160.npz
"""
import argparse, json, os

import numpy as np

SS = 4                                                                                  # subsamples per pixel side
WING = np.deg2rad(150)                                                                  # the two back corners


def glyph(t, s, vmax, side, grid):
    """Coverage in [0, 1] of the arrowhead's fill and of its outline, for a type and size value."""
    r = side * (0.16 + 0.32 * s / vmax)
    a = 2 * np.pi * t / (vmax + 1) - np.pi / 2
    corners = [(a, r), (a + WING, r * 0.8), (a - WING, r * 0.8)]
    x, y = grid
    inside = np.ones(x.shape, bool)
    inner = np.ones(x.shape, bool)
    width = side / 80
    for (a0, r0), (a1, r1) in zip(corners, corners[1:] + corners[:1]):
        p, q = np.array([r0 * np.cos(a0), r0 * np.sin(a0)]), np.array([r1 * np.cos(a1), r1 * np.sin(a1)])
        e = q - p
        d = (e[0] * (y - p[1]) - e[1] * (x - p[0])) / np.hypot(*e)                      # signed distance to the edge
        inside &= d >= 0
        inner &= d >= width
    cover = lambda m: m.reshape(side, SS, side, SS).mean((1, 3))
    return cover(inner), cover(inside & ~inner)


def glyphs(vmax, side):
    """Fill and outline coverage as uint8, [vmax + 1, vmax + 1, side, side], by type and size."""
    c = (np.arange(side * SS) + 0.5) / SS - side / 2
    y, x = np.meshgrid(c, c, indexing="ij")
    fill, edge = np.zeros((2, vmax + 1, vmax + 1, side, side), np.uint8)
    for t in range(vmax + 1):
        for s in range(vmax + 1):
            f, e = glyph(t, s, vmax, side, (x, y))
            fill[t, s], edge[t, s] = np.rint(255 * f), np.rint(255 * e)
    return fill, edge


def draw(vals, fill, edge, vmax):
    """Panels of values [..., 3] (type, size, colour), uint8: a higher colour value is darker, as in RAVEN."""
    t, s, c = vals[..., 0], vals[..., 1], vals[..., 2]
    grey = (235 - 195 * c / vmax)[..., None, None]
    f, e = fill[t, s].astype(np.float32) / 255, edge[t, s].astype(np.float32) / 255
    return np.rint(255 * (1 - f - e) + grey * f).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--size", type=int, default=80)
    ap.add_argument("--data-dir", default="experiments/baselines/data")
    args = ap.parse_args()
    z = np.load(os.path.join(args.data_dir, f"{args.name}.npz"))
    meta = json.loads(str(z["meta"]))
    vmax = meta["therm"]
    fill, edge = glyphs(vmax, args.size)
    out = {}
    for s in ("train", "sel", "unseen"):
        vals = z[f"{s}_vals"][:, :meta["n_ctx"], 0, 1:].astype(np.int64)
        out[f"{s}_img"] = np.concatenate([draw(vals[i:i + 200], fill, edge, vmax) for i in range(0, len(vals), 200)])
        out[f"{s}_y"] = z[f"{s}_y"]
        print(s, out[f"{s}_img"].shape, flush=True)
    name = f"{args.name}_images{args.size if args.size != 80 else ''}"
    tmp = os.path.join(args.data_dir, f"{name}.tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, os.path.join(args.data_dir, f"{name}.npz"))


if __name__ == "__main__":
    main()
