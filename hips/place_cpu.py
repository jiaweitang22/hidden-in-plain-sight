"""CPU-only Option A placement (no fitting): scatter Gaussians along secret-camera rays.

Produces the same scene.ply + scene.json contract as option_a.py, for viewer development
and quick tests without a GPU.
  python -m hips.place_cpu --target targets/eye.png --out web/scenes/test-eye
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from PIL import Image

from hips import cameras, splatio


def main(a):
    rng = np.random.default_rng(a.seed)
    im = Image.open(a.target).convert("RGBA")
    s = a.res / max(im.size)
    im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
    img = np.asarray(im, np.float32) / 255
    H, W = img.shape[:2]
    alpha = img[..., 3].ravel()
    p = (alpha > 0.02).astype(np.float64)
    idx = rng.choice(H * W, size=a.n, p=p / p.sum())
    py = idx // W + rng.random(a.n)
    px = idx % W + rng.random(a.n)
    dn, df = a.depth
    depth = np.exp(np.log(dn) + rng.random(a.n) * (np.log(df) - np.log(dn)))
    dc = math.sqrt(dn * df)
    center = np.zeros(3)
    eye = np.array([0.0, 0.0, -dc])
    secret = cameras.look_at(eye, center)
    K = cameras.intrinsics(W, H, a.fov)
    dirs = np.stack([(px - K[0, 2]) / K[0, 0], (py - K[1, 2]) / K[1, 1], np.ones(a.n)], -1)
    means = (dirs * depth[:, None]) @ secret[:3, :3].T + secret[:3, 3]
    col = img[np.clip(py.astype(int), 0, H - 1), np.clip(px.astype(int), 0, W - 1), :3]
    sc = np.repeat((a.max_px * 0.6 * depth / K[1, 1])[:, None], 3, 1)
    os.makedirs(a.out, exist_ok=True)
    splatio.write_ply(f"{a.out}/scene.ply", means, col, np.full(a.n, 0.95), sc)
    start = cameras.orbit(secret, center, 55, pitch_deg=-10)
    json.dump({
        "name": os.path.basename(a.out.rstrip("/")),
        "title": a.title or os.path.basename(a.out.rstrip("/")),
        "splat": "scene.ply",
        "convention": "opencv",
        "up": [0, -1, 0],
        "background": [0.04, 0.04, 0.06],
        "center": center.tolist(),
        "secret": {"c2w": secret.tolist(), "fov_y_deg": a.fov, "aspect": W / H},
        "start": {"c2w": start.tolist(), "fov_y_deg": a.fov},
        "num_gaussians": a.n,
    }, open(f"{a.out}/scene.json", "w"), indent=1)
    print("wrote", a.out, a.n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--n", type=int, default=120_000)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--depth", type=float, nargs=2, default=[3.0, 30.0])
    ap.add_argument("--max_px", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
