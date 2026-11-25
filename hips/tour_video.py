"""Tour video of a multi-painting scene: start pose -> secret 1 (hold) -> secret 2 (hold) -> ...

The camera moves on a sphere around the scene centre (radius interpolated), and its orientation is
slerped between the key poses, so each hold frame is exactly the secret camera the cloud was built for.
Also saves the key frames as stills. GPU (gsplat).

  python -m hips.tour_video --scene /content/out/mv3_art --out /content/out/tour_mv3_art
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from hips.eval_reveal import load_scene, render, torch


def segment(a, b, center, n):
    """n camera poses from c2w a to c2w b (excluding a), smoothstep-eased."""
    ra, rb = a[:3, 3] - center, b[:3, 3] - center
    da, db = np.linalg.norm(ra), np.linalg.norm(rb)
    ua, ub = ra / da, rb / db
    om = np.arccos(np.clip(ua @ ub, -1, 1))
    sl = Slerp([0, 1], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))
    out = []
    for k in range(1, n + 1):
        t = k / n
        t = t * t * (3 - 2 * t)
        u = ua if om < 1e-6 else (np.sin((1 - t) * om) * ua + np.sin(t * om) * ub) / np.sin(om)
        c = np.eye(4)
        c[:3, :3] = sl(t).as_matrix()
        c[:3, 3] = center + u * ((1 - t) * da + t * db)
        out.append(c)
    return out


def main(a):
    meta, g = load_scene(a.scene)
    center = np.array(meta["center"], np.float64)
    secrets = meta.get("secrets") or [meta["secret"]]
    fov = secrets[0]["fov_y_deg"]
    W = H = a.res
    bg = torch.tensor(meta.get("background", [0, 0, 0]), dtype=torch.float32, device="cuda")[None]
    keys = [np.array(meta["start"]["c2w"])] + [np.array(s["c2w"]) for s in secrets]
    os.makedirs(a.out, exist_ok=True)

    def shot(c):
        with torch.no_grad():
            return (render(g, c, fov, W, H, bg).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    frames = [shot(keys[0])] * a.hold
    Image.fromarray(frames[0]).save(f"{a.out}/key_0_start.jpg", quality=90)
    for i in range(1, len(keys)):
        path = segment(keys[i - 1], keys[i], center, a.move)
        for j, c in enumerate(path):
            f = shot(c)
            frames.append(f)
            if j == len(path) // 2:
                Image.fromarray(f).save(f"{a.out}/key_{i}_between.jpg", quality=90)
        Image.fromarray(frames[-1]).save(f"{a.out}/key_{i}_secret.jpg", quality=90)
        frames += [frames[-1]] * a.hold
    import imageio.v2 as imageio
    imageio.mimsave(f"{a.out}/tour.mp4", frames, fps=30, quality=8, macro_block_size=8)
    print("TOUR", f"{a.out}/tour.mp4", len(frames), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=720)
    ap.add_argument("--move", type=int, default=75)
    ap.add_argument("--hold", type=int, default=45)
    main(ap.parse_args())
