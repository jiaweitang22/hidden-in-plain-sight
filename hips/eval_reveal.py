"""Evaluate any exported scene (scene.ply + scene.json) the way the viewer shows it.

* Reveal-zone curves with a "recognisability" score: normalised cross-correlation (NCC) between the
  rendered view and the hidden picture, over the picture's ink region only. It is ~1 at the secret spot
  and ~0 when the specks are scrambled (unlike SSIM, which stays high on matching dark background).
* A fly-in video from the start pose to the secret pose (for the presentation).

GPU (gsplat). Usage:
  python -m hips.eval_reveal --scene web_scenes/a2_eye --target targets/eye.png --out /content/out/eval_a2_eye
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization

from hips import cameras, splatio

dev = "cuda"
C0 = 0.28209479177387814


def load_scene(d):
    meta = json.load(open(f"{d}/scene.json"))
    s = splatio.read_ply(f"{d}/{meta.get('splat', 'scene.ply')}")
    t = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
    sh0 = (t(s["rgb"]) - 0.5) / C0
    if s["sh_rest"] is not None:
        colors = torch.cat([sh0[:, None], t(s["sh_rest"])], 1)
        deg = int(math.isqrt(colors.shape[1]) - 1)
    else:
        colors, deg = sh0[:, None], 0
    return meta, dict(means=t(s["means"]), quats=F.normalize(t(s["quats"]), dim=-1), scales=t(s["scales"]),
                      opac=t(s["alpha"]), colors=colors, deg=deg)


def render(g, c2w, fov, W, H, bg):
    K = torch.tensor(cameras.intrinsics(W, H, fov), dtype=torch.float32, device=dev)[None]
    vm = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)[None]
    out, _, _ = rasterization(g["means"], g["quats"], g["scales"], g["opac"], g["colors"], vm, K, W, H,
                              sh_degree=g["deg"], backgrounds=bg)
    return out[0].clamp(0, 1)


def ncc(a, b, m):
    a = a.mean(-1)[m]
    b = b.mean(-1)[m]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main(a):
    meta, g = load_scene(a.scene)
    sec = np.array(meta["secret"]["c2w"])
    fov = meta["secret"]["fov_y_deg"]
    asp = meta["secret"].get("aspect", 1.0)
    H = a.res
    W = int(round(H * asp))
    bgc = meta.get("background", [0, 0, 0])
    bg = torch.tensor(bgc, dtype=torch.float32, device=dev)[None]
    up = tuple(meta.get("up", [0, -1, 0]))
    center = np.array(meta["center"])
    dist = np.linalg.norm(sec[:3, 3] - center)

    # hidden picture as it should appear in the secret view (fills view for A; centred `frac` for B)
    im = Image.open(a.target).convert("RGBA")
    s = a.frac * min(W, H) / max(im.size) if a.frac < 1 else min(W / im.size[0], H / im.size[1])
    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(im, ((W - im.size[0]) // 2, (H - im.size[1]) // 2))
    tg = torch.tensor(np.asarray(canvas, np.float32) / 255, device=dev)
    mask = tg[..., 3] > 0.5
    ys, xs = torch.nonzero(mask, as_tuple=True)
    box = torch.zeros_like(mask)
    box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True  # ink bounding box
    with torch.no_grad():
        ref = render(g, sec, fov, W, H, bg)  # the secret view itself is the reference picture
        res = {"shift": [], "orbit": []}
        for f in [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            c = cameras.look_at(sec[:3, 3] + f * dist * sec[:3, 0], center, up=up)
            res["shift"].append((f, ncc(render(g, c, fov, W, H, bg), ref, box)))
        for y in [0, 1, 2, 3, 5, 10, 25, 60, 120]:
            c = cameras.orbit(sec, center, y, up=up)
            res["orbit"].append((y, ncc(render(g, c, fov, W, H, bg), ref, box)))
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(f"{a.out}/reveal.json", "w"), indent=1)
        print("REVEAL", json.dumps(res), flush=True)

        if a.video:
            import imageio.v2 as imageio
            start = np.array(meta["start"]["c2w"])
            frames = []
            n = a.frames
            for k in range(n + 30):
                t = min(1.0, k / n)
                t = t * t * (3 - 2 * t)
                # orbit-interpolate around the centre so the path stays aimed at the scene
                eye = (1 - t) * start[:3, 3] + t * sec[:3, 3]
                d0, d1 = np.linalg.norm(start[:3, 3] - center), dist
                v = eye - center
                eye = center + v / np.linalg.norm(v) * ((1 - t) * d0 + t * d1)
                c = cameras.look_at(eye, center, up=up)
                frames.append((render(g, c, fov, W, H, bg).cpu().numpy() * 255).astype(np.uint8))
            imageio.mimsave(f"{a.out}/flyin.mp4", frames, fps=30, quality=8, macro_block_size=8)
            Image.fromarray(frames[0]).save(f"{a.out}/start.png")
            Image.fromarray(frames[-1]).save(f"{a.out}/secret.png")
            print("VIDEO", f"{a.out}/flyin.mp4", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0, help="<1: target centred at this fraction (option B)")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--frames", type=int, default=120)
    main(ap.parse_args())
