"""Option A: chaos -> image.

Scatter Gaussians along the secret camera's pixel rays at random depths, so the picture only
reassembles from that one viewpoint. Then a short gsplat fit cleans up overlaps/holes, optionally
using jittered cameras around the secret spot to control how wide the "reveal zone" is.

Runs on a CUDA machine with gsplat (Colab). Usage:
  python -m hips.option_a --target targets/foo.png --out results/foo --n 150000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization

from hips import cameras, splatio

dev = "cuda"


# ---------------------------------------------------------------- utils
def load_target(path, res):
    im = Image.open(path).convert("RGBA")
    w, h = im.size
    s = res / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255.0
    return torch.tensor(a[..., :3], device=dev), torch.tensor(a[..., 3], device=dev)


def ssim(x, y, win=11):
    """x, y: (H,W,3) in [0,1]. Plain gaussian-window SSIM, mean over image."""
    x = x.permute(2, 0, 1)[None]
    y = y.permute(2, 0, 1)[None]
    g = torch.exp(-((torch.arange(win, device=x.device) - win // 2) ** 2) / (2 * 1.5**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None]
    k = g.expand(3, 1, win, win)
    mu = lambda t: F.conv2d(t, k, padding=win // 2, groups=3)
    mx, my = mu(x), mu(y)
    sxx, syy, sxy = mu(x * x) - mx**2, mu(y * y) - my**2, mu(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return s.mean()


def to_viewmat(c2w):
    return torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)[None]


# ---------------------------------------------------------------- main
def build(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rgb_t, a_t = load_target(args.target, args.res)
    H, W = rgb_t.shape[:2]
    bg = torch.tensor(args.bg, device=dev, dtype=torch.float32)
    tgt = rgb_t * a_t[..., None] + bg * (1 - a_t[..., None])

    # Secret camera: at the origin side, looking at the cloud centre.
    dn, df = args.depth
    dc = math.sqrt(dn * df)  # geometric middle of the depth range
    center = np.array([0.0, 0.0, 0.0])
    eye = center - np.array([0.0, 0.0, dc])
    secret = cameras.look_at(eye, center)
    K = cameras.intrinsics(W, H, args.fov)
    fy = K[1, 1]
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)[None]

    # Sample subpixel locations, weighted by target alpha (only "ink" pixels get Gaussians).
    w = (a_t > 0.02).float().flatten()
    idx = torch.multinomial(w, args.n, replacement=True)
    py = (idx // W).float() + torch.rand(args.n, device=dev)
    px = (idx % W).float() + torch.rand(args.n, device=dev)
    dirs_cam = torch.stack([(px - K[0, 2]) / K[0, 0], (py - K[1, 2]) / fy, torch.ones_like(px)], -1)
    u = torch.rand(args.n, device=dev)
    if args.volume == "slab":
        # Depth log-uniform in [dn, df]: the cloud is the picture extruded along the view direction.
        depth = torch.exp(math.log(dn) + u * (math.log(df) - math.log(dn)))
    else:
        # "ball": each ray only gets specks where it passes through a sphere of radius R around the
        # centre, so from any other angle the cloud is a round ball of coloured dust.
        Rb = args.ball * dc
        dn_ = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)  # unit ray in camera frame; centre at (0,0,dc)
        b = dn_[:, 2] * dc  # projection of centre onto ray
        h2 = (Rb**2 - (dc**2 - b**2)).clamp(min=1e-6)
        t0, t1 = b - h2.sqrt(), b + h2.sqrt()
        # volume-uniform along the chord (cubic-in-radius would over-weight the rim); uniform t is fine
        depth = (t0 + u * (t1 - t0)) * dn_[:, 2]  # convert distance-along-ray to camera z
        dn, df = float(depth.min()), float(depth.max())
    R = torch.tensor(secret[:3, :3], dtype=torch.float32, device=dev)
    t = torch.tensor(secret[:3, 3], dtype=torch.float32, device=dev)
    means = (dirs_cam * depth[:, None]) @ R.T + t

    # Colour from the target (bilinear) at each subpixel.
    grid = torch.stack([px / W * 2 - 1, py / H * 2 - 1], -1)[None, None]
    col0 = F.grid_sample(rgb_t.permute(2, 0, 1)[None], grid, align_corners=False)[0, :, 0].T

    # Scale is capped to a fixed pixel footprint at each Gaussian's own depth -> stays a speck.
    px_world = depth / fy  # world size of one secret-view pixel at that depth
    s_max = (args.max_px * px_world)[:, None]
    s_min = (args.min_px * px_world)[:, None]  # size floor: no sub-pixel specks (Mip-Splatting)
    p_scale = torch.full((args.n, 1), 0.0, device=dev, requires_grad=True)  # sigmoid(0)=.5 of max
    p_col = torch.logit(col0.clamp(0.02, 0.98)).clone().requires_grad_(True)
    p_op = torch.full((args.n,), 2.0, device=dev, requires_grad=True)
    quats = torch.tensor([1.0, 0, 0, 0], device=dev).repeat(args.n, 1)

    def params():
        sc = (s_min + (s_max - s_min) * torch.sigmoid(p_scale)).expand(-1, 3)
        return sc, torch.sigmoid(p_op), torch.sigmoid(p_col)

    def render(c2w, w_=W, h_=H, K_=None):
        sc, op, col = params()
        out, alpha, _ = rasterization(
            means, quats, sc, op, col, to_viewmat(c2w), Kt if K_ is None else K_, w_, h_,
            backgrounds=bg[None], packed=False,
        )
        return out[0].clamp(0, 1)

    opt = torch.optim.Adam([
        {"params": [p_col], "lr": 0.02},
        {"params": [p_op], "lr": 0.05},
        {"params": [p_scale], "lr": 0.05},
    ])
    t0 = time.time()
    for it in range(args.steps):
        # Illusion3D-style scheduled jitter: radius grows over the first half of training.
        r = args.jitter * min(1.0, it / max(1, args.steps // 2))
        c2w = secret
        if r > 0:
            off = np.random.randn(3)
            off = off / np.linalg.norm(off) * r * np.random.rand() ** (1 / 3)
            c2w = cameras.look_at(eye + off, center)
        img = render(c2w)
        loss = (img - tgt).abs().mean() + 0.2 * (1 - ssim(img, tgt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 250 == 0 or it == args.steps - 1:
            print(f"it {it:5d} loss {loss.item():.4f} r {r:.3f} ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with torch.no_grad():
        sc, op, col = params()
        # ---- evaluation: reveal zone (translate sideways and re-aim) and orbit views
        rows = {}
        shifts = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]  # fraction of camera distance
        rows["shift"] = []
        for s in shifts:
            c = cameras.look_at(eye + np.array([s * dc, 0, 0]), center)
            im = render(c)
            rows["shift"].append((s, float(ssim(im, tgt)), im))
        yaws = [0, 3, 10, 25, 60, 120]
        rows["orbit"] = []
        for y in yaws:
            c = cameras.orbit(secret, center, y)
            im = render(c)
            rows["orbit"].append((y, float(ssim(im, tgt)), im))
        # contact sheet
        def strip(items):
            return torch.cat([im for _, _, im in items], 1)
        sheet = torch.cat([torch.cat([tgt, strip(rows["shift"])[:, : W * 6]], 1),
                           torch.cat([tgt, strip(rows["orbit"])], 1)], 0)
        Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8)).save(f"{args.out}/sheet.png")
        Image.fromarray((render(secret).cpu().numpy() * 255).astype(np.uint8)).save(f"{args.out}/secret.png")
        # orbit GIF: a slow arc that passes through the secret view
        frames = []
        for k in range(72):
            yaw = 40 * math.sin(2 * math.pi * k / 72)
            c = cameras.orbit(secret, center, yaw, pitch_deg=8 * math.sin(4 * math.pi * k / 72))
            im = render(c, W // 2, H // 2, Kt * torch.tensor([[[0.5], [0.5], [1.0]]], device=dev))
            frames.append(Image.fromarray((im.cpu().numpy() * 255).astype(np.uint8)))
        frames[0].save(f"{args.out}/orbit.gif", save_all=True, append_images=frames[1:], duration=60, loop=0)

        # ---- export splat + scene description for the web viewer
        keep = op > 0.05
        splatio.write_ply(
            f"{args.out}/scene.ply", means[keep].cpu().numpy(), col[keep].cpu().numpy(),
            op[keep].cpu().numpy(), sc[keep].cpu().numpy(),
        )
        start = cameras.orbit(secret, center, 55, pitch_deg=-10)
        scene = {
            "name": os.path.basename(args.out.rstrip("/")),
            "splat": "scene.ply",
            "convention": "opencv",
            "up": [0, -1, 0],
            "background": args.bg,
            "center": center.tolist(),
            "secret": {"c2w": secret.tolist(), "fov_y_deg": args.fov, "aspect": W / H},
            "start": {"c2w": start.tolist(), "fov_y_deg": args.fov},
            "reveal": {"shift_ssim": [(s, v) for s, v, _ in rows["shift"]],
                       "orbit_ssim": [(y, v) for y, v, _ in rows["orbit"]]},
            "params": vars(args),
            "num_gaussians": int(keep.sum()),
        }
        with open(f"{args.out}/scene.json", "w") as f:
            json.dump(scene, f, indent=1)
        print(json.dumps({"shift_ssim": scene["reveal"]["shift_ssim"],
                          "orbit_ssim": scene["reveal"]["orbit_ssim"],
                          "num_gaussians": scene["num_gaussians"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150_000)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--depth", type=float, nargs=2, default=[3.0, 30.0])
    ap.add_argument("--volume", choices=["ball", "slab"], default="ball")
    ap.add_argument("--ball", type=float, default=0.5, help="ball radius as a fraction of camera distance")
    ap.add_argument("--max_px", type=float, default=1.5, help="max Gaussian std in secret-view pixels")
    ap.add_argument("--min_px", type=float, default=0.4, help="min Gaussian std in secret-view pixels")
    ap.add_argument("--jitter", type=float, default=0.0, help="max camera jitter radius (world units)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bg", type=float, nargs=3, default=[0.04, 0.04, 0.06])
    ap.add_argument("--seed", type=int, default=0)
    build(ap.parse_args())
