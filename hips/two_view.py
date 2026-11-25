"""Two-view anamorphosis: one cloud of specks, two hidden pictures, two secret spots.

Picture A is seen from camera A, picture B from camera B (orbited `--angle` degrees around the centre).
A speck is only placed where camera A sees ink in A *and* camera B sees ink in B -- the intersection of the
two cones of sight (a visual hull, as in space carving / Shadow Art). This is stereo triangulation run
backwards: instead of recovering a 3D point from two matching pixels, we choose 3D points so that their two
projections land on the ink we want. Orbiting around the vertical axis keeps image rows (roughly, the
epipolar lines), so a pixel of A can only be covered if B has ink near the same row.

Each speck gets SH degree-1 colour so it can show A's colour toward camera A and B's colour toward camera B.
A short gsplat fit then optimizes colours/opacity/size against both views (optionally with camera jitter).

  python -m hips.two_view --a targets/hidden.png --b targets/sight.png --out /content/out/tv_words
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
from hips.option_a import load_target, ssim

dev = "cuda"
C0 = 0.28209479177387814


def project(pts, c2w, K):
    vm = np.linalg.inv(c2w)
    R = torch.tensor(vm[:3, :3], dtype=torch.float32, device=dev)
    t = torch.tensor(vm[:3, 3], dtype=torch.float32, device=dev)
    pc = pts @ R.T + t
    z = pc[:, 2]
    u = pc[:, 0] / z * K[0, 0] + K[0, 2]
    v = pc[:, 1] / z * K[1, 1] + K[1, 2]
    return u, v, z


def sample(rgb, alpha, u, v):
    H, W = alpha.shape
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ui, vi = u.long().clamp(0, W - 1), v.long().clamp(0, H - 1)
    return ok & (alpha[vi, ui] > 0.5), rgb[vi, ui]


def carve(src, dst, n, dc, Rb, K, rng_dev):
    """Sample points on ink rays of `src` camera inside the ball, keep those landing on ink in `dst`."""
    (rgbS, aS, camS), (rgbD, aD, camD) = src, dst
    H, W = aS.shape
    w = (aS > 0.5).float().flatten()
    keep_pts, keep_cS, keep_cD = [], [], []
    got, tries = 0, 0
    center = torch.zeros(3, device=dev)
    while got < n and tries < 60:
        m = 400_000
        idx = torch.multinomial(w, m, replacement=True)
        py = (idx // W).float() + torch.rand(m, device=dev)
        px = (idx % W).float() + torch.rand(m, device=dev)
        d = torch.stack([(px - K[0, 2]) / K[0, 0], (py - K[1, 2]) / K[1, 1], torch.ones_like(px)], -1)
        d = d / d.norm(dim=-1, keepdim=True)
        R = torch.tensor(camS[:3, :3], dtype=torch.float32, device=dev)
        o = torch.tensor(camS[:3, 3], dtype=torch.float32, device=dev)
        dw = d @ R.T
        b = ((center - o) * dw).sum(-1)
        h2 = (Rb**2 - ((center - o) ** 2).sum() + b**2).clamp(min=0)
        t = b - h2.sqrt() + torch.rand(m, device=dev) * 2 * h2.sqrt()
        pts = o + dw * t[:, None]
        u, v, z = project(pts, camD, K)
        okD, cD = sample(rgbD, aD, u, v)
        okD &= (h2 > 0) & (z > 0)
        cS = rgbS[py.long().clamp(0, H - 1), px.long().clamp(0, W - 1)]
        keep_pts.append(pts[okD]); keep_cS.append(cS[okD]); keep_cD.append(cD[okD])
        got += int(okD.sum()); tries += 1
    pts = torch.cat(keep_pts)[:n]
    return pts, torch.cat(keep_cS)[:n], torch.cat(keep_cD)[:n]


def main(a):
    torch.manual_seed(a.seed)
    rgbA, aA = load_target(a.a, a.res)
    rgbB, aB = load_target(a.b, a.res)
    H, W = aA.shape
    bg = torch.tensor(a.bg, device=dev)
    tA = rgbA * aA[..., None] + bg * (1 - aA[..., None])
    tB = rgbB * aB[..., None] + bg * (1 - aB[..., None])
    dc = a.dist
    center = np.zeros(3)
    camA = cameras.look_at(np.array([0, 0, -dc]), center)
    camB = cameras.orbit(camA, center, a.angle)
    Kn = cameras.intrinsics(W, H, a.fov)
    K = torch.tensor(Kn, dtype=torch.float32, device=dev)
    Rb = a.ball * dc

    # carve: half the specks seeded from A's ink rays, half from B's, all inside both cones
    pA, cAA, cAB = carve((rgbA, aA, camA), (rgbB, aB, camB), a.n // 2, dc, Rb, K, None)
    pB, cBB, cBA = carve((rgbB, aB, camB), (rgbA, aA, camA), a.n // 2, dc, Rb, K, None)
    means = torch.cat([pA, pB])
    colA = torch.cat([cAA, cBA])  # colour wanted toward camera A
    colB = torch.cat([cAB, cBB])  # colour wanted toward camera B
    N = means.shape[0]

    # coverage: fraction of each picture's ink pixels that some speck projects onto
    def coverage(c2w, alpha):
        u, v, _ = project(means, c2w, K)
        hit = torch.zeros_like(alpha, dtype=torch.bool)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit[v[ok].long(), u[ok].long()] = True
        ink = alpha > 0.5
        # dilate hits by 1px (specks have footprint)
        hit = F.max_pool2d(hit[None, None].float(), 3, 1, 1)[0, 0] > 0
        return float((hit & ink).sum() / ink.sum())
    covA, covB = coverage(camA, aA), coverage(camB, aB)
    print(f"{N} specks; ink coverage A {covA:.3f}, B {covB:.3f}", flush=True)

    # SH degree 1. Initialise DC as the mean colour, then let the fit split it by direction.
    sh0 = ((colA + colB) / 2 - 0.5) / C0
    p_sh0 = sh0.clone().requires_grad_(True)
    p_sh1 = torch.zeros(N, 3, 3, device=dev, requires_grad=True)
    dA = np.linalg.norm(camA[:3, 3] - means.cpu().numpy(), axis=1)
    dB = np.linalg.norm(camB[:3, 3] - means.cpu().numpy(), axis=1)
    foot = torch.tensor(np.minimum(dA, dB) / Kn[1, 1], dtype=torch.float32, device=dev)[:, None]  # world size of 1 px
    s_max, s_min = a.max_px * foot, a.min_px * foot  # size floor: no sub-pixel specks (Mip-Splatting)
    p_scale = torch.zeros(N, 1, device=dev, requires_grad=True)
    p_op = torch.full((N,), 2.0, device=dev, requires_grad=True)
    quats = torch.tensor([1.0, 0, 0, 0], device=dev).repeat(N, 1)
    Kt = K[None]

    def params():
        return (s_min + (s_max - s_min) * torch.sigmoid(p_scale)).expand(-1, 3), torch.sigmoid(p_op), torch.cat([p_sh0[:, None], p_sh1], 1)

    def render(c2w, w_=W, h_=H, K_=None):
        sc, op, col = params()
        vm = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)[None]
        out, al, _ = rasterization(means, quats, sc, op, col, vm, Kt if K_ is None else K_, w_, h_, sh_degree=1)
        return (out[0] + (1 - al[0]) * bg.view(1, 1, 3)).clamp(0, 1)

    opt = torch.optim.Adam([{"params": [p_sh0, p_sh1], "lr": 0.01},
                            {"params": [p_op], "lr": 0.05}, {"params": [p_scale], "lr": 0.05}])
    t0 = time.time()
    for it in range(a.steps):
        r = a.jitter * min(1.0, it / max(1, a.steps // 2))
        loss = 0
        for cam, tgt in ((camA, tA), (camB, tB)):
            c = cam
            if r > 0:
                off = np.random.randn(3); off = off / np.linalg.norm(off) * r * np.random.rand() ** (1 / 3)
                c = cameras.look_at(cam[:3, 3] + off, center)
            img = render(c)
            loss = loss + (img - tgt).abs().mean() + 0.2 * (1 - ssim(img, tgt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 250 == 0 or it == a.steps - 1:
            print(f"it {it:5d} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with torch.no_grad():
        yaws = [0, 10, 25, 45, 65, 80, 90, 135, 200]
        tiles = [render(cameras.orbit(camA, center, y)) for y in yaws]
        sheet = torch.cat([torch.cat([tA] + tiles[:4], 1), torch.cat([tiles[4], tiles[5], tiles[6], tB, tiles[7]], 1)], 0)
        Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8)).save(f"{a.out}/sheet.png")
        sA, sB = float(ssim(render(camA), tA)), float(ssim(render(camB), tB))
        # 360 orbit gif
        frames = []
        Kh = Kt * torch.tensor([[[0.5], [0.5], [1.0]]], device=dev)
        for k in range(120):
            y = 360 * k / 120
            im = render(cameras.orbit(camA, center, y), W // 2, H // 2, Kh)
            frames.append(Image.fromarray((im.cpu().numpy() * 255).astype(np.uint8)))
        frames[0].save(f"{a.out}/orbit.gif", save_all=True, append_images=frames[1:], duration=50, loop=0)
        sc, op, col = params()
        keep = op > 0.05
        rgb = (col[:, 0] * C0 + 0.5)
        splatio.write_ply(f"{a.out}/scene.ply", means[keep].cpu().numpy(), rgb[keep].cpu().numpy(),
                          op[keep].cpu().numpy(), sc[keep].cpu().numpy(), sh_rest=col[keep, 1:].cpu().numpy())
        start = cameras.orbit(camA, center, -40, pitch_deg=-12)
        scene = {
            "name": os.path.basename(a.out.rstrip("/")), "title": a.title, "splat": "scene.ply",
            "convention": "opencv", "up": [0, -1, 0], "background": a.bg, "center": center.tolist(),
            "secret": {"c2w": camA.tolist(), "fov_y_deg": a.fov, "aspect": W / H},
            "secrets": [{"c2w": camA.tolist(), "fov_y_deg": a.fov, "aspect": W / H, "label": a.label_a},
                        {"c2w": camB.tolist(), "fov_y_deg": a.fov, "aspect": W / H, "label": a.label_b}],
            "start": {"c2w": start.tolist(), "fov_y_deg": a.fov},
            "num_gaussians": int(keep.sum()),
            "metrics": {"coverage_a": covA, "coverage_b": covB, "ssim_a": sA, "ssim_b": sB},
            "params": vars(a),
        }
        json.dump(scene, open(f"{a.out}/scene.json", "w"), indent=1)
        print("RESULT", json.dumps(scene["metrics"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--label_a", default="Spot 1")
    ap.add_argument("--label_b", default="Spot 2")
    ap.add_argument("--angle", type=float, default=90.0)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--dist", type=float, default=10.0)
    ap.add_argument("--ball", type=float, default=0.45)
    ap.add_argument("--max_px", type=float, default=1.5)
    ap.add_argument("--min_px", type=float, default=0.4)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bg", type=float, nargs=3, default=[0.04, 0.04, 0.06])
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
