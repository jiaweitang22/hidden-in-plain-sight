"""Multi-view anamorphosis: one cloud of specks, K hidden pictures, K secret spots (K = 2 or 3).

Generalises two_view.py. A speck is only kept if every secret camera sees ink of its own picture where the
speck projects (the intersection of all K cones of sight -- a visual hull / space carving). Specks use SH
degree 1: its three linear terms let a speck show a different colour toward up to three roughly
perpendicular viewing directions (e.g. front, side, top). A joint gsplat fit on all K views cleans up.

Views are given as yaw,pitch pairs (degrees) relative to the first camera, which looks along +z at the centre.
Negative pitch raises the camera (cameras.orbit convention).

  python -m hips.multi_view --targets targets/art_great_wave.png targets/art_starry_night.png \
      targets/art_mona_lisa.png --views 0,0 90,0 0,-80 --out /content/out/mv3_art
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization

from hips import cameras, splatio
from hips.option_a import load_target, ssim
from hips.two_view import project, sample

dev = "cuda"
C0 = 0.28209479177387814


def carve_multi(src, others, n, center, Rb, K):
    """Sample points on `src`'s ink rays inside the ball; keep those on ink in every other view.
    Returns points and, per view (src first, then others), the colour each view wants there."""
    rgbS, aS, camS = src
    H, W = aS.shape
    w = (aS > 0.5).float().flatten()
    pts_all, cols_all = [], [[] for _ in range(len(others) + 1)]
    got, tries = 0, 0
    c = torch.tensor(center, dtype=torch.float32, device=dev)
    R = torch.tensor(camS[:3, :3], dtype=torch.float32, device=dev)
    o = torch.tensor(camS[:3, 3], dtype=torch.float32, device=dev)
    while got < n and tries < 80:
        m = 400_000
        idx = torch.multinomial(w, m, replacement=True)
        py = (idx // W).float() + torch.rand(m, device=dev)
        px = (idx % W).float() + torch.rand(m, device=dev)
        d = torch.stack([(px - K[0, 2]) / K[0, 0], (py - K[1, 2]) / K[1, 1], torch.ones_like(px)], -1)
        dw = (d / d.norm(dim=-1, keepdim=True)) @ R.T
        b = ((c - o) * dw).sum(-1)
        h2 = (Rb**2 - ((c - o) ** 2).sum() + b**2)
        ok = h2 > 0
        h = h2.clamp(min=0).sqrt()
        t = b - h + torch.rand(m, device=dev) * 2 * h
        pts = o + dw * t[:, None]
        cols = [rgbS[py.long().clamp(0, H - 1), px.long().clamp(0, W - 1)]]
        for rgbD, aD, camD in others:
            u, v, z = project(pts, camD, K)
            okD, cD = sample(rgbD, aD, u, v)
            ok &= okD & (z > 0)
            cols.append(cD)
        pts_all.append(pts[ok])
        for k in range(len(cols)):
            cols_all[k].append(cols[k][ok])
        got += int(ok.sum())
        tries += 1
    return torch.cat(pts_all)[:n], [torch.cat(cc)[:n] for cc in cols_all]


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    imgs = [load_target(t, a.res) for t in a.targets]
    H, W = imgs[0][1].shape
    assert all(al.shape == (H, W) for _, al in imgs), "all targets must have the same size (use square crops)"
    bg = torch.tensor(a.bg, device=dev)
    tgts = [rgb * al[..., None] + bg * (1 - al[..., None]) for rgb, al in imgs]
    dc = a.dist
    center = np.zeros(3)
    cam0 = cameras.look_at(np.array([0, 0, -dc]), center)
    views = [tuple(float(x) for x in v.split(",")) for v in a.views]
    cams = [cameras.orbit(cam0, center, yw, pitch_deg=pt) for yw, pt in views]
    Kn = cameras.intrinsics(W, H, a.fov)
    K = torch.tensor(Kn, dtype=torch.float32, device=dev)
    Rb = a.ball * dc
    V = len(cams)

    # carve: specks seeded evenly from each view's ink rays, all inside every cone
    means, cols = [], [[] for _ in range(V)]
    for s in range(V):
        others = [(imgs[k][0], imgs[k][1], cams[k]) for k in range(V) if k != s]
        p, cs = carve_multi((imgs[s][0], imgs[s][1], cams[s]), others, a.n // V, center, Rb, K)
        order = [s] + [k for k in range(V) if k != s]  # cs is (src, others...) -> map back to view index
        means.append(p)
        for j, k in enumerate(order):
            cols[k].append(cs[j])
    means = torch.cat(means)
    cols = [torch.cat(c) for c in cols]
    N = means.shape[0]

    def coverage(c2w, alpha):
        u, v, _ = project(means, c2w, K)
        hit = torch.zeros_like(alpha, dtype=torch.bool)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit[v[ok].long(), u[ok].long()] = True
        hit = F.max_pool2d(hit[None, None].float(), 3, 1, 1)[0, 0] > 0
        ink = alpha > 0.5
        return float((hit & ink).sum() / ink.sum())
    covs = [coverage(cams[k], imgs[k][1]) for k in range(V)]
    print(f"{N} specks; coverage " + ", ".join(f"{c:.3f}" for c in covs), flush=True)

    p_sh0 = ((sum(cols) / V - 0.5) / C0).clone().requires_grad_(True)
    nrest = (a.sh + 1) ** 2 - 1  # higher SH coefficients per colour channel (0 for view-independent colour)
    p_sh1 = torch.zeros(N, nrest, 3, device=dev, requires_grad=nrest > 0)
    mn = means.cpu().numpy()
    dmin = np.min([np.linalg.norm(c[:3, 3] - mn, axis=1) for c in cams], axis=0)
    foot = torch.tensor(dmin / Kn[1, 1], dtype=torch.float32, device=dev)[:, None]
    s_max, s_min = a.max_px * foot, a.min_px * foot
    p_scale = torch.zeros(N, 1, device=dev, requires_grad=True)
    p_op = torch.full((N,), 2.0, device=dev, requires_grad=True)
    quats = torch.tensor([1.0, 0, 0, 0], device=dev).repeat(N, 1)
    Kt = K[None]

    def params():
        return ((s_min + (s_max - s_min) * torch.sigmoid(p_scale)).expand(-1, 3), torch.sigmoid(p_op),
                torch.cat([p_sh0[:, None], p_sh1], 1))

    def render(c2w, w_=W, h_=H, K_=None):
        sc, op, col = params()
        vm = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)[None]
        out, al, _ = rasterization(means, quats, sc, op, col, vm, Kt if K_ is None else K_, w_, h_, sh_degree=a.sh)
        return (out[0] + (1 - al[0]) * bg.view(1, 1, 3)).clamp(0, 1)

    opt = torch.optim.Adam([{"params": [p_sh0] + ([p_sh1] if nrest > 0 else []), "lr": 0.01},
                            {"params": [p_op], "lr": 0.05}, {"params": [p_scale], "lr": 0.05}])
    t0 = time.time()
    for it in range(a.steps):
        loss = 0
        for cam, tgt in zip(cams, tgts):
            img = render(cam)
            loss = loss + (img - tgt).abs().mean() + 0.2 * (1 - ssim(img, tgt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 250 == 0 or it == a.steps - 1:
            print(f"it {it:5d} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with torch.no_grad():
        ss = [float(ssim(render(c), t)) for c, t in zip(cams, tgts)]
        # contact sheet: each target and its spot, then in-between views
        row1 = torch.cat([torch.cat([t, render(c)], 1) for t, c in zip(tgts, cams)], 1)
        mids = [cameras.orbit(cam0, center, y, pitch_deg=p) for y, p in [(20, 0), (45, 0), (70, 0), (45, -40), (0, -45), (135, -20)]]
        row2 = torch.cat([render(c) for c in mids], 1)
        wpad = row1.shape[1] - row2.shape[1]
        if wpad > 0:
            row2 = torch.cat([row2, bg.view(1, 1, 3).expand(H, wpad, 3)], 1)
        else:
            row1 = torch.cat([row1, bg.view(1, 1, 3).expand(H, -wpad, 3)], 1)
        sheet = torch.cat([row1, row2], 0)
        im = Image.fromarray((sheet.cpu().numpy() * 255).astype(np.uint8))
        im.thumbnail((3600, 3600))
        im.save(f"{a.out}/sheet.png")
        sc, op, col = params()
        keep = op > 0.05
        if not a.no_ply:
          splatio.write_ply(f"{a.out}/scene.ply", means[keep].cpu().numpy(), (col[keep, 0] * C0 + 0.5).cpu().numpy(),
                          op[keep].cpu().numpy(), sc[keep].cpu().numpy(),
                          sh_rest=col[keep, 1:].cpu().numpy() if nrest > 0 else None)
        labels = a.labels or [f"Spot {k + 1}" for k in range(V)]
        start = cameras.orbit(cam0, center, 45, pitch_deg=-35)
        spot = lambda c, l: {"c2w": c.tolist(), "fov_y_deg": a.fov, "aspect": W / H, "label": l}
        scene = {
            "name": os.path.basename(a.out.rstrip("/")), "title": a.title, "splat": "scene.ply",
            "convention": "opencv", "up": [0, -1, 0], "background": a.bg, "center": center.tolist(),
            "secret": spot(cams[0], labels[0]),
            "secrets": [spot(c, l) for c, l in zip(cams, labels)],
            "start": {"c2w": start.tolist(), "fov_y_deg": a.fov},
            "num_gaussians": int(keep.sum()),
            "metrics": {"coverage": covs, "ssim": ss},
            "params": vars(a),
        }
        json.dump(scene, open(f"{a.out}/scene.json", "w"), indent=1)
        print("RESULT", json.dumps(scene["metrics"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--views", nargs="+", required=True, help="yaw,pitch per target (degrees)")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--dist", type=float, default=10.0)
    ap.add_argument("--ball", type=float, default=0.45)
    ap.add_argument("--max_px", type=float, default=1.5)
    ap.add_argument("--min_px", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--sh", type=int, default=1, help="SH degree of speck colour (0 = same colour from every angle)")
    ap.add_argument("--no_ply", action="store_true", help="skip writing scene.ply (studies)")
    ap.add_argument("--bg", type=float, nargs=3, default=[0.04, 0.04, 0.06])
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
