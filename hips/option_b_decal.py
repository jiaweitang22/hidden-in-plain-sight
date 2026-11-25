"""Option B, decal version: paint a picture into a real 3DGS scene like Felice Varini paints on buildings.

The scene's own Gaussians stay frozen. We add new small, flat "paint" Gaussians lying on the scene's surfaces
exactly where the secret camera sees the picture's ink pixels:
  * depth from the secret view's expected-depth map; points pulled slightly toward the camera;
  * each paint Gaussian is a thin disc oriented along the surface normal (from depth-map gradients), so from
    other viewpoints it looks like paint on the surface, not a floating dot;
  * pixels on depth discontinuities or where the scene is see-through are skipped (no floaters);
  * colour/opacity/size are then fitted so the secret view matches the picture over the scene.
From the secret spot the painting snaps into place; from anywhere else you see painted fragments across
the garden while the garden itself is unchanged.

  python -m hips.option_b_decal --data /content/data/360_v2/garden --ckpt .../ckpt_19999_rank0.pt \
      --target targets/art_pearl_earring_cut.png --out /content/out/b2_decal
  # --preview: only render candidate secret viewpoints with the picture outlined, to choose a spot
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

from hips import cameras
from hips.option_a import ssim
from hips.option_b import Model, Parser, focus_point, load_splats, place_target, psnr

dev = "cuda"
SH_C0 = 0.28209479177387814


def secret_pose(c2ws, elev, azim, dist, shift):
    """Like option_b.choose_secret, but the look-at point can be shifted horizontally (fractions of radius)."""
    f = focus_point(c2ws)
    up = -c2ws[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    radius = float(np.median(np.linalg.norm(c2ws[:, :3, 3] - f, axis=1)))
    ref = c2ws[0, :3, 3] - f
    h1 = ref - up * (ref @ up)
    h1 /= np.linalg.norm(h1)
    h2 = np.cross(up, h1)
    f2 = f + radius * (shift[0] * h1 + shift[1] * h2)
    az, el = math.radians(azim), math.radians(elev)
    d = math.cos(el) * (math.cos(az) * h1 + math.sin(az) * h2) + math.sin(el) * up
    return cameras.look_at(f2 + d * radius * dist, f2, up=up), f2, up, radius


def rot_to_quat(R):
    """(N,3,3) rotation matrices -> (N,4) wxyz quaternions."""
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(R.shape[0], 4, device=R.device)
    s = torch.sqrt((1 + t).clamp(min=1e-8)) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (m[:, 2, 1] - m[:, 1, 2]) / s
    q[:, 2] = (m[:, 0, 2] - m[:, 2, 0]) / s
    q[:, 3] = (m[:, 1, 0] - m[:, 0, 1]) / s
    # fall back to a robust branch where the trace is small
    bad = t < 0
    if bad.any():
        mb = m[bad]
        i = torch.argmax(torch.stack([mb[:, 0, 0], mb[:, 1, 1], mb[:, 2, 2]], -1), -1)
        qb = torch.zeros(mb.shape[0], 4, device=R.device)
        for k in range(3):
            sel = i == k
            if not sel.any():
                continue
            a, b, c = k, (k + 1) % 3, (k + 2) % 3
            mm = mb[sel]
            s2 = torch.sqrt((1 + mm[:, a, a] - mm[:, b, b] - mm[:, c, c]).clamp(min=1e-8)) * 2
            qq = torch.zeros(mm.shape[0], 4, device=R.device)
            qq[:, 0] = (mm[:, c, b] - mm[:, b, c]) / s2
            qq[:, 1 + a] = 0.25 * s2
            qq[:, 1 + b] = (mm[:, b, a] + mm[:, a, b]) / s2
            qq[:, 1 + c] = (mm[:, c, a] + mm[:, a, c]) / s2
            qb[sel] = qq
        q[bad] = qb
    return F.normalize(q, dim=-1)


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    parser = Parser(data_dir=a.data, factor=a.factor, normalize=True, test_every=8)
    c2ws = parser.camtoworlds
    test_ids = [i for i in range(len(c2ws)) if i % 8 == 0]
    cam0 = parser.camera_ids[0]
    K = torch.tensor(parser.Ks_dict[cam0], dtype=torch.float32, device=dev)
    W, H = parser.imsize_dict[cam0]
    sp0 = load_splats(a.ckpt)
    deg = int(math.isqrt(sp0["shN"].shape[1] + 1) - 1)
    orig = Model({k: v.clone() for k, v in sp0.items()}, deg)

    if a.preview:  # candidate viewpoints with the picture's footprint outlined
        tiles = []
        t_rgb, t_a = place_target(a.target, W, H, a.frac)
        edge = (F.max_pool2d(t_a[None, None], 5, 1, 2)[0, 0] - t_a) > 0.3
        for el in a.preview_elev:
            for az in a.preview_azim:
                for sh in [(0, 0), (0.25, 0), (0, 0.25), (-0.25, 0), (0, -0.25)]:
                    sec, *_ = secret_pose(c2ws, el, az, a.dist, sh)
                    with torch.no_grad():
                        im = orig.render(sec, K, W, H)
                    im = torch.where(edge[..., None], torch.tensor([1.0, 0.2, 0.2], device=dev), im)
                    small = F.interpolate(im.permute(2, 0, 1)[None], scale_factor=0.35, mode="area")[0].permute(1, 2, 0)
                    tiles.append((f"el{el}_az{az}_sh{sh[0]}_{sh[1]}", small))
        cols = 5
        rows = [torch.cat([t for _, t in tiles[i:i + cols]], 1) for i in range(0, len(tiles), cols)]
        Image.fromarray((torch.cat(rows, 0).cpu().numpy() * 255).astype(np.uint8)).save(f"{a.out}/preview.png")
        json.dump([n for n, _ in tiles], open(f"{a.out}/preview.json", "w"))
        print("PREVIEW", f"{a.out}/preview.png", flush=True)
        return

    secret, focus, up, radius = secret_pose(c2ws, a.elev, a.azim, a.dist, a.shift)
    dots = [float(secret[:3, 2] @ c[:3, 2]) for c in c2ws]
    print(f"closest training view angle {math.degrees(math.acos(min(1, max(dots)))):.1f} deg", flush=True)
    vm = torch.linalg.inv(torch.as_tensor(secret, dtype=torch.float32, device=dev))
    with torch.no_grad():
        sec_orig = orig.render(secret, K, W, H)
        colors = torch.cat([sp0["sh0"], sp0["shN"]], 1)
        ed, acc, _ = rasterization(sp0["means"], F.normalize(sp0["quats"], dim=-1), torch.exp(sp0["scales"]),
                                   torch.sigmoid(sp0["opacities"]), colors, vm[None], K[None], W, H,
                                   sh_degree=deg, render_mode="ED")
        ed, acc = ed[0, ..., 0], acc[0, ..., 0]
    t_rgb, t_a = place_target(a.target, W, H, a.frac)
    tgt = t_rgb * t_a[..., None] + sec_orig * (1 - t_a[..., None])

    # ---- surface points and normals from the secret view's depth map
    Kinv = torch.linalg.inv(K)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")
    rays = torch.stack([xs, ys, torch.ones_like(xs)], -1) @ Kinv.T  # camera-frame rays with z = 1
    Pc = rays * ed[..., None]  # camera-frame surface points
    dPu = torch.zeros_like(Pc); dPv = torch.zeros_like(Pc)
    dPu[:, 1:-1] = (Pc[:, 2:] - Pc[:, :-2]) / 2
    dPv[1:-1] = (Pc[2:] - Pc[:-2]) / 2
    nrm = F.normalize(torch.cross(dPu, dPv, dim=-1), dim=-1)
    nrm = torch.where((nrm[..., 2:3] > 0), -nrm, nrm)  # face the camera
    # depth discontinuities: depth range in a (dz_win x dz_win) window, relative to depth
    pad = a.dz_win // 2
    dmax = F.max_pool2d(ed[None, None], a.dz_win, 1, pad)[0, 0]
    dmin = -F.max_pool2d(-ed[None, None], a.dz_win, 1, pad)[0, 0]
    dz = dmax - dmin
    cos_all = nrm[..., 2].abs()  # how face-on the surface is to the secret camera
    okpix = (t_a > 0.5) & (acc > 0.95) & (dz / ed.clamp(min=1e-6) < a.max_rel_dz) & (cos_all > a.min_cos)
    print(f"ink pixels {int((t_a > 0.5).sum())}, usable {int(okpix.sum())}", flush=True)

    # ---- sample paint Gaussians on usable ink pixels (subpixel), place on the surface
    w = okpix.float().flatten()
    n = min(a.n, int(w.sum() * a.per_pixel))
    idx = torch.multinomial(w, n, replacement=True)
    py, px = idx // W, idx % W
    jx = torch.rand(n, device=dev) - 0.5
    jy = torch.rand(n, device=dev) - 0.5
    ray = rays[py, px] + torch.stack([jx / K[0, 0], jy / K[1, 1], torch.zeros_like(jx)], -1)
    z = ed[py, px] * (1 - a.lift)
    pc = ray * z[:, None]
    Rc2w = torch.as_tensor(secret[:3, :3], dtype=torch.float32, device=dev)
    tc2w = torch.as_tensor(secret[:3, 3], dtype=torch.float32, device=dev)
    means_d = pc @ Rc2w.T + tc2w
    nw = nrm[py, px] @ Rc2w.T
    # disc frame: t1, t2 in the surface plane, n = thin axis
    tmp = torch.where(nw[:, :1].abs() < 0.9, torch.tensor([1.0, 0, 0], device=dev), torch.tensor([0, 1.0, 0], device=dev))
    t1 = F.normalize(torch.cross(nw, tmp, dim=-1), dim=-1)
    t2 = torch.cross(nw, t1, dim=-1)
    quat_d = rot_to_quat(torch.stack([t1, t2, nw], -1))
    foot = z / K[1, 1]  # world size of one secret-view pixel at that depth (fronto-parallel)
    # a disc seen at an angle looks smaller by |cos|; enlarge in-plane so it still covers its pixel
    cosv = (nrm[py, px][:, 2]).abs().clamp(min=a.min_cos)
    s_min = (a.min_px * foot / cosv)[:, None]
    s_max = (a.max_px * foot / cosv)[:, None]
    col0 = t_rgb[py, px]

    p_col = torch.logit(col0.clamp(0.02, 0.98)).clone().requires_grad_(True)
    p_op = torch.full((n,), 3.0, device=dev, requires_grad=True)
    p_sc = torch.zeros(n, 1, device=dev, requires_grad=True)
    K_ = sp0["shN"].shape[1]

    def decal_params():
        sc = s_min + (s_max - s_min) * torch.sigmoid(p_sc)
        scales = torch.cat([sc, sc, sc * a.thin], 1)
        sh0 = ((torch.sigmoid(p_col) - 0.5) / SH_C0)[:, None]
        return scales, torch.sigmoid(p_op), sh0

    def combined():
        sc, op, sh0 = decal_params()
        return {
            "means": torch.cat([sp0["means"], means_d]),
            "quats": torch.cat([F.normalize(sp0["quats"], dim=-1), quat_d]),
            "scales": torch.cat([sp0["scales"], torch.log(sc)]),
            "opacities": torch.cat([sp0["opacities"], torch.logit(op.clamp(1e-4, 1 - 1e-4))]),
            "sh0": torch.cat([sp0["sh0"], sh0]),
            "shN": torch.cat([sp0["shN"], torch.zeros(n, K_, 3, device=dev)]),
        }

    def render(c2w, K_r=K, W_r=W, H_r=H):
        return Model(combined(), deg).render(c2w, K_r, W_r, H_r)

    opt = torch.optim.Adam([{"params": [p_col], "lr": 0.03}, {"params": [p_op], "lr": 0.05},
                            {"params": [p_sc], "lr": 0.05}])
    rw = a.reveal * radius
    t0 = time.time()
    for it in range(a.steps):
        r = rw * min(1.0, it / max(1, a.steps // 2))
        cs = secret
        if r > 0:
            off = np.random.randn(3); off = off / np.linalg.norm(off) * r * np.random.rand() ** (1 / 3)
            cs = cameras.look_at(secret[:3, 3] + off, focus, up=up)
        img = render(cs)
        loss = (img - tgt).abs().mean() + 0.2 * (1 - ssim(img, tgt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == a.steps - 1:
            print(json.dumps({"it": it, "loss": round(float(loss), 4), "t": round(time.time() - t0)}), flush=True)

    res = {"num_decals": n}
    with torch.no_grad():
        sec = render(secret)
        ink = t_a > 0.5
        res["secret_ink_l1"] = float((sec - tgt)[ink].abs().mean())
        res["secret_ink_l1_before"] = float((sec_orig - tgt)[ink].abs().mean())
        res["secret_ssim_vs_target"] = float(ssim(sec, tgt))
        tp = [psnr(render(c2ws[i]), orig.render(c2ws[i], K, W, H)) for i in test_ids]
        res["heldout_psnr_vs_original"] = float(np.mean(tp))
        res["heldout_psnr_vs_original_min"] = float(np.min(tp))

        def save(img, name):
            Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(f"{a.out}/{name}")
        save(sec, "secret.png"); save(sec_orig, "secret_original.png"); save(tgt, "secret_target.png")
        for j, i in enumerate(test_ids[:4]):
            save(torch.cat([orig.render(c2ws[i], K, W, H), render(c2ws[i])], 1), f"heldout_{j}.png")
        sp = combined()
    json.dump({**res, "args": vars(a)}, open(f"{a.out}/metrics.json", "w"), indent=1)
    torch.save({"splats": {k: v.detach().cpu() for k, v in sp.items()}, "secret_c2w": secret, "focus": focus,
                "up": up, "radius": radius, "K": K.cpu().numpy(), "W": W, "H": H}, f"{a.out}/hidden.pt")
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--elev", type=float, default=65.0)
    ap.add_argument("--azim", type=float, default=30.0)
    ap.add_argument("--dist", type=float, default=0.8)
    ap.add_argument("--shift", type=float, nargs=2, default=[0.0, 0.0], help="look-at shift (fractions of radius)")
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--per_pixel", type=float, default=1.5)
    ap.add_argument("--lift", type=float, default=0.004, help="pull paint toward the camera (relative depth)")
    ap.add_argument("--thin", type=float, default=0.05, help="disc thickness relative to its radius")
    ap.add_argument("--min_px", type=float, default=0.5)
    ap.add_argument("--max_px", type=float, default=1.2)
    ap.add_argument("--max_rel_dz", type=float, default=0.02, help="skip pixels on depth discontinuities")
    ap.add_argument("--dz_win", type=int, default=5, help="window for the depth-discontinuity test")
    ap.add_argument("--min_cos", type=float, default=0.5, help="skip surfaces seen at grazing angles")
    ap.add_argument("--reveal", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview_elev", type=float, nargs="+", default=[50, 70])
    ap.add_argument("--preview_azim", type=float, nargs="+", default=[0, 90, 180, 270])
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
