"""Option B: hide an image inside a real, trained 3DGS scene.

Fine-tune an existing scene's Gaussians so that
  * from one secret viewpoint, the ink pixels of a target image appear over the scene, and
  * from every normal viewpoint, the scene looks unchanged.

Loss = alpha * L(secret render, target composited over the ORIGINAL secret render)
     + beta  * L(training-view renders, ORIGINAL renders)            (ComplicitSplat-style anchor)
     + gamma * L(guard-ring renders near the secret pose, ORIGINAL)   (bounds the reveal zone,
                                                                       blocks "screen in front of the
                                                                       camera" cheats)
Optimized: positions (displacement-capped), base colour sh0 (view-independent), opacity.
Higher SH bands (view-dependent colour) are frozen, so the picture has to come from where the
Gaussians are and what base colour they have -- not from angle-dependent colour tricks.

Run on the Colab VM:
  python -m hips.option_b --data /content/data/360_v2/garden \
      --ckpt /content/out/garden/ckpts/ckpt_19999_rank0.pt --target targets/eye.png --out /content/out/b1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization

sys.path.insert(0, "/content/gsplat/examples")
from datasets.colmap import Parser  # noqa: E402

from hips import cameras, splatio  # noqa: E402
from hips.option_a import ssim  # noqa: E402

dev = "cuda"
SH_C0 = 0.28209479177387814


# ------------------------------------------------------------------ scene helpers
def load_splats(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    sp = ck["splats"]
    return {k: v.detach().to(dev).float() for k, v in sp.items()}


def focus_point(c2ws):
    """Least-squares point closest to all optical axes."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c in c2ws:
        d = c[:3, 2] / np.linalg.norm(c[:3, 2])
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c[:3, 3]
    return np.linalg.solve(A, b)


def choose_secret(c2ws, elev_deg, azim_deg, dist_scale):
    f = focus_point(c2ws)
    up = -c2ws[:, :3, 1].mean(0)  # OpenCV y is down
    up /= np.linalg.norm(up)
    radius = np.median(np.linalg.norm(c2ws[:, :3, 3] - f, axis=1))
    # horizontal basis
    ref = c2ws[0, :3, 3] - f
    h1 = ref - up * (ref @ up)
    h1 /= np.linalg.norm(h1)
    h2 = np.cross(up, h1)
    az, el = math.radians(azim_deg), math.radians(elev_deg)
    d = math.cos(el) * (math.cos(az) * h1 + math.sin(az) * h2) + math.sin(el) * up
    eye = f + d * radius * dist_scale
    return cameras.look_at(eye, f, up=up), f, up, radius


class Model:
    def __init__(self, sp, sh_degree):
        self.sp = sp
        self.sh_degree = sh_degree

    def render(self, c2w, K, W, H, override=None, bg=None):
        sp = dict(self.sp)
        if override:
            sp.update(override)
        colors = torch.cat([sp["sh0"], sp["shN"]], 1)
        vm = torch.linalg.inv(torch.as_tensor(c2w, dtype=torch.float32, device=dev))[None]
        out, _, _ = rasterization(
            sp["means"], F.normalize(sp["quats"], dim=-1), torch.exp(sp["scales"]),
            torch.sigmoid(sp["opacities"]), colors, vm, K[None], W, H,
            sh_degree=self.sh_degree, backgrounds=bg,
        )
        return out[0].clamp(0, 1)


def place_target(path, W, H, frac):
    """Target RGBA scaled to `frac` of the secret view's short side, centred. Returns rgb, alpha (H,W)."""
    im = Image.open(path).convert("RGBA")
    s = frac * min(W, H) / max(im.size)
    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(im, ((W - im.size[0]) // 2, (H - im.size[1]) // 2))
    a = torch.tensor(np.asarray(canvas, np.float32) / 255, device=dev)
    return a[..., :3], a[..., 3]


def jitter_pose(c2w, focus, r, up):
    off = np.random.randn(3)
    off = off / np.linalg.norm(off) * r * np.random.rand() ** (1 / 3)
    return cameras.look_at(c2w[:3, 3] + off, focus, up=up)


def guard_ring(secret, focus, up, radius_world, n=8):
    """Cameras on a ring around the secret eye (perpendicular to the view), still aimed at focus."""
    x, y = secret[:3, 0], secret[:3, 1]
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        eye = secret[:3, 3] + radius_world * (math.cos(a) * x + math.sin(a) * y)
        out.append(cameras.look_at(eye, focus, up=up))
    return out


def l1_ssim(a, b):
    return (a - b).abs().mean() + 0.2 * (1 - ssim(a, b))


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean() + 1e-10))


# ------------------------------------------------------------------ main
def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    parser = Parser(data_dir=a.data, factor=a.factor, normalize=True, test_every=8)
    c2ws = parser.camtoworlds
    n_img = len(c2ws)
    test_ids = [i for i in range(n_img) if i % 8 == 0]
    train_ids = [i for i in range(n_img) if i % 8 != 0]
    cam0 = parser.camera_ids[0]
    K = torch.tensor(parser.Ks_dict[cam0], dtype=torch.float32, device=dev)
    W, H = parser.imsize_dict[cam0]
    if a.render_scale != 1.0:  # train at reduced resolution to save time/memory
        K = K.clone()
        K[:2] *= a.render_scale
        W, H = int(W * a.render_scale), int(H * a.render_scale)

    sp0 = load_splats(a.ckpt)
    sh_degree = int(math.isqrt(sp0["shN"].shape[1] + 1) - 1)
    print(f"{sp0['means'].shape[0]} Gaussians, sh_degree {sh_degree}, {n_img} images, {W}x{H}", flush=True)
    orig = Model({k: v.clone() for k, v in sp0.items()}, sh_degree)

    secret, focus, up, radius = choose_secret(c2ws, a.elev, a.azim, a.dist)
    # nearest training camera (angle between view directions), to report overlap
    dots = [float(secret[:3, 2] @ c[:3, 2]) for c in c2ws]
    print(f"secret eye {secret[:3,3].round(3)} focus {focus.round(3)} radius {radius:.3f}; "
          f"closest training view angle {math.degrees(math.acos(min(1, max(dots)))):.1f} deg", flush=True)

    # ---- cache original renders (uint8 on GPU) for anchor views
    with torch.no_grad():
        anchor = {i: (orig.render(c2ws[i], K, W, H) * 255).to(torch.uint8) for i in train_ids}
        sec_orig = orig.render(secret, K, W, H)
    t_rgb, t_a = place_target(a.target, W, H, a.frac)
    tgt = t_rgb * t_a[..., None] + sec_orig * (1 - t_a[..., None])
    rw = a.reveal * radius  # world-space jitter radius = reveal zone
    guards = guard_ring(secret, focus, up, a.guard * radius, n=8)
    with torch.no_grad():
        guard_orig = [orig.render(g, K, W, H) for g in guards]

    # ---- trainable params
    p = {k: sp0[k].clone().requires_grad_(k in ("means", "sh0", "opacities")) for k in sp0}
    means0 = sp0["means"].clone()
    max_disp = a.max_disp * radius
    model = Model(p, sh_degree)
    opt = torch.optim.Adam([
        {"params": [p["means"]], "lr": a.lr_means * radius},
        {"params": [p["sh0"]], "lr": 0.01},
        {"params": [p["opacities"]], "lr": 0.05},
    ])
    log = []
    t0 = time.time()
    for it in range(a.steps):
        r = rw * min(1.0, it / max(1, a.steps // 2))
        cs = jitter_pose(secret, focus, r, up) if r > 0 else secret
        loss_s = l1_ssim(model.render(cs, K, W, H), tgt)
        loss_a = 0
        for i in np.random.choice(train_ids, a.batch, replace=False):
            loss_a = loss_a + (model.render(c2ws[i], K, W, H) - anchor[i].float() / 255).abs().mean()
        loss_a = loss_a / a.batch
        g = np.random.randint(len(guards))
        loss_g = (model.render(guards[g], K, W, H) - guard_orig[g]).abs().mean()
        loss = a.alpha * loss_s + a.beta * loss_a + a.gamma * loss_g
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():  # displacement cap
            d = p["means"] - means0
            n = d.norm(dim=-1, keepdim=True)
            p["means"].copy_(means0 + d * torch.clamp(max_disp / (n + 1e-12), max=1.0))
        if it % 100 == 0 or it == a.steps - 1:
            rec = dict(it=it, secret=float(loss_s), anchor=float(loss_a), guard=float(loss_g), t=round(time.time() - t0))
            log.append(rec)
            print(json.dumps(rec), flush=True)

    # ---- evaluation
    res = {}
    with torch.no_grad():
        sec = model.render(secret, K, W, H)
        ink = t_a > 0.5
        res["secret_ink_l1"] = float((sec - tgt)[ink].abs().mean())
        res["secret_ssim_vs_target"] = float(ssim(sec, tgt))
        res["secret_ssim_orig_vs_target"] = float(ssim(sec_orig, tgt))
        tp, tp0 = [], []
        for i in test_ids:
            o = orig.render(c2ws[i], K, W, H)
            m = model.render(c2ws[i], K, W, H)
            tp.append(psnr(m, o))
        res["heldout_psnr_vs_original"] = float(np.mean(tp))
        res["heldout_psnr_vs_original_min"] = float(np.min(tp))
        # reveal zone: translate sideways from the secret pose (fraction of radius), re-aim at focus
        rz = []
        for s in [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
            c = cameras.look_at(secret[:3, 3] + s * radius * secret[:3, 0], focus, up=up)
            im = model.render(c, K, W, H)
            im0 = orig.render(c, K, W, H)
            # how much of the hidden image is visible = SSIM gain vs original over the ink region box
            rz.append((s, float(ssim(im, tgt)), float(ssim(im0, tgt))))
        res["reveal_shift"] = rz
        res["max_disp_used"] = float((p["means"] - means0).norm(dim=-1).max() / radius)
        res["frac_moved_gt_1pct"] = float(((p["means"] - means0).norm(dim=-1) > 0.01 * radius).float().mean())

        def save(img, name):
            Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(f"{a.out}/{name}")
        save(sec, "secret.png")
        save(sec_orig, "secret_original.png")
        save(tgt, "secret_target.png")
        for j, i in enumerate(test_ids[:3]):
            save(torch.cat([orig.render(c2ws[i], K, W, H), model.render(c2ws[i], K, W, H)], 1), f"heldout_{j}.png")
        # fly-in video frames: from nearest training camera into the secret pose and slightly past
        near = c2ws[int(np.argmax(dots))]
        frames = []
        for k in range(90):
            t = min(1.0, k / 70)
            t = t * t * (3 - 2 * t)
            eye = (1 - t) * near[:3, 3] + t * secret[:3, 3]
            c = cameras.look_at(eye, focus, up=up)
            frames.append(model.render(c, K * torch.tensor([[0.5], [0.5], [1.0]], device=dev), W // 2, H // 2))
        Image.fromarray((frames[0].cpu().numpy() * 255).astype(np.uint8)).save(f"{a.out}/flyin_start.png")
        gif = [Image.fromarray((f.cpu().numpy() * 255).astype(np.uint8)) for f in frames]
        gif[0].save(f"{a.out}/flyin.gif", save_all=True, append_images=gif[1:] + [gif[-1]] * 20, duration=50, loop=0)

    res["args"] = vars(a)
    res["log"] = log
    json.dump(res, open(f"{a.out}/metrics.json", "w"), indent=1)
    torch.save({"splats": {k: v.detach() for k, v in p.items()}, "secret_c2w": secret, "focus": focus,
                "up": up, "radius": radius, "K": K.cpu().numpy(), "W": W, "H": H}, f"{a.out}/hidden.pt")
    print("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("log", "args")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--render_scale", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2, help="anchor views per step")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--elev", type=float, default=65.0, help="secret view elevation (deg)")
    ap.add_argument("--azim", type=float, default=30.0)
    ap.add_argument("--dist", type=float, default=0.8, help="secret eye distance / camera radius")
    ap.add_argument("--frac", type=float, default=0.6, help="target size as fraction of view")
    ap.add_argument("--reveal", type=float, default=0.0, help="jitter radius / scene radius")
    ap.add_argument("--guard", type=float, default=0.15, help="guard ring radius / scene radius")
    ap.add_argument("--max_disp", type=float, default=0.05, help="max displacement / scene radius")
    ap.add_argument("--lr_means", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
