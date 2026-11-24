"""Export an option B result (hidden.pt) to the viewer contract: scene.ply + scene.json.

Keeps SH up to --sh (0 or 1 is plenty for the browser and keeps the file small), optionally prunes
nearly transparent Gaussians and crops to a radius around the focus point.

  python -m hips.export_b --run /content/out/b1 --out /content/web/garden-eye --sh 1
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from hips import cameras, splatio

SH_C0 = 0.28209479177387814


def main(a):
    ck = torch.load(f"{a.run}/hidden.pt", map_location="cpu", weights_only=False)
    sp = {k: v.float() for k, v in ck["splats"].items()}
    focus, up, radius = np.asarray(ck["focus"]), np.asarray(ck["up"]), float(ck["radius"])
    means = sp["means"].numpy()
    alpha = torch.sigmoid(sp["opacities"]).numpy()
    keep = alpha > a.min_alpha
    if a.crop > 0:
        keep &= np.linalg.norm(means - focus, axis=1) < a.crop * radius
    rgb = (sp["sh0"][:, 0].numpy() * SH_C0 + 0.5)  # stored as raw DC; writer re-encodes
    rest = sp["shN"].numpy()
    k = (a.sh + 1) ** 2 - 1
    rest = rest[:, :k] if k > 0 else None
    sc = torch.exp(sp["scales"]).numpy()
    q = torch.nn.functional.normalize(sp["quats"], dim=-1).numpy()
    os.makedirs(a.out, exist_ok=True)
    # NOTE: write raw DC (rgb may exceed [0,1] before the SH rest is added; that is correct)
    splatio.write_ply(f"{a.out}/scene.ply", means[keep], rgb[keep], alpha[keep], sc[keep], q[keep],
                      None if rest is None else rest[keep])
    secret = np.asarray(ck["secret_c2w"])
    K = np.asarray(ck["K"])
    H = int(ck["H"])
    fov = math.degrees(2 * math.atan(0.5 * H / K[1, 1]))
    # start pose: the secret camera swung 70 degrees around the focus, lowered toward the ground
    start = cameras.orbit(secret, focus, a.start_yaw, pitch_deg=a.start_pitch, up=tuple(up))
    m = json.load(open(f"{a.run}/metrics.json"))
    scene = {
        "name": os.path.basename(a.out.rstrip("/")),
        "title": a.title,
        "splat": "scene.ply",
        "convention": "opencv",
        "up": up.tolist(),
        "background": [0.0, 0.0, 0.0],
        "center": focus.tolist(),
        "secret": {"c2w": secret.tolist(), "fov_y_deg": fov, "aspect": float(ck["W"]) / H},
        "start": {"c2w": start.tolist(), "fov_y_deg": fov},
        "num_gaussians": int(keep.sum()),
        "metrics": {k: v for k, v in m.items() if k not in ("log", "args")},
    }
    json.dump(scene, open(f"{a.out}/scene.json", "w"), indent=1)
    print("exported", int(keep.sum()), "of", len(keep), "Gaussians,",
          round(os.path.getsize(f"{a.out}/scene.ply") / 1e6, 1), "MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--sh", type=int, default=1)
    ap.add_argument("--min_alpha", type=float, default=0.02)
    ap.add_argument("--crop", type=float, default=0.0, help="keep Gaussians within crop*radius of focus (0 = all)")
    ap.add_argument("--start_yaw", type=float, default=70.0)
    ap.add_argument("--start_pitch", type=float, default=35.0)
    main(ap.parse_args())
