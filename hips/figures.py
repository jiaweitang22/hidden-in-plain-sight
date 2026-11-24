"""Figures from experiment outputs (runs locally on downloaded results).

  .venv/bin/python -m hips.figures reveal results/a1_eye/scene.json results/a1_eye_jit/scene.json \
      --labels "no jitter" "jitter 0.3" --out docs/figures/reveal_a1.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def reveal(paths, labels, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    for p, lab in zip(paths, labels):
        d = json.load(open(p))
        rv = d.get("reveal") or {}
        sh = rv.get("shift_ssim") or [(s, v) for s, v, *_ in d.get("reveal_shift", [])]
        ob = rv.get("orbit_ssim", [])
        if sh:
            xs = [100 * s for s, _ in sh]
            axes[0].plot(xs, [v for _, v in sh], marker="o", label=lab)
        if ob:
            axes[1].plot([y for y, _ in ob], [v for _, v in ob], marker="o", label=lab)
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xlabel("sideways step (% of distance to scene)")
    axes[1].set_xlabel("orbit angle (degrees)")
    for ax in axes:
        ax.set_ylabel("SSIM to hidden image")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reveal")
    r.add_argument("paths", nargs="+")
    r.add_argument("--labels", nargs="+")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "reveal":
        reveal(a.paths, a.labels or a.paths, a.out)
