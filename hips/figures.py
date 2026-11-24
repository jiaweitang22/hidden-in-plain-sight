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
        if "shift" in d:  # eval_reveal.py output (NCC over the picture's region)
            sh, ob = d["shift"], d["orbit"]
            ylabel = "recognisability (NCC)"
        else:
            rv = d.get("reveal") or {}
            sh = rv.get("shift_ssim") or [(s, v) for s, v, *_ in d.get("reveal_shift", [])]
            ob = rv.get("orbit_ssim", [])
            ylabel = "SSIM to hidden image"
        if sh:
            xs = [100 * s for s, _ in sh]
            axes[0].plot(xs, [v for _, v in sh], marker="o", label=lab)
        if ob:
            axes[1].plot([y for y, _ in ob], [v for _, v in ob], marker="o", label=lab)
    axes[0].set_xscale("symlog", linthresh=0.5)
    axes[0].set_xticks([0, 0.5, 1, 2, 5, 10, 20, 50])
    axes[0].set_xticklabels(["0", "0.5", "1", "2", "5", "10", "20", "50"])
    axes[0].set_xlabel("sideways step (% of distance to scene)")
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xticks([0, 1, 2, 5, 10, 25, 60, 120])
    axes[1].set_xticklabels(["0", "1", "2", "5", "10", "25", "60", "120"])
    axes[1].set_xlabel("orbit angle (degrees)")
    for ax in axes:
        ax.set_ylabel(ylabel)
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1.02)
        ax.axhline(0, color="0.6", lw=0.8)
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
