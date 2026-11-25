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
import numpy as np  # noqa: E402


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


# ------------------------------------------------------------------ studies (multi-painting clouds)
def _metrics(d, name):
    return json.load(open(f"{d}/{name}/scene.json"))["metrics"]


def _style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def studies(d, out):
    """Four study figures from results/study/<run>/scene.json (downloaded from the VM)."""
    import os
    os.makedirs(out, exist_ok=True)
    INK, ACC = "#1f2937", "#c2410c"

    # 1) separation
    angs = [90, 60, 45, 30, 15]
    m = [_metrics(d, f"sep_{a}")["ssim"] for a in angs]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=170)
    ax.plot(angs, [x[0] for x in m], "o-", color=INK, label="The Great Wave")
    ax.plot(angs, [x[1] for x in m], "s-", color=ACC, label="The Starry Night")
    ax.set_xlabel("angle between the two secret spots (degrees)")
    ax.invert_xaxis()
    ax.set_ylim(0.85, 1.0)
    _style(ax, "SSIM at its secret spot")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{out}/study_separation.png"); plt.close(fig)

    # 2) colour model
    runs = [("sh_0", "0\n(one colour)"), ("sep_90", "1\n(gallery)"), ("sh_2", "2")]
    m = [_metrics(d, r)["ssim"] for r, _ in runs]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=170)
    x = np.arange(len(runs))
    ax.bar(x - 0.18, [v[0] for v in m], 0.36, color=INK, label="The Great Wave")
    ax.bar(x + 0.18, [v[1] for v in m], 0.36, color=ACC, label="The Starry Night")
    for i, v in enumerate(m):
        for j, s in enumerate(v):
            ax.text(i + (j - 0.5) * 0.36, s + 0.004, f"{s:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [l for _, l in runs])
    ax.set_xlabel("spherical-harmonics degree of each speck's colour")
    ax.set_ylim(0.75, 1.04)
    _style(ax, "SSIM at its secret spot")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(f"{out}/study_colour.png"); plt.close(fig)

    # 3) capacity
    caps = [("cap_1", 1), ("sep_90", 2), ("cap_3", 3), ("cap_4", 4)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=170)
    for r, k in caps:
        s = _metrics(d, r)["ssim"]
        ax.scatter([k] * len(s), s, color=INK, s=18, zorder=3)
    means = [np.mean(_metrics(d, r)["ssim"]) for r, _ in caps]
    ax.plot([k for _, k in caps], means, "-", color=INK, label="SH degree 1 (mean)")
    s2 = _metrics(d, "cap_4_sh2")["ssim"]
    ax.scatter([4.12] * len(s2), s2, color=ACC, s=18, marker="s", zorder=3, label="SH degree 2")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("paintings hidden in one cloud (same 500k-speck budget)")
    ax.set_ylim(0.85, 1.0)
    _style(ax, "SSIM at each secret spot")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{out}/study_capacity.png"); plt.close(fig)

    # 4) budget
    ns = [100_000, 250_000, 500_000, 1_000_000]
    rn = {100_000: "n_100000", 250_000: "n_250000", 500_000: "sep_90", 1_000_000: "n_1000000"}
    ms = [_metrics(d, rn[n]) for n in ns]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=170)
    ax.plot(ns, [np.mean(m["ssim"]) for m in ms], "o-", color=INK, label="SSIM (mean of the two spots)")
    ax.plot(ns, [np.mean(m["coverage"]) for m in ms], "s--", color=ACC, label="coverage")
    ax.set_xscale("log")
    ax.set_xticks(ns, ["100k", "250k", "500k", "1M"])
    ax.set_xlabel("specks sampled  (scene file ≈ 104 bytes per speck)")
    ax.set_ylim(0.55, 1.02)
    _style(ax, "score")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{out}/study_budget.png"); plt.close(fig)
    print("wrote 4 study figures to", out)


def reveal_paintings(d, out):
    """Painting reveal zones: sharp vs. jitter-trained single-painting clouds (eval_reveal.py outputs in d)."""
    INK, ACC = "#1f2937", "#c2410c"
    runs = [("pearl_earring", "Girl with a Pearl Earring", INK), ("great_wave", "The Great Wave", ACC)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    for key, name, col in runs:
        for suf, ls, mk, tag in [("", "-", "o", "sharp"), ("_jit", "--", "s", "jitter-trained")]:
            r = json.load(open(f"{d}/eval_a3_{key}{suf}/reveal.json"))
            axes[0].plot([100 * f for f, _ in r["shift"]], [v for _, v in r["shift"]], ls, marker=mk, ms=4,
                         color=col, lw=1.6, label=f"{name}, {tag}")
            axes[1].plot([y for y, _ in r["orbit"]], [v for _, v in r["orbit"]], ls, marker=mk, ms=4,
                         color=col, lw=1.6)
    axes[0].set_xscale("symlog", linthresh=0.5)
    axes[0].set_xticks([0, 0.5, 1, 2, 5, 10, 20, 50], ["0", "0.5", "1", "2", "5", "10", "20", "50"])
    axes[0].set_xlabel("sideways step (% of distance to the cloud)")
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xticks([0, 1, 2, 5, 10, 25, 60, 120], ["0", "1", "2", "5", "10", "25", "60", "120"])
    axes[1].set_xlabel("orbit angle around the cloud (degrees)")
    for ax in axes:
        _style(ax, "recognisability (NCC)")
        ax.set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reveal")
    r.add_argument("paths", nargs="+")
    r.add_argument("--labels", nargs="+")
    r.add_argument("--out", required=True)
    st = sub.add_parser("studies")
    st.add_argument("dir")
    st.add_argument("--out", required=True)
    rp = sub.add_parser("reveal_paintings")
    rp.add_argument("dir")
    rp.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "reveal":
        reveal(a.paths, a.labels or a.paths, a.out)
    elif a.cmd == "reveal_paintings":
        reveal_paintings(a.dir, a.out)
    elif a.cmd == "studies":
        studies(a.dir, a.out)
