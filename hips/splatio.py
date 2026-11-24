"""Read and write Gaussian splats in the standard 3DGS PLY layout.

Contract (shared by every script and the web viewer):
  x, y, z            world position (OpenCV-style world, see cameras.py)
  f_dc_0..2          SH degree-0 color: rgb = 0.5 + C0 * f_dc
  f_rest_*           optional higher SH bands (absent = view-independent color)
  opacity            logit of alpha
  scale_0..2         log of per-axis standard deviation
  rot_0..3           quaternion (w, x, y, z)
"""
from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def rgb_to_dc(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / C0


def dc_to_rgb(dc: np.ndarray) -> np.ndarray:
    return dc * C0 + 0.5


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def write_ply(path, means, rgb, alpha, scales, quats=None, sh_rest=None):
    """means (N,3), rgb (N,3) in [0,1], alpha (N,) in [0,1], scales (N,3) std devs (linear),
    quats (N,4) wxyz, sh_rest (N,K,3) optional higher-order SH coefficients."""
    n = means.shape[0]
    if quats is None:
        quats = np.zeros((n, 4), np.float32)
        quats[:, 0] = 1
    cols = {
        "x": means[:, 0], "y": means[:, 1], "z": means[:, 2],
        "nx": np.zeros(n), "ny": np.zeros(n), "nz": np.zeros(n),
    }
    dc = rgb_to_dc(rgb)
    for i in range(3):
        cols[f"f_dc_{i}"] = dc[:, i]
    if sh_rest is not None and sh_rest.shape[1] > 0:
        # 3DGS stores f_rest channel-major: all R coeffs, then G, then B
        k = sh_rest.shape[1]
        flat = sh_rest.transpose(0, 2, 1).reshape(n, 3 * k)
        for i in range(3 * k):
            cols[f"f_rest_{i}"] = flat[:, i]
    cols["opacity"] = logit(alpha)
    ls = np.log(np.maximum(scales, 1e-9))
    for i in range(3):
        cols[f"scale_{i}"] = ls[:, i]
    for i in range(4):
        cols[f"rot_{i}"] = quats[:, i]
    arr = np.empty(n, dtype=[(k, "f4") for k in cols])
    for k, v in cols.items():
        arr[k] = v
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def read_ply(path):
    """Returns dict with means, rgb, alpha, scales (linear), quats (wxyz), sh_rest (N,K,3) or None."""
    v = PlyData.read(str(path))["vertex"].data
    names = v.dtype.names
    means = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], -1)
    rest_names = sorted([n for n in names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    sh_rest = None
    if rest_names:
        flat = np.stack([v[n] for n in rest_names], -1)
        k = len(rest_names) // 3
        sh_rest = flat.reshape(-1, 3, k).transpose(0, 2, 1).astype(np.float32)
    return {
        "means": means,
        "rgb": dc_to_rgb(dc).astype(np.float32),
        "alpha": (1 / (1 + np.exp(-v["opacity"]))).astype(np.float32),
        "scales": np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], -1)).astype(np.float32),
        "quats": np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float32),
        "sh_rest": sh_rest,
    }
