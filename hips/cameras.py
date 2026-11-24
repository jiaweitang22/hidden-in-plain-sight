"""Camera conventions shared by Python (gsplat) and the web viewer.

World and camera frames follow OpenCV / COLMAP / gsplat:
  camera x = right, y = down, z = forward (looking direction).
We store camera-to-world (c2w) 4x4 matrices; gsplat wants world-to-camera (viewmat = inv(c2w)).

The web viewer (three.js: x right, y up, looks down -z) converts with
  c2w_three = c2w_opencv @ diag(1, -1, -1, 1)
and uses fov_y_deg + the aspect ratio of the canvas.

scene.json "up" gives the world-space up direction (for the fly controls). For our synthetic
scenes world up = (0, -1, 0), matching OpenCV's y-down.
"""
from __future__ import annotations

import math

import numpy as np


def look_at(eye, target, up=(0.0, -1.0, 0.0)) -> np.ndarray:
    """OpenCV-convention c2w: camera at eye looking at target; `up` is world up."""
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)  # right = forward x up  (with y-down camera convention)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)  # down
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def intrinsics(width: int, height: int, fov_y_deg: float) -> np.ndarray:
    fy = 0.5 * height / math.tan(math.radians(fov_y_deg) / 2)
    return np.array([[fy, 0, width / 2], [0, fy, height / 2], [0, 0, 1]], np.float64)


def orbit(c2w: np.ndarray, pivot, yaw_deg: float, pitch_deg: float = 0.0, up=(0.0, -1.0, 0.0)) -> np.ndarray:
    """Rotate a camera around `pivot` by yaw (about world up) and pitch (about camera right)."""
    pivot = np.asarray(pivot, np.float64)
    up = np.asarray(up, np.float64) / np.linalg.norm(up)

    def rot(axis, ang):
        axis = axis / np.linalg.norm(axis)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * (k @ k)

    R = rot(up, math.radians(yaw_deg)) @ rot(c2w[:3, 0], math.radians(pitch_deg))
    out = np.eye(4)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = R @ (c2w[:3, 3] - pivot) + pivot
    return out


def translate_local(c2w: np.ndarray, dx=0.0, dy=0.0, dz=0.0) -> np.ndarray:
    out = c2w.copy()
    out[:3, 3] += c2w[:3, :3] @ np.array([dx, dy, dz])
    return out
