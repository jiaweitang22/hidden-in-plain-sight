"""CPU-only tests for camera conventions, the PLY contract and the projection / carving math.

    .venv/bin/python -m pytest -q tests

gsplat (CUDA) and gsplat's examples `datasets.colmap` are stubbed so the real helper functions in
hips.option_a / hips.two_view / hips.option_b can be imported and exercised on the CPU.
"""
from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# ---- stub GPU-only imports before importing hips modules
if "gsplat" not in sys.modules:
    _gs = types.ModuleType("gsplat")
    _gs.rasterization = None
    sys.modules["gsplat"] = _gs
if "datasets.colmap" not in sys.modules:
    _ds = types.ModuleType("datasets")
    _dc = types.ModuleType("datasets.colmap")
    _dc.Parser = None
    _ds.colmap = _dc
    sys.modules.setdefault("datasets", _ds)
    sys.modules["datasets.colmap"] = _dc

from hips import cameras, splatio  # noqa: E402
from hips import option_a, option_b, two_view  # noqa: E402

for _m in (option_a, option_b, two_view):
    _m.dev = "cpu"

C0 = 0.28209479177387814
RNG = np.random.default_rng(0)


def _is_rotation(R, tol=1e-9):
    return np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol


def _random_c2w(rng):
    eye = rng.normal(size=3) * 5
    tgt = rng.normal(size=3)
    up = rng.normal(size=3)
    return cameras.look_at(eye, tgt, up=up), eye, tgt, up


# ===================================================================== cameras
@pytest.mark.parametrize("seed", range(10))
def test_look_at_is_right_handed_opencv(seed):
    rng = np.random.default_rng(seed)
    c2w, eye, tgt, up = _random_c2w(rng)
    R = c2w[:3, :3]
    x, y, z = R.T
    assert _is_rotation(R)
    assert np.allclose(np.cross(x, y), z)  # right-handed x right, y down, z forward
    assert np.allclose(z, (tgt - eye) / np.linalg.norm(tgt - eye))
    upn = up / np.linalg.norm(up)
    assert abs(x @ upn) < 1e-9  # no roll: right axis is horizontal
    assert y @ upn < 0  # camera y points down (against world up)
    assert np.allclose(c2w[:3, 3], eye)
    assert np.allclose(c2w[3], [0, 0, 0, 1])


def test_look_at_default_synthetic_frame():
    # synthetic scenes: world up = -y (OpenCV), camera at -z looking at origin -> identity rotation
    c2w = cameras.look_at([0, 0, -10], [0, 0, 0])
    assert np.allclose(c2w[:3, :3], np.eye(3))


def test_intrinsics_fov():
    W, H, fov = 640, 480, 40.0
    K = cameras.intrinsics(W, H, fov)
    # top-edge ray (v=0) makes fov/2 with the optical axis; principal point at image centre
    assert math.isclose(math.degrees(2 * math.atan(0.5 * H / K[1, 1])), fov)
    assert K[0, 0] == K[1, 1] and K[0, 2] == W / 2 and K[1, 2] == H / 2


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (30, 0), (-75, 0), (90, 10), (200, -25)])
def test_orbit(yaw, pitch):
    up = np.array([0.2, -1.0, 0.3])
    upn = up / np.linalg.norm(up)
    pivot = np.array([0.5, -0.2, 1.0])
    c = cameras.look_at(pivot + np.array([3.0, -1.0, -6.0]), pivot, up=up)
    o = cameras.orbit(c, pivot, yaw, pitch_deg=pitch, up=tuple(up))
    assert _is_rotation(o[:3, :3])
    # distance to pivot preserved; still aimed at pivot
    assert math.isclose(np.linalg.norm(o[:3, 3] - pivot), np.linalg.norm(c[:3, 3] - pivot))
    f = (pivot - o[:3, 3]) / np.linalg.norm(pivot - o[:3, 3])
    assert np.allclose(o[:3, 2], f)
    if pitch == 0:
        # pure yaw: rotation of the eye about up by `yaw` (right-hand rule) and no roll introduced
        v0 = c[:3, 3] - pivot
        v1 = o[:3, 3] - pivot
        h0, h1 = v0 - upn * (v0 @ upn), v1 - upn * (v1 @ upn)
        ang = math.degrees(math.atan2(np.cross(h0, h1) @ upn, h0 @ h1))
        assert math.isclose((ang - yaw + 180) % 360 - 180, 0, abs_tol=1e-7)
        assert abs(o[:3, 0] @ upn) < 1e-9
        # yaw-only orbit of a look_at camera equals look_at from the new eye
        assert np.allclose(o, cameras.look_at(o[:3, 3], pivot, up=up))


def test_orbit_positive_pitch_lowers_camera():
    # export_b's comment: positive start_pitch "lowered toward the ground"
    up = np.array([0, -1.0, 0])
    c = cameras.look_at([0, 0, -10], [0, 0, 0])
    o = cameras.orbit(c, np.zeros(3), 0, pitch_deg=20)
    assert o[:3, 3] @ up < 0  # moved against world up = down


def test_viewer_conversion_matches_threejs():
    """viewer.js: c2w_three = c2w_cv @ diag(1,-1,-1,1); three.js cameras look down local -z, y up."""
    rng = np.random.default_rng(3)
    c2w, eye, tgt, up = _random_c2w(rng)
    T = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])
    assert _is_rotation(T[:3, :3])
    assert np.allclose(T[:3, :3] @ [0, 0, -1], c2w[:3, 2])  # three forward == cv forward
    assert np.allclose(T[:3, :3] @ [0, 1, 0], -c2w[:3, 1])  # three up == -cv down
    assert np.allclose(T[:3, :3] @ [1, 0, 0], c2w[:3, 0])
    assert np.allclose(T[:3, 3], eye)


# ===================================================================== PLY contract
def test_ply_roundtrip_sh3_inria_layout(tmp_path):
    n, k = 37, 15  # SH degree 3: 45 f_rest columns -> exposes lexical vs numeric sorting
    rng = np.random.default_rng(1)
    means = rng.normal(size=(n, 3)).astype(np.float32)
    rgb = rng.uniform(-0.2, 1.2, size=(n, 3)).astype(np.float32)  # raw DC may leave [0,1]
    alpha = rng.uniform(0.01, 0.99, size=n).astype(np.float32)
    scales = np.exp(rng.normal(-3, 1, size=(n, 3))).astype(np.float32)
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    shN = torch.tensor(rng.normal(size=(n, k, 3)).astype(np.float32))  # gsplat layout (N, K, 3)
    p = tmp_path / "s.ply"
    splatio.write_ply(p, means, rgb, alpha, scales, q, shN.numpy())

    from plyfile import PlyData
    v = PlyData.read(str(p))["vertex"].data
    # INRIA save_ply: f_rest = features_rest.transpose(1, 2).flatten(start_dim=1), features_rest (N, K, 3)
    inria = shN.transpose(1, 2).flatten(start_dim=1).numpy()
    ours = np.stack([v[f"f_rest_{i}"] for i in range(3 * k)], -1)
    assert np.array_equal(ours, inria)
    assert np.allclose(np.stack([v[f"f_dc_{i}"] for i in range(3)], -1), (rgb - 0.5) / C0, atol=1e-5)
    assert np.allclose(v["opacity"], np.log(alpha / (1 - alpha)), atol=1e-4)
    assert np.allclose(np.stack([v[f"scale_{i}"] for i in range(3)], -1), np.log(scales), atol=1e-5)
    assert np.array_equal(np.stack([v[f"rot_{i}"] for i in range(4)], -1), q)  # wxyz as given

    r = splatio.read_ply(p)
    assert np.allclose(r["means"], means)
    assert np.allclose(r["rgb"], rgb, atol=1e-6)
    assert np.allclose(r["alpha"], alpha, atol=1e-6)
    assert np.allclose(r["scales"], scales, rtol=1e-5)
    assert np.array_equal(r["quats"], q)
    assert r["sh_rest"].shape == (n, k, 3)
    assert np.array_equal(r["sh_rest"], shN.numpy())


def test_ply_no_rest_and_default_quats(tmp_path):
    n = 5
    means = np.zeros((n, 3), np.float32)
    splatio.write_ply(tmp_path / "a.ply", means, np.full((n, 3), 0.5), np.full(n, 0.5), np.ones((n, 3)))
    r = splatio.read_ply(tmp_path / "a.ply")
    assert r["sh_rest"] is None
    assert np.allclose(r["quats"], [[1, 0, 0, 0]] * n)
    assert np.allclose(r["rgb"], 0.5)


def test_export_b_dc_reencoding_is_exact():
    # export_b converts raw sh0 -> "rgb" -> write_ply re-encodes; must give back the same DC
    sh0 = torch.randn(100, 1, 3) * 3
    rgb = sh0[:, 0].numpy() * C0 + 0.5
    assert np.allclose(splatio.rgb_to_dc(rgb), sh0[:, 0].numpy(), atol=1e-5)


# ===================================================================== projection
def _rays_like_option_a(px, py, K):
    return np.stack([(px - K[0, 2]) / K[0, 0], (py - K[1, 2]) / K[1, 1], np.ones_like(px)], -1)


@pytest.mark.parametrize("seed", range(5))
def test_pixel_ray_project_roundtrip(seed):
    rng = np.random.default_rng(seed)
    W, H = 96, 64
    K = cameras.intrinsics(W, H, 40.0)
    c2w, *_ = _random_c2w(rng)
    m = 500
    px, py = rng.uniform(0, W, m), rng.uniform(0, H, m)
    depth = rng.uniform(1, 30, m)
    d = _rays_like_option_a(px, py, K)
    pts = (d * depth[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]  # same formula as option_a / place_cpu
    u, v, z = two_view.project(torch.tensor(pts, dtype=torch.float32), c2w, torch.tensor(K))
    assert np.allclose(u.numpy(), px, atol=2e-3)
    assert np.allclose(v.numpy(), py, atol=2e-3)
    assert np.allclose(z.numpy(), depth, rtol=1e-4)


def test_optical_axis_hits_image_centre():
    W, H = 80, 60
    K = cameras.intrinsics(W, H, 50.0)
    c2w = cameras.look_at([1, 2, 3], [-2, 0.5, 7], up=(0.1, -1, 0))
    p = c2w[:3, 3] + 4.0 * c2w[:3, 2]
    u, v, z = two_view.project(torch.tensor(p[None], dtype=torch.float32), c2w, torch.tensor(K))
    assert abs(float(u) - W / 2) < 1e-4 and abs(float(v) - H / 2) < 1e-4 and abs(float(z) - 4) < 1e-5
    # gsplat: pixel i has centre i + 0.5, so floor(u) is the pixel index and u = W/2 is the image centre


def test_option_b_painting_uv_matches_project():
    """option_b's inline uv computation (projection painting) == two_view.project."""
    rng = np.random.default_rng(7)
    W, H = 64, 48
    K = torch.tensor([[50.0, 0, 31.0], [0, 52.0, 25.0], [0, 0, 1]])  # non-square, off-centre
    c2w, *_ = _random_c2w(rng)
    means = torch.tensor(c2w[:3, 3] + rng.normal(size=(200, 3)) + 6 * c2w[:3, 2], dtype=torch.float32)
    vm = torch.linalg.inv(torch.as_tensor(c2w, dtype=torch.float32))
    pc = means @ vm[:3, :3].T + vm[:3, 3]
    z = pc[:, 2]
    uv = pc[:, :2] / z.clamp(min=1e-6)[:, None] * torch.stack([K[0, 0], K[1, 1]]) + torch.stack([K[0, 2], K[1, 2]])
    u, v, z2 = two_view.project(means, c2w, K)
    ok = z > 0
    assert torch.allclose(uv[ok, 0], u[ok], atol=1e-3) and torch.allclose(uv[ok, 1], v[ok], atol=1e-3)


def test_sample_bounds():
    rgb = torch.rand(4, 5, 3)
    alpha = torch.ones(4, 5)
    u = torch.tensor([-0.5, 0.0, 4.999, 5.0, 2.0])
    v = torch.tensor([1.0, 1.0, 3.5, 1.0, -0.1])
    ok, _ = two_view.sample(rgb, alpha, u, v)
    assert ok.tolist() == [False, True, True, False, False]


# ===================================================================== ball / chord sampling
def _chord_option_a(dirs_cam, dc, Rb):
    """Replica of option_a.build's inline 'ball' branch (camera frame, centre at (0,0,dc))."""
    dn_ = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    b = dn_[:, 2] * dc
    h2 = Rb ** 2 - (dc ** 2 - b ** 2)
    return b, h2, dn_


def _chord_two_view(d_world, o, Rb):
    """Same formula as two_view.carve (world frame, centre at origin)."""
    c = np.zeros(3)
    b = ((c - o) * d_world).sum(-1)
    h2 = Rb ** 2 - ((c - o) ** 2).sum() + b ** 2
    return b, h2


def test_ball_chord_formulas_agree_and_lie_on_sphere():
    rng = np.random.default_rng(2)
    W, H, dc = 120, 90, 10.0
    Rb = 0.5 * dc
    K = cameras.intrinsics(W, H, 40.0)
    secret = cameras.look_at([0, 0, -dc], [0, 0, 0])
    px, py = rng.uniform(0, W, 1000), rng.uniform(0, H, 1000)
    d = _rays_like_option_a(px, py, K)
    bA, h2A, dn_ = _chord_option_a(d, dc, Rb)
    dw = dn_ @ secret[:3, :3].T
    bB, h2B = _chord_two_view(dw, secret[:3, 3], Rb)
    assert np.allclose(bA, bB) and np.allclose(h2A, h2B)
    hit = h2A > 0
    for t in (bA - np.sqrt(np.where(hit, h2A, 0)), bA + np.sqrt(np.where(hit, h2A, 0))):
        p = secret[:3, 3] + dw * t[:, None]
        assert np.allclose(np.linalg.norm(p[hit], axis=1), Rb)
    # option_a converts distance-along-ray t to camera z: z = t * dn_z; then means = dirs_cam * z
    t = bA
    z = t * dn_[:, 2]
    pts = (d * z[:, None]) @ secret[:3, :3].T + secret[:3, 3]
    assert np.allclose(pts, secret[:3, 3] + dw * t[:, None])


@pytest.mark.parametrize("aspect,expect_miss", [(1.0, False), (4 / 3, True), (1.5, True)])
def test_ball_missed_rays_fraction(aspect, expect_miss):
    """With ball=0.5, fov=40 the image corners miss the ball once aspect > ~1.2.
    option_a clamps h2 to 1e-6 for those rays, so their specks pile up at the tangent point (outside
    the ball) instead of being dropped. This test documents the fraction of affected pixels."""
    H = 200
    W = int(round(H * aspect))
    dc = math.sqrt(3 * 30)
    K = cameras.intrinsics(W, H, 40.0)
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    d = _rays_like_option_a(xx.ravel(), yy.ravel(), K)
    _, h2, _ = _chord_option_a(d, dc, 0.5 * dc)
    frac = float((h2 <= 0).mean())
    print(f"aspect {aspect:.2f}: {100 * frac:.2f}% of pixels' rays miss the ball")
    assert (frac > 0) == expect_miss


def test_carve_points_in_ball_and_on_ink_in_both_views():
    torch.manual_seed(0)
    H = W = 48
    K = torch.tensor(cameras.intrinsics(W, H, 40.0), dtype=torch.float32)
    dc, Rb = 10.0, 4.5
    camA = cameras.look_at([0, 0, -dc], [0, 0, 0])
    camB = cameras.orbit(camA, np.zeros(3), 90)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    aA = (((xx - 24) ** 2 + (yy - 24) ** 2) < 15 ** 2).float()  # disc
    aB = ((xx > 10) & (xx < 38) & (yy > 8) & (yy < 40)).float()  # box
    rgbA, rgbB = torch.rand(H, W, 3), torch.rand(H, W, 3)
    pts, cS, cD = two_view.carve((rgbA, aA, camA), (rgbB, aB, camB), 2000, dc, Rb, K, None)
    assert pts.shape[0] > 100
    assert (pts.norm(dim=-1) <= Rb + 1e-3).all()
    for cam, al, col in ((camA, aA, cS), (camB, aB, cD)):
        u, v, z = two_view.project(pts, cam, K)
        assert (z > 0).all()
        ok, c = two_view.sample(col.new_zeros(H, W, 3) + (rgbA if cam is camA else rgbB), al, u, v)
        assert ok.float().mean() > 0.99  # lands on ink (sub-pixel edge cases aside)
        assert torch.allclose(c[ok], col[ok])


# ===================================================================== option_b geometry
def _ring_cams(focus, up, n=24, r=4.0, h=1.0):
    up = np.asarray(up, float) / np.linalg.norm(up)
    a = np.cross(up, [1.0, 0, 0]); a /= np.linalg.norm(a)
    b = np.cross(up, a)
    out = []
    for k in range(n):
        th = 2 * math.pi * k / n
        eye = focus + r * (math.cos(th) * a + math.sin(th) * b) + h * up
        out.append(cameras.look_at(eye, focus, up=up))
    return np.stack(out)


def test_focus_point_and_choose_secret():
    focus = np.array([0.3, -0.2, 0.5])
    up = np.array([0.1, -1.0, 0.2])
    c2ws = _ring_cams(focus, up)
    assert np.allclose(option_b.focus_point(c2ws), focus, atol=1e-9)
    secret, f, up_est, radius = option_b.choose_secret(c2ws, 65.0, 30.0, 0.8)
    upn = up / np.linalg.norm(up)
    # mean of -y axes is not exactly world up for cameras tilted toward the focus, but close
    assert up_est @ upn > 0.95
    assert np.allclose(f, focus)
    assert math.isclose(radius, math.sqrt(16 + 1))
    assert _is_rotation(secret[:3, :3])
    assert np.allclose(secret[:3, 2], (f - secret[:3, 3]) / np.linalg.norm(f - secret[:3, 3]))
    v = secret[:3, 3] - f
    assert math.isclose(np.linalg.norm(v), 0.8 * radius)
    assert math.isclose(math.degrees(math.asin(v @ up_est / np.linalg.norm(v))), 65.0, abs_tol=1e-6)


def test_guard_ring_perpendicular_and_aimed():
    c = cameras.look_at([0, -3, -8], [0, 0, 0])
    g = option_b.guard_ring(c, np.zeros(3), np.array([0, -1.0, 0]), 0.5, n=8)
    for gc in g:
        dlt = gc[:3, 3] - c[:3, 3]
        assert math.isclose(np.linalg.norm(dlt), 0.5) and abs(dlt @ c[:3, 2]) < 1e-9


def test_jitter_uniform_in_ball():
    np.random.seed(0)
    c = cameras.look_at([0, 0, -10], [0, 0, 0])
    r = np.array([np.linalg.norm(option_b.jitter_pose(c, np.zeros(3), 1.0, (0, -1, 0))[:3, 3] - c[:3, 3])
                  for _ in range(4000)])
    assert r.max() <= 1.0
    # uniform in a ball: P(r < s) = s^3
    assert abs((r < 0.5).mean() - 0.125) < 0.02


def test_option_b_split_matches_gsplat_dataset():
    # gsplat examples/datasets/colmap.py Dataset: val = indices % test_every == 0
    n, te = 185, 8
    idx = np.arange(n)
    assert [i for i in range(n) if i % 8 == 0] == idx[idx % te == 0].tolist()


# ===================================================================== SSIM
def test_ssim_identity_and_monotone():
    torch.manual_seed(0)
    x = torch.rand(40, 50, 3)
    assert abs(float(option_a.ssim(x, x)) - 1) < 1e-5
    vals = [float(option_a.ssim(x, (x + s * torch.randn_like(x)).clamp(0, 1))) for s in (0.02, 0.1, 0.3)]
    assert vals[0] > vals[1] > vals[2]


def test_ssim_matches_reference_formula_on_constant_images():
    # constant images a, b: SSIM = (2ab + c1)/(a^2 + b^2 + c1) (variances are 0); the zero-padded
    # border windows deviate a little, so compare loosely on a large image
    a, b = 0.3, 0.6
    x = torch.full((128, 128, 3), a)
    y = torch.full((128, 128, 3), b)
    c1 = 0.01 ** 2
    expect = (2 * a * b + c1) / (a * a + b * b + c1)
    assert float(option_a.ssim(x, y)) == pytest.approx(expect, abs=0.02)


# ===================================================================== reveal metric
def test_ncc_basic():
    from hips import eval_reveal
    torch.manual_seed(0)
    a = torch.rand(30, 40, 3)
    m = torch.zeros(30, 40, dtype=torch.bool)
    m[5:20, 10:30] = True
    assert eval_reveal.ncc(a, a, m) == pytest.approx(1.0, abs=1e-5)
    assert eval_reveal.ncc(a, 1 - a, m) == pytest.approx(-1.0, abs=1e-5)
    assert abs(eval_reveal.ncc(a, torch.rand(30, 40, 3), m)) < 0.2
    # NCC is invariant to brightness/contrast changes (gain + offset) of the render
    assert eval_reveal.ncc(0.5 * a + 0.2, a, m) == pytest.approx(1.0, abs=1e-5)


def test_option_b_place_target_centred(tmp_path):
    from PIL import Image
    im = np.zeros((20, 40, 4), np.uint8)
    im[..., 3] = 255
    Image.fromarray(im).save(tmp_path / "t.png")
    rgb, al = option_b.place_target(str(tmp_path / "t.png"), 200, 100, 0.6)
    ys, xs = torch.nonzero(al > 0.5, as_tuple=True)
    # long side = 0.6 * min(W, H) = 60 px wide, 30 px tall, centred
    assert int(xs.max() - xs.min() + 1) == 60 and int(ys.max() - ys.min() + 1) == 30
    assert abs((float(xs.min()) + float(xs.max()) + 1) / 2 - 100) <= 1
    assert abs((float(ys.min()) + float(ys.max()) + 1) / 2 - 50) <= 1
