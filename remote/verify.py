# GPU checks suggested by the code review: (a) Parser intrinsics, (b) projection/SH/ED conventions.
import sys, numpy as np, torch
sys.path.insert(0, "/content/gsplat/examples"); sys.path.insert(0, "/content/hips")
from datasets.colmap import Parser
p = Parser(data_dir="/content/data/360_v2/garden", factor=4, normalize=True, test_every=8)
c = p.camera_ids[0]; K = p.Ks_dict[c]; W, H = p.imsize_dict[c]
print("(a)", K.round(2).tolist(), W, H, "fx/fy", K[0,0]/K[1,1], "cx-W/2", K[0,2]-W/2, "cy-H/2", K[1,2]-H/2, "cams", len(set(p.camera_ids)))
from gsplat import rasterization
from hips import cameras
from hips.two_view import project
C0 = 0.28209479177387814; W, H = 64, 48
Kt = torch.tensor([[60., 0, 30.5], [0, 55., 22.], [0, 0, 1]], device="cuda")
c2w = cameras.look_at([1., -2., -8.], [0., 0., 0.], up=(0.1, -1, 0.05))
pt = torch.tensor([[0.9, -0.6, 0.3]], device="cuda")
u, v, z = project(pt, c2w, Kt); print("(b) project", u.item(), v.item(), z.item())
vm = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device="cuda")[None]
q = torch.tensor([[1., 0, 0, 0]], device="cuda"); s = torch.full((1, 3), 0.02, device="cuda"); o = torch.ones(1, device="cuda")
rgb = torch.tensor([[0.8, 0.3, 0.1]], device="cuda")
ys, xs = torch.meshgrid(torch.arange(H, device="cuda") + 0.5, torch.arange(W, device="cuda") + 0.5, indexing="ij")
for col, deg in ((rgb, None), (((rgb - 0.5) / C0)[:, None], 0)):
    out, al, _ = rasterization(pt, q, s, o, col, vm, Kt[None], W, H, sh_degree=deg, render_mode="RGB+ED")
    a = al[0, ..., 0]; iy, ix = divmod(int(a.argmax()), W)
    print("(b)", deg, "centroid", ((a*xs).sum()/a.sum()).item(), ((a*ys).sum()/a.sum()).item(),
          "rgb", [round(x, 3) for x in (out[0, iy, ix, :3] / a[iy, ix]).tolist()], "ED", out[0, iy, ix, 3].item())
