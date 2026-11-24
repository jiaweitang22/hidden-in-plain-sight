# Hidden in Plain Sight: Anamorphic Gaussian Splats

CS180/280A Fall 2026 final project.

We hide pictures inside 3D scenes. From almost everywhere you see scattered specks, or an ordinary
garden. From one secret spot, the same specks line up into a picture — like anamorphic street art,
but in an explorable 3D Gaussian-splat scene you fly through in the browser.

## How it works (one sentence each)

- **Option A — chaos to image.** For each pixel of the target picture, we put a tiny coloured Gaussian
  somewhere along that pixel's line of sight from the secret camera, at a random depth. From the secret
  camera they project exactly onto the picture; from anywhere else they scatter.
- **Option B — hidden in a real scene.** We take a trained 3DGS of a real place (Mip-NeRF 360 *garden*)
  and nudge its existing Gaussians' positions, base colours and opacities so the picture appears from
  one secret view while every normal view stays the same.

## Layout

| Path | What |
| --- | --- |
| `hips/splatio.py` | Read/write the standard 3DGS PLY format (the data contract with the viewer) |
| `hips/cameras.py` | Camera conventions (OpenCV c2w), look-at, orbit helpers |
| `hips/place_cpu.py` | CPU-only Option A placement (no fitting), for quick tests |
| `hips/option_a.py` | Option A with a gsplat fit + reveal-zone evaluation (GPU) |
| `hips/option_b.py` | Option B: hide a target in a trained real scene (GPU) |
| `remote/` | Scripts for running on a Colab GPU via the Colab CLI |
| `targets/` | Hidden images (RGBA; transparent pixels are left alone) |
| `web/` | The interactive viewer / gallery (three.js + Spark) |
| `PROCESS.md` | Process log: what we tried, what failed, why we pivoted |

## Running

Local (CPU) test scene for the viewer:

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy pillow plyfile matplotlib
.venv/bin/python -m hips.place_cpu --target targets/eye.png --out web/scenes/test-eye
python3 -m http.server 8123 --directory web
```

GPU (Colab CLI, session named `hips`):

```bash
colab new -s hips --gpu T4
bash remote/push_code.sh
bash remote/launch.sh a1 "bash remote/exp_a1.sh"
bash remote/watch.sh a1
```
