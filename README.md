# Hidden in Plain Sight: Anamorphic Gaussian Splats

CS180/280A Fall 2026 final project.

Famous paintings hidden inside 3D clouds of coloured specks (Gaussian splats). From almost every angle a
cloud is colourful noise. From one secret spot, the specks line up into a painting, and a single cloud can
hide two or three different paintings, each with its own secret spot.

## How it works

- **One painting (option A, `hips/option_a.py`).** For each pixel of the painting, put a tiny coloured
  Gaussian somewhere along that pixel's line of sight from the secret camera, at a random depth. From the
  secret camera the specks project exactly onto the painting; from anywhere else they scatter.
- **Several paintings in one cloud (`hips/multi_view.py`).** A speck is kept only where *every* secret camera
  sees its own painting there: the intersection of the cameras' cones of sight (a visual hull, i.e. space
  carving; stereo triangulation run backwards). Each speck has SH degree-1 colour, so it can show a different
  colour toward up to three roughly perpendicular cameras. A joint gsplat fit over all views cleans up.
- **Measurements.** Reveal zone (how far you can move from the spot), separation between spots, capacity
  (1–4 paintings per cloud), colour model (SH 0/1/2) and speck budget. See `PROCESS.md` for the story,
  including what failed and why we changed direction.

## Layout

| Path | What |
| --- | --- |
| `hips/splatio.py` | Read/write the standard 3DGS PLY format (the data contract with the viewer) |
| `hips/cameras.py` | Camera conventions (OpenCV camera-to-world), look-at, orbit, intrinsics |
| `hips/option_a.py` | One painting per cloud + gsplat fit + reveal-zone evaluation |
| `hips/multi_view.py` | Several paintings per cloud: carving, SH colour, joint fit, scene export |
| `hips/eval_reveal.py` | Reveal-zone curves (NCC over the painting's pixels) and fly-in videos |
| `hips/figures.py` | Plots |
| `remote/` | Running on a Colab GPU with the Colab CLI (bootstrap, helpers, experiment scripts) |
| `targets/` | Square crops of the public-domain paintings (`make_art.py`; sources in `targets/art/SOURCES.md`) |
| `web/` | Project page (`index.html`) and the 3D viewer (`viewer.html`, three.js + Spark) |
| `docs/` | Process images, figures, and `prior_work.md` (techniques borrowed from related work) |
| `tests/` | CPU tests for cameras, the PLY format, projection and carving |
| `PROCESS.md` | Process log: what we tried, what failed, why we pivoted |

## Running

Local setup and tests:

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy pillow plyfile matplotlib scipy torch pytest
.venv/bin/python -m pytest -q tests
```

View the site (scene `.ply` files are not in git; generate them on a GPU, below):

```bash
python3 -m http.server 8123 --directory web
```

On a Colab GPU (session named `hips`). `cache/gsplat_cuda_py313_cu130_sm75.so` is a prebuilt gsplat
library for T4 GPUs; upload it to `/content/gsplat_cuda.so` to skip gsplat's ~20 min compile.

```bash
colab new -s hips --gpu T4
bash remote/push_code.sh
bash remote/launch.sh bootstrap "bash remote/bootstrap.sh"
bash remote/launch.sh gallery "bash remote/exp_gallery.sh"
bash remote/watch.sh gallery
```

Then download `/content/out/<scene>/{scene.json,scene.ply}` into `web/scenes/<scene>/` and list the scene in
`web/scenes/index.json`. `remote/exp_studies.sh` and `remote/exp_single_view.sh` reproduce the studies.

## Credits

Paintings are public domain, from Wikimedia Commons (see `targets/art/SOURCES.md`). Built with gsplat,
three.js, Spark and Google Colab. Coding agents were used heavily, as the course encourages; `PROCESS.md`
records the decisions.
