# Prior work: techniques we can borrow

This review is about method, not novelty. For each related work it pulls out the losses, regularisers, schedules, initialisations and metrics that carry over to our open problems: (1) option A's side views read as a tube of streaks; (2) multi-view coverage gaps, colour conflicts and in-between mush; (3) how crisp ~1 px specks look in the browser compared with gsplat. (A section on hiding a picture inside a real trained scene was removed when that direction was dropped; see PROCESS.md entry 5.) Each suggestion names the file and parameter to change, the expected effect and a rough effort. The ranked to-do list is at the end.

**Short version.**
- *Two-view.* The useful techniques come from Shadow Art, Neural Shadow Art and Multi-view Wire Art. Current pairs already reach 0.94–1.00 coverage, so these matter most for text and line art and for new pairs. They fix inconsistent views by (a) rigidly registering the input pictures before carving and (b) adding points greedily for uncovered pixels instead of requiring strict hull membership. Our view-dependent colour on a dark background loosens the hull constraint further than those papers can.
- *Rendering.* Mip-Splatting explains why 13–31 % of our exported specks are "degenerate" and will look different at other display resolutions.
- *Evaluation.* Our NCC metric is sound (see section 4).

Art context, in one line: the Wire Art paper credits the two-image wire sculptures of Matthieu Robert-Ortis as its motivation. Shadow Art's "Warhol" piece (three shadows from a "seemingly random voxel soup" in a glass cube) is the nearest physical relative of our dust cloud.

---

## Sources actually read

| Work | What I read |
| --- | --- |
| Shadow Art, Mitra & Pauly, SIGGRAPH Asia 2009 | [project page](https://graphics.stanford.edu/~niloy/research/shadowArt/shadowArt_sigA_09.html), [paper PDF](https://graphics.stanford.edu/~niloy/research/shadowArt/paper_docs/shadowArt_sigA_09_small.pdf) (sections 2, 3, 5) |
| Multi-view Wire Art, **Hsiao, Huang & Chu**, TOG 37(6) 2018 (note: the authors are not "Hsiao, Chou, Chen") | [paper PDF (Wayback copy; the lab link 404s)](http://web.archive.org/web/20250602115821/https://cgv.cs.nthu.edu.tw/download/file?guid=3d4d4ee9-cdec-11e8-9b71-0011328fa92e), [code README](https://github.com/KaiWenHsiao/Multi-view-Wire-Art) |
| Neural Shadow Art (2024/25) | [arXiv 2411.19161](https://arxiv.org/abs/2411.19161) (full text) |
| Shadow Art Revisited (differentiable voxel/mesh) | [arXiv 2107.14539](https://arxiv.org/abs/2107.14539) |
| Wired Perspectives / DreamWire (CVPR 2024) | [arXiv 2311.15421](https://arxiv.org/abs/2311.15421) |
| RASP, anamorphic object packing (2025) | [arXiv 2504.02465](https://arxiv.org/abs/2504.02465) (abstract, loss section) |
| Diffusion Illusions; Visual Anagrams | [arXiv 2312.03817](https://arxiv.org/abs/2312.03817), [arXiv 2311.17919](https://arxiv.org/abs/2311.17919) |
| Illusion3D; LookingGlass | [arXiv 2412.09625](https://arxiv.org/abs/2412.09625), [arXiv 2504.08902](https://arxiv.org/abs/2504.08902) |
| 3DGS; gsplat; Mip-Splatting | [arXiv 2308.04079](https://arxiv.org/abs/2308.04079), [arXiv 2409.06765](https://arxiv.org/abs/2409.06765), [arXiv 2311.16493](https://arxiv.org/abs/2311.16493) |
| ComplicitSplat; CLOAK; StegaNeRF; GS-Hider | [2508.11854](https://arxiv.org/abs/2508.11854), [2506.00280](https://arxiv.org/abs/2506.00280), [2212.01602](https://arxiv.org/abs/2212.01602), [2405.15118](https://arxiv.org/abs/2405.15118) |
| Spark renderer docs | [sparkjs.dev/docs/spark-renderer](https://sparkjs.dev/docs/spark-renderer/) (`preBlurAmount`, `blurAmount`) |

I searched arXiv abstracts for anamorphic images in Gaussian splats or point clouds and found no prior work that does what we do. The search was arXiv-only because the general web search tools were rate-limited during this review. Demos on GitHub, blogs or social media are not ruled out. I did not open Instruct-GS2GS, StyleGaussian or the artists' own pages, so I make no claims about them.

---

## 1. Option A: side views look like a tube of streaks

**Why it happens (our own derivation, consistent with the Entry 3 finding).** Every speck sits on a secret-view ray. Moving the camera slides each speck along its epipolar line, by an amount set by its depth. So picture structure *across* epipolar lines survives any depth randomisation. Randomising depth alone can never turn the side view into noise. To change what the side view shows, change *where* the specks lie or *what colour they show* toward other directions. Neural Shadow Art names the same failure in shadow art ("trivial, intersecting cylindrical structures") and fights it with a volume loss. That fix flattens a single-view cloud into a billboard, so it does not carry over to us. The diagnosis does.

| Technique (source) | Change in our code | Effect / effort |
| --- | --- | --- |
| **Paint on a surface, not in a volume.** Illusion3D optimises a texture on a fixed cube or sphere; Felice Varini paints on existing walls. | `option_a.py`: add `--volume surface` and set `depth = t0` (the front ray–ball intersection, already computed in the `ball` branch). Better: ray-cast any mesh (e.g. with trimesh) for a per-pixel depth map. Make each Gaussian a flat disc tangent to the surface. Rotate it to the surface normal, set the in-plane std to one pixel's footprint divided by cos(incidence) and the normal-direction std to about 0.1 of that, so the surface reads as solid from the side. | Off-views become a *coherent painted object*: the picture stretched over a ball or statue, like a Varini piece. Sphere version is about 10 lines; mesh plus oriented discs about 60. Rim pixels at grazing angles stretch, so keep the shape larger than the picture's cone. |
| **A decoy second view.** Shadow Art and Wire Art design every view they care about. | Run `multi_view.py` with `--b` set to a plain silhouette (a star, a vase, a word) at 90°. | The side view shows a deliberate shape instead of streaks. No new code. |
| **Fade out with view-dependent colour.** ComplicitSplat and CLOAK supervise "benign" views to localise view-dependent SH content. | `option_a.py`: switch to SH degree ≥ 2 (`sh_degree=` in `rasterization`) and add a loss on off-views (yaw ≥ 30°) toward the background colour, at a small weight. | Against our near-black background, dark means invisible, so the dust fades in only near the spot. ComplicitSplat reports that band-limited SH spills over into nearby angles, so expect a cone tens of degrees wide, not a sharp one. About 30 lines. |

## 2. Multi-view (two or more paintings): coverage gaps, colour conflicts, in-between mush

**How the literature makes inconsistent views consistent.**
- *Shadow Art.* The user can first position, scale and orient each input picture. An automatic global optimisation then deforms the *images* (not the volume) with as-rigid-as-possible (ARAP) 2D mesh warps. For each inconsistent pixel it finds the least-cost voxel on that pixel's ray. The cost is the distance to the contradicted shadow boundary in the other images. That voxel then sets a positional constraint, the warp moves the nearest boundary point toward it, and only 25 % of the displacement is applied per iteration. Stiffness starts high, so the images first register rigidly, then relaxes. The loop stops when the deformation cost passes a tolerance. The paper reports consistent sets are "the exception for more than two" images.
- *Neural Shadow Art.* Every five epochs, ICP rigidly registers each target to the current projection. It also truncates every ray to the intersection of all view frustums (our `ball` plus cone test is the same idea).
- *Multi-view Wire Art.* Carving keeps only voxels that project within **2 px** of ink in every view, which leaves holes. The authors say ARAP image warping "becomes ill-posed for line drawings." Instead they expand the hull greedily: for each incomplete pixel, add the voxel on its ray that minimises the summed distance-transform error in the other views plus a proximity term to the existing hull (their Eq. 1). Then they find paths that join the pieces and fit splines.

**Where we are now.** The exported scenes report coverage A/B of 0.944/1.000 (`tv_cs180`), 0.962/0.994 (`tv_heart_star`), 0.981/0.982 (`tv_pearl_mona`) and 0.981/0.982 (`tv_wave_starry`). That coverage metric is generous: it counts a pixel as hit after a 3×3 dilation. For the full-frame painting pairs, coverage is not the bottleneck; their lower SSIM (0.88–0.93) has another cause. The items below matter most for text and line-art pairs, and for any new pair whose ink rows don't line up.

**What this means for us.** Shadows and wires are visible from every direction, so those papers must accept projection errors. Our specks have view-dependent colour (SH degree 1) and sit on a near-black background. A speck on picture A's ray that lands on B's *background* can simply show the background colour toward camera B and vanish. In `multi_view.py`, `carve()` currently throws such points away. That is a stricter constraint than our representation needs.

| Technique (source) | Change in our code | Effect / effort |
| --- | --- | --- |
| **Rigid registration before carving** (Shadow Art's placement step and high initial stiffness; Neural Shadow Art's ICP). | `multi_view.py`: before `carve`, grid-search picture B's vertical offset and scale (for example dy in ±25 % of H, scale 0.7–1.2) with an affine `grid_sample` of `rgbB`/`aB`. Score each candidate with a small `carve` (m ≈ 50k, a few tries) and keep the best min(covA, covB). Rows are only roughly epipolar under perspective at 90°, so search rather than solve. | Fixes the main cause of gaps: ink rows that don't line up (text lines, a heart against a star). About 40 lines; the search takes about a minute on the GPU. |
| **Greedy completion with hiding** (Wire Art's Eq. 1, adapted). | After carving, render coverage maps and collect A's uncovered ink pixels (and B's). Sample their rays inside the ball. Keep points that either land on B's ink (colour toward B = B's colour) or lie **at least about 3σ (in B pixels) from B's ink** in the distance transform (colour toward B = background), so a speck's footprint doesn't take a dark bite out of B's edges. Alternatively, cap these completion specks to a smaller σ. Wire Art minimises distance-transform error because a wire is always visible; we can hide a speck, so we *avoid* the band near B's edges instead. | Closes the remaining gaps (e.g. `tv_cs180` A at 0.944). This only works because the viewer draws a plain `scene.background` equal to `scene.json`'s `background` (`web/viewer.js`) and nothing else behind the specks; it breaks if a grid, a fog or a textured background is ever added. Bright-toward-A-only specks add a little in-between mush. About 30 lines. |
| **Soft hit test.** Wire Art labels a pixel complete if it lies within 2 px of a projected voxel. | `sample()` uses a hard nearest-pixel `alpha > 0.5`. Accept a dilated ink mask (max-pool 3–5 px) when carving. | More carve yield at a 1–2 px edge error, which the fit then cleans up. A few lines. |
| **Non-rigid warp** (Shadow Art ARAP). | Only for solid silhouettes. A cheap 1-D version is a smooth monotone vertical remap of B's rows onto A's ink rows. | Wire Art says this is ill-posed for line art, so try it after the two items above. Medium effort. |
| **In-between views.** ComplicitSplat's benign supervision "especially near region boundaries". | Add renders at yaw 30°/45°/60° with a weak loss toward the background or toward low contrast, and raise `sh_degree` to 2–3. | Uncertain. With SH degree 1, colour is linear in view direction. When the two views are 90° apart, the midpoint colour is b(1−√2) + (c_A+c_B)/√2. Making it dark forces a large base b, which turns the back side bright after clamping. So SH degree 1 cannot fix mush cleanly; degree 2–3 might, with ComplicitSplat's spillover caveat. Low priority. |

Colour conflicts at the two secret views themselves are not a real problem. Each colour channel has four SH degree-1 coefficients and only two constraints, so both colours can be matched exactly. The 3DGS trick of adding one SH band every 1000 iterations (3DGS §7) is an optional way to stabilise the start.

## 3. Rendering ~1 px specks: gsplat vs Spark

- **Background.** 3DGS's code adds 0.3 to the 2D covariance (the paper doesn't mention it; Mip-Splatting points this out). gsplat exposes this as the "classic" mode. Its "antialiased" mode is Mip-Splatting's 2D filter: Σ + sI, with opacity multiplied by √(|Σ|/|Σ+sI|), where s is sized to cover about one pixel. Spark's docs say `preBlurAmount` (no compensation) is "typically 0.3" for scenes trained *without* anti-aliasing. `blurAmount` (with compensation) is "typically 0.3" for scenes trained *with* it. Our pairing of classic training with `preBlurAmount 0.3, blurAmount 0` is therefore consistent. Spark's default pairs with gsplat's antialiased mode.
- **Measured problem.** I read the exported PLYs and assumed a 768-px-tall secret view. Specks with σ < 0.3 secret-view px make up 13 % of `a2_eye`, 31 % of `a2_eye_jit` and 7 % of `tv_cs180`. These are Mip-Splatting's Fig. 1 "degenerate" Gaussians. What you see on screen is mostly the 0.3 dilation kernel, so they fatten and brighten when the canvas is smaller than 768 px × DPR and look different when it is larger. The cause is that `s_max·sigmoid(p_scale)` in `option_a.py`/`multi_view.py` has no floor.
- **Fix, either of:**
  - (a) Add a floor, `s = s_min + (s_max − s_min)·sigmoid(p)` with `s_min ≈ 0.3–0.5 · depth / fy`. This is Mip-Splatting's 3D smoothing filter (variance s·(d/f)² from the highest-rate view) applied by hand. Our only design view is the secret camera.
  - (b) Train with `rasterize_mode="antialiased"` and use Spark's default blur. Zooming out then dims specks instead of fattening them, and the optimiser compensates for the dimming during training.

  Either way, add a multi-resolution check like Mip-Splatting's: `eval_reveal.py` at `--res 256/512/1024`, compared at the same angular size.
- The PLY SH layout in `splatio.py` is correct: channel-major f_rest, the same as 3DGS.

## 4. Evaluation

- **The NCC metric is sensible.** It compares what the picture adds to a view (render minus background) with the picture itself, over the picture's pixels, and goes to about 0 when scrambled. Its weakness is that it is pixel-aligned: a recognisable picture shifted by 2 px scores low. Report the maximum NCC over ±3 px shifts, or add Wire Art's **bidirectional distance-transform error**, deviation (mean D_ink over rendered ink) plus incompleteness (mean D_render over target ink), which tolerates 1–2 px misalignment. For binary ink, also report **IoU/Dice** (Neural Shadow Art).
- **Multi-view alignment and concealment.** Visual Anagrams builds S_ij = similarity(view i, prompt j) and reports A = min diag(S) and concealment C = (1/N)·tr(softmax(S/τ)). Use NCC (or CLIP image-to-image) between each secret render and each picture. This catches B "leaking" into A's view, which a mean SSIM hides. Diffusion Illusions' independence score is the same idea.
- **Perceptual recognisability and a small user study.** Illusion3D, Visual Anagrams and Diffusion Illusions all use CLIP scores plus a user study. A 10-person "can you read it?" test at each reveal-zone step would make the reveal-zone curve much more convincing.

A related warning: Diffusion Illusions found that optimising in pixel space hid information in high frequencies that "requir[ed] pixel-perfect alignment". This is the same brittleness as our un-jittered reveal zone. Their fix was a low-frequency parameterisation (Fourier-feature networks). Ours is camera jitter. Illusion3D's jitter also perturbs **rotation and field of view**, with a *linear* ramp over all of training, not translation only over the first half as in `option_a.py`. Adding FOV jitter is a cheap way to cover the forward/back direction of the reveal zone.

---

## Comparison

| Work | Representation | Multi-view conflict handling | Transferable piece for us |
| --- | --- | --- | --- |
| Shadow Art | binary voxels, shadow hull = AND of cones | rigid placement, then iterative ARAP image warps (25 % steps, stiffness relaxed) | registration before carving; warp for solid silhouettes |
| Neural Shadow Art | occupancy MLP, ray-occupancy loss | periodic ICP target registration; frustum-intersection truncation; volume, cohesion, smoothness losses | registration; IoU/Dice; names the "cylinder" failure |
| Multi-view Wire Art | voxels → curve skeleton → splines | greedy hull expansion by distance-transform cost, 2 px tolerance; no warping (ill-posed for lines) | greedy completion; 2 px tolerance; bidirectional DT metric |
| DreamWire | 3D Bézier curves, SDS | MST connectivity loss | (only if we ever want connected strands) |
| Diffusion Illusions / Visual Anagrams | 2D images, diffusion prior | joint denoising or score distillation over views | alignment/concealment metrics; warning about high-frequency brittleness |
| Illusion3D | neural texture on fixed 3D shape, VSD | jitter (R, T, FOV, linear ramp), progressive resolution, patches | surface placement; FOV jitter; patch supervision |
| LookingGlass | latent flow, Laplacian pyramid warping | level-of-detail-aware warping and blending of views | area-averaged colour per Gaussian |
| ComplicitSplat / CLOAK | trained 3DGS, view-dependent SH | adversarial vs benign view sets, benign views near region boundary | multi-radius guard and novel benign poses; SH ablation |
| Mip-Splatting / gsplat | 3DGS + 3D and 2D filters | n/a | scale floor or antialiased mode; multi-resolution test |

---

## Ranked improvements (status as of 2026-11-24)

Done:
- **Size floor** (Mip-Splatting): specks are never smaller than 0.4 secret-view pixels (`--min_px` in `option_a.py`, `multi_view.py`).
- **Target-based reveal metric**: NCC of (render − background) against (picture − background) over the picture's pixels (`eval_reveal.py`).

Open, in order of impact relative to effort:
1. **Surface placement for option A** (`option_a.py --volume surface`): specks at the front ray–sphere (or a mesh) intersection as surface-aligned discs, so side views read as a painted object instead of a tube. ~10 lines for the sphere.
2. **Rigid registration before carving** (`multi_view.py`, Shadow Art / Neural Shadow Art): grid-search each picture's vertical offset and scale to maximise the worst view's coverage. Matters for pictures whose content does not fill the same rows. ~40 lines.
3. **Greedy completion with a 2 px tolerance** (`multi_view.carve_multi`, Multi-view Wire Art): add specks for uncovered pixels that land on the other views' ink, or just off it with background colour toward those cameras. ~30 lines.
4. **FOV and rotation jitter** (Illusion3D) in addition to translation jitter, with a linear ramp over all of training; covers the forward/back direction of the reveal zone.
5. **Off-view losses with SH degree 2–3** to clean up in-between angles; speculative because higher SH spills over (ComplicitSplat), and SH degree 1 cannot do it cleanly.

Cheap extras: IoU and distance-transform metrics for line-art pictures; a per-view confusion matrix (how much of painting B shows at spot A).
