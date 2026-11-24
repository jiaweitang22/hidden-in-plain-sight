// Hidden in Plain Sight: web viewer.
// three.js + Spark (Gaussian splats). Static, no build step.

import * as THREE from "three";
import { SparkRenderer, SplatMesh, SparkControls } from "@sparkjsdev/spark";

const params = new URLSearchParams(location.search);
const DEBUG = params.get("debug") === "1";
// ?filter=spark switches to Spark's default (opacity-compensated) blur, for A/B comparison.
const FILTER = params.get("filter") === "spark" ? "spark" : "classic";
const DPR = params.has("dpr") ? Number(params.get("dpr")) : Math.min(window.devicePixelRatio || 1, 2);

const REVEAL_SECONDS = 2.5;
const FOUND_POS = 0.02; // relative distance (fraction of secret-eye-to-centre)
const FOUND_ANGLE = 2.0; // degrees

const $ = (id) => document.getElementById(id);
const ui = {
  title: $("title"),
  hint: $("hint"),
  meter: $("meter"),
  meterFill: document.querySelector(".meter-fill"),
  meterLabel: document.querySelector(".meter-label"),
  reveal: $("btn-reveal"),
  reset: $("btn-reset"),
  scenes: $("btn-scenes"),
  gallery: $("gallery"),
  galleryList: $("gallery-list"),
  galleryClose: $("gallery-close"),
  loading: $("loading"),
  loadingFill: document.querySelector(".loading-fill"),
  loadingLabel: document.querySelector(".loading-label"),
  frame: $("frame"),
  debug: $("debug"),
};
const HINT = ui.hint.textContent;

// ---------------------------------------------------------------------------
// Camera convention helpers
// ---------------------------------------------------------------------------

// OpenCV camera (x right, y down, z forward) -> three.js camera (x right, y up, looking -z).
const CV_TO_GL = new THREE.Matrix4().makeScale(1, -1, -1);

/** Convert a row-major 4x4 OpenCV c2w (nested lists) into a three.js pose. */
function poseFromC2W(c2w, fovYDeg) {
  const m = new THREE.Matrix4().set(...c2w.flat()).multiply(CV_TO_GL);
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  m.decompose(position, quaternion, scale);
  return { position, quaternion, fov: fovYDeg };
}

/**
 * Vertical FOV to actually use so a frame of `frameAspect` with vertical FOV `fovY`
 * always fits in the window (never cropped left/right on narrow screens).
 */
function fitFov(fovY, frameAspect, windowAspect) {
  if (!frameAspect || windowAspect >= frameAspect) return fovY;
  const t = Math.tan(THREE.MathUtils.degToRad(fovY) / 2) * (frameAspect / windowAspect);
  return THREE.MathUtils.radToDeg(2 * Math.atan(t));
}

const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

// ---------------------------------------------------------------------------
// Scene list / gallery
// ---------------------------------------------------------------------------

async function fetchJSON(url) {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

function buildGallery(list, current) {
  ui.galleryList.innerHTML = "";
  for (const s of list) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    const p = new URLSearchParams(location.search);
    p.set("scene", s.name);
    a.href = `?${p}`;
    if (s.name === current) a.className = "current";
    const t = document.createElement("span");
    t.className = "g-title";
    t.textContent = s.title || s.name;
    const n = document.createElement("span");
    n.className = "g-name";
    n.textContent = s.name;
    a.append(t, n);
    li.append(a);
    ui.galleryList.append(li);
  }
}

function openGallery(open) {
  ui.gallery.hidden = !open;
  if (open) ui.galleryClose.focus();
}

// ---------------------------------------------------------------------------
// Renderer setup
// ---------------------------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, powerPreference: "high-performance" });
renderer.setPixelRatio(DPR);
renderer.setSize(window.innerWidth, window.innerHeight);
$("stage").append(renderer.domElement);
renderer.domElement.tabIndex = 0;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.01, 2000);

// Filter settings. Our Gaussians were optimised with gsplat "classic" rasterisation:
// a 0.3 px^2 dilation added to the 2D covariance with NO opacity compensation.
// Spark's default is blurAmount=0.3 WITH opacity compensation (alpha *= sqrt(det/det')),
// which dims tiny specks (about -27% for a 0.9 px sigma speck). preBlurAmount is
// Spark's uncompensated dilation, i.e. exactly gsplat classic eps2d.
const filterOptions =
  FILTER === "classic"
    ? { preBlurAmount: 0.3, blurAmount: 0.0, maxStdDev: 3.0, sortRadial: false }
    : {}; // Spark defaults: preBlurAmount 0, blurAmount 0.3, maxStdDev sqrt(8)

const spark = new SparkRenderer({
  renderer,
  ...filterOptions,
  focalAdjustment: 1.0,
  falloff: 1.0,
  enableLod: false, // LoD would merge tiny specks into larger blobs
  accumExtSplats: true, // float32 centres in the accumulator, not float16
});
scene.add(spark);

const controls = new SparkControls({ canvas: renderer.domElement });

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  meta: null,
  secret: null, // { position, quaternion, fov }
  start: null,
  secretAspect: 1,
  center: new THREE.Vector3(),
  secretDist: 1,
  nominalFov: 40, // vertical FOV before window fitting
  anim: null, // { from, to, t0 }
  heat: 0,
  found: false,
  ready: false,
};

function applyPose(pose) {
  camera.position.copy(pose.position);
  camera.quaternion.copy(pose.quaternion);
  state.nominalFov = pose.fov;
  updateProjection();
}

function updateProjection() {
  const aspect = window.innerWidth / window.innerHeight;
  camera.aspect = aspect;
  camera.fov = fitFov(state.nominalFov, state.secretAspect, aspect);
  camera.updateProjectionMatrix();
  layoutFrame();
}

/** Position the secret-frame overlay (the region the hidden image fills at the secret pose). */
function layoutFrame() {
  const W = window.innerWidth, H = window.innerHeight;
  const a = state.secretAspect || W / H;
  // Fraction of the window height covered by the secret frame's vertical FOV.
  const tNom = Math.tan(THREE.MathUtils.degToRad(state.secret ? state.secret.fov : camera.fov) / 2);
  const tCam = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
  const h = H * Math.min(1, tNom / tCam);
  const w = Math.min(W, h * a);
  Object.assign(ui.frame.style, {
    width: `${w}px`,
    height: `${h}px`,
    left: `${(W - w) / 2}px`,
    top: `${(H - h) / 2}px`,
  });
}

function resetVelocities() {
  const pc = controls.pointerControls;
  pc.rotateVelocity.set(0, 0, 0);
  pc.moveVelocity.set(0, 0, 0);
  pc.scroll.set(0, 0, 0);
  controls.fpsMovement.extraMove?.set(0, 0, 0);
}

function setControlsEnabled(on) {
  controls.fpsMovement.enable = on;
  controls.pointerControls.enable = on;
}

function flyTo(target, seconds = REVEAL_SECONDS) {
  if (!state.ready) return;
  resetVelocities();
  setControlsEnabled(false);
  state.anim = {
    from: { position: camera.position.clone(), quaternion: camera.quaternion.clone(), fov: state.nominalFov },
    to: target,
    t0: performance.now(),
    dur: seconds * 1000,
  };
}

function stepAnimation(now) {
  const a = state.anim;
  if (!a) return;
  const raw = Math.min(1, (now - a.t0) / a.dur);
  const t = easeInOutCubic(raw);
  camera.position.lerpVectors(a.from.position, a.to.position, t);
  camera.quaternion.slerpQuaternions(a.from.quaternion, a.to.quaternion, t);
  state.nominalFov = THREE.MathUtils.lerp(a.from.fov, a.to.fov, t);
  updateProjection();
  if (raw >= 1) {
    applyPose(a.to); // land exactly
    state.anim = null;
    resetVelocities();
    setControlsEnabled(true);
  }
}

// ---------------------------------------------------------------------------
// Warmer / colder
// ---------------------------------------------------------------------------

const _fwd = new THREE.Vector3();
const _sfwd = new THREE.Vector3();

function closeness() {
  const s = state.secret;
  const posErr = camera.position.distanceTo(s.position) / state.secretDist;
  camera.getWorldDirection(_fwd);
  _sfwd.set(0, 0, -1).applyQuaternion(s.quaternion);
  const angErr = THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(_fwd.dot(_sfwd), -1, 1)));
  // Normalised so (2%, 2 deg) -> d = 1. Log scale so the bar responds both far and near.
  const d = Math.hypot(posErr / FOUND_POS, angErr / FOUND_ANGLE) / Math.SQRT2;
  const heat = THREE.MathUtils.clamp(1 - Math.log1p(d) / Math.log1p(150), 0, 1);
  const found = posErr < FOUND_POS && angErr < FOUND_ANGLE;
  return { posErr, angErr, heat, found };
}

function updateMeter(c, dt) {
  state.heat += (c.heat - state.heat) * (1 - Math.exp(-dt * 8));
  const pct = Math.round(state.heat * 100);
  ui.meterFill.style.width = `${pct}%`;
  ui.meter.setAttribute("aria-valuenow", String(pct));
  if (c.found !== state.found) {
    state.found = c.found;
    document.body.classList.toggle("found", c.found);
    ui.hint.textContent = c.found ? "Found it. This is the spot." : HINT;
  }
  const label = c.found ? "Found it" : state.heat > 0.8 ? "Hot" : state.heat > 0.62 ? "Warmer" : state.heat > 0.4 ? "Cool" : "Colder";
  if (ui.meterLabel.textContent !== label) ui.meterLabel.textContent = label;
}

// ---------------------------------------------------------------------------
// Loading a scene
// ---------------------------------------------------------------------------

async function loadScene() {
  let list = [];
  try {
    list = (await fetchJSON("scenes/index.json")).scenes || [];
  } catch (e) {
    console.warn("Could not read scenes/index.json", e);
  }
  const name = params.get("scene") || list[0]?.name;
  buildGallery(list, name);
  if (!name) throw new Error("No scene given and scenes/index.json is empty.");

  const base = `scenes/${encodeURIComponent(name)}/`;
  const meta = await fetchJSON(`${base}scene.json`);
  if (meta.convention && meta.convention !== "opencv") {
    console.warn(`Unexpected camera convention "${meta.convention}", treating as opencv.`);
  }
  state.meta = meta;
  document.title = `${meta.title || meta.name} · Hidden in Plain Sight`;
  ui.title.textContent = meta.title || meta.name;

  const bg = meta.background || [0, 0, 0];
  const bgColor = new THREE.Color().setRGB(bg[0], bg[1], bg[2], THREE.SRGBColorSpace);
  scene.background = bgColor;
  document.documentElement.style.setProperty("--bg", `#${bgColor.getHexString(THREE.SRGBColorSpace)}`);

  // World up from scene.json (synthetic scenes are y-down). Spark's controls read camera.up.
  const up = new THREE.Vector3().fromArray(meta.up || [0, 1, 0]).normalize();
  camera.up.copy(up);

  state.center.fromArray(meta.center || [0, 0, 0]);
  state.secret = poseFromC2W(meta.secret.c2w, meta.secret.fov_y_deg);
  state.secretAspect = meta.secret.aspect || 1;
  state.start = meta.start ? poseFromC2W(meta.start.c2w, meta.start.fov_y_deg ?? meta.secret.fov_y_deg) : state.secret;
  state.secretDist = Math.max(1e-6, state.secret.position.distanceTo(state.center));

  // Scale navigation speeds to the scene.
  const s = state.secretDist / 5;
  controls.fpsMovement.moveSpeed = state.secretDist / 4;
  const pc = controls.pointerControls;
  pc.slideSpeed *= s;
  pc.scrollSpeed *= s;
  if (pc.pressMoveSpeed) pc.pressMoveSpeed *= s;
  if (pc.doublePressMoveSpeed) pc.doublePressMoveSpeed *= s;
  if (pc.triplePressMoveSpeed) pc.triplePressMoveSpeed *= s;

  applyPose(state.start);

  const mesh = new SplatMesh({
    url: base + (meta.splat || "scene.ply"),
    extSplats: true, // float32 centres: avoids float16 jitter on far specks
    lod: false,
    onProgress: (ev) => {
      if (ev.lengthComputable && ev.total) ui.loadingFill.style.width = `${(100 * ev.loaded) / ev.total}%`;
    },
  });
  scene.add(mesh);
  await mesh.initialized;
  state.mesh = mesh;
  ui.loadingFill.style.width = "100%";
  state.ready = true;
  ui.loading.classList.add("done");
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

ui.reveal.addEventListener("click", () => flyTo(state.secret));
ui.reset.addEventListener("click", () => flyTo(state.start, 1.6));
ui.scenes.addEventListener("click", () => openGallery(true));
ui.galleryClose.addEventListener("click", () => openGallery(false));
ui.gallery.addEventListener("click", (e) => {
  if (e.target === ui.gallery) openGallery(false);
});

// Keep buttons from holding keyboard focus (so Space/Enter don't re-trigger them).
for (const b of document.querySelectorAll("#dock button")) {
  b.addEventListener("pointerup", () => b.blur());
}

window.addEventListener("keydown", (e) => {
  if (e.metaKey || e.altKey || e.ctrlKey) return;
  if (e.code === "Escape") return openGallery(false);
  if (!ui.gallery.hidden) return;
  if (e.code === "KeyH") flyTo(state.secret);
  else if (e.code === "KeyR") flyTo(state.start, 1.6);
});

window.addEventListener("resize", () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  updateProjection();
});

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

let last = performance.now();
let fpsFrames = 0, fpsT0 = last, fps = 0;
const _euler = new THREE.Euler();

if (DEBUG) ui.debug.hidden = false;

renderer.setAnimationLoop((now) => {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;

  // Always call update (even when disabled) so its internal clock stays current.
  controls.update(camera);
  stepAnimation(now);

  renderer.render(scene, camera);

  if (state.ready) {
    const c = closeness();
    updateMeter(c, dt);
    if (DEBUG) {
      fpsFrames++;
      if (now - fpsT0 > 500) {
        fps = (fpsFrames * 1000) / (now - fpsT0);
        fpsFrames = 0;
        fpsT0 = now;
      }
      const p = camera.position, q = camera.quaternion;
      camera.getWorldDirection(_fwd);
      _euler.setFromQuaternion(q, "YXZ");
      const db = renderer.getDrawingBufferSize(new THREE.Vector2());
      ui.debug.textContent = [
        `fps     ${fps.toFixed(1)}`,
        `splats  ${state.mesh?.extSplats?.numSplats ?? state.mesh?.packedSplats?.numSplats ?? state.meta?.num_gaussians ?? "?"}`,
        `filter  ${FILTER}  dpr ${DPR}`,
        `canvas  ${db.x}x${db.y}`,
        `pos     ${p.x.toFixed(3)} ${p.y.toFixed(3)} ${p.z.toFixed(3)}`,
        `quat    ${q.x.toFixed(3)} ${q.y.toFixed(3)} ${q.z.toFixed(3)} ${q.w.toFixed(3)}`,
        `fwd     ${_fwd.x.toFixed(3)} ${_fwd.y.toFixed(3)} ${_fwd.z.toFixed(3)}`,
        `fov     ${camera.fov.toFixed(2)} (nominal ${state.nominalFov.toFixed(2)})`,
        `posErr  ${(c.posErr * 100).toFixed(2)}%`,
        `angErr  ${c.angErr.toFixed(2)} deg`,
        `heat    ${c.heat.toFixed(3)}${c.found ? "  FOUND" : ""}`,
      ].join("\n");
    }
  }
});

// Expose for debugging from the console.
window.hips = { THREE, camera, scene, spark, controls, state, flyTo, renderer };

loadScene().catch((err) => {
  console.error(err);
  ui.loading.classList.add("error");
  ui.loadingLabel.textContent = `Could not load scene: ${err.message}`;
});
