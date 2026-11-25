// Hidden in Plain Sight: web viewer.
// three.js + Spark (Gaussian splats). Static, no build step.
//
// Navigation: ORBIT (default; three.js OrbitControls around the scene centre) or
// FREE FLY (Spark's SparkControls: WASD/QE + drag to look). Only one of the two is
// ever enabled. Automatic camera moves ("Show me the spot", Reset, the scrub slider,
// snap-to-spot) suspend both, and any pointer/wheel input on the canvas hands control
// straight back to the user at the current pose.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh, SparkControls } from "@sparkjsdev/spark";

const params = new URLSearchParams(location.search);
const DEBUG = params.get("debug") === "1";
// ?filter=spark switches to Spark's default (opacity-compensated) blur, for A/B comparison.
const FILTER = params.get("filter") === "spark" ? "spark" : "classic";
const DPR = params.has("dpr") ? Number(params.get("dpr")) : Math.min(window.devicePixelRatio || 1, 2);
// ?snap=0 disables the gentle snap onto a spot when the user stops very close to it.
const SNAP = params.get("snap") !== "0";
const START_MODE = params.get("mode") === "fly" ? "fly" : "orbit";

const REVEAL_SECONDS = 2.5; // "Show me the spot" flight (also the cap for other automatic moves)
const FOUND_POS = 0.02; // relative distance (fraction of secret-eye-to-centre)
const FOUND_ANGLE = 2.0; // degrees
const SNAP_D = 3; // snap onto a spot when the closeness distance d drops below this (d = 1 ~ found)

// Orbit tuning.
const ORBIT_ROTATE = 0.7; // full canvas height drag = 0.7 * 360 deg
const ORBIT_ZOOM = 0.8;
const ORBIT_DAMPING = 0.1;

const IS_TOUCH = matchMedia("(hover: none) and (pointer: coarse)").matches;

const $ = (id) => document.getElementById(id);
const ui = {
  stage: $("stage"),
  title: $("title"),
  hint: $("hint"),
  meter: $("meter"),
  ringFill: document.querySelector("#meter .ring-fill"),
  meterLabel: document.querySelector("#meter .meter-label"),
  reveal: $("btn-reveal"),
  revealLabel: document.querySelector("#btn-reveal .label"),
  revealCount: document.querySelector("#btn-reveal .count"),
  reset: $("btn-reset"),
  modeOrbit: $("mode-orbit"),
  modeFly: $("mode-fly"),
  helpBtn: $("btn-help"),
  help: $("help"),
  helpClose: $("help-close"),
  scrub: $("scrub"),
  scrubRange: $("scrub-range"),
  scrubSpot: $("scrub-spot"),
  scrubName: document.querySelector("#scrub-spot .scrub-name"),
  scrubThumb: document.querySelector("#scrub-spot .scrub-thumb"),
  sceneList: $("scene-list"),
  caption: $("caption"),
  capImg: document.querySelector("#caption .cap-img"),
  capKicker: document.querySelector("#caption .cap-kicker"),
  capTitle: document.querySelector("#caption .cap-title"),
  capArtist: document.querySelector("#caption .cap-artist"),
  capNext: document.querySelector("#caption .cap-next"),
  capAction: $("cap-action"),
  capScene: $("cap-scene"),
  loading: $("loading"),
  loadingFill: document.querySelector(".loading-fill"),
  loadingLabel: document.querySelector(".loading-label"),
  frame: $("frame"),
  debug: $("debug"),
  side: $("side"),
  stepNum: $("step-num"),
  progressDots: document.querySelector("#progress .dots"),
  progressText: document.querySelector("#progress .progress-text"),
  helpGo: $("help-go"),
  helpForget: $("help-forget"),
};
const MODE_HINT = {
  orbit: IS_TOUCH
    ? "Drag to circle the cloud, pinch to zoom."
    : "Drag to circle the cloud, scroll to zoom. Somewhere, a painting lines up.",
  fly: IS_TOUCH
    ? "Free fly: drag to turn, two fingers to move."
    : "Free fly: drag to look, WASD to move, Q/E down and up. F goes back to orbit.",
};
let HINT = MODE_HINT[START_MODE];

/** Painting info (title, artist, year, thumb) for a secret; see paintings.js. */
const paintingOf = (secret) =>
  window.hipsPainting ? window.hipsPainting(secret) : { label: secret?.label || "", title: secret?.label || "A hidden picture", artist: "", year: "", thumb: "" };
/** Secret spots of a scene.json (tagged with their target image when paintings.js is loaded). */
const secretsOf = (meta) =>
  window.hipsSecrets ? window.hipsSecrets(meta) : meta ? (meta.secrets && meta.secrets.length ? meta.secrets : [meta.secret || {}]) : [];
/** Found paintings are remembered per scene (paintings.js, localStorage); they stay hidden until then. */
const foundStore = window.hipsFound || { get: () => new Set(), add() {}, clear() {} };

const COLD_RGB = [91, 140, 255];
const WARM_RGB = [255, 181, 71];
const RING_C = 2 * Math.PI * 15; // ring circumference (r = 15 in the SVG)

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
const smoothstep = (a, b, x) => {
  const t = THREE.MathUtils.clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

// ---------------------------------------------------------------------------
// Scene list / gallery
// ---------------------------------------------------------------------------

async function fetchJSON(url) {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

let sceneList = []; // entries of scenes/index.json
let sceneName = "";

function sceneHref(name) {
  const p = new URLSearchParams(location.search);
  p.set("scene", name);
  return `?${p}`;
}

const countWord = (n) => (n === 1 ? "1 painting" : `${n} paintings`);

/**
 * A small tile for one painting. Until it has been found it is a plain "?" tile with no
 * hint of the painting (no image, no title); once found it shows the painting.
 */
function paintingTile(secret, i, found) {
  const tile = document.createElement("span");
  tile.className = "tile";
  tile.dataset.idx = String(i);
  fillTile(tile, secret, found);
  return tile;
}

function fillTile(tile, secret, found) {
  const p = paintingOf(secret);
  const shown = found && p.thumb;
  if (tile.dataset.shown === String(!!found)) return;
  // Animate only a tile that flips from "?" to the painting while the visitor watches.
  tile.classList.toggle("reveal", !!found && tile.dataset.shown === "false");
  tile.dataset.shown = String(!!found);
  tile.querySelector("img")?.remove();
  tile.classList.toggle("found", !!found);
  tile.classList.toggle("blank", !shown);
  if (shown) {
    const img = document.createElement("img");
    img.src = p.thumb;
    img.alt = "";
    img.decoding = "async";
    img.addEventListener("error", () => { img.remove(); tile.classList.add("blank"); });
    tile.append(img);
  }
  tile.title = found ? (p.artist ? `${p.title}, ${p.artist}` : p.title) : "Not found yet";
}

/**
 * Scene rail. Cards come from scenes/index.json; each card's painting thumbnails and
 * count come from that scene's scene.json (secrets[].label), fetched in parallel.
 */
async function buildSceneRail(list, current, currentMeta) {
  ui.sceneList.innerHTML = "";
  const metas = await Promise.all(
    list.map((s) =>
      s.name === current && currentMeta
        ? currentMeta
        : fetchJSON(`scenes/${encodeURIComponent(s.name)}/scene.json`).catch(() => null)
    )
  );
  list.forEach((s, k) => {
    const meta = metas[k];
    const secrets = secretsOf(meta);
    const found = s.name === current ? state.foundSet : foundStore.get(s.name);
    const a = document.createElement("a");
    a.className = "scene-card";
    a.href = sceneHref(s.name);
    a.dataset.scene = s.name;
    if (s.name === current) {
      a.classList.add("current");
      a.setAttribute("aria-current", "page");
    }
    const art = document.createElement("span");
    art.className = `sc-art n${Math.min(secrets.length || 1, 4)}`;
    if (secrets.length) secrets.slice(0, 4).forEach((sp, i) => art.append(paintingTile(sp, i, found.has(i))));
    else art.append(Object.assign(document.createElement("span"), { className: "tile blank" }));
    const text = document.createElement("span");
    text.className = "sc-text";
    const t = document.createElement("span");
    t.className = "sc-title";
    t.textContent = s.title || meta?.title || s.name;
    const m = document.createElement("span");
    m.className = "sc-meta";
    m.textContent = secrets.length ? progressWord(secrets.length, [...found].filter((i) => i < secrets.length).length) : "Hidden paintings";
    a.classList.toggle("complete", secrets.length > 0 && found.size >= secrets.length);
    text.append(t, m);
    a.append(art, text);
    ui.sceneList.append(a);
  });
  // On phones the rail scrolls sideways: keep the current scene in view.
  ui.sceneList.querySelector(".current")?.scrollIntoView({ block: "nearest", inline: "center" });
  updateRailProgress();
}

function progressWord(n, k) {
  return k === 0 ? `${countWord(n)} hidden here` : k >= n ? (n === 2 ? "Both found" : n === 1 ? "Found" : `All ${n} found`) : `${k} of ${n} found`;
}

/** Reflect found paintings: the current scene's card, the progress dots, the scrub pill, storage. */
function updateRailProgress() {
  const n = state.secrets.length;
  if (!n) return;
  const k = state.foundSet.size;
  if (sceneName && k) foundStore.add(sceneName, [...state.foundSet]);
  // progress dots in the controls card
  if (ui.progressDots.childElementCount !== n) {
    ui.progressDots.innerHTML = "";
    for (let i = 0; i < n; i++) ui.progressDots.append(document.createElement("i"));
  }
  [...ui.progressDots.children].forEach((d, i) => d.classList.toggle("on", state.foundSet.has(i)));
  const pt = n === 1 ? (k ? "Found" : "1 painting hidden") : `${k} of ${n} found`;
  if (ui.progressText.textContent !== pt) ui.progressText.textContent = pt;
  setScrubIdx(state.scrubIdx); // the pill may now name the painting
  const card = ui.sceneList.querySelector(".scene-card.current");
  if (!card) return;
  const meta = card.querySelector(".sc-meta");
  const text = progressWord(n, k);
  if (meta && meta.textContent !== text) meta.textContent = text;
  card.classList.toggle("complete", k >= n);
  for (const tile of card.querySelectorAll(".tile")) {
    const i = Number(tile.dataset.idx);
    fillTile(tile, state.secrets[i], state.foundSet.has(i));
  }
}

const HELP_SEEN = "hips.helpSeen.v1";
function openHelp(open) {
  if (!open) try { localStorage.setItem(HELP_SEEN, "1"); } catch (e) { /* ignore */ }
  ui.help.hidden = !open;
  ui.helpBtn.setAttribute("aria-expanded", String(open));
  if (open) ui.helpClose.focus();
}

// ---------------------------------------------------------------------------
// Renderer setup
// ---------------------------------------------------------------------------

// The canvas fills the #stage box (between the header and the dock), not the window,
// so the paintings are never covered by the UI.
const stageSize = () => ({
  w: Math.max(1, ui.stage.clientWidth),
  h: Math.max(1, ui.stage.clientHeight),
});

const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, powerPreference: "high-performance" });
renderer.setPixelRatio(DPR);
renderer.setSize(stageSize().w, stageSize().h);
ui.stage.prepend(renderer.domElement);
renderer.domElement.tabIndex = 0;
renderer.domElement.setAttribute("aria-label", "3D view of the scene");
const canvas = renderer.domElement;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, stageSize().w / stageSize().h, 0.01, 2000);

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

// Free-fly controls (Spark). Disabled unless the mode is "fly".
const controls = new SparkControls({ canvas });
// Orbit controls: created in loadScene(), because OrbitControls caches camera.up at
// construction and synthetic scenes are y-down.
let orbit = null;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  meta: null,
  secret: null, // { position, quaternion, fov, aspect, label } -- the spot currently nearest
  secrets: [], // all secret spots in the scene
  revealIdx: 0,
  nearIdx: 0,
  start: null,
  secretAspect: 1,
  center: new THREE.Vector3(),
  secretDist: 1,
  nominalFov: 40, // vertical FOV before window fitting
  mode: START_MODE, // "orbit" | "fly"
  anim: null, // automatic flight { from, to, target, t0, dur, onDone }
  scrub: null, // slider-driven move { from, blend, shown, target, idx }
  scrubIdx: 0, // which spot the slider leads to
  orbitLive: false, // OrbitControls is being driven (user input or damping still settling)
  dragging: false,
  interactD0: Infinity, // closeness at the start of the last orbit gesture (for snap)
  heat: 0,
  found: false,
  foundSet: new Set(),
  captionIdx: -1, // painting the caption card currently describes
  lastC: null,
  ready: false,
};

function applyPose(pose) {
  camera.position.copy(pose.position);
  camera.quaternion.copy(pose.quaternion);
  state.nominalFov = pose.fov;
  updateProjection();
}

function currentPose() {
  return { position: camera.position.clone(), quaternion: camera.quaternion.clone(), fov: state.nominalFov };
}

function updateProjection() {
  const { w, h } = stageSize();
  const aspect = w / h;
  camera.aspect = aspect;
  camera.fov = fitFov(state.nominalFov, state.secretAspect, aspect);
  camera.updateProjectionMatrix();
  layoutFrame();
}

/** Position the secret-frame overlay (the region the hidden image fills at the secret pose). */
function layoutFrame() {
  const { w: W, h: H } = stageSize();
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
  layoutCaption({ left: (W - w) / 2, top: (H - h) / 2, w, h, W, H });
}

let lastFrameBox = null;

/**
 * Place the museum label: beside the painting when there is room (wide screens),
 * below it when there is room (tall phone screens), otherwise over its lower edge.
 */
const SIDE_LABEL = matchMedia("(min-width: 761px)");

function layoutCaption(box = lastFrameBox) {
  if (!box) return;
  lastFrameBox = box;
  const c = ui.caption;
  // Wide screens: the label hangs in the right column under the controls card, like a
  // wall label beside the painting, so it never covers the picture.
  if (SIDE_LABEL.matches && ui.side) {
    if (c.parentNode !== ui.side) ui.side.append(c);
    c.classList.remove("beside", "below", "over");
    c.classList.add("aside");
    c.style.width = c.style.left = c.style.top = "";
    return;
  }
  if (c.parentNode !== ui.stage) ui.stage.insertBefore(c, ui.loading);
  c.classList.remove("aside");
  const side = box.W - (box.left + box.w);
  const below = box.H - (box.top + box.h);
  c.classList.remove("beside", "below", "over");
  c.style.width = "";
  let mode;
  if (side >= 210) {
    mode = "beside";
    c.style.width = `${Math.min(300, side - 40)}px`;
  } else {
    c.style.width = `${Math.min(440, box.W - 32)}px`;
    mode = below >= c.offsetHeight + 24 ? "below" : "over";
    if (mode === "over") c.style.width = ""; // compact label, shrink-to-fit (see style.css)
  }
  c.classList.add(mode);
  const ch = c.offsetHeight, cw = c.offsetWidth;
  let left, top;
  if (mode === "beside") {
    left = box.left + box.w + 20;
    top = box.top + box.h - ch;
  } else if (mode === "below") {
    left = (box.W - cw) / 2;
    top = box.top + box.h + Math.min(24, (below - ch) / 2);
  } else {
    left = Math.max(16, box.left + 16);
    top = box.top + box.h - ch - 16;
  }
  c.style.left = `${Math.round(left)}px`;
  c.style.top = `${Math.round(Math.max(8, top))}px`;
}

// ---------------------------------------------------------------------------
// Orbit-space poses: spherical coordinates about the scene centre with scene "up" as
// the pole, plus the camera's rotation relative to "looking at the centre". Moving
// through this space keeps the horizon level and arcs around the scene instead of
// cutting through it; endpoints are reproduced exactly.
// ---------------------------------------------------------------------------

const _Y = new THREE.Vector3(0, 1, 0);
const upQuat = new THREE.Quaternion(); // scene up -> +Y
const upQuatInv = new THREE.Quaternion();
const _m4 = new THREE.Matrix4();
const _v = new THREE.Vector3();
const _q = new THREE.Quaternion();

function lookQuat(pos, target, out) {
  _m4.lookAt(pos, target, camera.up); // same convention as Object3D.lookAt for cameras
  return out.setFromRotationMatrix(_m4);
}

function toOrbit(pose) {
  _v.copy(pose.position).sub(state.center).applyQuaternion(upQuat);
  const sph = new THREE.Spherical().setFromVector3(_v);
  const look = sph.radius > 1e-9 ? lookQuat(pose.position, state.center, new THREE.Quaternion()) : pose.quaternion.clone();
  const rel = look.invert().multiply(pose.quaternion);
  return { sph, rel, fov: pose.fov };
}

function fromOrbit(o, out = { position: new THREE.Vector3(), quaternion: new THREE.Quaternion(), fov: 40 }) {
  _v.setFromSpherical(o.sph).applyQuaternion(upQuatInv);
  out.position.copy(state.center).add(_v);
  lookQuat(out.position, state.center, out.quaternion).multiply(o.rel);
  out.fov = o.fov;
  return out;
}

function lerpOrbit(a, b, t) {
  let dTheta = b.sph.theta - a.sph.theta;
  dTheta = Math.atan2(Math.sin(dTheta), Math.cos(dTheta)); // shortest way round
  const ra = a.sph.radius, rb = b.sph.radius;
  const r = ra > 1e-6 && rb > 1e-6 ? ra * Math.pow(rb / ra, t) : THREE.MathUtils.lerp(ra, rb, t);
  const phi = THREE.MathUtils.clamp(THREE.MathUtils.lerp(a.sph.phi, b.sph.phi, t), 1e-3, Math.PI - 1e-3);
  return {
    sph: new THREE.Spherical(r, phi, a.sph.theta + dTheta * t),
    rel: a.rel.clone().slerp(b.rel, t),
    fov: THREE.MathUtils.lerp(a.fov, b.fov, t),
  };
}

/** Rough "how far apart" of two orbit poses (radians + log-radius), for flight durations. */
function orbitSeparation(a, b) {
  _v.setFromSpherical(a.sph).normalize();
  const u = new THREE.Vector3().setFromSpherical(b.sph).normalize();
  const ang = Math.acos(THREE.MathUtils.clamp(_v.dot(u), -1, 1));
  const rr = a.sph.radius > 1e-6 && b.sph.radius > 1e-6 ? Math.abs(Math.log(b.sph.radius / a.sph.radius)) : 1;
  return ang + rr + a.rel.angleTo(b.rel);
}

// ---------------------------------------------------------------------------
// Controls hand-off
// ---------------------------------------------------------------------------

/**
 * Spark's PointerControls keeps recording pointer positions and wheel deltas while
 * disabled, but only consumes them in update() when enabled. Re-enabling after a
 * flight (or after orbit mode) would then replay all of it at once: the "jump".
 * Call this every frame while Spark is disabled.
 */
function syncSparkInput() {
  const pc = controls.pointerControls;
  if (pc.rotating) pc.rotating.last.copy(pc.rotating.position);
  if (pc.sliding) pc.sliding.last.copy(pc.sliding.position);
  pc.scroll.set(0, 0, 0);
  pc.rotateVelocity.set(0, 0, 0);
  pc.moveVelocity.set(0, 0, 0);
  controls.fpsMovement.extraMove?.set(0, 0, 0);
}

/** Forget any OrbitControls motion in flight (damping residue, half-finished drag). */
function clearOrbitMotion() {
  if (!orbit) return;
  orbit._sphericalDelta?.set(0, 0, 0);
  orbit._panOffset?.set(0, 0, 0);
  if ("_scale" in orbit) orbit._scale = 1;
  // A drag that began before/while the camera was flying has a stale start point;
  // drop it (the user simply presses again) rather than jumping.
  orbit.state = -1; // _STATE.NONE
  state.orbitLive = false;
}

/**
 * Orbit target for the current view: the point on the view ray closest to the scene
 * centre. Equals the centre whenever the camera looks at it (start and secret poses of
 * synthetic scenes), and never makes OrbitControls' lookAt() turn the camera.
 */
function setOrbitTargetFromView() {
  if (!orbit) return;
  camera.getWorldDirection(_v);
  const along = _v.dot(new THREE.Vector3().subVectors(state.center, camera.position));
  const dist = along > 0.1 * state.secretDist ? along : state.secretDist;
  orbit.target.copy(camera.position).addScaledVector(_v, dist);
}

const isAuto = () => !!(state.anim || state.scrub);

/** Enable exactly one of the two control schemes (or none during automatic moves). */
function applyControlState() {
  const auto = isAuto();
  const flyOn = state.mode === "fly" && !auto;
  const orbitOn = state.mode === "orbit" && !auto;
  controls.fpsMovement.enable = flyOn;
  controls.pointerControls.enable = flyOn;
  if (orbit) orbit.enabled = orbitOn;
}

/** Give control back to the user at the current pose. */
function handBack() {
  syncSparkInput();
  clearOrbitMotion();
  setOrbitTargetFromView();
  applyControlState();
}

/** Stop any automatic camera move where it is. */
function stopAuto() {
  if (!isAuto()) return;
  state.anim = null;
  state.scrub = null;
  handBack();
}

function flyTo(target, { seconds, onDone } = {}) {
  if (!state.ready) return;
  state.scrub = null;
  const from = toOrbit(currentPose());
  const to = toOrbit(target);
  const dur = seconds ?? THREE.MathUtils.clamp(0.5 + 1.0 * orbitSeparation(from, to), 0.5, REVEAL_SECONDS);
  state.anim = { from, to, target, t0: performance.now(), dur: dur * 1000, onDone };
  clearOrbitMotion();
  applyControlState();
}

const _pose = { position: new THREE.Vector3(), quaternion: new THREE.Quaternion(), fov: 40 };

function stepAnimation(now) {
  const a = state.anim;
  if (!a) return;
  const raw = Math.min(1, (now - a.t0) / a.dur);
  if (raw >= 1) {
    applyPose(a.target); // land exactly
    state.anim = null;
    handBack();
    a.onDone?.();
    return;
  }
  applyPose(fromOrbit(lerpOrbit(a.from, a.to, easeInOutCubic(raw)), _pose));
}

// ---------------------------------------------------------------------------
// Scrub slider: glide along the arc from the start pose (0) to a spot (1).
// ---------------------------------------------------------------------------

const SCRUB_MAX = Number(ui.scrubRange.max) || 1000;
let startOrbit = null;
let spotOrbits = [];

function setScrubIdx(i) {
  state.scrubIdx = i;
  const n = state.secrets.length;
  const s = state.secrets[i];
  const p = paintingOf(s);
  const known = state.foundSet.has(i);
  const name = known ? s.label || p.title : n > 1 ? `Painting ${i + 1}` : "the painting";
  if (ui.scrubName.textContent !== name) ui.scrubName.textContent = name;
  if (known && p.thumb) {
    if (!ui.scrubThumb.src.endsWith(p.thumb)) ui.scrubThumb.src = p.thumb;
    ui.scrubThumb.hidden = false;
  } else {
    ui.scrubThumb.hidden = true;
  }
  ui.scrubSpot.title = n > 1 ? `The slider assembles "${name}". Click to switch paintings.` : "";
  ui.scrubSpot.setAttribute("aria-label", n > 1 ? `Slider leads to ${name}. Switch painting.` : `Slider leads to ${name}`);
  ui.scrubRange.setAttribute("aria-valuetext", name);
}

function setScrubValue(t, attached = true) {
  ui.scrubRange.value = String(Math.round(t * SCRUB_MAX));
  ui.scrub.classList.toggle("detached", !attached);
}

function detachScrub() {
  if (!ui.scrub.classList.contains("detached")) ui.scrub.classList.add("detached");
}

function scrubPathOrbit(idx, t) {
  return lerpOrbit(startOrbit, spotOrbits[idx], t);
}

/** Begin (or retarget) a slider-driven move towards slider value t. */
function scrubTo(t) {
  if (!state.ready) return;
  if (!state.scrub) {
    // Is the camera already on the slider's path (where the slider last put it)?
    const onPath = !ui.scrub.classList.contains("detached") && !state.anim;
    state.anim = null;
    state.scrub = {
      idx: state.scrubIdx,
      from: toOrbit(currentPose()),
      // On the path: continue smoothly from there. Off it: blend in from the current pose.
      blend: onPath ? 1 : 0,
      shown: onPath ? lastScrubT : t,
      target: t,
    };
    clearOrbitMotion();
    applyControlState();
  }
  state.scrub.target = t;
  ui.scrub.classList.remove("detached");
}

let lastScrubT = 0;

function stepScrub(dt) {
  const s = state.scrub;
  if (!s) return;
  // Smooth the slider so dragging looks like a camera move, not a stutter.
  s.shown += (s.target - s.shown) * (1 - Math.exp(-dt * 10));
  if (Math.abs(s.target - s.shown) < 2e-4) s.shown = s.target;
  s.blend = Math.min(1, s.blend + dt / 0.5);
  const pathO = scrubPathOrbit(s.idx, s.shown);
  const o = s.blend < 1 ? lerpOrbit(s.from, pathO, easeInOutCubic(s.blend)) : pathO;
  applyPose(fromOrbit(o, _pose));
  lastScrubT = s.shown;
  if (s.shown === s.target && s.blend >= 1) {
    // Settled. Land exactly on the endpoints so the picture is crisp.
    if (s.target >= 1) applyPose(state.secrets[s.idx]);
    else if (s.target <= 0) applyPose(state.start);
    state.scrub = null;
    handBack();
  }
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

function setMode(mode) {
  if (mode !== "orbit" && mode !== "fly") return;
  state.mode = mode;
  document.body.dataset.mode = mode;
  ui.modeOrbit.setAttribute("aria-pressed", String(mode === "orbit"));
  ui.modeFly.setAttribute("aria-pressed", String(mode === "fly"));
  HINT = MODE_HINT[mode];
  if (!state.found) ui.hint.textContent = HINT;
  if (!state.ready) return applyControlState();
  if (isAuto()) return applyControlState(); // takes effect when the move ends
  if (mode === "orbit") {
    // Orbit needs the centre in front of us at a sensible distance; turn smoothly if not.
    const toC = new THREE.Vector3().subVectors(state.center, camera.position);
    const dist = toC.length();
    camera.getWorldDirection(_v);
    const ang = THREE.MathUtils.radToDeg(_v.angleTo(toC));
    const r = THREE.MathUtils.clamp(dist, orbit.minDistance, orbit.maxDistance);
    if (ang > 3 || Math.abs(r - dist) > 1e-6) {
      const pos = dist > 1e-6
        ? state.center.clone().addScaledVector(toC.normalize(), -r)
        : state.start.position.clone();
      flyTo({ position: pos, quaternion: lookQuat(pos, state.center, new THREE.Quaternion()), fov: state.nominalFov }, { seconds: 0.8 });
      return;
    }
  }
  handBack();
}

// ---------------------------------------------------------------------------
// Warmer / colder
// ---------------------------------------------------------------------------

const _fwd = new THREE.Vector3();
const _sfwd = new THREE.Vector3();

function closenessTo(s) {
  const posErr = camera.position.distanceTo(s.position) / state.secretDist;
  camera.getWorldDirection(_fwd);
  _sfwd.set(0, 0, -1).applyQuaternion(s.quaternion);
  const angErr = THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(_fwd.dot(_sfwd), -1, 1)));
  // Normalised so (2%, 2 deg) -> d = 1. Log scale so the bar responds both far and near.
  const d = Math.hypot(posErr / FOUND_POS, angErr / FOUND_ANGLE) / Math.SQRT2;
  const heat = THREE.MathUtils.clamp(1 - Math.log1p(d) / Math.log1p(150), 0, 1);
  const found = posErr < FOUND_POS && angErr < FOUND_ANGLE;
  return { posErr, angErr, heat, found, d };
}

/** Closeness to the nearest secret spot; scenes may hide several pictures (scene.json "secrets"). */
function closeness() {
  let best = null;
  state.secrets.forEach((s, i) => {
    const c = closenessTo(s);
    if (!best || c.d < best.d) best = { ...c, idx: i };
  });
  if (state.secrets[best.idx] !== state.secret) {
    state.secret = state.secrets[best.idx];
    state.secretAspect = state.secret.aspect;
    updateProjection();
  }
  state.nearIdx = best.idx;
  return best;
}

function foundHint() {
  const n = state.secrets.length;
  const left = n - state.foundSet.size;
  if (n < 2) return "You found it. Move away and watch it scatter.";
  return left > 0
    ? `${left === 1 ? "One more painting is" : `${left} more paintings are`} hiding here.`
    : n === 2 ? "You found both paintings in this scene." : `You found all ${n} paintings in this scene.`;
}

/** The museum label for the painting currently in view. */
function updateCaption() {
  const n = state.secrets.length;
  const idx = state.nearIdx;
  const p = paintingOf(state.secrets[idx]);
  const k = state.foundSet.size;
  ui.capKicker.textContent = n > 1 ? `${k} of ${n} found` : "Found";
  ui.capTitle.textContent = p.title;
  if (p.thumb) {
    if (!ui.capImg.src.endsWith(p.thumb)) ui.capImg.src = p.thumb;
    ui.capImg.alt = p.title;
    ui.capImg.hidden = false;
  } else {
    ui.capImg.hidden = true;
  }
  ui.capArtist.textContent = [p.artist, p.year].filter(Boolean).join(", ");
  ui.capArtist.hidden = !ui.capArtist.textContent;

  let next = "", action = "", sceneLink = null;
  if (n > 1 && k < n) {
    next = n - k === 1 ? "There’s another painting hiding here." : `${n - k} more paintings are hiding here.`;
    action = "Find the next one";
  } else if (n > 1) {
    next = n === 2 ? "You found both." : `You found all ${n}.`;
    action = "See them again";
  } else {
    action = "";
  }
  if ((k >= n) && sceneList.length > 1) {
    const i = sceneList.findIndex((s) => s.name === sceneName);
    const nx = sceneList[(i + 1) % sceneList.length];
    if (nx && nx.name !== sceneName) sceneLink = nx;
  }
  ui.capNext.textContent = next;
  ui.capNext.hidden = !next;
  ui.capAction.textContent = action ? `${action} →` : "";
  ui.capAction.hidden = !action || !!sceneLink;
  if (sceneLink) {
    ui.capScene.href = sceneHref(sceneLink.name);
    ui.capScene.textContent = `Next scene: ${sceneLink.title || sceneLink.name} →`;
    ui.capScene.hidden = false;
  } else {
    ui.capScene.hidden = true;
  }
  ui.caption.classList.toggle("no-more", !next && !action && !sceneLink);
  layoutCaption();
}

function updateMeter(c, dt) {
  state.heat += (c.heat - state.heat) * (1 - Math.exp(-dt * 8));
  const pct = Math.round(state.heat * 100);
  const h = state.heat;
  // Thin ring: fills and warms from blue to amber as you get closer.
  const rgb = COLD_RGB.map((v, i) => Math.round(v + (WARM_RGB[i] - v) * smoothstep(0.3, 0.9, h)));
  ui.ringFill.style.strokeDashoffset = String(RING_C * (1 - h));
  ui.ringFill.style.stroke = `rgb(${rgb.join(",")})`;
  ui.stage.style.setProperty("--glow", (smoothstep(0.55, 1, h) * 0.9).toFixed(3));
  ui.meter.setAttribute("aria-valuenow", String(pct));
  const before = state.foundSet.size;
  if (c.found) state.foundSet.add(c.idx);
  const newlyFound = state.foundSet.size !== before;
  if (c.found !== state.found || (c.found && c.idx !== state.captionIdx) || newlyFound) {
    state.found = c.found;
    state.captionIdx = c.found ? c.idx : -1;
    document.body.classList.toggle("found", c.found);
    if (c.found) updateCaption();
    if (c.found && !state.scrub) {
      // The slider now leads to this painting and sits at its end.
      setScrubIdx(c.idx);
      setScrubValue(1);
      lastScrubT = 1;
    }
    updateRevealLabel();
    updateRailProgress();
  }
  const hint = c.found ? foundHint() : HINT;
  if (ui.hint.textContent !== hint) ui.hint.textContent = hint;
  const label = c.found ? "Found" : h > 0.8 ? "Hot" : h > 0.62 ? "Warmer" : h > 0.4 ? "Cool" : "Cold";
  if (ui.meterLabel.textContent !== label) ui.meterLabel.textContent = label;
}

// ---------------------------------------------------------------------------
// Loading a scene
// ---------------------------------------------------------------------------

function initOrbit() {
  orbit = new OrbitControls(camera, canvas); // reads camera.up now
  orbit.enableDamping = true;
  orbit.dampingFactor = ORBIT_DAMPING;
  orbit.rotateSpeed = ORBIT_ROTATE;
  orbit.zoomSpeed = ORBIT_ZOOM;
  orbit.panSpeed = 0.8;
  orbit.zoomToCursor = false; // zoom toward the centre
  orbit.screenSpacePanning = true;
  const D = state.secretDist;
  const radii = [D, state.start.position.distanceTo(state.center), ...state.secrets.map((s) => s.position.distanceTo(state.center))];
  orbit.minDistance = Math.min(0.15 * D, ...radii.map((r) => 0.9 * r));
  orbit.maxDistance = Math.max(3 * D, ...radii.map((r) => 1.3 * r));
  // Keep panning from losing the scene: the target stays within one secret distance of the centre.
  orbit.cursor.copy(state.center);
  orbit.maxTargetRadius = D;

  orbit.addEventListener("start", () => {
    state.orbitLive = true;
    state.dragging = true;
    state.interactD0 = state.lastC ? state.lastC.d : Infinity;
    detachScrub();
  });
  orbit.addEventListener("end", () => {
    state.dragging = false;
  });
}

async function loadScene() {
  let list = [];
  try {
    list = (await fetchJSON("scenes/index.json")).scenes || [];
  } catch (e) {
    console.warn("Could not read scenes/index.json", e);
  }
  const name = params.get("scene") || list[0]?.name;
  sceneList = list;
  sceneName = name || "";
  if (!name) throw new Error("No scene given and scenes/index.json is empty.");

  const base = `scenes/${encodeURIComponent(name)}/`;
  let meta;
  try {
    meta = await fetchJSON(`${base}scene.json`);
  } catch (e) {
    buildSceneRail(list, name, null).catch((err) => console.warn(err));
    throw e;
  }
  if (meta.convention && meta.convention !== "opencv") {
    console.warn(`Unexpected camera convention "${meta.convention}", treating as opencv.`);
  }
  state.meta = meta;
  document.title = `${meta.title || meta.name} · Hidden in Plain Sight`;
  ui.title.textContent = meta.title || meta.name;

  const bg = meta.background || [0, 0, 0];
  const bgColor = new THREE.Color().setRGB(bg[0], bg[1], bg[2], THREE.SRGBColorSpace);
  scene.background = bgColor;
  document.documentElement.style.setProperty("--stage-bg", `#${bgColor.getHexString(THREE.SRGBColorSpace)}`);
  const k = list.findIndex((e) => e.name === name);
  ui.stepNum.textContent = String(k >= 0 ? k + 1 : 1).padStart(2, "0");

  // World up from scene.json (synthetic scenes are y-down). Spark's and three's controls read camera.up.
  const up = new THREE.Vector3().fromArray(meta.up || [0, 1, 0]).normalize();
  camera.up.copy(up);
  upQuat.setFromUnitVectors(up, _Y);
  upQuatInv.copy(upQuat).invert();

  state.center.fromArray(meta.center || [0, 0, 0]);
  const spots = secretsOf(meta);
  state.secrets = spots.map((sp) => Object.assign(poseFromC2W(sp.c2w, sp.fov_y_deg),
    { aspect: sp.aspect || 1, label: sp.label || "", image: sp.image || "" }));
  for (const i of foundStore.get(name)) if (i < state.secrets.length) state.foundSet.add(i);
  state.secret = state.secrets[0];
  state.secretAspect = state.secret.aspect;
  state.revealIdx = 0;
  updateRevealLabel();
  buildSceneRail(list, name, meta).catch((err) => console.warn("Scene rail:", err));
  state.start = meta.start ? poseFromC2W(meta.start.c2w, meta.start.fov_y_deg ?? meta.secret.fov_y_deg) : state.secret;
  state.secretDist = Math.max(1e-6, state.secret.position.distanceTo(state.center));

  // Scale free-fly speeds to the scene.
  const s = state.secretDist / 5;
  controls.fpsMovement.moveSpeed = state.secretDist / 4;
  const pc = controls.pointerControls;
  pc.slideSpeed *= s;
  pc.scrollSpeed *= s;
  if (pc.pressMoveSpeed) pc.pressMoveSpeed *= s;
  if (pc.doublePressMoveSpeed) pc.doublePressMoveSpeed *= s;
  if (pc.triplePressMoveSpeed) pc.triplePressMoveSpeed *= s;
  pc.rotateInertia = 0.08; // less glide after letting go of a look-drag (default 0.15 s overshoots)

  initOrbit();
  applyPose(state.start);
  setOrbitTargetFromView();

  // Scrub path (start -> each spot), in orbit space.
  startOrbit = toOrbit(state.start);
  spotOrbits = state.secrets.map((sp) => toOrbit(sp));
  setScrubIdx(0);
  setScrubValue(0);
  updateRailProgress();
  ui.scrubSpot.disabled = state.secrets.length < 2;
  ui.scrubSpot.classList.toggle("single", state.secrets.length < 2);
  ui.scrub.hidden = state.secrets.every((sp) => sp.position.distanceTo(state.start.position) < 1e-6);

  setMode(state.mode);

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
  handBack();
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/** Index of the painting "Show me a painting" will fly to next (mirrors revealNext). */
function nextRevealIdx() {
  const n = state.secrets.length;
  if (!n) return 0;
  return state.found && n > 1 ? (state.nearIdx + 1) % n : state.revealIdx % n;
}

function updateRevealLabel() {
  const n = state.secrets.length;
  const label = n < 2 ? "Show me the painting" : state.found ? "Show me the next one" : "Show me a painting";
  if (ui.revealLabel.textContent !== label) ui.revealLabel.textContent = label;
  const count = n > 1 ? `${nextRevealIdx() + 1} of ${n}` : "";
  if (ui.revealCount.textContent !== count) ui.revealCount.textContent = count;
  ui.revealCount.hidden = !count;
}

function revealNext() {
  if (!state.ready) return;
  const n = state.secrets.length;
  // If we are already at a spot, go to the next one; otherwise go to the next in order.
  if (state.found && n > 1) state.revealIdx = (state.nearIdx + 1) % n;
  const idx = state.revealIdx % n;
  const target = state.secrets[idx];
  state.revealIdx = (state.revealIdx + 1) % n;
  updateRevealLabel();
  setScrubIdx(idx);
  flyTo(target, {
    seconds: REVEAL_SECONDS,
    onDone: () => {
      setScrubValue(1);
      lastScrubT = 1;
    },
  });
}

function resetView() {
  if (!state.ready) return;
  flyTo(state.start, {
    seconds: 1.6,
    onDone: () => {
      setScrubValue(0);
      lastScrubT = 0;
    },
  });
}

ui.reveal.addEventListener("click", revealNext);
ui.capAction.addEventListener("click", revealNext);
ui.reset.addEventListener("click", resetView);
ui.helpBtn.addEventListener("click", () => openHelp(ui.help.hidden));
ui.helpClose.addEventListener("click", () => openHelp(false));
ui.helpGo.addEventListener("click", () => openHelp(false));
ui.helpForget.addEventListener("click", () => {
  foundStore.clear();
  location.reload();
});
ui.help.addEventListener("click", (e) => {
  if (e.target === ui.help) openHelp(false);
});
ui.modeOrbit.addEventListener("click", () => setMode("orbit"));
ui.modeFly.addEventListener("click", () => setMode("fly"));

ui.scrubRange.addEventListener("input", () => scrubTo(Number(ui.scrubRange.value) / SCRUB_MAX));
ui.scrubRange.addEventListener("change", () => ui.scrubRange.blur());
ui.scrubSpot.addEventListener("click", () => {
  const n = state.secrets.length;
  if (n < 2) return;
  setScrubIdx((state.scrubIdx + 1) % n);
  // Re-aim the slider: glide onto the new path at the same slider position.
  const t = Number(ui.scrubRange.value) / SCRUB_MAX;
  if (t > 0) {
    state.scrub = null;
    ui.scrub.classList.add("detached");
    scrubTo(t);
  }
});

// Keep buttons from holding keyboard focus (so Space/Enter don't re-trigger them,
// and arrows/WASD don't drive both a focused control and the camera).
for (const b of document.querySelectorAll("#dock button, #dock input, #caption button")) {
  b.addEventListener("pointerup", () => b.blur());
}

// Any pointer or wheel input on the canvas takes over from an automatic move. Capture
// phase, so this runs before OrbitControls'/Spark's own pointerdown handlers, which then
// start the user's gesture cleanly from the current pose.
canvas.addEventListener("pointerdown", () => { stopAuto(); canvas.focus({ preventScroll: true }); }, { capture: true });
canvas.addEventListener("wheel", stopAuto, { capture: true, passive: true });

const FLY_KEYS = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"]);

window.addEventListener("keydown", (e) => {
  if (e.metaKey || e.altKey || e.ctrlKey) return;
  if (e.code === "Escape") {
    openHelp(false);
    return;
  }
  if (e.key === "?" || (e.code === "Slash" && e.shiftKey)) return openHelp(ui.help.hidden);
  if (!ui.help.hidden) return;
  if (e.repeat) return;
  if (e.code === "KeyH") revealNext();
  else if (e.code === "KeyR") resetView();
  else if (e.code === "KeyF") setMode(state.mode === "orbit" ? "fly" : "orbit");
  else if (state.mode === "fly" && FLY_KEYS.has(e.code)) stopAuto();
  else if (state.mode === "orbit" && FLY_KEYS.has(e.code) && state.ready && !isAuto()) {
    // WASD from orbit: the visitor wants to walk. Switch to free fly.
    setMode("fly");
  }
});

function onStageResize() {
  const { w, h } = stageSize();
  renderer.setSize(w, h);
  updateProjection();
}
SIDE_LABEL.addEventListener?.("change", () => layoutCaption());
if ("ResizeObserver" in window) new ResizeObserver(onStageResize).observe(ui.stage);
else window.addEventListener("resize", onStageResize);

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

  // Spark: always call update (even when disabled) so its internal clock stays current,
  // and drain input it recorded while disabled.
  if (!controls.pointerControls.enable) syncSparkInput();
  if (controls.update(camera) && state.mode === "fly") detachScrub();

  if (state.anim) stepAnimation(now);
  else if (state.scrub) stepScrub(dt);
  else if (orbit && state.mode === "orbit" && state.orbitLive) {
    // Only drive OrbitControls while the user is using it (or damping settles), so a
    // landed pose is never touched by its lookAt().
    const near = state.lastC ? smoothstep(0.55, 0.9, state.heat) : 0;
    orbit.rotateSpeed = ORBIT_ROTATE * (1 - 0.6 * near); // finer control near a spot
    orbit.zoomSpeed = ORBIT_ZOOM * (1 - 0.7 * near);
    const moved = orbit.update(dt);
    if (!moved && !state.dragging) {
      state.orbitLive = false;
      maybeSnap();
    }
  }

  renderer.render(scene, camera);

  if (state.ready) {
    const c = closeness();
    state.lastC = c;
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
        `mode    ${state.mode}${state.anim ? "  (flying)" : state.scrub ? "  (scrub)" : ""}`,
        `pos     ${p.x.toFixed(3)} ${p.y.toFixed(3)} ${p.z.toFixed(3)}`,
        `quat    ${q.x.toFixed(3)} ${q.y.toFixed(3)} ${q.z.toFixed(3)} ${q.w.toFixed(3)}`,
        `fwd     ${_fwd.x.toFixed(3)} ${_fwd.y.toFixed(3)} ${_fwd.z.toFixed(3)}`,
        `fov     ${camera.fov.toFixed(2)} (nominal ${state.nominalFov.toFixed(2)})`,
        `spot    ${c.idx + 1}/${state.secrets.length} ${state.secret.label || ""}`,
        `posErr  ${(c.posErr * 100).toFixed(2)}%`,
        `angErr  ${c.angErr.toFixed(2)} deg`,
        `heat    ${c.heat.toFixed(3)}${c.found ? "  FOUND" : ""}`,
      ].join("\n");
    }
  }
});

/** After an orbit gesture settles very close to a spot (and closer than it began), ease onto it. */
function maybeSnap() {
  const c = state.lastC;
  if (!SNAP || !c || isAuto()) return;
  if (c.d < SNAP_D && c.d > 0.02 && c.d < state.interactD0) {
    const idx = c.idx;
    flyTo(state.secrets[idx], {
      seconds: 0.6,
      onDone: () => {
        setScrubIdx(idx);
        setScrubValue(1);
        lastScrubT = 1;
      },
    });
  }
}

// Expose for debugging from the console.
window.hips = {
  THREE, camera, scene, spark, controls, state, flyTo, renderer, setMode, scrubTo, revealNext, resetView,
  get orbit() { return orbit; },
};

setMode(state.mode);
try {
  if (!localStorage.getItem(HELP_SEEN) && !params.has("nohelp")) openHelp(true);
} catch (e) { /* no storage: skip the first-visit help */ }
loadScene().catch((err) => {
  console.error(err);
  ui.loading.classList.add("error");
  ui.loadingLabel.textContent = `Could not load scene: ${err.message}`;
});
