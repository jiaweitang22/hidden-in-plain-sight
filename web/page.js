// Hidden in Plain Sight: small enhancements for the write-up page. No dependencies.
// The page reads fine without this script; it adds the interactive diagram, tap-to-peek
// on touch screens, and the top bar's border.

(() => {
  "use strict";

  // ---------- top bar border once scrolled ----------
  const bar = document.querySelector(".topbar");
  const onScroll = () => bar && bar.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // ---------- hero video: make sure it plays (some browsers need muted set from script) ----------
  const hero = document.querySelector(".hero-media video");
  if (hero) {
    hero.muted = true;
    const p = hero.play && hero.play();
    if (p && p.catch) p.catch(() => { hero.controls = true; });
  }

  // ---------- gallery: on touch screens, first tap peeks at the hidden picture ----------
  const touch = window.matchMedia("(hover: none)").matches;
  if (touch) {
    for (const card of document.querySelectorAll(".card")) {
      card.querySelector(".pics")?.addEventListener("click", (e) => {
        if (!card.classList.contains("peek")) {
          e.preventDefault();
          for (const c of document.querySelectorAll(".card.peek")) c.classList.remove("peek");
          card.classList.add("peek");
        }
      });
    }
  }

  // ---------- "how it works" diagram: move the second camera around the cloud ----------
  // Geometry must match the generator that drew the static SVG (seen from above):
  // secret camera S looks at the cloud centre C; the moving camera orbits C at the same radius.
  const svg = document.getElementById("diagram");
  const slider = document.getElementById("orbit");
  if (!svg || !slider) return;

  const S = [130, 250], C = [600, 250], R = C[0] - S[0];
  const RAY_STEP = 4;   // degrees between neighbouring picture dots
  const HALF = 13.5;    // half field of view shown in the strips (degrees)
  const rad = (d) => (d * Math.PI) / 180;
  const tanH = Math.tan(rad(HALF));
  const pixelW = Math.tan(rad(RAY_STEP)) / tanH; // width of one picture dot, in strip units

  const specks = [...svg.querySelectorAll(".specks circle")].map((c) => ({
    x: +c.getAttribute("cx"),
    y: +c.getAttribute("cy"),
    t: +c.dataset.t,
    fill: c.getAttribute("fill"),
  }));

  const camAt = (phi) => [C[0] - R * Math.cos(rad(phi)), C[1] + R * Math.sin(rad(phi))];

  function stripMarkup(phi) {
    const P = camAt(phi);
    let fx = C[0] - P[0], fy = C[1] - P[1];
    const n = Math.hypot(fx, fy);
    fx /= n; fy /= n;
    const items = specks.map((s) => {
      const dx = s.x - P[0], dy = s.y - P[1];
      const z = dx * fx + dy * fy;
      const lat = -dx * fy + dy * fx;
      return { z, u: lat / z / tanH, w: pixelW * 0.55, fill: s.fill }; // fixed tick width: specks are tiny, so near ones do not hide the rest
    });
    items.sort((a, b) => b.z - a.z); // far first, near specks drawn on top
    return items
      .map((it) => `<rect x="${(it.u - it.w / 2).toFixed(4)}" y="0" width="${it.w.toFixed(4)}" height="1" fill="${it.fill}"/>`)
      .join("");
  }

  const camG = document.getElementById("cam2-pos");
  const fovG = document.getElementById("cam2-fov");
  const lbl = document.getElementById("cam2-lbl");
  const youStrip = document.querySelector(".strip.you");
  const card = svg.closest(".diagram-card");
  const status = document.getElementById("orbit-status");
  const snap = document.getElementById("orbit-snap");

  function draw(phi) {
    const P = camAt(phi);
    const a = Math.atan2(C[1] - P[1], C[0] - P[0]);
    camG.setAttribute("transform", `translate(${P[0].toFixed(1)},${P[1].toFixed(1)}) rotate(${((a * 180) / Math.PI).toFixed(2)})`);
    const L = 560, h = rad(HALF);
    const lines = fovG.querySelectorAll("line");
    [-1, 1].forEach((sg, k) => {
      lines[k].setAttribute("x1", P[0].toFixed(1));
      lines[k].setAttribute("y1", P[1].toFixed(1));
      lines[k].setAttribute("x2", (P[0] + L * Math.cos(a + sg * h)).toFixed(1));
      lines[k].setAttribute("y2", (P[1] + L * Math.sin(a + sg * h)).toFixed(1));
    });
    // Keep the "you" label below the camera when it is low, above it when it is high.
    lbl.setAttribute("x", P[0].toFixed(1));
    lbl.setAttribute("y", (phi >= -4 ? P[1] + 46 : P[1] - 30).toFixed(1));
    lbl.style.opacity = Math.abs(phi) < 3 ? 0 : 1; // overlaps the secret camera's label at the spot
    youStrip.innerHTML = stripMarkup(phi);
    const found = Math.abs(phi) < 0.75;
    card.classList.toggle("found", found);
    const deg = Math.abs(phi);
    status.textContent = found ? "Found it: the picture." : deg < 4 ? "Very close…" : deg < 10 ? "Getting warmer" : "Scrambled";
  }

  slider.addEventListener("input", () => draw(+slider.value));

  let anim = 0;
  snap?.addEventListener("click", () => {
    cancelAnimationFrame(anim);
    const from = +slider.value, t0 = performance.now(), dur = 1600;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const step = (now) => {
      const r = reduce ? 1 : Math.min(1, (now - t0) / dur);
      const e = r < 0.5 ? 4 * r * r * r : 1 - Math.pow(-2 * r + 2, 3) / 2;
      const v = from * (1 - e);
      slider.value = v.toFixed(1);
      draw(r >= 1 ? 0 : v);
      if (r < 1) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  });

  draw(+slider.value);
})();
