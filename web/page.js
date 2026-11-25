// Hidden in Plain Sight: enhancements for the write-up page. No dependencies.
// Builds the gallery cards and the results table from scenes/index.json + each
// scene's scene.json, swaps in the hero video when one is supplied, highlights the
// current section in the nav, and drives the "how it works" diagram.

(() => {
  "use strict";

  // ---------- top bar: border once scrolled, highlight the section in view ----------
  const bar = document.querySelector(".topbar");
  const onScroll = () => bar && bar.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const navLinks = [...document.querySelectorAll(".topbar nav a[href^='#']")];
  const sections = navLinks.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  if ("IntersectionObserver" in window && sections.length) {
    const visible = new Map();
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) visible.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0);
        let best = null, bestR = 0;
        for (const s of sections) {
          const r = visible.get(s.id) || 0;
          if (r > bestR) { best = s.id; bestR = r; }
        }
        for (const a of navLinks) a.classList.toggle("active", best !== null && a.getAttribute("href") === `#${best}`);
      },
      { rootMargin: "-30% 0px -55% 0px", threshold: [0, 0.01, 0.2, 0.5, 1] }
    );
    sections.forEach((s) => io.observe(s));
  }

  // ---------- hero: use the video if one has been supplied (see the HOOK in index.html) ----------
  const video = document.getElementById("hero-video");
  if (video && (video.getAttribute("src") || video.querySelector("source"))) {
    video.hidden = false;
    video.closest(".hero-stage")?.classList.add("has-video");
    video.muted = true;
    video.autoplay = true;
    const p = video.play && video.play();
    if (p && p.catch) p.catch(() => { video.controls = true; });
  }

  // ---------- gallery + results table, from scenes/index.json ----------
  const painting = (s) =>
    window.hipsPainting ? window.hipsPainting(s) : { title: s?.label || "A hidden painting", artist: "", year: "", thumb: "" };
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const fetchJSON = (url) => fetch(url, { cache: "no-cache" }).then((r) => {
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return r.json();
  });
  const secretsOf = (meta) =>
    window.hipsSecrets ? window.hipsSecrets(meta) : meta ? (meta.secrets && meta.secrets.length ? meta.secrets : meta.secret ? [meta.secret] : []) : [];
  // Paintings stay "?" in the gallery cards until the visitor has found them in the viewer.
  const foundIn = (name) => (window.hipsFound ? window.hipsFound.get(name) : new Set());
  const plural = (n) => (n === 1 ? "1 painting hidden" : `${n} paintings hidden`);
  const surname = (artist) => {
    if (!artist) return "";
    const parts = artist.split(" ");
    return /da Vinci$/.test(artist) ? "Leonardo" : /van Gogh$/.test(artist) ? "Van Gogh" : parts[parts.length - 1];
  };

  function sceneCard(entry, meta) {
    const secrets = secretsOf(meta);
    const found = foundIn(entry.name);
    const nFound = secrets.filter((_, i) => found.has(i)).length;
    const allFound = secrets.length > 0 && nFound >= secrets.length;
    const a = el("a", "scene-card");
    a.href = `viewer.html?scene=${encodeURIComponent(entry.name)}`;

    const visual = el("div", "sc-visual");
    if (entry.poster) {
      const img = el("img", "sc-poster");
      img.src = entry.poster;
      img.alt = "";
      img.loading = "lazy";
      img.addEventListener("error", () => img.remove());
      visual.append(img);
    }
    const row = el("div", `sc-paintings n${Math.min(Math.max(secrets.length, 1), 4)}`);
    (secrets.length ? secrets.slice(0, 4) : [{}]).forEach((s, i) => {
      const p = painting(s);
      const tile = el("span", "sc-tile");
      if (found.has(i) && p.thumb) {
        const img = el("img");
        img.src = p.thumb;
        img.alt = "";
        img.loading = "lazy";
        img.addEventListener("error", () => { img.remove(); tile.classList.add("blank"); });
        tile.append(img);
      } else {
        tile.classList.add("blank");
      }
      row.append(tile);
    });
    visual.append(row);

    const body = el("div", "sc-body");
    body.append(el("p", "sc-count",
      !secrets.length ? "Hidden paintings" : nFound === 0 ? plural(secrets.length) : allFound ? "All found" : `${nFound} of ${secrets.length} found`));
    body.append(el("h3", null, entry.title || meta?.title || entry.name));
    if (secrets.length) {
      const ul = el("ul", "sc-list");
      secrets.forEach((s, i) => {
        const li = el("li");
        if (found.has(i)) {
          const p = painting(s);
          li.append(el("i", null, p.title));
          if (p.artist) li.append(document.createTextNode(` · ${surname(p.artist)}`));
        } else {
          li.className = "unknown";
          li.textContent = `Painting ${i + 1}: not found yet`;
        }
        ul.append(li);
      });
      body.append(ul);
    }
    // Blurbs name the artists, so they wait until every painting in the scene is found.
    if (entry.blurb && allFound) body.append(el("p", "sc-blurb", entry.blurb));
    const go = el("span", "sc-go", "Enter the scene ");
    go.append(Object.assign(el("span", null, "→"), { ariaHidden: "true" }));
    body.append(go);

    a.append(visual, body);
    a.setAttribute("aria-label", `${entry.title || entry.name}: ${secrets.map((s, i) => (found.has(i) ? painting(s).title : "a hidden painting")).join(", ")}`);
    return a;
  }

  function metricsRow(entry, meta) {
    const secrets = secretsOf(meta);
    const m = (meta && meta.metrics) || {};
    const keys = "abcdefgh".split("");
    const scores = secrets
      .map((s, i) => ({ name: s.label || painting(s).title, v: m[`ssim_${keys[i]}`] }))
      .filter((x) => typeof x.v === "number");
    if (!scores.length) return null;
    const tr = el("tr");
    tr.append(el("td", null, entry.title || meta.title || entry.name));
    const n = meta.num_gaussians;
    tr.append(el("td", null, n ? `${Math.round(n / 1000)}k` : "–"));
    const td = el("td");
    scores.forEach((x, i) => {
      if (i) td.append(document.createTextNode(" · "));
      td.append(el("i", null, x.name), document.createTextNode(" "), el("span", "score", x.v.toFixed(2)));
    });
    tr.append(td);
    return tr;
  }

  async function buildGallery() {
    const grid = document.getElementById("scene-grid");
    if (!grid) return;
    let list = [];
    try {
      list = (await fetchJSON("scenes/index.json")).scenes || [];
    } catch (e) {
      console.warn("Could not read scenes/index.json; keeping the plain link.", e);
      return;
    }
    if (!list.length) return;
    const metas = await Promise.all(
      list.map((s) => fetchJSON(`scenes/${encodeURIComponent(s.name)}/scene.json`).catch(() => null))
    );
    grid.replaceChildren(...list.map((s, i) => sceneCard(s, metas[i])));

    const tbody = document.querySelector("#metrics-table tbody");
    if (tbody) {
      const rows = list.map((s, i) => metricsRow(s, metas[i])).filter(Boolean);
      if (rows.length) tbody.replaceChildren(...rows);
    }
  }
  buildGallery();

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
