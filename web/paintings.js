// Hidden in Plain Sight: catalogue of the public-domain paintings hidden in our scenes,
// plus a small "which paintings has this visitor found" store.
// Shared by the viewer (caption card, scene rail) and the write-up page (gallery cards).
//
// A scene.json secret is matched to an entry here by its "label", loosely: case,
// punctuation, accents and a leading "the"/"a" are ignored, and each entry lists a few
// aliases ("Grande Jatte", "Impression Sunrise", ...). If the label does not match (for
// example the default "Spot 1"), the painting is matched by the target image the scene
// was trained on (scene.json params.a / params.b / params.targets, e.g.
// "targets/art_the_kiss.png" -> "the_kiss"). A secret may also carry its own "title",
// "artist", "year", "thumb" or "source" fields; those win over this table.
//
// Sources: targets/art/SOURCES.md (all public domain, Wikimedia Commons).

window.HIPS_PAINTINGS = {
  "The Great Wave": {
    slug: "great_wave",
    aliases: ["The Great Wave off Kanagawa", "Great Wave", "Under the Wave off Kanagawa"],
    title: "The Great Wave off Kanagawa",
    artist: "Katsushika Hokusai",
    year: "c. 1831",
    thumb: "media/art_great_wave.jpg",
    source: "https://commons.wikimedia.org/wiki/File:Tsunami_by_hokusai_19th_century.jpg",
  },
  "The Starry Night": {
    slug: "starry_night",
    aliases: ["Starry Night"],
    title: "The Starry Night",
    artist: "Vincent van Gogh",
    year: "1889",
    thumb: "media/art_starry_night.jpg",
    source: "https://commons.wikimedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
  },
  "Girl with a Pearl Earring": {
    slug: "pearl_earring",
    aliases: ["Pearl Earring", "Girl with the Pearl Earring"],
    title: "Girl with a Pearl Earring",
    artist: "Johannes Vermeer",
    year: "c. 1665",
    thumb: "media/art_pearl_earring.jpg",
    source: "https://commons.wikimedia.org/wiki/File:1665_Girl_with_a_Pearl_Earring.jpg",
  },
  "Mona Lisa": {
    slug: "mona_lisa",
    aliases: ["La Gioconda", "La Joconde"],
    title: "Mona Lisa",
    artist: "Leonardo da Vinci",
    year: "c. 1503–1519",
    thumb: "media/art_mona_lisa.jpg",
    source: "https://commons.wikimedia.org/wiki/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg",
  },
  "A Sunday on La Grande Jatte": {
    slug: "grande_jatte",
    aliases: ["Grande Jatte", "La Grande Jatte", "A Sunday Afternoon on the Island of La Grande Jatte", "Sunday on La Grande Jatte"],
    title: "A Sunday on La Grande Jatte",
    artist: "Georges Seurat",
    year: "1884–1886",
    thumb: "media/art_grande_jatte.jpg",
    source: "https://commons.wikimedia.org/wiki/File:A_Sunday_on_La_Grande_Jatte,_Georges_Seurat,_1884.jpg",
  },
  "Impression, Sunrise": {
    slug: "impression_sunrise",
    aliases: ["Impression Sunrise", "Impression, soleil levant", "Sunrise"],
    title: "Impression, Sunrise",
    artist: "Claude Monet",
    year: "1872",
    thumb: "media/art_impression_sunrise.jpg",
    source: "https://commons.wikimedia.org/wiki/File:Claude_Monet,_Impression,_soleil_levant.jpg",
  },
  "The Kiss": {
    slug: "the_kiss",
    aliases: ["Kiss", "Der Kuss", "Lovers"],
    title: "The Kiss",
    artist: "Gustav Klimt",
    year: "1907–1908",
    thumb: "media/art_the_kiss.jpg",
    source: "https://commons.wikimedia.org/wiki/File:The_Kiss_-_Gustav_Klimt_-_Google_Cultural_Institute.jpg",
  },
  "The Scream": {
    slug: "the_scream",
    aliases: ["Scream", "Skrik", "Der Schrei der Natur"],
    title: "The Scream",
    artist: "Edvard Munch",
    year: "1893",
    thumb: "media/art_the_scream.jpg",
    source: "https://commons.wikimedia.org/wiki/File:Edvard_Munch,_1893,_The_Scream,_oil,_tempera_and_pastel_on_cardboard,_91_x_73_cm,_National_Gallery_of_Norway.jpg",
  },
  "The Birth of Venus": {
    slug: "birth_of_venus",
    aliases: ["Birth of Venus", "La nascita di Venere", "Venus"],
    title: "The Birth of Venus",
    artist: "Sandro Botticelli",
    year: "c. 1485",
    thumb: "media/art_birth_of_venus.jpg",
    source: "https://commons.wikimedia.org/wiki/File:Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg",
  },
  "The Night Watch": {
    slug: "night_watch",
    aliases: ["Night Watch", "De Nachtwacht", "Militia Company of District II"],
    title: "The Night Watch",
    artist: "Rembrandt van Rijn",
    year: "1642",
    thumb: "media/art_night_watch.jpg",
    source: "https://commons.wikimedia.org/wiki/File:The_Night_Watch_-_HD.jpg",
  },
};

(() => {
  // "The Great Wave off Kanagawa!" -> "great wave off kanagawa"
  const norm = (s) =>
    String(s || "")
      .normalize("NFD").replace(/[̀-ͯ]/g, "")
      .toLowerCase()
      .replace(/[_\-]+/g, " ")
      .replace(/[^a-z0-9 ]+/g, " ")
      .replace(/^\s*(the|a|an)\s+/, "")
      .replace(/\s+/g, " ")
      .trim();

  const byName = new Map();
  for (const [key, p] of Object.entries(window.HIPS_PAINTINGS)) {
    for (const name of [key, p.title, p.slug, ...(p.aliases || [])]) {
      const k = norm(name);
      if (k && !byName.has(k)) byName.set(k, p);
    }
  }

  /** Catalogue entry for a label or an image path such as "targets/art_the_kiss.png". */
  function lookup(name) {
    if (!name) return null;
    const direct = byName.get(norm(name));
    if (direct) return direct;
    const file = String(name).split(/[\\/]/).pop().replace(/\.[a-z0-9]+$/i, "").replace(/^art_/, "");
    return byName.get(norm(file)) || null;
  }

  /**
   * Painting info for one scene.json secret: its own fields, then the catalogue (by label,
   * then by the target image it was trained on, `image`), then the label itself.
   */
  window.hipsPainting = function hipsPainting(secret, image) {
    const s = secret || {};
    const cat = lookup(s.label) || lookup(s.image || image) || {};
    return {
      label: s.label || "",
      title: s.title || cat.title || s.label || "A hidden picture",
      artist: s.artist || cat.artist || "",
      year: s.year || cat.year || "",
      thumb: s.thumb || cat.thumb || "",
      source: s.source || cat.source || "",
    };
  };

  /** The secret spots of a scene.json, each tagged with the target image it was trained on. */
  window.hipsSecrets = function hipsSecrets(meta) {
    if (!meta) return [];
    const list = meta.secrets && meta.secrets.length ? meta.secrets : meta.secret ? [meta.secret] : [];
    const p = meta.params || {};
    const images = Array.isArray(p.targets) ? p.targets : [p.a, p.b, p.c, p.d].filter(Boolean);
    return list.map((s, i) => (s.image || !images[i] ? s : { ...s, image: images[i] }));
  };

  // ---- Which paintings this visitor has found, per scene (localStorage; optional). ----
  // Paintings stay hidden ("?") everywhere until found in the viewer.
  const KEY = "hips.found.v1";
  const readAll = () => {
    try {
      const v = JSON.parse(localStorage.getItem(KEY) || "{}");
      return v && typeof v === "object" ? v : {};
    } catch (e) {
      return {};
    }
  };
  window.hipsFound = {
    /** Set of found spot indices for a scene. */
    get(scene) {
      const v = readAll()[scene];
      return new Set(Array.isArray(v) ? v.filter(Number.isInteger) : []);
    },
    /** Record found spot indices for a scene (merged with what is stored). */
    add(scene, indices) {
      try {
        const all = readAll();
        const set = new Set([...(all[scene] || []), ...indices]);
        all[scene] = [...set].sort((a, b) => a - b);
        localStorage.setItem(KEY, JSON.stringify(all));
      } catch (e) { /* private mode etc.: found state just isn't remembered */ }
    },
    /** Forget everything found (all scenes). */
    clear() {
      try { localStorage.removeItem(KEY); } catch (e) { /* ignore */ }
    },
  };
})();
