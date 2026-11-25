// Hidden in Plain Sight: catalogue of the public-domain paintings hidden in our scenes.
// Shared by the viewer (caption card, scene rail) and the write-up page (gallery cards).
//
// Keyed by the exact secrets[].label strings in each scenes/<name>/scene.json.
// A scene.json secret may also carry its own "title", "artist", "year", "thumb" or
// "source" fields; those win over this table. Labels not listed here still work:
// the caption then shows the label only, and thumbnails fall back to a plain tile.
//
// Sources: targets/art/SOURCES.md (all public domain, Wikimedia Commons).

window.HIPS_PAINTINGS = {
  "The Great Wave": {
    title: "The Great Wave off Kanagawa",
    artist: "Katsushika Hokusai",
    year: "c. 1831",
    thumb: "media/art_great_wave.jpg",
    source: "https://commons.wikimedia.org/wiki/File:Tsunami_by_hokusai_19th_century.jpg",
  },
  "The Starry Night": {
    title: "The Starry Night",
    artist: "Vincent van Gogh",
    year: "1889",
    thumb: "media/art_starry_night.jpg",
    source: "https://commons.wikimedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
  },
  "Girl with a Pearl Earring": {
    title: "Girl with a Pearl Earring",
    artist: "Johannes Vermeer",
    year: "c. 1665",
    thumb: "media/art_pearl_earring.jpg",
    source: "https://commons.wikimedia.org/wiki/File:1665_Girl_with_a_Pearl_Earring.jpg",
  },
  "Mona Lisa": {
    title: "Mona Lisa",
    artist: "Leonardo da Vinci",
    year: "c. 1503–1519",
    thumb: "media/art_mona_lisa.jpg",
    source: "https://commons.wikimedia.org/wiki/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg",
  },
};

/** Painting info for one scene.json secret: its own fields, then the catalogue, then the label. */
window.hipsPainting = function hipsPainting(secret) {
  const s = secret || {};
  const cat = (s.label && window.HIPS_PAINTINGS[s.label]) || {};
  return {
    label: s.label || "",
    title: s.title || cat.title || s.label || "A hidden picture",
    artist: s.artist || cat.artist || "",
    year: s.year || cat.year || "",
    thumb: s.thumb || cat.thumb || "",
    source: s.source || cat.source || "",
  };
};
