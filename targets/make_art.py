"""Square 768 px crops of the public-domain paintings in targets/art/ (sources: targets/art/SOURCES.md).

Square crops matter for the multi-painting clouds: every painting must cover the same image rows.
  .venv/bin/python targets/make_art.py
"""
from PIL import Image

S = 768
CROPS = {  # name: (left, top) of the square crop as fractions of width/height; side = the short side
    "great_wave": (0.02, 0.0),
    "starry_night": (0.10, 0.0),
    "pearl_earring": (0.0, 0.08),
    "mona_lisa": (0.0, 0.02),
    "grande_jatte": (0.18, 0.0),
    "the_kiss": (0.0, 0.0),
    "the_scream": (0.0, 0.08),
    "birth_of_venus": (0.19, 0.0),
    "impression_sunrise": (0.12, 0.0),
    "night_watch": (0.16, 0.0),
}

for name, (fx, fy) in CROPS.items():
    im = Image.open(f"targets/art/{name}.jpg").convert("RGB")
    w, h = im.size
    s = min(w, h)
    x, y = min(int(fx * w), w - s), min(int(fy * h), h - s)
    im.crop((x, y, x + s, y + s)).resize((S, S), Image.LANCZOS).save(f"targets/art_{name}.png")
print("wrote", len(CROPS), "crops")
