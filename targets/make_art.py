"""Square crops of the public-domain paintings (two-view pairs need ink in the same rows)."""
from PIL import Image
S = 768
def crop(src, dst, box_frac):
    im = Image.open(src).convert("RGB"); w, h = im.size
    x0, y0, side = box_frac  # left, top, side as fractions of width (side) -- square crop
    s = int(side * min(w, h) if side <= 1 else side); x = int(x0 * w); y = int(y0 * h)
    im.crop((x, y, x + s, y + s)).resize((S, S), Image.LANCZOS).save(dst)
crop("targets/art/great_wave.jpg", "targets/art_great_wave.png", (0.02, 0.0, 1.0))
crop("targets/art/starry_night.jpg", "targets/art_starry_night.png", (0.10, 0.0, 1.0))
crop("targets/art/pearl_earring.jpg", "targets/art_pearl_earring.png", (0.0, 0.08, 1.0))
crop("targets/art/mona_lisa.jpg", "targets/art_mona_lisa.png", (0.0, 0.02, 1.0))
print("ok")
