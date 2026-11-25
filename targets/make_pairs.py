"""Target pairs for two-view anamorphosis (ink in similar rows in both pictures)."""
import math
from PIL import Image, ImageDraw, ImageFont
S = 768
def font(sz, f="/System/Library/Fonts/Supplemental/Arial Black.ttf"):
    try: return ImageFont.truetype(f, sz)
    except OSError: return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", sz)
def word(txt, col, path, sz=230):
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    f = font(sz)
    while d.textlength(txt, font=f) > S * 0.92: sz -= 8; f = font(sz)
    bb = d.textbbox((0, 0), txt, font=f); w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((S - w) / 2 - bb[0], (S - h) / 2 - bb[1]), txt, font=f, fill=col)
    im.save(path)
word("CS", (255, 205, 60, 255), "targets/pair_cs.png", 420)
word("180", (90, 180, 255, 255), "targets/pair_180.png", 420)
word("LOOK", (255, 255, 255, 255), "targets/pair_look.png")
word("AGAIN", (255, 90, 120, 255), "targets/pair_again.png")
# heart and star (solid shapes spanning similar rows)
im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
pts = []
for k in range(400):
    t = 2 * math.pi * k / 400
    x = 16 * math.sin(t) ** 3; y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
    pts.append((S / 2 + 19 * x, S / 2 - 19 * y - 20))
d.polygon(pts, fill=(235, 60, 90, 255)); im.save("targets/pair_heart.png")
im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
pts = [(S / 2 + (300 if k % 2 == 0 else 125) * math.sin(math.pi * k / 5), S / 2 + 20 - (300 if k % 2 == 0 else 125) * math.cos(math.pi * k / 5)) for k in range(10)]
d.polygon(pts, fill=(255, 210, 70, 255)); im.save("targets/pair_star.png")
print("ok")
