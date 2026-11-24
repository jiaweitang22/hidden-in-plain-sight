"""Procedural test targets (RGBA; transparent pixels get no Gaussians)."""
import math
from PIL import Image, ImageDraw, ImageFont
S = 768
F = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

def font(sz):
    try: return ImageFont.truetype(F, sz)
    except OSError: return ImageFont.load_default(size=sz)

# 1. flat-colour smiley (the blob test)
im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
d.ellipse([134, 134, 634, 634], fill=(250, 200, 40, 255))
d.ellipse([270, 280, 330, 370], fill=(25, 25, 25, 255)); d.ellipse([438, 280, 498, 370], fill=(25, 25, 25, 255))
d.arc([250, 330, 518, 540], 20, 160, fill=(25, 25, 25, 255), width=26)
im.save("targets/smiley.png")

# 2. text (high frequency, thin strokes)
im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
for i, (txt, sz) in enumerate([("HIDDEN", 150), ("IN PLAIN", 120), ("SIGHT", 160)]):
    w = d.textlength(txt, font=font(sz)); d.text(((S - w) / 2, 110 + i * 190), txt, font=font(sz), fill=(255, 255, 255, 255))
im.save("targets/text.png")

# 3. line-art eye (thin coloured strokes)
im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
pts_top = [(100 + t * 568, 384 - 210 * math.sin(math.pi * t)) for t in [i / 60 for i in range(61)]]
pts_bot = [(100 + t * 568, 384 + 190 * math.sin(math.pi * t)) for t in [i / 60 for i in range(61)]]
d.line(pts_top, fill=(240, 240, 255, 255), width=14); d.line(pts_bot, fill=(240, 240, 255, 255), width=14)
for r, c in [(150, (60, 150, 255, 255)), (110, (30, 90, 200, 255))]:
    d.ellipse([384 - r, 384 - r, 384 + r, 384 + r], outline=c, width=18)
d.ellipse([334, 334, 434, 434], fill=(15, 15, 30, 255))
for k in range(24):
    a = 2 * math.pi * k / 24
    d.line([(384 + 160 * math.cos(a), 384 + 160 * math.sin(a)), (384 + 120 * math.cos(a), 384 + 120 * math.sin(a))], fill=(120, 200, 255, 255), width=5)
for k in range(9):
    t = 0.1 + 0.8 * k / 8; x = 100 + t * 568; y = 384 - 210 * math.sin(math.pi * t)
    d.line([(x, y), (x + (t - 0.5) * 60, y - 60)], fill=(240, 240, 255, 255), width=8)
im.save("targets/eye.png")
print("ok")
