"""Cut a portrait out of a dark background: dark pixels connected to the border become transparent."""
import sys
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

src, dst, thr = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.16
im = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255
lum = im @ np.array([0.299, 0.587, 0.114])
dark = lum < thr
lab, _ = ndimage.label(dark)
border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
bg = np.isin(lab, border[border > 0])
bg = ndimage.binary_opening(bg, iterations=2)
fg = ~bg
fg = ndimage.binary_fill_holes(ndimage.binary_closing(fg, iterations=3))
keep, n = ndimage.label(fg)
if n > 1:  # keep the largest component
    sizes = ndimage.sum(fg, keep, range(1, n + 1)); fg = keep == (1 + int(np.argmax(sizes)))
alpha = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
out = Image.fromarray((im * 255).astype(np.uint8)).convert("RGBA"); out.putalpha(alpha); out.save(dst)
print("foreground fraction", round(float(fg.mean()), 3))
