#!/usr/bin/env python3
"""logo_finder._edges (sliced Sobel, integer for 8-bit gray) must give what ndimage.sobel
gives, borders included, since check_frame runs it on cropped boxes."""
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import logo_finder as lf

rng = np.random.default_rng(7)
failed = 0
cases = []
for shape in [(120, 200), (37, 41), (2, 5), (5, 2), (2, 2), (1080, 1920)]:
    cases.append(("noise", rng.integers(0, 256, shape, dtype=np.uint8)))
    smooth = np.cumsum(rng.integers(-2, 3, shape), axis=1).clip(0, 255).astype(np.uint8)
    smooth[shape[0]//3:, shape[1]//2:] = 255 - smooth[shape[0]//3:, shape[1]//2:]
    cases.append(("steps", smooth))
cases.append(("black/white", np.where(np.indices((64, 64)).sum(0) % 16 < 8, 0, 255).astype(np.uint8)))
# contrast steps either side of the threshold
for step in (10, 11, 12, 13, 14, 48, 49):
    x = np.full((30, 30), 100, np.uint8)
    x[:, 15:] = 100 + step
    cases.append((f"step {step}", x))

for name, img in cases:
    f = img.astype(np.float32)
    want = np.hypot(ndimage.sobel(f, 0), ndimage.sobel(f, 1)) / 4 > lf._LOGO_EDGE_THRESHOLD
    got = lf._edges(img)
    if got.shape != want.shape or (got != want).any():
        print(f"FAIL {name} {img.shape}: {np.count_nonzero(got != want)} pixels differ")
        failed += 1
# non-integer gray (what packed pixel formats give); no ties to hit, so float rounding can't matter
img = rng.uniform(0, 255, (90, 130))
f = img.astype(np.float32)
want = np.hypot(ndimage.sobel(f, 0), ndimage.sobel(f, 1)) / 4 > lf._LOGO_EDGE_THRESHOLD
if (lf._edges(img) != want).any():
    print("FAIL float input")
    failed += 1
if failed:
    sys.exit(1)
print("ok")
