#!/usr/bin/env python3
"""Checks for the logo search, especially the stuck-overlay path.

`_analyze` is a pure function of (logo_sum, fcount, shape), so it can be tested against
saved persistence maps without any video.  That matters here: the recordings these came
from will be deleted eventually, and several of them are the only examples we have of a
real weather alert.

`fixtures/*.npz` hold those maps, scrubbed of on-screen text (see README) and trimmed of
everything below .40 of the frame count --
nothing under that can reach a mask (the .55 gate less the .15 band) or a stuck core
(.60), so the trim cannot change a decision, and it takes the set from 29MB to 0.6MB.
`fixtures/expected.json` records what each one produced, and for nine of them what a human
saw when they watched the video.

Run it directly; it exits non-zero if anything fails.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import logo_finder as lf
from pycommflag.feature_span import SceneType

_HERE = os.path.dirname(os.path.abspath(__file__))
# The committed fixtures have had on-screen text scrubbed and one case dropped, because a
# persistence map renders as a readable picture and these came off someone's recordings.
# A full-fidelity copy can live in tests/private (gitignored) or wherever
# PYCF_TEST_FIXTURES points; it is preferred when present and must never be published.
_PRIVATE = os.environ.get('PYCF_TEST_FIXTURES') or os.path.join(_HERE, 'private')
FX = _PRIVATE if os.path.exists(os.path.join(_PRIVATE, 'expected.json')) \
     else os.path.join(_HERE, 'fixtures')
STUCK_PERSIST = .85   # mirrors the constant inside _find_stuck

_fails = []
print(f'fixtures: {os.path.relpath(FX, _HERE)}'
      + ('  (full fidelity)' if FX == _PRIVATE else '  (public, scrubbed)'))

def check(name, ok, detail=''):
    print(('  pass  ' if ok else '  FAIL  ') + name + (f'   {detail}' if detail else ''))
    if not ok:
        _fails.append(name)

def fixtures():
    exp = json.load(open(os.path.join(FX, 'expected.json')))
    for name in sorted(exp):
        z = np.load(os.path.join(FX, name + '.npz'))
        yield (name, z['persist'].astype(np.uint32), int(z['fcount']),
               tuple(int(x) for x in z['shape']), exp[name])

def summarize(logo):
    if logo is None:
        return None
    (t, l), (b, r), mask, thresh, *stuck = logo
    return {'bbox': [t, l, b, r], 'mask_px': int(np.count_nonzero(mask)), 'thresh': int(thresh),
            'stuck': [[list(a), list(c)] for a, c in stuck]}

def blanked(sum_, fcount, shape):
    return lf._blank_margins((sum_.astype(np.float64) / fcount).copy(), shape)


print('=== recordings a human watched: logo or no logo ===')
for name, s, fc, shape, e in fixtures():
    if not e['human_verdict']:
        continue
    got = lf._analyze(s.copy(), fc, shape)
    want_logo = e['expect'] is not None
    check(f'{name}: {"a logo is found" if want_logo else "no logo is returned"}',
          (got is not None) == want_logo,
          e['human_verdict'][:72])

print('\n=== every fixture still produces exactly what it did ===')
for name, s, fc, shape, e in fixtures():
    got = summarize(lf._analyze(s.copy(), fc, shape))
    check(f'{name}', got == e['expect'],
          '' if got == e['expect'] else f'got {got} want {e["expect"]}')

print('\n=== the two confirmed weather alerts are bounded as one wide region ===')
for name in ('alert-band-1080', 'alert-band-720'):
    z = np.load(os.path.join(FX, name + '.npz'))
    s, fc = z['persist'].astype(np.uint32), int(z['fcount'])
    shape = tuple(int(x) for x in z['shape'])
    remove, report = lf._find_stuck(blanked(s, fc, shape), shape)
    widest = max(((r - l) / shape[1] for (_, l), (_, r) in remove), default=0)
    check(f'{name}: one stuck region spanning most of the frame width',
          len(remove) == 1 and widest > 0.90, f'{len(remove)} region(s), widest {widest*100:.0f}% of width')
    # the gate must not come anywhere near rejecting a real alert's cores
    core = blanked(s, fc, shape) >= STUCK_PERSIST
    lab, k = ndimage.label(core, structure=np.ones((3, 3), bool))
    largest = int(np.bincount(lab.ravel())[1:].max()) if k else 0
    check(f'{name}: its largest core clears the 100px gate by a wide margin',
          largest >= 1000, f'largest core {largest}px')

print('\n=== a logo whose own peak crosses the line is not mistaken for an overlay ===')
# This is what _STUCK_CORE_PX exists for.  logo-handflagged is a hand-confirmed working
# logo whose per-pixel persistence reaches .909; without the gate it is deleted outright.
for name, s, fc, shape, e in fixtures():
    if e['expect'] is None:
        continue
    p = blanked(s, fc, shape)
    core = p >= STUCK_PERSIST
    if not core.any():
        continue
    lab, k = ndimage.label(core, structure=np.ones((3, 3), bool))
    largest = int(np.bincount(lab.ravel())[1:].max())
    remove, _ = lf._find_stuck(p, shape)
    check(f'{name}: peak {p.max():.3f} over the line, still returns a logo',
          lf._analyze(s.copy(), fc, shape) is not None,
          f'largest core {largest}px, {len(remove)} stuck region(s)')

print('\n=== the hand-flagged recording, scored against its true tags ===')
z = np.load(os.path.join(FX, 'logo-handflagged.counts.npz'), allow_pickle=True)
frac = z['counts'] / int(z['lmc'])
fps = float(z['samples_per_sec'])
t = np.arange(len(frac)) / fps
com = np.zeros(len(t), bool); nonshow = np.zeros(len(t), bool)
for ty, (a, b) in json.loads(str(z['tags'])):
    sel = (t >= a) & (t < b)
    if ty == SceneType.COMMERCIAL.value:
        com |= sel
    nonshow |= sel
show = ~nonshow
on = frac >= 2/3
check('logo is on for most of the hand-flagged show', on[show].mean() > 0.85,
      f'{on[show].mean()*100:.1f}% of show')
check('logo is off for nearly all hand-flagged commercial', on[com].mean() < 0.05,
      f'{on[com].mean()*100:.1f}% of commercial')

print('\n=== synthetic overlays composited onto the real maps ===')
BG = 'logo-sd'   # small crisp logo, no stuck region of its own
z = np.load(os.path.join(FX, BG + '.npz'))
bg, bgfc = z['persist'].astype(np.uint32), int(z['fcount'])
bgshape = tuple(int(x) for x in z['shape'])

def band(s, fc, shape, flush_bottom=False):
    """Two high-persistence edges with scrolling text between, plus a static blob -- the
    pieces are deliberately not connected to each other, as in the real 2513 alert."""
    h, w = shape
    th = max(60, h // 12)
    y0 = h - th if flush_bottom else max(2, h // 20)
    x0, x1 = w // 8, w - w // 8
    el = max(3, h // 270)
    s[y0:y0+el, x0:x1] = int(0.995 * fc)
    s[y0+th-el:y0+th, x0:x1] = int(0.99 * fc)
    rng = np.random.default_rng(3)
    s[y0+el:y0+th-el, x0:x1] = ((rng.random((th-2*el, x1-x0)) * 0.3 + 0.3) * fc).astype(np.uint32)
    rs = max(24, h // 30)
    s[y0+th//2-rs//2:y0+th//2+rs//2, x0+40:x0+40+rs] = int(0.98 * fc)
    return y0, y0 + th

for label, flush in (('scrolling band', False), ('band flush with the frame edge', True)):
    s = bg.copy()
    y0, y1 = band(s, bgfc, bgshape, flush_bottom=flush)
    remove, _ = lf._find_stuck(blanked(s, bgfc, bgshape), bgshape)
    ok = len(remove) == 1
    detail = f'{len(remove)} region(s)'
    if ok:
        (t_, l_), (b_, r_) = remove[0]
        ok = t_ <= y0 and b_ >= y1
        detail = f'box {t_},{l_}->{b_},{r_} vs band {y0}..{y1}'
        if flush:
            ok = ok and b_ == bgshape[0]
            detail += f' (snapped to {bgshape[0]})'
    check(f'{label} bounded as one box', ok, detail)
    s2 = s.astype(np.float64); lf._blank_margins(s2, bgshape)
    for (t_, l_), (b_, r_) in remove:
        s2[t_:b_, l_:r_] = 0
    check(f'{label}: nothing above the cutoff survives removal',
          (s2 / bgfc).max() < STUCK_PERSIST, f'peak {(s2/bgfc).max():.3f}')

for size, expect in ((1, False), (49, False), (100, True), (400, True)):
    s = bg.copy()
    side = int(np.ceil(np.sqrt(size)))
    y, x = bgshape[0] // 8, bgshape[1] // 2
    s[y:y+side, x:x+side] = int(0.999 * bgfc)
    remove, _ = lf._find_stuck(blanked(s, bgfc, bgshape), bgshape)
    check(f'a {side}x{side}={side*side}px core is {"bounded" if expect else "ignored"}',
          (len(remove) > 0) == expect, f'{len(remove)} region(s)')

# Modelled on a real recording (not kept as a fixture, see README): the leftover sat 1px
# below a removed banner with nothing above .50 joining them, so growth could never reach
# it, and a scatter of edges was returned as the logo.
for gap, expect_logo in ((2, False), (60, True)):
    s = (bg.astype(np.float64) * 0.15).astype(np.uint32)
    h, w = bgshape
    th = max(60, h // 12); y0 = h - th
    s[y0:y0+3, w//8:w-w//8] = int(0.99 * bgfc)
    s[y0+th-3:y0+th, w//8:w-w//8] = int(0.99 * bgfc)
    rs = max(24, h // 30)
    s[y0+th//2-rs//2:y0+th//2+rs//2, w//8+40:w//8+40+rs] = int(0.98 * bgfc)
    rng = np.random.default_rng(11)
    yy = y0 - gap - 40
    blob = rng.random((38, 180)) < 0.05
    s[yy:yy+38, w-400:w-220] = np.where(blob, int(0.78 * bgfc), s[yy:yy+38, w-400:w-220])
    got = lf._analyze(s.copy(), bgfc, bgshape)
    check(f'a scatter {gap}px from a banner is {"accepted" if expect_logo else "rejected"}',
          (got is not None) == expect_logo,
          'none' if got is None else f'bbox {got[0]}->{got[1]}')

s = bg.copy()
cy, cx = bgshape[0] // 6, bgshape[1] // 3
s[cy:cy+22, cx:cx+22] = int(0.99 * bgfc)
for i in range(1, 16):
    v = int((0.98 - 0.23 * i / 15) * bgfc)
    sl = (slice(cy-i, cy+22+i), slice(cx-i, cx+22+i))
    s[sl] = np.maximum(s[sl], v)
remove, _ = lf._find_stuck(blanked(s, bgfc, bgshape), bgshape)
left = s.astype(np.float64); lf._blank_margins(left, bgshape)
for (t_, l_), (b_, r_) in remove:
    left[t_:b_, l_:r_] = 0
peak = (left / bgfc)[cy-20:cy+42, cx-20:cx+42].max()
check('a graded skirt is absorbed, leaving nothing to mistake for a logo',
      len(remove) >= 1 and peak < 0.60, f'leftover peak {peak:.3f}')

s = bg.copy()
s[int(bgshape[0]*0.02):int(bgshape[0]*0.50), :] = int(0.99 * bgfc)
remove, report = lf._find_stuck(blanked(s, bgfc, bgshape), bgshape)
check('an implausibly large region is removed but not reported for blank detection',
      len(remove) >= 1 and len(report) == 0, f'{len(remove)} removed, {len(report)} reported')

print()
if _fails:
    print(f'{len(_fails)} FAILED: ' + ', '.join(_fails))
    sys.exit(1)
print('all checks passed')
