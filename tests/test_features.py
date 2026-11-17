#!/usr/bin/env python3
"""Checks for the captions recorded by extraction.

`processor._cc_data` keeps a frame's caption bytes from its A53 side data (cc_data in the
feature log) and `processor.cc_chars` counts the CEA-608 characters in them (cc_text); neither
goes into the model, yet.

Run it directly; it exits non-zero if anything fails.
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import neural, processor

_fails = []

def check(name, ok, detail=''):
    print(('  pass  ' if ok else '  FAIL  ') + name + (f'   {detail}' if detail else ''))
    if not ok:
        _fails.append(name)

class SideData:
    def __init__(self, data, type='A53_CC'):
        self.type, self._data = type, bytes(data)
    def __bytes__(self):
        return self._data

def frame(*side_data):
    return types.SimpleNamespace(side_data=list(side_data))

def triplet(c1, c2, valid=True, cc_type=0):
    return [0xf8 | (0x04 if valid else 0) | cc_type, c1, c2]

def cc_text(f):
    return processor.cc_chars(processor._cc_data(f))

# 'H','i' with odd parity set on the 'i' byte, as broadcast
text = triplet(ord('H'), ord('i') | 0x80)
check('caption text counts both characters', cc_text(frame(SideData(text))) == 2)
check('a control code (resume caption loading) does not count',
      cc_text(frame(SideData(triplet(0x14, 0x20)))) == 0)
check('padding does not count', cc_text(frame(SideData(triplet(0x80, 0x80)))) == 0)
check('an invalid triplet does not count',
      cc_text(frame(SideData(triplet(ord('H'), ord('i'), valid=False)))) == 0)
check('field 2 and 708 data do not count',
      cc_text(frame(SideData(triplet(ord('H'), ord('i'), cc_type=1)
                                        + triplet(ord('H'), ord('i'), cc_type=2)))) == 0)
check('other side data is ignored',
      cc_text(frame(SideData(text, type='MOTION_VECTORS'))) == 0)
check('a frame without side data has none', cc_text(frame()) == 0)
check('several triplets add up',
      cc_text(frame(SideData(text + triplet(0x14, 0x2f) + text))) == 4)

# what is kept: every valid construct but 608 null padding, whole, in order, parity and all
PAC = triplet(0x91, 0x70)                       # field 1 preamble: row 1, indent 16
POPON = triplet(0x94, 0x20)                     # resume caption loading (pop-on)
XDS = triplet(0x01, 0x83, cc_type=1)            # field 2: XDS start, programme name
DTV = triplet(0x02, 0x21, cc_type=3) + triplet(0x8f, 0x00, cc_type=2)  # a 708 packet; its 00 stays
kept = POPON + PAC + text + XDS + DTV
sd = SideData(triplet(0x80, 0x80) + POPON + PAC + triplet(0x80, 0x80, cc_type=1) + text
              + triplet(ord('Z'), ord('Z'), valid=False) + XDS + DTV + triplet(0, 0, valid=False, cc_type=2))
check('caption bytes: mode, position, field 2 and 708 kept; padding and invalid ones dropped',
      processor._cc_data(frame(sd)) == bytes(kept), processor._cc_data(frame(sd)).hex())
check('caption bytes from several A53 side data, in order',
      processor._cc_data(frame(SideData(PAC), SideData(text, type='MOTION_VECTORS'), SideData(text))) == bytes(PAC + text))
check('no captions, no bytes', processor._cc_data(frame(SideData(triplet(0x80, 0x80)))) == b'')

# A plain 300s log, to check that an extra cc_text column is ignored.
FPS = 29.97
times, t = [], 0.0
while t < 300:
    times.append(round(t, 5))
    t += 1/FPS
rows = [[t, 0, 0, 1.0, .1, 0, 0, 1, 0, 0] for t in times]
flog = {'file_version': 10, 'duration': 300.0, 'frame_rate': FPS, 'logo': None,
        'frames_header': ['time', 'logo_present', 'is_blank', 'diff', 'fvol', 'rvol',
                          'silence', 'speech', 'music', 'noise'],
        'frames': [list(r) for r in rows]}
c = neural.load_nonpersistent(dict(flog, frames=[list(r) for r in rows]))
check('condensed rows have the bookkeeping columns after the features',
      c is not None and c.shape[1] == neural.WEIGHTS + 1, '' if c is None else str(c.shape))
if c is not None:
    # file_version 11 logs carry cc_text after the raw ten; the model must not see it
    flog11 = dict(flog, file_version=11, frames_header=flog['frames_header'] + ['cc_text'],
                  frames=[list(r) + [37] for r in rows])
    c11 = neural.load_nonpersistent(flog11)
    check('an extra cc_text column changes nothing the model sees', c11 is not None and np.array_equal(c, c11))

    # a 130ms dip to silence (4 frames) becomes SILENCE; a 60ms one (2 frames) does not
    def second_of(c, t):
        return c[np.flatnonzero((c[:, neural.TIMESTAMPS] >= t) & (c[:, neural.TIMESTAMPS] < t + 1))[0]]
    dipped = [list(r) for r in rows]
    i130, i60 = times.index(next(x for x in times if x >= 100.3)), times.index(next(x for x in times if x >= 200.3))
    for i in range(i130, i130 + 4):
        dipped[i][4] = 0
    for i in range(i60, i60 + 2):
        dipped[i][4] = 0
    cd = neural.load_nonpersistent(dict(flog, frames=dipped))
    s130, s60, s0 = second_of(cd, 100), second_of(cd, 200), second_of(c, 100)
    share = 4 / np.count_nonzero((np.array(times) >= 100) & (np.array(times) < 101))
    check('a 130ms silence is labelled SILENCE for its share of the second',
          abs(s130[neural.SILENCE] - share) < 1e-4 and abs(s130[neural.SPEECH] - (1 - share)) < 1e-4,
          f'silence {s130[neural.SILENCE]:.4f} speech {s130[neural.SPEECH]:.4f} want {share:.4f}')
    check('a 60ms silence is not', s60[neural.SILENCE] == 0 and s60[neural.SPEECH] == s0[neural.SPEECH],
          f'silence {s60[neural.SILENCE]:.4f}')

    # condensing goes by time, one row a second (the last second can be a partial one)
    body = c[neural.WINDOW_BEFORE:-neural.WINDOW_AFTER]
    step = np.diff(body[:-1, neural.TIMESTAMPS])
    check('one condensed row per second', np.abs(step - 1).max() < .05,
          f'steps {step.min():.3f}..{step.max():.3f}')

# All film: 24 frames a second (gaps alternating 50/33ms) at a nominal 29.97, the logo on until
# 100s, and a break from 150s to 210s.
times, t = [], 0.0
while t < 300:
    times.append(round(t, 5))
    t += 0.05005 if len(times) % 2 else 0.03337
film = [[t, int(t < 100), 0, 1.0, .1, 0, 0, 1, 0, 0] for t in times]
flog = dict(flog, logo=[1], tags=[(3, (150.0, 210.0))])
c = neural.load_nonpersistent(dict(flog, frames=[list(r) for r in film]))
if c is not None:
    body = c[neural.WINDOW_BEFORE:-neural.WINDOW_AFTER]
    ts = body[:, neural.TIMESTAMPS]
    check('all-film log condenses to one row a second', abs(len(body) - 300) <= 1, f'{len(body)} rows')
    k = int(np.flatnonzero((ts >= 110) & (ts < 111))[0])
    # LOGO_RUN is taken at a row's last frame, which is just short of 111s
    want = neural._scale_dist(np.array([110.96 - 99.98]))[0]
    check('LOGO_RUN counts seconds, not frames at the nominal rate', abs(body[k, neural.LOGO_RUN] - want) < .005,
          f'{body[k, neural.LOGO_RUN]:.3f} vs {want:.3f}')
else:
    check('all-film log condenses', False)

c = neural.load_nonpersistent(dict(flog, frames=[list(r) for r in film]), for_training=True)
if c is not None:
    body = c[neural.WINDOW_BEFORE:-neural.WINDOW_AFTER]
    ts, w = body[:, neural.TIMESTAMPS], body[:, neural.WEIGHTS]
    def weight_at(t): return float(w[np.argmin(np.abs(ts - t))])
    # 1 + ((60 - dt)/60)^2, maxed over the second: dt reaches 29.5 inside the row centred at 120.5
    check('boundary upweighting reaches 60s out by time', abs(weight_at(120.5) - (1 + (30.5/60)**2)) < .02,
          f'{weight_at(120.5):.3f}')
    check('...and stops there', weight_at(85) == 1.0 and weight_at(275) == 1.0,
          f'{weight_at(85):.3f} {weight_at(275):.3f}')
else:
    check('all-film log condenses for training', False)

# condense's per-second stats against plain numpy over each second's frames
rng = np.random.default_rng(7)
t = np.cumsum(rng.choice([0.03337, 0.05005], 4000))
fr = rng.random((len(t), neural.GENERATED_FEATURES_START)).astype('float32')
w = rng.choice([0, .75, 1, 1.5], (len(t), 1)).astype('float32')
got = neural.condense(fr, t, np.zeros((len(t), 1), 'float32'), w, 1.0)
sec = np.floor(t - t[0]).astype(int)
bad = []
for k, s in enumerate(np.unique(sec)):
    f, ww = fr[sec == s], w[sec == s, 0]
    want = {neural.DIFF_90TH: np.percentile(f[:, neural.DIFF], 90), neural.DIFF_MAX: f[:, neural.DIFF].max(),
            neural.RVOL_STDDEV: np.std(f[:, neural.RVOL]), neural.FVOL_MAX: f[:, neural.FVOL].max(),
            neural.LOGO: f[:, neural.LOGO].mean(), neural.LOGO_RUN: f[-1, neural.LOGO_RUN],
            neural.WEIGHTS: 0 if ww.min() < .1 else ww.max()}
    bad += [col for col, v in want.items() if abs(got[k, col] - v) > 1e-5]
check("condense's per-second stats match numpy's", len(got) == len(np.unique(sec)) and not bad, str(sorted(set(bad))))

# the DIFF column counts cuts a second, whatever the frame rate: the same cuts read the same at
# 24 (uneven), 30 and 60 fps, and a cut that spans two frames is still one
cut_times = np.array([2.2, 3.5, 3.9, 7.1, 9.6])
def cuts_at(rate, split, uneven=False):
    t = np.arange(0, 12, 1 / rate)
    if uneven:
        t = np.cumsum(rng.choice([0.03337, 0.05005], 400))[:int(12 * 24)]
    f = np.zeros((len(t), neural.GENERATED_FEATURES_START), 'float32')
    for c in cut_times:
        i = int(np.searchsorted(t, c))
        f[i, neural.DIFF] = 1
        if split:
            f[i + 1, neural.DIFF] = .8
    z = np.zeros((len(t), 1), 'float32')
    return neural.condense(f, t, z, z + 1, 1.0)[:, neural.DIFF]
want = np.bincount(cut_times.astype(int), minlength=12)
for name, got in (('30 fps', cuts_at(29.97, False)), ('30 fps, cuts split over two frames', cuts_at(29.97, True)),
                  ('59.94 fps', cuts_at(59.94, False)), ('uneven ~24 fps', cuts_at(24, False, True))):
    n = min(len(got), len(want))
    check(f'DIFF is cuts a second at {name}', np.allclose(got[:n], want[:n]), f'{got[:n]} want {want[:n]}')

if _fails:
    print(f'{len(_fails)} failed')
    sys.exit(1)
print('all passed')
