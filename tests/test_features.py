#!/usr/bin/env python3
"""Checks for the caption count recorded by extraction.

`processor._cc_text` counts CEA-608 caption characters per frame from its A53 side data;
it goes into the feature log (cc_text) but not, yet, into the model.

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

# 'H','i' with odd parity set on the 'i' byte, as broadcast
text = triplet(ord('H'), ord('i') | 0x80)
check('caption text counts both characters', processor._cc_text(frame(SideData(text))) == 2)
check('a control code (resume caption loading) does not count',
      processor._cc_text(frame(SideData(triplet(0x14, 0x20)))) == 0)
check('padding does not count', processor._cc_text(frame(SideData(triplet(0x80, 0x80)))) == 0)
check('an invalid triplet does not count',
      processor._cc_text(frame(SideData(triplet(ord('H'), ord('i'), valid=False)))) == 0)
check('field 2 and 708 data do not count',
      processor._cc_text(frame(SideData(triplet(ord('H'), ord('i'), cc_type=1)
                                        + triplet(ord('H'), ord('i'), cc_type=2)))) == 0)
check('other side data is ignored',
      processor._cc_text(frame(SideData(text, type='MOTION_VECTORS'))) == 0)
check('a frame without side data has none', processor._cc_text(frame()) == 0)
check('several triplets add up',
      processor._cc_text(frame(SideData(text + triplet(0x14, 0x2f) + text))) == 4)

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

if _fails:
    print(f'{len(_fails)} failed')
    sys.exit(1)
print('all passed')
