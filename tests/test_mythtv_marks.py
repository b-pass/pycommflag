#!/usr/bin/env python3
"""Checks for the MythTV time <-> frame number mapping used for commbreak marks.

MythTV numbers the frames it actually recorded, so where a channel sends film with soft 3:2
pulldown the count advances ~24 a second, not at the nominal 29.97.  The type 33 seek table
(frame number -> ms at each keyframe) is what tells them apart; this uses a synthetic one, so
it needs no database (or mysqlclient).

Run it directly; it exits non-zero if anything fails.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import mythtv

_fails = []

def check(name, ok, detail=''):
    print(('  pass  ' if ok else '  FAIL  ') + name + (f'   {detail}' if detail else ''))
    if not ok:
        _fails.append(name)

RATE = 29.97
# 60s of video (a keyframe every 15 frames at 29.97), then 60s of film (every 12 frames at 23.976)
secs, frames = [], []
for k in range(120):
    secs.append(k * 15 / 29.97)
    frames.append(k * 15)
f0, t0 = frames[-1] + 15, secs[-1] + 15 / 29.97
for k in range(120):
    secs.append(t0 + k * 12 / 23.976)
    frames.append(f0 + k * 12)
dmap = (secs, frames)

def true_frame(t):
    return t * 29.97 if t < t0 else f0 + (t - t0) * 23.976

for t in (10.0, 30.123):
    got = mythtv._frame_for_time(dmap, t, RATE)
    check(f'time -> frame in video at {t}s', abs(got - true_frame(t)) <= 1, f'{got} vs {true_frame(t):.1f}')
for t in (65.0, 87.77):
    got = mythtv._frame_for_time(dmap, t, RATE)
    check(f'time -> frame in film at {t}s', abs(got - true_frame(t)) <= 1, f'{got} vs {true_frame(t):.1f}')
for t in (12.3, 77.7):
    back = mythtv._time_for_frame(dmap, mythtv._frame_for_time(dmap, t, RATE), RATE)
    check(f'frame -> time inverts it at {t}s', abs(back - t) <= 1 / 23.976, f'{back:.3f}')

end = secs[-1] + 2
got = mythtv._frame_for_time(dmap, end, RATE)
check('past the table, extrapolates at the nominal rate', got == round(frames[-1] + 2 * RATE), str(got))
got = mythtv._frame_for_time(([5.0, 6.0], [150, 180]), 1.0, RATE)
check('before the table, extrapolates back from its first row', got == round(150 - 4 * RATE), str(got))
check('with no table, the nominal rate', mythtv._frame_for_time(([], []), 10.0, RATE) == round(10 * RATE)
      and abs(mythtv._time_for_frame(([], []), 300, RATE) - 300 / RATE) < 1e-9)
check('never a negative frame', mythtv._frame_for_time(([5.0], [10]), 0.0, RATE) == 0)

if _fails:
    print(f'{len(_fails)} failed')
    sys.exit(1)
print('all passed')
