#!/usr/bin/env python3
"""Checks where the editor's review queue (gui.review_items) stops, on made-up tags and features.

It must stop at the recording's start and end and at every show/break edge (and not at intro or
credits, which train as show), and away from the edges wherever the tags might miss a break: the
model disagreeing with them, the logo gone from show, or blanks inside show spaced like the ends
of ads.  Imports tkinter but opens no window.

Run it directly; it exits non-zero if anything fails.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import gui
from pycommflag.feature_span import SceneType

_fails = []

def check(name, ok, detail=''):
    print(('  pass  ' if ok else '  FAIL  ') + name + (f'   {detail}' if detail else ''))
    if not ok:
        _fails.append(name)

opts = SimpleNamespace(break_min_len=59, break_max_len=335, show_min_len=59)
S, B, I, C = SceneType.SHOW, SceneType.COMMERCIAL, SceneType.INTRO, SceneType.CREDITS
DUR = 1800.0
# intro 0-60, show, a break 600-780, show, credits 1740-1800
tags = [(I, (0, 60)), (S, (60, 600)), (B, (600, 780)), (S, (780, 1740)), (C, (1740, DUR))]

def prob(t):
    if 600 <= t < 780: return 0.97
    if 1200 <= t < 1290: return 0.9   # a break the tags miss
    return 0.03
raw = [(t + 0.5, prob(t + 0.5)) for t in range(int(DUR))]
logo = [(True, (0, 600)), (False, (600, 780)), (True, (780, 1000)), (False, (1000, 1030)), (True, (1030, DUR))]
blanks = [1500.0, 1500.3, 1530.0, 1560.0]      # a fade read as two blanks, then ads' spacing

items = gui.review_items(tags, raw, logo, blanks, opts)
times = [t for (t, _, _) in items]
print('\n'.join(f'    {t:7.1f} {why}' for (t, why, _) in items))

check('stops at both edges of the break', 600 in times and 780 in times)
check('no stop at the intro or credits', not any(t in (60, 1740) for t in times))
check('the missed break is a stop', any(1190 <= t <= 1210 for t in times))
check('logo gone from show is a stop', any(abs(t - 1000) < 1 for t in times))
check('blanks spaced like ads are a stop', any(abs(t - 1500) < 1 for t in times))
check('stops at the start and the last second', 0 in times and DUR - 1 in times)
check('nothing else', len(items) == 7, str(len(items)))

# the same blanks without the third one are just two fades
items = gui.review_items(tags, raw, [], blanks[:3], opts)
check('two blanks are not a stop', not any(abs(t - 1500) < 1 for (t, _, _) in items))

# a 30s break is short
items = gui.review_items([(S, (0, 600)), (B, (600, 630)), (S, (630, DUR))], [], [], [], opts)
check('a short break is noted at its edge', any(t == 600 and 'short break' in why for (t, why, _) in items))

if _fails:
    print(f'{len(_fails)} FAILED')
    sys.exit(1)
print('all passed')
