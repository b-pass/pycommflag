#!/usr/bin/env python3
"""Checks that AudioProc puts audio on the right timeline.

Audio segments arrive as (main, surround, time, rate), with times measured from the video
start, so they can be negative (audio that starts before the video), repeat audio already
seen (replays after a recovery seek), or leave holes.  Each case feeds 1kHz bursts at
known times and checks where the RMS volume says they landed.

The segmenter and TensorFlow are stubbed out: only the timeline is under test.

Run it directly; it exits non-zero if anything fails.
"""
import os
import sys
import types

import numpy as np

# AudioProc.__init__ imports tensorflow only to limit its threads
_tf = types.ModuleType('tensorflow')
_tf.config = types.SimpleNamespace(threading=types.SimpleNamespace(
    set_intra_op_parallelism_threads=lambda n: None,
    set_inter_op_parallelism_threads=lambda n: None))
sys.modules.setdefault('tensorflow', _tf)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import processor

class _Seg:
    def __call__(self, samples):
        return [('speech', 0.0, len(samples)/16000.0)]
processor.ina_foss.Segmenter = _Seg

SR = 48000
_fails = []

def check(name, ok, detail=''):
    print(('  pass  ' if ok else '  FAIL  ') + name + (f'   {detail}' if detail else ''))
    if not ok:
        _fails.append(name)

def tone(t0, t1, bursts):
    """samples covering [t0,t1): silence, with a burst over each (start,end) in bursts"""
    t = t0 + np.arange(int(round((t1 - t0)*SR)))/SR
    x = np.zeros(len(t), 'float32')
    for (b0, b1) in bursts:
        on = (t >= b0) & (t < b1)
        x[on] = 0.5*np.sin(2*np.pi*1000*t[on])
    return x

def run(segments):
    ap = processor.AudioProc()
    for s in segments:
        ap.queue.put(s)
    ap.stop()
    ap.run()
    return np.array(ap.rms)

def onsets(rms, col, thresh=.1):
    """window-end times where the column first rises over thresh"""
    on = rms[:, col] > thresh
    rise = np.flatnonzero(on[1:] & ~on[:-1]) + 1
    return rms[rise, 0]

def near(got, want, tol=.05):
    return len(got) == len(want) and all(abs(g - w) <= tol for g, w in zip(got, want))

# audio starts half a second before the video; a burst at 2.0s must land at 2.0s
segs = [(tone(-.5, 4, [(2, 2.5)]), None, -.5, SR)]
got = onsets(run(segs), 1)
check('audio before the video start is dropped, not prepended', near(got, [2.0]), f'burst at {got}')

# a chunk of audio arrives twice (a replay after a seek); the copy must be dropped without
# moving anything after it
bursts = [(1.5, 1.7), (3.0, 3.2)]
segs = [(tone(0, 1, bursts), None, 0.0, SR),
        (tone(1, 2, bursts), None, 1.0, SR),
        (tone(1.2, 1.8, bursts), None, 1.2, SR),  # entirely stale
        (tone(2, 4, bursts), None, 2.0, SR)]
got = onsets(run(segs), 1)
check('a fully replayed segment does not shift later audio', near(got, [1.5, 3.0]), f'bursts at {got}')

# a hole in the audio: both channels must be padded at the front of the next segment
bursts = [(2.0, 2.2)]
segs = [(tone(0, 1, []), tone(0, 1, []), 0.0, SR),
        (tone(1.5, 3, bursts), tone(1.5, 3, bursts), 1.5, SR)]
rms = run(segs)
fg, rg = onsets(rms, 1), onsets(rms, 2)
check('front lands after a hole', near(fg, [2.0]), f'burst at {fg}')
check('surround lands with the front after a hole', near(rg, [2.0]), f'burst at {rg}')

if _fails:
    print(f'{len(_fails)} failed')
    sys.exit(1)
print('all passed')
