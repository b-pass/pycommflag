#!/usr/bin/env python3
"""Checks that Player and AudioProc put audio on the right timeline.

Audio segments arrive as (main, surround, time, rate), with times measured from the video
start, so they can be negative (audio that starts before the video), repeat audio already
seen (replays after a recovery seek), or leave holes.  Each case feeds 1kHz bursts at
known times and checks where the RMS volume says they landed.

The last cases feed fake audio frames through Player._queue_audio, where a frame stamped far
ahead or behind must be skipped rather than silencing everything after it.

The segmenter and TensorFlow are stubbed out: only the timeline is under test.

Run it directly; it exits non-zero if anything fails.
"""
import os
import sys
import types
from fractions import Fraction

import numpy as np

# AudioProc.__init__ imports tensorflow only to limit its threads
_tf = types.ModuleType('tensorflow')
_tf.config = types.SimpleNamespace(threading=types.SimpleNamespace(
    set_intra_op_parallelism_threads=lambda n: None,
    set_inter_op_parallelism_threads=lambda n: None))
sys.modules.setdefault('tensorflow', _tf)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import processor
from pycommflag.player import Player

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

# --- Player._queue_audio: timestamps far from the audio and video ---

AF = 1536  # samples per frame, as AC-3 has

def make_player():
    p = Player.__new__(Player)
    p.aq = []
    p.streams = {'video': 0, 'audio': 0}
    p.vt_start = 0.0
    p.vtime_base = Fraction(1, 90000)
    p.vpts = 0
    p._a_end = None
    p._a_off = 0
    return p

def frame(stamp, x):
    """a fake stereo AudioFrame stamped `stamp` holding samples x"""
    d = np.vstack([x, x]).astype('float32')
    return types.SimpleNamespace(time=stamp, sample_rate=SR, to_ndarray=lambda: d,
                                 layout=types.SimpleNamespace(channels=[None, None]))

def feed(p, t0, t1, bursts, stamp=lambda i, t: t, video=lambda t: t, drain=1.0):
    """frames covering [t0,t1), the video kept at video(t); returns the drained segments"""
    out = []
    x = tone(t0, t1, bursts)
    nexttake = t0 + drain
    for i in range(len(x) // AF):
        t = t0 + i*AF/SR
        p.vpts = int(round(video(t) * 90000))
        p._queue_audio(frame(stamp(i, t), x[i*AF:(i+1)*AF]))
        if t >= nexttake:
            out += p.move_audio()
            nexttake += drain
    return out + p.move_audio()

# one frame at 5s stamped 12 hours ahead: the burst at 15s must still land at 15s
p = make_player()
bad = int(5*SR/AF)
segs = feed(p, 0, 20, [(15, 15.5)], stamp=lambda i, t: t + 12*3600 if i == bad else t)
latest = max(s[2] for s in segs)
check('a lone far-future audio frame is never queued', latest < 20, f'latest segment at {latest:.0f}s')
got = onsets(run(segs), 1)
check('audio after a far-future frame lands on time', near(got, [15.0]), f'burst at {got}')

# one frame at 5s stamped 12 hours back, then one at 8s stamped 1 hour back with the video
# still at 0 (no help from the video position): neither queued, and later audio on time
p = make_player()
bad2 = int(8*SR/AF)
segs = feed(p, 0, 20, [(15, 15.5)],
            stamp=lambda i, t: t - 12*3600 if i == bad else (t - 3600 if i == bad2 else t),
            video=lambda t: 0.0)
earliest = min(s[2] for s in segs)
check('a lone far-past audio frame is never queued', earliest >= 0, f'earliest segment at {earliest:.0f}s')
got = onsets(run(segs), 1)
check('audio after a far-past frame lands on time', near(got, [15.0]), f'burst at {got}')

# a short replay (2s back, as after a recovery seek) still reaches AudioProc to be trimmed
p = make_player()
feed(p, 0, 5, [])
segs = feed(p, 3, 4, [])
n = sum(len(s[0]) for s in segs)
check('a short replay is still queued', n == int(SR/AF)*AF, f'{n} samples')

# the audio clock really moves an hour ahead of the video: taken from the 5th frame on
p = make_player()
segs = feed(p, 0, 1, [], stamp=lambda i, t: t + 3600, video=lambda t: 0.0)
n = sum(len(s[0]) for s in segs)
want = (int(SR/AF) - (Player.PTS_JUMP_CONFIRM - 1)) * AF
check('a lasting audio jump is followed after PTS_JUMP_CONFIRM frames', n == want, f'{n} samples, want {want}')
check('...at its own time', abs(segs[0][2] - (3600 + (Player.PTS_JUMP_CONFIRM-1)*AF/SR)) < 1e-6,
      f'starts at {segs[0][2]:.3f}')

# audio stops for 30s while the video runs on, then resumes in step with the video: nothing lost
p = make_player()
feed(p, 0, 1, [])
segs = feed(p, 31, 32, [])
n = sum(len(s[0]) for s in segs)
check('audio resuming with the video after a gap is taken at once', n == int(SR/AF)*AF, f'{n} samples')

# a seek clears the reference
p = make_player()
p.graph = None
p._skip_frames = lambda mode: None
feed(p, 0, 1, [])
p._a_off = 3
p._flush()
check('_flush resets the audio jump guard', p._a_end is None and p._a_off == 0)

if _fails:
    print(f'{len(_fails)} failed')
    sys.exit(1)
print('all passed')
