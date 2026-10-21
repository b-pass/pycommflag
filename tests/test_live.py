#!/usr/bin/env python3
"""Checks for flagging a recording while it is still being written.

Makes a small synthetic .ts, "records" it by appending to a second file from a thread, and
checks that following the growing file gives exactly what reading the finished one does.
Needs no real video, but the extraction check runs the audio segmenter, so it needs its
model (downloaded on first use).

Run it directly; it exits non-zero if anything fails.
"""
import fractions
import os
import shutil
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import av
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import live
from pycommflag.player import Player

live.POLL_SECS = 0.1
live.SETTLE_SECS = 0.5
live.IDLE_SECS = 2
live.HEADSTART = 50
live.HEADSTART_POLL_SECS = 0.5
live.INTERVAL = 30

_fails = []

def check(name, got, want):
    ok = got == want
    print(f'{"ok  " if ok else "FAIL"} {name}' + ('' if ok else f': got {got!r}, want {want!r}'))
    if not ok:
        _fails.append(name)

def make_ts(path, seconds, fps=30, w=320, h=180):
    """Segments of different brightness and tone, a black silent gap between them, and a
    white box in the corner for the logo search to find (it goes away for 15s of every 40,
    or the search would take it for a stuck overlay)."""
    out = av.open(path, 'w', format='mpegts')
    vs = out.add_stream('mpeg2video', rate=fps)
    vs.width, vs.height, vs.pix_fmt, vs.bit_rate = w, h, 'yuv420p', 800000
    aus = out.add_stream('mp2', rate=48000)
    aus.layout = 'stereo'
    asamp = 0
    for i in range(seconds*fps):
        t = i/fps
        seg = int(t//20)
        gap = t % 20 < 0.5
        img = np.zeros((h,w,3), 'uint8')
        if not gap:
            img[:] = (seg*40)%200 + 30
            x = int((t*40) % (w-40))
            img[60:100, x:x+40] = 250
            if t % 40 < 25:
                img[20:45, w-60:w-20] = 255
        f = av.VideoFrame.from_ndarray(img, format='rgb24')
        f.pts, f.time_base = i, fractions.Fraction(1, fps)
        for p in vs.encode(f):
            out.mux(p)
        while asamp < (i+1)/fps*48000:
            tt = np.arange(asamp, asamp+1152)/48000.0
            s = np.zeros(len(tt), 'float32') if gap else (0.2*np.sin(2*np.pi*(220+seg*110)*tt)).astype('float32')
            af = av.AudioFrame.from_ndarray(np.stack([s,s]).reshape(1,-1), format='flt', layout='stereo')
            af.sample_rate, af.pts, af.time_base = 48000, asamp, fractions.Fraction(1, 48000)
            for p in aus.encode(af):
                out.mux(p)
            asamp += 1152
    for st in (vs, aus):
        for p in st.encode():
            out.mux(p)
    out.close()

class Recorder(threading.Thread):
    """Copies src to dst a bit at a time, `speed` times faster than real time."""
    def __init__(self, src, dst, seconds, speed, start_frac=0.1):
        super().__init__(daemon=True)
        self.data = open(src, 'rb').read()
        self.dst = dst
        self.step = 188*100
        self.delay = seconds / speed / (len(self.data) / self.step)
        self.pos = int(len(self.data) * start_frac) // 188 * 188
        with open(dst, 'wb') as f:
            f.write(self.data[:self.pos])

    def run(self):
        with open(self.dst, 'ab') as f:
            while self.pos < len(self.data):
                f.write(self.data[self.pos:self.pos+self.step])
                f.flush()
                self.pos += self.step
                time.sleep(self.delay)

def frame_times(player, follow=None):
    if follow is not None:
        follow.follow = True
    player.enable_audio()
    return [round(float(f.time), 5) for f in player.frames()]

tmp = tempfile.mkdtemp(prefix='pycf_test_live_')
try:
    full = os.path.join(tmp, 'full.ts')
    make_ts(full, 60)

    # --- following a growing file decodes the same frames as the finished file
    grow = os.path.join(tmp, 'grow.ts')
    rec = live.Recording(grow, lambda r: r.idle_for() < live.IDLE_SECS)
    writer = Recorder(full, grow, 60, speed=10)
    p = Player(grow, no_deinterlace=True, source=rec)
    before = sum(1 for _ in p.frames_stride(15))   # stops at the current end, like the logo search
    p.seek(0)
    writer.start()
    got = frame_times(p, follow=rec)
    writer.join()
    want = frame_times(Player(full, no_deinterlace=True))
    check('stride read stopped early', 0 < before < len(want)//15, True)
    check('frame count', len(got), len(want))
    check('frame times', got == want, True)

    # --- a live extraction writes the same log as one over the finished recording
    from pycommflag import processor
    full = os.path.join(tmp, 'full150.ts')
    make_ts(full, 150)
    grow = os.path.join(tmp, 'grow150.ts')
    def opts_for(fn):
        return SimpleNamespace(filename=fn, no_logo=False, quiet=True, no_deinterlace=True,
                               logo_samples=2, chanid=None, starttime=None, mythjob=None)
    rec = live.Recording(grow, lambda r: r.idle_for() < live.IDLE_SECS)
    writer = Recorder(full, grow, 150, speed=5, start_frac=0.02)
    writer.start()
    live_log = os.path.join(tmp, 'live.json')
    snaps = []
    processor.process_video(grow, live_log, opts_for(grow), live=rec, on_snapshot=snaps.append)
    writer.join()

    # with the log already there, the normal run reuses its logo, which the live one found
    # from the head start only
    shutil.copyfile(grow, full)
    full_log = os.path.join(tmp, 'full.json')
    shutil.copyfile(live_log, full_log)
    processor.process_video(full, full_log, opts_for(full))

    a = processor.read_feature_log(live_log)
    b = processor.read_feature_log(full_log)
    check('log keys', sorted(a), sorted(b))
    check('filename', (a['filename'], b['filename']), (os.path.realpath(grow), os.path.realpath(full)))
    check('duration', a['duration'], b['duration'])
    check('logo found', a['logo'] is not None, True)
    check('logo', a['logo'], b['logo'])
    check('frame count', len(a['frames']), len(b['frames']))
    check('frames', a['frames'] == b['frames'], True)

    # --- interim snapshots along the way
    # the audio is final a 60s chunk at a time, so 150s gives two
    check('snapshots', len(snaps) >= 2, True)
    check('snapshot times', [s['duration'] for s in snaps] == sorted(s['duration'] for s in snaps), True)
    check('snapshot spacing', all(n['duration'] - p['duration'] >= live.INTERVAL for p, n in zip(snaps, snaps[1:])), True)
    for i, snap in enumerate(snaps):
        fr = snap['frames']
        check(f'snapshot {i} header', snap['frames_header'], a['frames_header'])
        check(f'snapshot {i} stops at its duration', fr[-1][0] < snap['duration'] <= a['frames'][len(fr)][0], True)
        # all but the volume, which is scaled by the loudest moment so far
        check(f'snapshot {i} frames', [f[:4] + f[6:] for f in fr] == [f[:4] + f[6:] for f in a['frames'][:len(fr)]], True)
except BaseException:
    # extraction leaves its worker threads running when it dies, so don't wait for them
    import traceback
    traceback.print_exc()
    shutil.rmtree(tmp, ignore_errors=True)
    os._exit(1)
shutil.rmtree(tmp, ignore_errors=True)

if _fails:
    print(f'\n{len(_fails)} failed')
    sys.exit(1)
