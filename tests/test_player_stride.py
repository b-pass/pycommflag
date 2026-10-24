#!/usr/bin/env python3
"""Checks that Player.frames_stride hands back correctly decoded frames, often enough.

frames_stride decodes only the reference frames (I and P).  Each sample must be identical
to the same frame from a full decode, and a long-GOP recording must still give at least one
sample a second.  The test videos are synthesized here (MPEG-2, as on most OTA channels), so this
needs no recordings.

Run it directly; it exits non-zero if anything fails.
"""
import os
import sys
import tempfile

from fractions import Fraction

import av
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag.player import Player

FPS, SECS, W, H = 30, 40, 320, 180
_fails = []

def check(name, ok, detail=''):
    print(('  pass  ' if ok else '  FAIL  ') + name + (f'   {detail}' if detail else ''))
    if not ok:
        _fails.append(name)

def make_video(path, gop):
    rng = np.random.default_rng(gop)
    out = av.open(path, 'w', format='mpegts')
    # no scene-change keyframes: the panning texture would otherwise get one every few frames
    st = out.add_stream('mpeg2video', rate=FPS, options={'sc_threshold': '1000000000'})
    st.width, st.height, st.pix_fmt = W, H, 'yuv420p'
    st.codec_context.gop_size = gop
    st.codec_context.max_b_frames = 2
    st.codec_context.bit_rate = 2_000_000
    bg = rng.uniform(0, 255, (H, W * 3)).astype('uint8')
    for i in range(SECS * FPS):
        img = np.repeat(np.roll(bg, 3 * i, axis=1)[:, :W, None], 3, axis=2)  # panning texture
        img[10:40, 250:300] = 235                                          # a static "logo"
        f = av.VideoFrame.from_ndarray(img, format='rgb24')
        f.pts = i
        f.time_base = Fraction(1, FPS)
        for pkt in st.encode(f):
            out.mux(pkt)
    for pkt in st.encode(None):
        out.mux(pkt)
    out.close()

def luma(f):
    return f.to_ndarray()[:f.height].astype('int16')

tmp = tempfile.mkdtemp(prefix='pycf-stride-')
for label, gop in (('3s GOP', 3 * FPS), ('0.5s GOP', FPS // 2)):
    path = os.path.join(tmp, f'gop{gop}.ts')
    make_video(path, gop)

    p = Player(path, no_deinterlace=True)
    p.seek(0)
    samples = {f.pts: luma(f) for f in p.frames_stride(round(FPS / 2))}
    full = {}
    q = Player(path, no_deinterlace=True)
    q.seek(0)
    for f in q.frames():
        if f.pts in samples:
            full[f.pts] = luma(f)
    rate = len(samples) / SECS
    check(f'{label}: at least one sample a second', rate >= 1.0, f'{rate:.2f}/s')
    check(f'{label}: no more than the 2/s asked for', rate <= 2.1, f'{rate:.2f}/s')
    wrong = [k for k in samples if k not in full or not np.array_equal(samples[k], full[k])]
    check(f'{label}: every sample matches a full decode', not wrong, f'{len(wrong)} of {len(samples)} differ')
    check(f'{label}: the decoder is back to decoding everything', p._video_stream().codec_context.skip_frame == 'DEFAULT',
          str(p._video_stream().codec_context.skip_frame))

if _fails:
    print(f'{len(_fails)} failed')
    sys.exit(1)
print('all passed')
