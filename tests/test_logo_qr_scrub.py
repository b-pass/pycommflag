#!/usr/bin/env python3
"""tools/logo_qr_scrub.py: which stretches a run of QR keyframes clears, and that clearing only
turns logo_present 1 into 0 inside them.  No video: the keyframe readings are made up."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import logo_qr_scrub as S

fails = 0
def check(name, cond, detail=''):
    global fails
    print(('ok   ' if cond else 'FAIL ') + name + (f': {detail}' if detail and not cond else ''))
    fails += not cond

times = [float(t) for t in range(0, 200, 4)]            # a keyframe every 4 s
def readings(**at):
    return [at.get(f't{int(t)}', 0) for t in times]

# a QR ad from the cut at 41.5 to the cut at 101: keyframes 44..100 read it
qr = [1 if 44 <= t <= 100 else 0 for t in times]
spans = S.qr_spans(times, qr, cuts=[10.0, 41.5, 60.0, 101.0, 150.0])
check('run grows to the cuts either side', spans == [(41.5, 101.0)], spans)

spans = S.qr_spans(times, qr, cuts=[10.0, 150.0])
check('no cut nearby: stays on its keyframes', spans == [(44.0, 100.0)], spans)

spans = S.qr_spans(times, qr, cuts=[39.0, 41.5, 103.9, 105.0])
check("a cut beyond the neighbouring keyframe isn't used", spans == [(41.5, 103.9)], spans)

check('one undecodable find is ignored', S.qr_spans(times, readings(t80=1), cuts=[78.0, 83.0]) == [])
spans = S.qr_spans(times, readings(t80=2), cuts=[78.0, 83.0])
check('one cleanly decoded code counts', spans == [(78.0, 83.0)], spans)

qr2 = list(qr); qr2[times.index(72.0)] = 0              # one keyframe misread mid-ad
spans = S.qr_spans(times, qr2, cuts=[41.5, 70.0, 74.0, 101.0])
check('a misread keyframe does not split the span', spans == [(41.5, 101.0)], spans)

spans = S.qr_spans(times, readings(t20=1, t24=1, t120=1, t124=1), cuts=[])
check('separate ads stay separate', spans == [(20.0, 24.0), (120.0, 124.0)], spans)

spans = S.qr_spans(times, readings(t0=1, t4=1, t192=1, t196=1), cuts=[5.0, 190.0])
check('runs at the ends of the video', spans == [(0.0, 5.0), (190.0, 196.0)], spans)

check('nothing found, nothing cleared', S.qr_spans(times, readings(), cuts=[50.0]) == [] and S.qr_spans([], [], []) == [])

# clearing: only logo_present, only 1 -> 0, only inside the span; gaps in the frame list survive
flog = {'frames': [[t / 2, int(t % 7 != 0), 0, 1.5, 0.2] for t in range(400)]}
flog['frames'][50] = None
before = [r and list(r) for r in flog['frames']]
n = S.clear_spans(flog, [(41.5, 101.0)])
inside = lambda r: 41.5 <= r[0] <= 101.0
ok = all((a is None and b is None) or (a[:1] + a[2:] == b[:1] + b[2:] and (b[1] == 0 if inside(a) else b[1] == a[1]))
         for a, b in zip(before, flog['frames']))
check('clear_spans changes only logo_present inside the span', ok)
check('clear_spans counts the frames it changed',
      n == [sum(1 for a in before if a and inside(a) and a[1])], n)
check('clearing again changes nothing', S.clear_spans(flog, [(41.5, 101.0)]) == [0])

# sizes tried for an upscaled SD recording: the video's own first, then SD sizes the box fits in
logo = [[388, 605], [425, 677], None, 100]
check('upscale candidates', S.sizes_for(1080, 1920, logo) == [(1080, 1920), (480, 720), (480, 704), (576, 720), (720, 1280)],
      S.sizes_for(1080, 1920, logo))
check('own-size video has one candidate', S.sizes_for(480, 720, logo) == [(480, 720)])

sys.exit(1 if fails else 0)
