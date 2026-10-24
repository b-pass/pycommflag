#!/usr/bin/env python3
"""Move the audio columns of old feature logs back into line with the video.

Before 2026-10-23 extraction timed audio from the audio stream's own start and video from
the video stream's, so every audio column (fvol, rvol, silence/speech/music/noise) sits late
by video_start - audio_start: 0.2-1s on OTA .ts recordings, different for each one.  The
logs don't record either start, so the offset comes from one of:

  probe     the log's video still exists: its stream start times, read the way Player
            reads them, which is exactly what the fixed extractor corrects for
  estimate  the log itself: at break boundaries the picture goes black and the sound
            goes quiet together, so the lag that best lines up blank frames with low
            volume is the offset (within ~0.1s of the probe on 15 recordings)
  default   too few blanks, an implausible estimate, or an ambiguous one (two lags that
            line up nearly equally well): DEFAULT_SHIFT, the median offset

A probe that disagrees with the estimate by more than PROBE_TRUST is not used, since it
can be thrown off by a hole in the audio at the start of the recording.

The repaired log records the shift in "audio_shift" (seconds the audio was moved earlier)
and where it came from in "audio_shift_from".  Logs that already have "audio_shift" (repaired,
or made by the fixed extractor, which writes 0) are left alone.  The neural caches beside a
repaired log (<stem>.data.npy, <stem>.nologo.data.npy) are deleted so they rebuild.

    tools/repair_audio_shift.py [--dry-run] LOG...
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import processor

AUDIO_COLS = ['fvol', 'rvol', 'silence', 'speech', 'music', 'noise']
DEFAULT_SHIFT = 0.65         # median of probed offsets and of estimates over data/test+blah
PLAUSIBLE = (0.0, 1.3)       # probed offsets ran 0.18-0.96s; outside this, distrust the estimate
MIN_BLANK_FRAMES = 10
MAX_LAG = 2.0                # seconds searched either way
PROBE_TRUST = 0.5            # seconds
AMBIGUOUS_SPACING = 0.2      # seconds; a rival peak this far from the best...
AMBIGUOUS_RATIO = 0.95       # ...and this nearly as strong makes the estimate a guess

def estimate(frames:np.ndarray, frame_rate:float) -> tuple[float|None, str]:
    """Lag (seconds the audio is late) that best lines blank frames up with low volume, or
    None and the reason there isn't a trustworthy one."""
    if np.count_nonzero(frames[:, 2] > .5) < MIN_BLANK_FRAMES:
        return None, 'too few blanks'
    # On a uniform time grid, not per frame: film-cadence recordings flagged at 29.97 have
    # uneven frame gaps averaging up to ~0.04s, so a lag counted in frames reads ~20% short.
    t = frames[:, 0]
    step = 1.0 / frame_rate
    grid = np.arange(t[0], t[-1], step)
    nearest = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    blank, fvol = frames[nearest, 2], frames[nearest, 4]
    # only look within a couple of seconds of a blank run, so ordinary quiet moments in the
    # show don't dilute the alignment
    near = np.convolve(blank, np.ones(int(frame_rate*4)), 'same') > 0
    x = np.where(near, blank, 0)
    y = np.where(near, -np.log10(np.maximum(fvol, 1e-5)), 0)
    x = x - x.mean()
    y = y - y.mean()
    maxlag = int(round(MAX_LAG * frame_rate))
    lags = np.arange(-maxlag, maxlag+1)
    c = np.array([np.dot(x[max(0, -L):len(x)-max(0, L)], y[max(0, L):len(y)-max(0, -L)]) for L in lags])
    best = int(np.argmax(c))
    # Some recordings line up nearly as well at two lags a fraction of a second apart (or
    # more); picking between them would be arbitrary, so don't.  Rivals must be peaks of
    # their own: the shoulders of one broad peak are not a second opinion.
    peaks = np.flatnonzero((c[1:-1] > c[:-2]) & (c[1:-1] >= c[2:])) + 1
    rivals = peaks[np.abs(lags[peaks] - lags[best]) * step > AMBIGUOUS_SPACING]
    if c[best] <= 0 or (len(rivals) and c[rivals].max() >= AMBIGUOUS_RATIO * c[best]):
        return None, 'ambiguous'
    return float(lags[best] * step), ''

def probe(filename:str) -> float|None:
    if not filename or not os.path.exists(filename):
        return None
    try:
        import av
        with av.open(filename) as c:
            v, a = c.streams.video[0], c.streams.audio[0]
            return float(v.start_time * v.time_base - a.start_time * a.time_base)
    except Exception:
        return None

def choose_shift(flog:dict, frames:np.ndarray) -> tuple[float, str, str]:
    est, why = estimate(frames, flog.get('frame_rate', 29.97))
    est_ok = est is not None and PLAUSIBLE[0] <= est <= PLAUSIBLE[1]
    prb = probe(flog.get('filename'))
    note = f"estimate={why or f'{est:+.3f}'} probe={'none' if prb is None else f'{prb:+.3f}'}"
    if prb is not None and (not est_ok or abs(prb - est) <= PROBE_TRUST) and PLAUSIBLE[0] <= prb <= PLAUSIBLE[1]:
        return prb, 'probe', note
    if est_ok:
        return est, 'estimate', note
    return DEFAULT_SHIFT, 'default', note

def source_rows(t:np.ndarray, shift:float) -> np.ndarray:
    """For each frame, the frame nearest `shift` seconds later (by timestamp, so dropped
    frames don't throw it off); past the end, the last frame.  Estimated shifts are whole
    frames, so the target lands on a frame, not between two where timestamp jitter would
    pick."""
    target = t + shift
    hi = np.minimum(np.searchsorted(t, target), len(t) - 1)
    lo = np.maximum(hi - 1, 0)
    return np.where(np.abs(t[lo] - target) <= np.abs(t[hi] - target), lo, hi)

def shift_audio(rows:list, shift:float) -> None:
    """In place: each frame takes the audio columns of the frame `shift` seconds later; the
    last `shift` seconds, whose audio isn't in the log, repeat the last frame's."""
    t = np.array([r[0] for r in rows], dtype='float64')
    src = source_rows(t, shift)
    audio = [r[4:10] for r in rows]
    for r, j in zip(rows, src):
        r[4:10] = audio[j]

def repair(path:str, dry_run:bool) -> str:
    flog = processor.read_feature_log(path)
    if 'audio_shift' in flog:
        return f"skip (already has audio_shift={flog['audio_shift']})"
    if flog.get('frames_header', [])[4:10] != AUDIO_COLS or not flog.get('frames'):
        return f"skip (unexpected frames_header {flog.get('frames_header')})"

    rows = flog['frames']
    lead = 1 if rows[0] is None else 0
    body = rows[lead:]
    shift, source, note = choose_shift(flog, np.array(body, dtype='float64'))
    shift = round(shift, 4)  # applied exactly as recorded
    msg = f"shift {shift:+.3f}s from {source:8s} ({note})"
    if dry_run:
        return msg

    shift_audio(body, shift)

    # put the new keys up with the other timing fields rather than after the frames
    out = {}
    for k, v in flog.items():
        out[k] = v
        if k == 'frame_rate':
            out['audio_shift'] = shift
            out['audio_shift_from'] = source
    if 'audio_shift' not in out:
        out['audio_shift'] = shift
        out['audio_shift_from'] = source

    mode = os.stat(path).st_mode & 0o777
    d, base = os.path.split(path)
    tmp = os.path.join(d, '.repairing-' + base)  # same suffix, so .gz still means gzip
    processor.write_feature_log(out, tmp)
    os.chmod(tmp, mode)
    os.replace(tmp, path)

    stem = path
    for ext in ('.gz', '.json'):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
    for cache in (stem + '.data.npy', stem + '.nologo.data.npy'):
        if os.path.exists(cache):
            os.remove(cache)
    return msg

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dry-run', action='store_true', help='report the shifts, change nothing')
    ap.add_argument('logs', nargs='+')
    args = ap.parse_args()
    failed = 0
    for path in args.logs:
        try:
            print(f'{path}: {repair(path, args.dry_run)}', flush=True)
        except Exception as e:
            failed += 1
            print(f'{path}: FAILED {e!r}', flush=True)
    return 1 if failed else 0

if __name__ == '__main__':
    sys.exit(main())
