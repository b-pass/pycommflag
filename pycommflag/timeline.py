"""Where a feature log's times fall in another copy of its video, such as the .mkv a .ts was
transcoded to.

Every time in a log is seconds from its video stream's start_time, as Player counts it.  A
recording starts mid-GOP, so the first frames of a .ts can't be decoded, and a re-encode drops
them: the transcode's video starts at the first keyframe instead (0.5-1.1s later), and its clock
starts there too.  The audio is copied as it was, so the shift shows in the stream start times:
it is how much further the copy's video start sits from its audio start than the original's
did.  Logs made before file_version 11 don't record the original's start times, so for those
it is measured, by lining up scene cuts in the log's `diff` column with cuts decoded from the
copy.  Not every transcode moves (the HD ones measured so far kept the original's timeline), so
it is never assumed.

The shift is recorded in a log as `mkv_offset` (log time - the .mkv's time) and where it came
from in `mkv_offset_from` (`starts` or `cuts`).
"""
import logging as log
import os

import numpy as np

CUT_DIFF = 15.0       # a log frame's diff this high is a cut (neural.DIFF_THRESHOLD)
VIDEO_CUT_DIFF = 9.0  # lower for the copy's, decoded smaller: extra cuts there only add noise
MAX_OFFSET = 3.0      # seconds searched either way
WINDOW = 40.0         # seconds of video decoded per window
WINDOWS = 2           # windows decoded, the busiest stretches of cuts in the log (2x40s: ~20 CPU-s
                      # on a 1080p .mkv, and the same answer as 3x60s to within a frame)
MIN_WINDOW_CUTS = 5   # log cuts a window needs to be worth decoding
MIN_MATCHED = 8       # log cuts that must agree on the offset...
MIN_SHARE = 0.5       # ...and the share of all those decoded that must
DECODE_SIZE = (320, 180)

def same_video(flog:dict, video:str) -> bool:
    fn = flog.get('filename') or ''
    return not fn or not video or os.path.realpath(fn) == os.path.realpath(video)

def stream_starts(video:str) -> tuple[float, float]|None:
    """(video, audio) start_time in seconds of the first video and audio streams, or None."""
    try:
        import av
        with av.open(video) as c:
            v, a = c.streams.video[0], c.streams.audio[0]
            return float(v.start_time * v.time_base), float(a.start_time * a.time_base)
    except Exception:
        return None

def offset_from_starts(flog:dict, video:str) -> float|None:
    """log time - `video`'s time, from the stream start times the log recorded, or None."""
    if flog.get('video_start') is None or flog.get('audio_start') is None:
        return None
    starts = stream_starts(video)
    if starts is None:
        return None
    return round((starts[0] - starts[1]) - (flog['video_start'] - flog['audio_start']), 4)

def _peaks(t:np.ndarray, d:np.ndarray, threshold:float) -> np.ndarray:
    """Times of the frames whose diff is at least `threshold` and a local maximum."""
    if len(d) < 3:
        return np.zeros(0)
    prev = np.concatenate(([-np.inf], d[:-1]))
    nxt = np.concatenate((d[1:], [-np.inf]))
    return t[(d >= threshold) & (d > prev) & (d >= nxt)]

def match_cuts(log_cuts:list[np.ndarray], video_cuts:list[np.ndarray], frame_rate:float) -> float|None:
    """log time - video time that lines the most cuts up, from per-window lists of cut times
    (log cuts in log time, the video's in its own), or None without a clear answer.

    Every pairing within MAX_OFFSET votes for its difference, in bins a frame wide; the winning
    bin needs MIN_MATCHED log cuts (and MIN_SHARE of them) from at least two windows.  Cuts in
    the copy often land a frame off (blended by a deinterlacer or scaler), so the answer is the
    median of the votes within a frame and a half of the winner."""
    step = 1.0 / frame_rate
    votes, owners = [], []  # (video - log) per pairing, and (window, log cut) it came from
    for w, (lc, vc) in enumerate(zip(log_cuts, video_cuts)):
        for i, c in enumerate(lc):
            d = vc - c
            d = d[np.abs(d) <= MAX_OFFSET]
            votes.extend(d.tolist())
            owners.extend([(w, i)] * len(d))
    if not votes:
        return None
    votes = np.array(votes)
    bins = np.round(votes / step).astype(int)
    best = np.bincount(bins - bins.min()).argmax() + bins.min()
    near = np.abs(votes - best * step) <= 1.5 * step
    matched = {owners[k] for k in np.flatnonzero(near)}
    total = sum(len(lc) for lc in log_cuts)
    if len(matched) < max(MIN_MATCHED, MIN_SHARE * total) or len({w for w, _ in matched}) < min(2, len(log_cuts)):
        return None
    return round(-float(np.median(votes[near])), 4)

def _windows(t:np.ndarray, cuts:np.ndarray) -> list[float]:
    """Starts of up to WINDOWS non-overlapping WINDOW-second stretches holding the most cuts."""
    lo, hi = t[0] + MAX_OFFSET + 1, t[-1] - WINDOW - MAX_OFFSET - 1
    starts = cuts[(cuts >= lo) & (cuts <= hi)]
    counts = np.searchsorted(cuts, starts + WINDOW) - np.searchsorted(cuts, starts)
    picked = []
    for k in np.argsort(-counts, kind='stable'):
        if counts[k] < MIN_WINDOW_CUTS or len(picked) >= WINDOWS:
            break
        if all(abs(starts[k] - s) >= WINDOW for s in picked):
            picked.append(float(starts[k]))
    return sorted(picked)

def measure_offset(flog:dict, video:str) -> float|None:
    """log time - `video`'s time, by lining up scene cuts, or None if they don't line up clearly.
    Decodes about WINDOWS * WINDOW seconds of video."""
    from .player import Player
    from .processor import mean_axis1
    rows = np.array([r[:4] for r in flog.get('frames', []) if r is not None], dtype='float64')
    if len(rows) < 2:
        return None
    t, d = rows[:, 0], rows[:, 3]
    cuts = _peaks(t, d, CUT_DIFF)
    windows = _windows(t, cuts)
    if not windows:
        return None
    player = Player(video, no_deinterlace=True)
    vt_start = float(player.vt_start)
    log_cuts, video_cuts = [], []
    for s in windows:
        e = s + WINDOW
        log_cuts.append(cuts[(cuts >= s) & (cuts <= e)])
        player.seek(max(0.0, s - MAX_OFFSET - 1))
        vt, vd, prev = [], [], None
        frames = player.frames()
        try:
            for f in frames:
                if f.time is None:
                    continue
                ft = float(f.time) - vt_start
                if ft > e + MAX_OFFSET:
                    break
                col = mean_axis1(f.to_ndarray(format='rgb24', width=DECODE_SIZE[0], height=DECODE_SIZE[1]), dtype='int16')
                if prev is not None and ft >= s - MAX_OFFSET:
                    vt.append(ft)
                    vd.append(float(np.mean(np.std(np.abs(col - prev), 0))))
                prev = col
        finally:
            frames.close()
        video_cuts.append(_peaks(np.array(vt), np.array(vd), VIDEO_CUT_DIFF))
    return match_cuts(log_cuts, video_cuts, float(flog.get('frame_rate') or player.frame_rate))

def video_offset(flog:dict, video:str, measure:bool=True) -> tuple[float|None, str]:
    """(log time - `video`'s time, where that came from): 0 for the log's own video, else the
    recorded `mkv_offset`, else from the stream start times, else measured (unless `measure` is
    False).  None if none of those could say."""
    if same_video(flog, video):
        return 0.0, 'own'
    fn = flog.get('filename') or ''
    if flog.get('mkv_offset') is not None and os.path.splitext(fn)[0] == os.path.splitext(video)[0]:
        return float(flog['mkv_offset']), flog.get('mkv_offset_from') or 'recorded'
    if (off := offset_from_starts(flog, video)) is not None:
        return off, 'starts'
    if measure:
        try:
            if (off := measure_offset(flog, video)) is not None:
                return off, 'cuts'
        except Exception:
            log.exception(f'measuring where {fn} falls in {video} failed')
    return None, 'unknown'

def record_offset(flog:dict, offset:float, source:str) -> dict:
    """A copy of flog with mkv_offset/mkv_offset_from set, up with the other timing keys."""
    from .processor import set_header_key
    after = next((k for k in ('audio_shift_from', 'audio_shift', 'frame_rate') if k in flog), 'filename')
    flog = set_header_key(flog, 'mkv_offset', offset, after=after)
    return set_header_key(flog, 'mkv_offset_from', source, after='mkv_offset')

def shift_tags(tags:list, offset:float) -> list:
    """Tags moved from log time to a video `offset` seconds behind it (none before 0)."""
    if not offset:
        return tags
    return [(t, (max(0.0, b - offset), max(0.0, e - offset))) for (t, (b, e)) in tags]
