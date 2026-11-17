#!/usr/bin/env python3
"""Bring old feature logs up to date: the .mkv's lag, the audio timing and the captions.

Each step is skipped when the log already has what it adds, so this can be run on any log.  The
log is read once and rewritten once (in place, same compression and permissions).

mkv-lag  (`mkv_lag`, `mkv_lag_from`) When the log's own video is gone and only the .mkv it
         was transcoded to is left: how far the .mkv's clock is behind the log's (see
         pycommflag/timeline.py).  The other steps need it whenever they read the .mkv, and the
         editor and the outputs read it instead of measuring again.

audio    (`audio_shift`, `audio_shift_from`) Before 2026-10-23 extraction timed audio from the
         audio stream's own start and video from the video stream's, so every audio column
         (fvol, rvol, silence/speech/music/noise) sits late by video_start - audio_start: 0.2-1s
         on OTA .ts recordings, different for each one.  The logs don't record either start, so
         the offset comes from one of:

  probe     the log's video still exists: its stream start times, read the way Player reads
            them, which is exactly what the fixed extractor corrects for
  probe-mkv the same, from the .mkv, less the .mkv's lag (its audio is the original's,
            copied, but its video starts at the first keyframe; without a measured lag
            the .mkv isn't used)
  estimate  the log itself: at break boundaries the picture goes black and the sound goes
            quiet together, so the lag that best lines up blank frames with low volume is
            the offset (within 0.1s of the probe on 93% of 1197 recordings, 0.2s on 98%;
            overriding it with the channel's typical offset only did worse)
  channel   too few blanks, an implausible estimate, or an ambiguous one (two lags that line
            up nearly equally well): the median probed offset of other recordings of the same
            channel still beside the log's own (offsets are set by the station's encoder; this
            beat DEFAULT_SHIFT on the same 1197)
  default   the same, where no other recordings of the channel are left: DEFAULT_SHIFT, the
            median offset overall

         A probe that disagrees with the estimate by more than PROBE_TRUST is not used, since it
         can be thrown off by a hole in the audio at the start of the recording.  Logs made by
         the fixed extractor write audio_shift 0.  The neural caches beside a repaired log
         (<stem>.data.npy, <stem>.nologo.data.npy) are deleted so they rebuild.  With
         --redo-from BACKUP_DIR, already repaired logs are repaired again from the original
         audio columns in their backup (found under BACKUP_DIR by the log's path, .json or
         .json.gz), keeping everything else in the current log, such as tags edited since.

cc       (`cc_text`, `cc_text_from`, `cc_data`, `cc_data_from`) Logs before file_version 11
         have no captions, and those before 12 only count them (cc_text), without the caption
         bytes (cc_data) that hold their modes, positions and styles.  Broadcast captions (ATSC
         A/53 "GA94" user data: MPEG-2 picture user data, or an H.264/HEVC SEI message) ride in
         each video packet as plain bytes, so they can be read without decoding: a few seconds
         per hour of video, where re-extracting the log takes ~60 CPU-minutes.  Each frame gets
         the number of CEA-608 field-1 characters it carries (processor.cc_chars) and the bytes
         processor.cc_keep keeps.  From the log's own video every packet's timestamp is a
         frame's (checked identical to an extraction's cc_text on H.264 and MPEG-2, 29.97 and
         59.94), and a packet with no frame (one the decoder dropped) counts for nothing; the
         `_from` keys are `packets`.  From the .mkv (`packets-mkv`), a packet counts for the
         frame shown at its time, plus the lag: 10s totals then match an extraction's within
         ~1%, also through a 29.97 -> 59.94 transcode, except where a soft-pulldown film transcode
         re-timed the frames and dropped some, with their captions (~4% fewer characters, and
         ~5% of the log's frames with no packet).  Whichever of the two a log already has
         is left alone (an extraction's cc_text beats one from an .mkv).  Some transcodes
         dropped the captions, so a video whose total disagrees with the log's cc_text by over
         CC_AGREE, or an .mkv with none at all, is skipped rather than recorded as captionless.
         Sets file_version 12 once the log has cc_data.  --redo-cc redoes both, for logs whose captions came from an
         .mkv before the lag was applied.

    tools/upgrade_log.py [--dry-run] [--only mkv-lag|audio|cc]... [--redo-from BACKUP_DIR] [--redo-cc] LOG...
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import processor, timeline

STEPS = ('mkv-lag', 'audio', 'cc')

# --- audio ----------------------------------------------------------------------------------

AUDIO_COLS = ['fvol', 'rvol', 'silence', 'speech', 'music', 'noise']
DEFAULT_SHIFT = 0.65         # median of probed offsets and of estimates over data/test+blah
PLAUSIBLE = (0.0, 1.3)       # probed offsets ran 0.18-0.96s; outside this, distrust the estimate
MIN_BLANK_FRAMES = 10
MAX_LAG = 2.0                # seconds searched either way
PROBE_TRUST = 0.5            # seconds
CHANNEL_PROBES = 25          # other recordings probed for a channel's typical offset
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
    """video_start - audio_start of `filename`, or None."""
    if not filename or not os.path.exists(filename):
        return None
    starts = timeline.stream_starts(filename)
    return None if starts is None else starts[0] - starts[1]

_channel_offsets = {}

def channel_offset(flog:dict) -> float|None:
    """Median probed offset of other recordings of the log's channel in the directory its
    own recording was in (up to CHANNEL_PROBES, spread across them), or None."""
    fn = flog.get('filename') or ''
    m = re.match(r'(\d{4,6})_\d{12,}', os.path.basename(fn))
    ch = str(flog.get('chanid') or (m.group(1) if m else ''))
    d = os.path.dirname(fn)
    if not ch or not d:
        return None
    if (d, ch) not in _channel_offsets:
        files = sorted(f for f in glob.glob(os.path.join(d, f'{ch}_*.ts')) if f != fn)
        picks = [files[int(i)] for i in np.linspace(0, len(files) - 1, min(CHANNEL_PROBES, len(files)))] if files else []
        offs = [o for o in map(probe, dict.fromkeys(picks)) if o is not None and PLAUSIBLE[0] <= o <= PLAUSIBLE[1]]
        _channel_offsets[(d, ch)] = float(np.median(offs)) if len(offs) >= 3 else None
    return _channel_offsets[(d, ch)]

def choose_shift(flog:dict, frames:np.ndarray, video:str|None, lag:float|None) -> tuple[float, str, str]:
    """(shift, source, note) for a log whose video is `video` (its own, its .mkv, or None),
    `lag` seconds behind it (None if unknown)."""
    est, why = estimate(frames, flog.get('frame_rate', 29.97))
    est_ok = est is not None and PLAUSIBLE[0] <= est <= PLAUSIBLE[1]
    prb, src, shown = None, 'probe', 'none'
    if video and timeline.same_video(flog, video):
        prb = probe(video)
        shown = 'none' if prb is None else f'{prb:+.3f}'
    elif video:
        src = 'probe-mkv'
        if lag is None:
            shown = 'mkv lag unknown'
        elif (raw := probe(video)) is not None:
            prb = raw - lag
            shown = f'{prb:+.3f} (mkv {raw:+.3f} less its lag)'
    note = f"estimate={why or f'{est:+.3f}'} probe={shown}"
    if prb is not None and (not est_ok or abs(prb - est) <= PROBE_TRUST) and PLAUSIBLE[0] <= prb <= PLAUSIBLE[1]:
        return prb, src, note
    if est_ok:
        return est, 'estimate', note
    ch = channel_offset(flog)
    if ch is not None:
        return ch, 'channel', note
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

def shift_audio(rows:list, shift:float, original:list|None=None) -> None:
    """In place: each frame takes the audio columns of the frame `shift` seconds later in
    `original` (by default `rows` itself, which must have the same timestamps); the last
    `shift` seconds, whose audio isn't in the log, repeat the last frame's."""
    original = rows if original is None else original
    t = np.array([r[0] for r in original], dtype='float64')
    src = source_rows(t, shift)
    audio = [list(r[4:10]) for r in original]
    for r, j in zip(rows, src):
        r[4:10] = audio[j]

def _backup_of(path:str, backup_dir:str) -> str|None:
    """The log's original under backup_dir: the longest trailing part of its path that
    exists there, with or without .gz."""
    parts = os.path.normpath(path).split(os.sep)
    for k in range(len(parts), 0, -1):
        rel = os.path.join(*parts[-k:])
        for cand in (rel, rel[:-3] if rel.endswith('.gz') else rel + '.gz'):
            if os.path.exists(os.path.join(backup_dir, cand)):
                return os.path.join(backup_dir, cand)
    return None

def _body(flog:dict) -> list:
    rows = flog['frames']
    return rows[1:] if rows and rows[0] is None else rows

def repair_audio(flog:dict, path:str, video:str|None, lag:float|None, redo_from:str|None=None) -> tuple[dict, str, bool]:
    """(the log, what happened, whether it changed); the log's frames are changed in place."""
    if redo_from is None:
        if 'audio_shift' in flog:
            return flog, f"skip (already has audio_shift={flog['audio_shift']})", False
        orig = flog
    else:
        if 'audio_shift' not in flog:
            return flog, "skip (never repaired; repair it without --redo-from)", False
        backup = _backup_of(path, redo_from)
        if backup is None:
            return flog, f"skip (no original under {redo_from})", False
        orig = processor.read_feature_log(backup)
    if orig.get('frames_header', [])[4:10] != AUDIO_COLS or not orig.get('frames'):
        return flog, f"skip (unexpected frames_header {orig.get('frames_header')})", False

    body, obody = _body(flog), _body(orig)
    if obody is not body and [r[0] for r in body] != [r[0] for r in obody]:
        return flog, "skip (frame times differ from the original's)", False
    shift, source, note = choose_shift(orig, np.array([r[:10] for r in obody], dtype='float64'), video, lag)
    shift = round(shift, 4)  # applied exactly as recorded
    was = f" (was {flog['audio_shift']:+.3f} from {flog.get('audio_shift_from')})" if redo_from else ''
    msg = f"shift {shift:+.3f}s from {source}{was} ({note})"

    shift_audio(body, shift, obody if obody is not body else None)
    flog = processor.set_header_key(flog, 'audio_shift', shift, after='frame_rate')
    flog = processor.set_header_key(flog, 'audio_shift_from', source, after='audio_shift')
    return flog, msg, True

# --- captions -------------------------------------------------------------------------------

GA94 = re.compile(rb'GA94\x03')
NAL_CODECS = ('h264', 'hevc')   # caption bytes sit in a NAL unit, behind emulation prevention
MIN_MATCHED = 0.99              # share of the log's frames that must line up with a packet...
MIN_MATCHED_MKV = 0.9           # ...or a transcode's (film transcodes dropped ~5% of the frames)
CC_AGREE = 0.1                  # a transcode's caption total is within ~1% of the extraction's (4% for film)

def packet_cc(data:bytes, nal:bool) -> bytes:
    """The caption bytes in one video packet, as processor.cc_keep keeps them."""
    if b'GA94' not in data:
        return b''
    if nal:
        data = data.replace(b'\x00\x00\x03', b'\x00\x00')
    out = b''
    for m in GA94.finditer(data):
        p = m.start()
        # what precedes it: the T.35 country/provider code (0xB5, 0x0031) in an SEI, or the
        # user_data start code in MPEG-2; anything else is a chance match in picture data
        if data[p-4 + nal:p] != (b'\xb5\x00\x31' if nal else b'\x00\x00\x01\xb2'):
            continue
        q = m.end()
        if q + 2 > len(data) or not (data[q] & 0x40):
            continue  # no process_cc_data_flag
        count = data[q] & 0x1f
        out += processor.cc_keep(data[q+2:q+2+count*3])  # skip the flags byte and em_data
    return out

def read_packets(video:str) -> tuple[np.ndarray, list[bytes]]:
    """(times from the stream's start, caption bytes) per packet of the first video stream, the one Player decodes."""
    import av
    times, data = [], []
    with av.open(video) as c:
        vs = c.streams.video[0]
        nal = vs.codec_context.name in NAL_CODECS
        if not nal and vs.codec_context.name != 'mpeg2video':
            raise ValueError(f"don't know where {vs.codec_context.name} keeps captions")
        start, tb = vs.start_time or 0, vs.time_base
        for pkt in c.demux(vs):
            if pkt.pts is None or pkt.size == 0:
                continue
            times.append(float((pkt.pts - start) * tb))
            data.append(packet_cc(bytes(pkt), nal))
    order = np.argsort(times, kind='stable')
    return np.asarray(times)[order], [data[i] for i in order]

def assign(row_times:np.ndarray, times:np.ndarray, data:list[bytes], exact:bool, frame_rate:float) -> tuple[list[bytes], float]:
    """Caption bytes per row (its packets', in time order), and the share of rows that a packet
    lined up with.

    exact: packets count only for the row with their own timestamp (within 1ms; the log
    rounds times to 5 places).  Otherwise each counts for the row shown at its time, unless
    that row is more than 1.5 frames older (a gap in the log)."""
    out = [b''] * len(row_times)
    idx = np.searchsorted(row_times, times + 0.001, side='right') - 1
    ok = idx >= 0
    lag = times - row_times[np.maximum(idx, 0)]
    ok &= (np.abs(lag) <= 0.001) if exact else (lag <= 1.5 / frame_rate)
    for i in np.flatnonzero(ok):
        if data[i]:
            out[idx[i]] += data[i]
    hit = np.zeros(len(row_times), dtype=bool)
    hit[idx[ok]] = True
    return out, float(hit.mean()) if len(hit) else 0.0

def add_cc(flog:dict, video:str|None, lag:float|None, redo:bool=False) -> tuple[dict, str, bool]:
    """(the log, what happened, whether it changed); the log's frames are changed in place."""
    header = list(flog.get('frames_header', []))
    want_text = 'cc_text' not in header or redo
    want_data = 'cc_data' not in flog or redo
    if not (want_text or want_data):
        return flog, 'skip (already has cc_text and cc_data)', False
    if header[:1] != ['time']:
        return flog, f'skip (unexpected frames_header {header})', False
    if video is None:
        return flog, f"skip (no video: {flog.get('filename')})", False
    exact = timeline.same_video(flog, video)
    if not exact and lag is None:
        return flog, f'skip (where the log falls in {os.path.basename(video)} is unknown)', False

    body = _body(flog)
    col = header.index('cc_text') if 'cc_text' in header else None
    row_times = np.array([r[0] for r in body], dtype='float64')
    times, data = read_packets(video)
    if not exact:
        times = times + lag  # into the log's time
    cc, matched = assign(row_times, times, data, exact, flog.get('frame_rate', 29.97))
    chars = [processor.cc_chars(b) for b in cc]
    how = 'own video' if exact else f'{os.path.basename(video)}, {lag:+.3f}s'
    msg = (f'{sum(chars)} chars on {sum(1 for n in chars if n)} of {len(cc)} frames, '
           f'{sum(map(len, cc))} bytes on {sum(1 for b in cc if b)}, {matched:.1%} matched ({how})')
    if matched < (MIN_MATCHED if exact else MIN_MATCHED_MKV):
        return flog, f'skip (only {msg}; not this log\'s video?)', False
    # Some transcodes drop the captions: an empty cc_data would then claim the recording had none
    if col is not None:
        had = sum(r[col] for r in body)
        if abs(sum(chars) - had) > CC_AGREE * had + 50:
            return flog, f"skip ({msg}, but the log's cc_text has {had} chars: the video lost captions?)", False
    elif not exact and not any(cc):
        return flog, f'skip ({msg}: none in a transcode, which may have dropped them)', False

    source = 'packets' if exact else 'packets-mkv'
    after = next(k for k in ('mkv_lag_from', 'audio_shift_from', 'audio_shift', 'frame_rate', 'filename') if k in flog or k == 'filename')
    done = []
    if want_text:
        for r, n in zip(body, chars):
            if col is None:
                r.append(n)
            else:
                r[col] = n
        if col is None:
            flog['frames_header'] = header + ['cc_text']
        flog['file_version'] = max(11, flog.get('file_version', 0))
        flog = processor.set_header_key(flog, 'cc_text_from', source, after=after)
        done.append('cc_text' if col is None else 'cc_text re-counted')
    if want_data:
        had = 'cc_data' in flog
        # after the frames, as extraction writes it: it is as long as they are
        flog.pop('cc_data', None)
        flog['cc_data'] = [[r[0], b.hex()] for r, b in zip(body, cc) if b]
        flog['file_version'] = max(12, flog.get('file_version', 0))
        flog = processor.set_header_key(flog, 'cc_data_from', source,
                                        after='cc_text_from' if 'cc_text_from' in flog else after)
        done.append('cc_data redone' if had else 'cc_data')
    return flog, f"{' and '.join(done)}: {msg}", True

# --- the log --------------------------------------------------------------------------------

def find_video(flog:dict) -> str|None:
    """The log's own video, else the .mkv it was transcoded to, else None."""
    fn = flog.get('filename') or ''
    if fn and os.path.exists(fn):
        return fn
    if fn.endswith('.ts') and os.path.isfile(fn[:-3] + '.mkv'):
        return fn[:-3] + '.mkv'
    return None

def upgrade(path:str, steps=STEPS, dry_run:bool=False, redo_from:str|None=None, redo_cc:bool=False) -> list[str]:
    """Run `steps` on the log at `path`; one line per step on what it did."""
    flog = processor.read_feature_log(path)
    video = find_video(flog)
    out, audio_changed, changed = [], False, False

    lag = None
    if video and timeline.same_video(flog, video):
        lag = 0.0
        if 'mkv-lag' in steps:
            out.append(f'mkv lag: skip (not transcoded: {os.path.basename(video)} still exists, so the log lines up with it)')
    elif video and flog.get('mkv_lag') is not None:
        lag = float(flog['mkv_lag'])
        if 'mkv-lag' in steps:
            out.append(f"mkv lag: skip (already has mkv_lag={lag})")
    elif video:
        header = flog.get('frames_header', [])
        needed = ('mkv-lag' in steps
                  or ('audio' in steps and ('audio_shift' not in flog or redo_from))
                  or ('cc' in steps and ('cc_text' not in header or 'cc_data' not in flog or redo_cc)))
        if needed:
            # recorded whichever step asked for it, so nothing has to measure it again
            lag, source = timeline.video_lag(flog, video)
            if lag is None:
                out.append(f'mkv lag: FAILED (the cuts in {os.path.basename(video)} did not line up with the log\'s)')
            else:
                flog = timeline.record_lag(flog, lag, source)
                changed = True
                out.append(f'mkv lag: {os.path.basename(video)} is {lag:+.3f}s behind the log (from {source})')
    elif 'mkv-lag' in steps:
        out.append(f"mkv lag: skip (no video: {flog.get('filename')})")

    if 'audio' in steps:
        flog, msg, audio_changed = repair_audio(flog, path, video, lag, redo_from)
        changed |= audio_changed
        out.append(f'audio: {msg}')
    if 'cc' in steps:
        flog, msg, did = add_cc(flog, video, lag, redo_cc)
        changed |= did
        out.append(f'cc: {msg}')

    if changed and not dry_run:
        mode = os.stat(path).st_mode & 0o777
        d, base = os.path.split(path)
        tmp = os.path.join(d, '.upgrading-' + base)  # same suffix, so .gz still means gzip
        processor.write_feature_log(flog, tmp)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        if audio_changed:
            stem = path
            for ext in ('.gz', '.json'):
                if stem.endswith(ext):
                    stem = stem[:-len(ext)]
            for cache in (stem + '.data.npy', stem + '.nologo.data.npy'):
                if os.path.exists(cache):
                    os.remove(cache)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dry-run', action='store_true', help='report what would change, change nothing')
    ap.add_argument('--only', action='append', choices=STEPS, metavar='STEP',
                    help=f'run only this step ({", ".join(STEPS)}); may be repeated')
    ap.add_argument('--redo-from', metavar='BACKUP_DIR',
                    help='repair the audio of already repaired logs again, from the originals in this directory')
    ap.add_argument('--redo-cc', action='store_true', help='read the captions again even if the log has them')
    ap.add_argument('logs', nargs='+')
    args = ap.parse_args()
    failed = 0
    for path in args.logs:
        try:
            lines = upgrade(path, args.only or STEPS, args.dry_run, args.redo_from, args.redo_cc)
            failed += any('FAILED' in s for s in lines)
            print(f'{path}:' + ''.join(f'\n  {s}' for s in lines), flush=True)
        except Exception as e:
            failed += 1
            print(f'{path}: FAILED {e!r}', flush=True)
    return 1 if failed else 0

if __name__ == '__main__':
    sys.exit(main())
