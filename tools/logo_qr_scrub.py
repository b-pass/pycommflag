#!/usr/bin/env python3
"""Clear `logo_present` where a feature log counted a QR code over the logo as the logo.

Logs written before the extractor refused such matches have the logo "present" all through an
ad that parks a QR code on it, and the model then calls a long one show.  This decodes only the
keyframes of each log's video (its own, or the .mkv it was transcoded to), reads the logo's
quarter of the screen with zxing-cpp, and where a code is found while more than half of the
logo's box off its mask is lit (the extractor's own condition), sets `logo_present` to 0 from
the scene cut before the first such keyframe to the cut after the last.  It only ever turns 1
into 0 and touches no other column.

It does not re-run the logo match: a log's saved threshold was sized for the match rule of its
day (older ones counted edges anywhere in the box), so today's `check_frame` reads those logos
as absent through much of the show.

The .mkv of an SD recording is an upscale while the logo's box is in the original's pixels, and
the original size isn't logged: the first keyframes are tried at each plausible size and the one
that lights the mask best where the log has the logo is kept.

  logo_qr_scrub.py --out DIR LOG...                         report: DIR/logs.tsv, DIR/spans.tsv
  logo_qr_scrub.py --out DIR --apply --backup-dir B LOG...  also rewrite the logs with spans

With --apply a log is first copied to B and compared, rewritten through a temporary file, then
read back and checked against the copy; a log that fails the check is restored from it.  A
rewritten log gets a `logo_qr_scrub` key listing the spans, and is skipped from then on.
"""
import os
import sys

for _k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse
import filecmp
import multiprocessing
import shutil
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycommflag import logo_finder, processor, timeline

SD_SIZES = [(480, 720), (480, 704), (480, 640), (480, 544), (480, 528), (576, 720), (720, 1280)]
CALIBRATE = 150      # keyframes tried at every candidate size
OFF_LIT = 0.50       # as in logo_finder.check_frame: the QR test applies above this
JOIN_GAP = 8.0       # runs of QR keyframes closer than this, or one keyframe apart, are one span (a misread)
MAX_REACH = 10.0     # a span's edge moves at most this far from its QR keyframe to reach a cut
KEY = 'logo_qr_scrub'

def sizes_for(vh, vw, logo):
    (bottom, right) = logo[1]
    return [(vh, vw)] + [(h, w) for h, w in SD_SIZES if h < vh and bottom <= h and right <= w]

def box_edges(frame, logo, size):
    """(share of the mask lit, share of the rest of the box lit) with the frame scaled to `size`."""
    if (frame.height, frame.width) != size:
        frame = frame.reformat(width=size[1], height=size[0])
    (top, left), (bottom, right), lmask = logo[:3]
    edges = logo_finder._edges(logo_finder._gray(frame, [top, bottom, left, right]))
    if edges.shape != lmask.shape:
        return 0.0, 0.0
    off = ~lmask
    return (np.count_nonzero(edges & lmask) / max(1, np.count_nonzero(lmask)),
            np.count_nonzero(edges & off) / max(1, np.count_nonzero(off)))

def qr_in_quarter(frame, logo, size):
    """0: no QR code in the logo's quarter of the screen (read at the video's own resolution);
    1: one found that doesn't decode cleanly; 2: one that does."""
    (top, left), (bottom, right) = logo[0], logo[1]
    h, w = frame.height // 2, frame.width // 2
    y = h if top + bottom >= size[0] else 0
    x = w if left + right >= size[1] else 0
    q = np.ascontiguousarray(logo_finder._gray(frame, [y, y + h, x, x + w]), dtype=np.uint8)
    found = logo_finder.zxingcpp.read_barcodes(q, formats=logo_finder.zxingcpp.BarcodeFormat.QRCode, return_errors=True)
    return 0 if not found else 2 if any(r.valid for r in found) else 1

def keyframes(video):
    """(seconds from the video stream's start, frame) for every keyframe."""
    import av
    with av.open(video) as c:
        s = c.streams.video[0]
        s.codec_context.skip_frame = 'NONKEY'
        start = float(s.start_time * s.time_base) if s.start_time is not None else 0.0
        bad = 0
        packets = c.demux(s)
        while True:
            try:
                frames = next(packets).decode()
            except StopIteration:
                return
            except Exception:
                bad += 1
                if bad > 2000:
                    return
                continue
            for f in frames:
                if f.time is not None and f.key_frame:
                    yield float(f.time) - start, f

def qr_spans(times, qr, cuts):
    """(start, end) spans to clear, from keyframe `times`, their `qr` readings (0, 1, 2 as
    qr_in_quarter, already 0 where the box wasn't lit) and the log's scene-cut times.

    A run of QR keyframes counts if it has two of them or one cleanly decoded code: a lone
    undecodable find can be a texture.  The code was up for some of the time before the run's
    first keyframe and after its last, so each edge moves out to the nearest cut on its side of
    the neighbouring keyframe, MAX_REACH at most, and stays put if there is none."""
    times, qr, cuts = np.asarray(times, float), np.asarray(qr), np.sort(np.asarray(cuts, float))
    hit = qr > 0
    d = np.diff(np.concatenate(([0], hit.astype(np.int8), [0])))
    runs = [[a, b] for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    joined = []
    for a, b in runs:
        if joined and (a - joined[-1][1] <= 1 or times[a] - times[joined[-1][1] - 1] < JOIN_GAP):
            joined[-1][1] = b
        else:
            joined.append([a, b])
    out = []
    for a, b in joined:
        if hit[a:b].sum() < 2 and qr[a:b].max() < 2:
            continue
        lo = max(times[a - 1] if a else -np.inf, times[a] - MAX_REACH)
        hi = min(times[b] if b < len(times) else np.inf, times[b - 1] + MAX_REACH)
        before = cuts[(cuts > lo) & (cuts <= times[a])]
        after = cuts[(cuts >= times[b - 1]) & (cuts < hi)]
        out.append((float(before[-1]) if len(before) else float(times[a]),
                    float(after[0]) if len(after) else float(times[b - 1])))
    return out

def clear_spans(flog, spans):
    """Set logo_present to 0 in `spans`, in place; the number of frames changed per span."""
    changed = []
    rows = flog['frames']
    for s, e in spans:
        n = 0
        for r in rows:
            if r is not None and s <= r[0] <= e and r[1]:
                r[1] = 0
                n += 1
        changed.append(n)
    return changed

def scan(flog, video, lag):
    """(spans, facts) for one log: the spans to clear, in log time."""
    logo = logo_finder.from_json(flog.get('logo'))
    rows = np.array([r[:4] for r in flog['frames'] if r is not None], dtype='float64')
    lt, lp = rows[:, 0], rows[:, 1] > 0
    cuts = timeline._peaks(lt, rows[:, 3], timeline.CUT_DIFF)
    size, cands, trial = None, None, []
    T, Q, L = [], [], []
    def keep(t, logged, found, off):
        T.append(t); L.append(logged); Q.append(found if off > OFF_LIT else 0)
    for vt, f in keyframes(video):
        t = vt + lag
        if t < lt[0] - 0.5 or t > lt[-1] + 0.5:
            continue
        i = min(len(lt) - 1, int(np.searchsorted(lt, t)))
        if i and t - lt[i - 1] < lt[i] - t:
            i -= 1
        logged = bool(lp[i])
        if cands is None:
            cands = sizes_for(f.height, f.width, logo)
            if len(cands) == 1:
                size = cands[0]
        if size is None:
            trial.append((t, logged, [box_edges(f, logo, c) for c in cands], qr_in_quarter(f, logo, cands[-1])))
            if len(trial) < CALIBRATE and sum(x[1] for x in trial) < CALIBRATE // 2:
                continue
            size = _pick_size(cands, trial)
        if trial:
            k = cands.index(size)
            for x in trial:
                keep(x[0], x[1], x[3], x[2][k][1])
            trial = []
            continue
        keep(t, logged, qr_in_quarter(f, logo, size), box_edges(f, logo, size)[1])
    if trial:
        size = _pick_size(cands, trial)
        k = cands.index(size)
        for x in trial:
            keep(x[0], x[1], x[3], x[2][k][1])
    T, Q, L = np.array(T), np.array(Q), np.array(L, bool)
    spans = qr_spans(T, Q, cuts) if len(T) else []
    return spans, dict(size=f'{size[1]}x{size[0]}' if size else '', scaled=int(bool(cands) and size != cands[0]), keyframes=len(T),
                       gap=round(float(np.median(np.diff(T))), 2) if len(T) > 1 else 0,
                       qr_keyframes=int((Q > 0).sum()), qr_logged_present=int(((Q > 0) & L).sum()))

def _pick_size(cands, trial):
    """The candidate size at which the mask is best lit on the keyframes the log has the logo on."""
    on = [x for x in trial if x[1]] or trial
    return cands[int(np.argmax([np.mean([x[2][k][0] for x in on]) for k in range(len(cands))]))]

def _verify(path, backup, spans):
    """'' if the rewritten log differs from its backup only by logo_present going 1 -> 0 inside spans."""
    a, b = processor.read_feature_log(backup), processor.read_feature_log(path)
    if a.get('tags') != b.get('tags'):
        return 'tags differ'
    for k, v in a.items():
        if k != 'frames' and b.get(k) != v:
            return f'key {k} differs'
    if set(b) - set(a) != {KEY}:
        return 'unexpected keys'
    if len(a['frames']) != len(b['frames']):
        return 'frame count differs'
    for r, s in zip(a['frames'], b['frames']):
        if r == s:
            continue
        if r is None or s is None or r[:1] + r[2:] != s[:1] + s[2:] or not (r[1] and not s[1]) \
                or not any(lo <= r[0] <= hi for lo, hi in spans):
            return f'frame at {r and r[0]} differs outside the rule'
    return ''

def scrub(job):
    path, apply, backup_dir = job
    t0 = time.time()
    info = {'log': path, 'status': 'ok'}
    try:
        flog = processor.read_feature_log(path)
        if KEY in flog:
            return dict(info, status='already scrubbed'), []
        if not flog.get('logo') or not logo_finder.from_json(flog.get('logo')):
            return dict(info, status='no logo'), []
        import upgrade_log
        video = upgrade_log.find_video(flog)
        if not video:
            return dict(info, status='no video'), []
        lag, _ = timeline.video_lag(flog, video)
        if lag is None:
            return dict(info, status='lag unknown'), []
        spans, facts = scan(flog, video, lag)
        info.update(facts, video=video)
        com = [(b, e) for t, (b, e) in flog.get('tags') or [] if int(t) == 3]
        changed = clear_spans(flog, spans)   # on the dict in memory; written only with --apply
        fr = float(flog.get('frame_rate') or 29.97)
        rows = []
        for (s, e), n in zip(spans, changed):
            inc = sum(max(0.0, min(e, ce) - max(s, cs)) for cs, ce in com)
            rows.append(dict(log=path, start=round(s, 2), end=round(e, 2), secs=round(e - s, 1), frames_cleared=n,
                             tagged_commercial=round(inc / max(e - s, 1e-6), 2)))
        info.update(spans=len(spans), span_secs=round(sum(e - s for s, e in spans), 1), frames_cleared=sum(changed),
                    secs_cleared=round(sum(changed) / fr, 1))
        if apply and sum(changed):
            bk = os.path.join(backup_dir, os.path.abspath(path).lstrip('/'))
            if not os.path.exists(bk):
                os.makedirs(os.path.dirname(bk), exist_ok=True)
                shutil.copy2(path, bk)
            if not filecmp.cmp(path, bk, shallow=False):
                return dict(info, status='SKIPPED backup differs from log', secs=round(time.time() - t0, 1)), rows
            flog = processor.set_header_key(flog, KEY, [[round(s, 3), round(e, 3)] for s, e in spans],
                                            after='logo' if 'logo' in flog else 'filename')
            mode = os.stat(path).st_mode & 0o777
            d, base = os.path.split(path)
            tmp = os.path.join(d, '.scrubbing-' + base)  # same suffix, so .gz still means gzip
            processor.write_feature_log(flog, tmp)
            os.chmod(tmp, mode)
            os.replace(tmp, path)
            bad = _verify(path, bk, spans)
            if bad:
                shutil.copy2(bk, path)
                return dict(info, status=f'RESTORED {bad}', secs=round(time.time() - t0, 1)), rows
            info['status'] = 'rewritten'
            stem = path
            for ext in ('.gz', '.json'):
                if stem.endswith(ext):
                    stem = stem[:-len(ext)]
            for cache in (stem + '.data.npy', stem + '.nologo.data.npy'):
                if os.path.exists(cache):
                    os.remove(cache)
        return dict(info, secs=round(time.time() - t0, 1)), rows
    except Exception as e:
        return dict(info, status=f'FAILED {e!r}', secs=round(time.time() - t0, 1)), []

LOG_COLS = ['log', 'status', 'video', 'size', 'scaled', 'keyframes', 'gap', 'qr_keyframes', 'qr_logged_present', 'spans', 'span_secs',
            'frames_cleared', 'secs_cleared', 'secs']
SPAN_COLS = ['log', 'start', 'end', 'secs', 'frames_cleared', 'tagged_commercial']

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('logs', nargs='+', help='feature logs, or @FILE listing them')
    ap.add_argument('--out', required=True, help='directory for logs.tsv and spans.tsv (appended to)')
    ap.add_argument('--apply', action='store_true', help='rewrite the logs that have spans')
    ap.add_argument('--backup-dir', help='where --apply copies a log before rewriting it')
    ap.add_argument('--jobs', type=int, default=2)
    args = ap.parse_args()
    if args.apply and not args.backup_dir:
        ap.error('--apply needs --backup-dir')
    if logo_finder.zxingcpp is None:
        sys.exit('zxing-cpp is not installed')
    logs = []
    for a in args.logs:
        logs += open(a[1:]).read().split() if a.startswith('@') else [a]
    os.makedirs(args.out, exist_ok=True)
    os.nice(15)
    fresh = [not os.path.exists(os.path.join(args.out, n)) for n in ('logs.tsv', 'spans.tsv')]
    with open(os.path.join(args.out, 'logs.tsv'), 'a', buffering=1) as fl, open(os.path.join(args.out, 'spans.tsv'), 'a', buffering=1) as fs, \
         multiprocessing.Pool(args.jobs, maxtasksperchild=4) as pool:
        if fresh[0]: fl.write('\t'.join(LOG_COLS) + '\n')
        if fresh[1]: fs.write('\t'.join(SPAN_COLS) + '\n')
        for info, spans in pool.imap_unordered(scrub, [(p, args.apply, args.backup_dir) for p in logs]):
            fl.write('\t'.join(str(info.get(c, '')) for c in LOG_COLS) + '\n')
            for s in spans:
                fs.write('\t'.join(str(s[c]) for c in SPAN_COLS) + '\n')
            print(f"{os.path.basename(info['log'])}: {info['status']} {info.get('secs', '')}s, {info.get('spans', 0)} spans, "
                  f"{info.get('secs_cleared', 0)}s cleared", flush=True)

if __name__ == '__main__':
    main()
