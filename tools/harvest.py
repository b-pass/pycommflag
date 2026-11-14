#!/usr/bin/env python3
"""Training labels from machine-flagged feature logs, and where the models are weak.

Offline data curation only; nothing here is used for flagging.  Two independent sources of evidence:

fingerprints  An ad that airs twice has the same run of scene-cut intervals and the same front
              volume envelope both times, and the feature logs hold both (`diff`, `fvol`), so
              repeats are found across all logs without decoding video.  Five consecutive cut
              intervals that match within PATTERN_TOL propose a match; matches with one partner
              at one time offset form a run, kept when the two volume envelopes correlate over
              it (best lag within MAX_LAG: logs without `audio_shift` have audio 0.2-1s late).
              First tier: a second is proven an ad when kept runs from at least two distinct
              other titles cover it.  Titles are not programme identities (the guide spells one
              series two ways; an overrun puts one programme under another's title), so one
              partner is never enough.  Second tier, kept apart because a false mark on a title
              sequence is the costly error: other episodes of the log's own series count as
              partners too, but not for what the series repeats in many episodes, not in the
              first minutes, and only beside first-tier marks.  A dense run (matches no more
              than DENSE_GAP apart) of LONG_RUN seconds or more is the same programme or feed,
              not an ad, and nothing inside it is marked; matches far apart at one offset stay
              separate runs, since a marathon repeats its ad slots at one offset.
consensus     Several trained models (different seeds) over each log; where they agree with
              confidence, that is a label.

Stages, each with its own subcommand; the expensive ones cache under --cache:

  titles OUT.tsv            title/subtitle/programid of every recording, from MythTV (read only)
  index                     one pass over the logs: header facts, cuts, volume envelope
  marks                     per log, the cut-pattern runs with every other log, audio-checked
  score-marks               fingerprint marks against the hand labels; the lag report
  ensemble --model M...     per-second probabilities of each model
  label                     the per-second label rule, scored on the test logs
  select                    the manifest of logs to write, under an hours budget
  write                     labelled copies of the manifest's logs into --out-dir
  review                    spans and channels for a human to look at; possible label errors
  silver --model M...       a model on held-out machine logs, judged by the fingerprints

Source logs are only ever opened for reading; `write` refuses any output path that is one.
Recordings of channels MythTV has as commercial-free (or with flagging off) are never labelled,
reviewed or used by silver: their older logs still carry false breaks.  Copies are written only
of logs that have `audio_shift`.
"""
import os
import sys

# numpy's BLAS would otherwise start a thread per core in every worker; the CPU budget is
# --jobs processes (and --threads for the models)
for _k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse
import collections
import copy
import functools
import hashlib
import json
import re
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycommflag import processor, neural
from pycommflag.feature_span import SceneType

COM = SceneType.COMMERCIAL.value
DNU = SceneType.DO_NOT_USE.value
UND = SceneType.TRANSITION.value   # the type training reads as "no trusted label"
HUMAN = ('train', 'test')

CUT_DIFF = neural.DIFF_THRESHOLD  # a frame with this diff is a cut
CUT_MERGE = 0.1        # cuts closer than this are one
PATTERN_CUTS = 5       # intervals per pattern
PATTERN_MIN_SPAN = 3.0 # seconds a pattern must cover
PATTERN_TOL = 0.05     # each of a pattern's cut times (from its first cut) must agree this well
PATTERN_MAX_HITS = 400 # a pattern matching more patterns than this is periodic, not a fingerprint
OFFSET_TOL = 0.15      # matches with one partner whose offsets differ by more are different runs
DENSE_GAP = 15.0       # ...and so are matches further apart in time than this
ENV_HZ = 20            # volume envelope grid
MAX_LAG = 1.5          # seconds of audio lag searched either way
MIN_AUDIO_SPAN = 2.0   # seconds a run needs for its audio to be compared

RULE_DEFAULTS = dict(
    rmin=0.8,          # envelope correlation a run needs...
    rmin_short=0.9,    # ...or this when it is shorter than short_span
    short_span=6.0,
    min_titles=2,      # distinct other titles that must cover a second
    long_run=150.0,    # a run this long is shared programme content
    min_patterns=1,    # patterns a run needs
    audio=True,        # False: no envelope check (then use min_patterns 3)
    # the second tier: same-series partners (other episodes of the log's own title)
    series_need=3,     # partners (other titles and other episodes together) a point needs
    series_frac=0.15,  # ...from at most this share of the title's other recordings (at least 2):
                       # what a series repeats in most episodes (title sequence, recap) is not an ad
    series_skip_start=180.0,     # seconds at the start of a recording with no second-tier marks
    series_break='component',    # off | component | flanked: also need first-tier marks nearby
    series_gap=10.0,   # seconds bridged when grouping second-tier points with first-tier ones
)

# --- small helpers -----------------------------------------------------------------------------

REC_RE = re.compile(r'(\d{4,6})_(\d{14})')

def norm_title(t):
    return re.sub(r'[^a-z0-9]', '', (t or '').lower()) or None

def log_key(path):
    base = os.path.basename(path)
    for ext in ('.gz', '.json'):
        if base.endswith(ext):
            base = base[:-len(ext)]
    return base + '-' + hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:8]

def holdout(meta, mod):
    """Machine logs whose recording hashes to 0 mod `mod` are held out of `select` for `silver`."""
    if not mod or meta['role'] != 'machine':
        return False
    return int(hashlib.sha1((meta['rec'] or meta['path']).encode()).hexdigest()[:8], 16) % mod == 0

def runs_of(mask):
    """(start, end) index pairs of the True runs in a boolean array."""
    d = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

def tag_type(t):
    return t.value if isinstance(t, SceneType) else int(t)

def labels_at(tags, times):
    """Per time: 1 inside a commercial tag, -1 inside DO_NOT_USE (which wins), else 0."""
    y = np.zeros(len(times), dtype=np.int8)
    for want, val in ((COM, 1), (DNU, -1)):
        for t, (b, e) in tags:
            if tag_type(t) == want:
                y[(times >= b) & (times < e)] = val
    return y

def boundary_distance(tags, times):
    edges = np.array(sorted(x for t, (b, e) in tags if tag_type(t) == COM for x in (b, e)))
    if not len(edges):
        return np.full(len(times), np.inf)
    i = np.searchsorted(edges, times)
    return np.minimum(np.abs(times - edges[np.clip(i - 1, 0, len(edges) - 1)]),
                      np.abs(edges[np.clip(i, 0, len(edges) - 1)] - times))

def length_opts(args=None):
    from pycommflag import options
    p = options.get_options()
    o = argparse.Namespace(**{k: p.get_default(k) for k in ('show_min_len', 'break_min_len', 'break_max_len')})
    for k in vars(o):
        if args is not None and getattr(args, k, None) is not None:
            setattr(o, k, getattr(args, k))
    return o

def nice(n):
    try:
        os.nice(n)
    except OSError:
        pass

def limit_threads(n):
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS'):
        os.environ[k] = str(n)
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'

def pool(jobs):
    import multiprocessing
    return multiprocessing.get_context('fork').Pool(jobs)

# --- titles ------------------------------------------------------------------------------------

def cmd_titles(args):
    from pycommflag import mythtv
    conn = mythtv._open()
    if conn is None:
        sys.exit('no MythTV database')
    out = {}
    with conn.cursor() as c:
        for table in ('recorded', 'oldrecorded'):
            c.execute(f"SELECT chanid, DATE_FORMAT(starttime,'%Y%m%d%H%i%s'), title, subtitle, programid FROM {table}")
            for ch, st, t, s, pid in c.fetchall():
                out.setdefault(f'{ch}_{st}', (t or '', s or '', pid or ''))
    with open(args.out, 'w') as f:
        f.write('rec\ttitle\tsubtitle\tprogramid\n')
        for k, v in sorted(out.items()):
            f.write(k + '\t' + '\t'.join(x.replace('\t', ' ').replace('\n', ' ') for x in v) + '\n')
    print(f'{len(out)} recordings -> {args.out}')

def read_titles(path):
    out = {}
    if not path:
        return out
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 4:
                out[p[0]] = (p[1] or None, p[2] or None, p[3] or None)
    return out

# --- index -------------------------------------------------------------------------------------

def reference_tags(flog, spans):
    """The hand labels as --eval scores them (neural._score_models): lead-in split, slips merged,
    commercials snapped, DO_NOT_USE stretched to the ends, and nothing before the first frame."""
    tags = neural._split_lead_in(flog.get('tags', []), flog)
    tags = neural._merge_tag_gaps([(t, (b, e)) for t, (b, e) in tags if tag_type(t) in (COM, DNU)])
    dur = flog.get('duration', 0.00001)
    com, dnu = [], []
    for t, (b, e) in tags:
        if tag_type(t) == DNU:
            if e + 10 > dur and e < dur and b < dur:
                e = dur
            if b < 10 and e > 0:
                b = 0
            dnu.append((DNU, (b, e)))
        else:
            com.append((COM, (b, e)))
    first = next((f[0] for f in flog['frames'] if f is not None), 0)
    covered = max([e for _, (b, e) in dnu if b <= 0] + [0])
    if covered < first:
        dnu.append((DNU, (covered, first)))
    com = neural._adjust_tags(com, spans.get('blank', []), spans.get('diff', []))
    return sorted([(tag_type(t), (float(b), float(e))) for t, (b, e) in com + dnu], key=lambda x: x[1][0])

def envelope(t, fvol):
    """Front volume on an ENV_HZ grid from floor(t[0]): log10 of volume over its median, float16."""
    t0 = float(np.floor(t[0]))
    b = ((t - t0) * ENV_HZ).astype(np.int64)
    n = int(b[-1]) + 1
    s = np.bincount(b, fvol, n)
    c = np.bincount(b, None, n)
    e = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    bad = np.isnan(e)
    if bad.any():
        e[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), e[~bad])
    med = np.median(fvol[fvol > 0]) if (fvol > 0).any() else 1.0
    return np.log10(e / med + 1e-3).astype(np.float16), t0

def _index_one(job):
    role, path, logdir = job
    key = log_key(path)
    out = os.path.join(logdir, key + '.npz')
    try:
        st = os.stat(path)
        stamp = [st.st_size, st.st_mtime]
        if os.path.exists(out):
            try:
                with np.load(out) as z:
                    meta = json.loads(str(z['meta']))
                if meta.get('stamp') == stamp and meta.get('role') == role:
                    return meta
            except Exception:
                pass
        flog = processor.read_feature_log(path)   # opened for reading only
        rows = [r for r in (flog.get('frames') or []) if r is not None]
        if len(rows) < 100:
            return {'error': 'no frames', 'path': path}
        hdr = flog.get('frames_header') or []
        if hdr[:5] != ['time', 'logo_present', 'is_blank', 'diff', 'fvol']:
            return {'error': f'frames_header {hdr[:5]}', 'path': path}
        a = np.array([r[:10] for r in rows], dtype=np.float64)
        t = a[:, 0]
        ct = t[a[:, 3] >= CUT_DIFF]
        if len(ct):
            ct = ct[np.concatenate(([True], np.diff(ct) > CUT_MERGE))]
        env, t0 = envelope(t, a[:, 4])
        spans = processor.read_feature_spans(flog, 'blank', 'diff')
        blank = np.array([(b, e) for v, (b, e) in spans.get('blank', []) if v], dtype=np.float64).reshape(-1, 2)
        dsel = a[:, 3] >= CUT_DIFF
        fn = flog.get('filename') or ''
        m = REC_RE.search(os.path.basename(fn)) or REC_RE.search(os.path.basename(path))
        video = fn if fn and os.path.exists(fn) else None
        if video is None and fn.endswith('.ts') and os.path.exists(fn[:-3] + '.mkv'):
            video = fn[:-3] + '.mkv'
        meta = dict(
            key=key, path=os.path.abspath(path), role=role, stamp=stamp, filename=fn,
            rec=f'{m.group(1)}_{m.group(2)}' if m else None,
            chanid=str(flog.get('chanid') or (m.group(1) if m else '')) or None,
            start=m.group(2) if m else None,
            frame_rate=flog.get('frame_rate'), file_version=flog.get('file_version', 10),
            has_shift='audio_shift' in flog, audio_shift=flog.get('audio_shift'),
            audio_shift_from=flog.get('audio_shift_from'), mkv_lag=flog.get('mkv_lag'),
            video=video, duration=float(flog.get('duration') or t[-1]), first=float(t[0]), last=float(t[-1]),
            nframes=len(rows), has_logo=bool(flog.get('logo')), edited=flog.get('edited'),
            tags=[(tag_type(x), (float(b), float(e))) for x, (b, e) in (flog.get('tags') or [])],
            n_cuts=int(len(ct)),
        )
        if not meta['has_shift']:
            import upgrade_log
            est, why = upgrade_log.estimate(a, meta['frame_rate'] or 29.97)
            meta['shift_estimate'] = est
        if role in HUMAN:
            meta['ref_tags'] = reference_tags(flog, spans)
        tmp = out + '.tmp'
        with open(tmp, 'wb') as f:
            np.savez(f, meta=json.dumps(meta), cuts=ct, env=env, env_t0=t0, blank=blank,
                     diff_t=t[dsel], diff_v=a[dsel, 3])
        os.replace(tmp, out)
        return meta
    except Exception as e:
        return {'error': repr(e), 'path': path}

def scan(args):
    jobs = []
    for role in ('train', 'test', 'machine'):
        for d in getattr(args, role) or []:
            files = sorted(f for f in os.listdir(d) if f.endswith('.json') or f.endswith('.json.gz'))
            jobs += [(role, os.path.join(d, f)) for f in files if os.path.isfile(os.path.join(d, f))]
    # a log listed under both train and test only validates (as in training)
    test = {os.path.realpath(p) for r, p in jobs if r == 'test'}
    return [(r, p) for r, p in jobs if r == 'test' or os.path.realpath(p) not in test]

def cmd_index(args):
    nice(15)
    logdir = os.path.join(args.cache, 'logs')
    os.makedirs(logdir, exist_ok=True)
    jobs = scan(args)
    if args.limit:
        jobs = jobs[:args.limit]
    t0 = time.time()
    with pool(args.jobs) as p:
        res = []
        for i, r in enumerate(p.imap(_index_one, [(r, path, logdir) for r, path in jobs], chunksize=4)):
            res.append(r)
            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(jobs)} {time.time()-t0:.0f}s', flush=True)
    dt = time.time() - t0
    metas = [r for r in res if 'error' not in r]
    bad = [r for r in res if 'error' in r]
    titles = read_titles(args.titles)
    for m in metas:
        m['title'], m['subtitle'], m['programid'] = titles.get(m['rec'], (None, None, None))
    if not args.limit:
        with open(os.path.join(args.cache, 'index.json'), 'w') as f:
            json.dump(dict(metas=metas, unreadable=bad), f)
    print(f'{len(jobs)} logs in {dt:.0f}s ({dt/max(1,len(jobs)):.2f}s each with {args.jobs} jobs)')
    c = collections.Counter(m['role'] for m in metas)
    print('  ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items())),
          f"| machine with video {sum(1 for m in metas if m['role']=='machine' and m['video'])}",
          f"| with audio_shift {sum(1 for m in metas if m['has_shift'])}",
          f"| titled {sum(1 for m in metas if m['title'])}")
    for b in bad:
        print(f"  unreadable: {b['path']}: {b['error']}")

@functools.lru_cache(maxsize=1)
def load_index(cache):
    with open(os.path.join(cache, 'index.json')) as f:
        d = json.load(f)
    for i, m in enumerate(d['metas']):
        m['i'] = i
        m['ntitle'] = norm_title(m.get('title'))
        # same episode: same title and subtitle; without a subtitle every recording is its own
        m['episode'] = (m['ntitle'] + '|' + (norm_title(m.get('subtitle')) or 'rec:' + (m['rec'] or m['key']))) if m['ntitle'] else None
    members = set(corpus(d['metas']))
    per_title = collections.Counter(m['ntitle'] for m in d['metas'] if m['i'] in members and m['ntitle'])
    # without a MythTV database check_method says every channel is flagged
    from pycommflag import mythtv
    commfree = {ch: bool(ch) and not mythtv.check_method(ch, commfree_is_off=True) for ch in {m['chanid'] for m in d['metas']}}
    for m in d['metas']:
        m['title_recordings'] = per_title.get(m['ntitle'], 0)
        # flagging off or commercial-free in MythTV: no breaks, so never labels, review or silver
        m['commfree'] = commfree.get(m['chanid'], False)
    return d['metas'], d['unreadable']

def log_npz(cache, key):
    return np.load(os.path.join(cache, 'logs', key + '.npz'))

# --- marks -------------------------------------------------------------------------------------

def patterns(ct):
    """(cut offsets from the pattern's first cut, first cut, last cut) of each pattern in `ct`."""
    if len(ct) <= PATTERN_CUTS:
        return np.zeros((0, PATTERN_CUTS), np.float32), np.zeros(0), np.zeros(0)
    st = ct[:-PATTERN_CUTS]
    rel = np.stack([ct[k:len(ct) - PATTERN_CUTS + k] - st for k in range(1, PATTERN_CUTS + 1)], axis=1)
    ok = rel[:, -1] >= PATTERN_MIN_SPAN
    return rel[ok].astype(np.float32), st[ok], ct[PATTERN_CUTS:][ok]

_G = {}

def corpus(metas):
    """One log per recording: a hand-labelled one over a machine one."""
    pri = {'test': 0, 'train': 1, 'machine': 2}
    seen = {}
    for m in sorted(metas, key=lambda m: (pri[m['role']], m['path'])):
        seen.setdefault(m['rec'] or m['path'], m['i'])
    return sorted(seen.values())

def build_pattern_index(cache, metas):
    from scipy.spatial import cKDTree
    members = corpus(metas)
    rels, starts, ends, owner = [], [], [], []
    cuts = {}
    for i in members:
        with log_npz(cache, metas[i]['key']) as z:
            ct = z['cuts']
        cuts[i] = ct
        r, s, e = patterns(ct)
        rels.append(r); starts.append(s); ends.append(e); owner.append(np.full(len(r), i, np.int32))
    rels = np.concatenate(rels); starts = np.concatenate(starts); ends = np.concatenate(ends); owner = np.concatenate(owner)
    return dict(tree=cKDTree(rels), starts=starts, ends=ends, owner=owner, members=set(members), cuts=cuts)

@functools.lru_cache(maxsize=300)
def load_env(cache, key):
    with log_npz(cache, key) as z:
        e = z['env'].astype(np.float32)
        t0 = float(z['env_t0'])
    return np.convolve(e, np.ones(3) / 3, 'same'), t0   # 150 ms smoothing

def env_corr(A, B, a, b, off, max_lag=MAX_LAG):
    """Best Pearson r between envelope A over [a, b] and B over [a-off, b-off] shifted by up to
    max_lag, and how many seconds A's audio is late relative to B's there; (nan, nan) if they
    don't cover it."""
    (ea, ta), (eb, tb) = A, B
    ia = int(round((a - ta) * ENV_HZ)); ib = int(round((b - ta) * ENV_HZ))
    n = ib - ia
    if ia < 0 or ib > len(ea) or n < MIN_AUDIO_SPAN * ENV_HZ:
        return np.nan, np.nan
    L = int(round(max_lag * ENV_HZ))
    ja = int(round((a - off - tb) * ENV_HZ))
    lo, hi = max(-L, -ja), min(L, len(eb) - n - ja)
    if hi < lo:
        return np.nan, np.nan
    x = ea[ia:ib] - ea[ia:ib].mean()
    sx = np.sqrt((x * x).sum())
    W = np.lib.stride_tricks.sliding_window_view(eb[ja + lo:ja + hi + n], n)
    Wm = W - W.mean(1, keepdims=True)
    r = (Wm @ x) / (sx * np.sqrt((Wm * Wm).sum(1)) + 1e-9)
    k = int(np.argmax(r))
    best = float(k)
    if 0 < k < len(r) - 1:   # parabolic peak, for a lag finer than the grid
        d = r[k - 1] - 2 * r[k] + r[k + 1]
        if d < 0:
            best += 0.5 * (r[k - 1] - r[k + 1]) / d
    # B's lag lo+best ahead of A's matches: B is lagged by L = d_B - d_A, so A is late by -L
    return float(r[k]), -(lo + best) / ENV_HZ

def find_runs(cache, metas, i):
    """Every run of cut-pattern matches between log i and another recording, as rows of
    (partner, offset, start, end, patterns, r, late) in log i's time; offset is i's time minus
    the partner's, late is how far i's audio trails the partner's."""
    G = _G
    m = metas[i]
    if i in G['cuts']:
        ct = G['cuts'][i]
    else:
        with log_npz(cache, m['key']) as z:
            ct = z['cuts']
    Q, qs, qe = patterns(ct)
    if not len(Q):
        return np.zeros((0, 7))
    hits = G['tree'].query_ball_point(Q, PATTERN_TOL, p=np.inf)
    own = m['rec'] or m['path']
    rows_q, rows_j = [], []
    for q, js in enumerate(hits):
        if 0 < len(js) <= PATTERN_MAX_HITS:
            rows_q.append(np.full(len(js), q)); rows_j.append(np.asarray(js))
    if not rows_q:
        return np.zeros((0, 7))
    q = np.concatenate(rows_q); j = np.concatenate(rows_j)
    o = G['owner'][j]
    recs = G['recs']
    keep = recs[o] != own
    q, j, o = q[keep], j[keep], o[keep]
    if not len(q):
        return np.zeros((0, 7))
    off = qs[q] - G['starts'][j]
    s, e = qs[q], qe[q]
    # runs: one partner, one offset (to OFFSET_TOL), matches no more than DENSE_GAP apart
    order = np.lexsort((off, o))
    o, off, s, e, q = o[order], off[order], s[order], e[order], q[order]
    cl = np.concatenate(([0], np.cumsum((np.diff(o) != 0) | (np.diff(off) > OFFSET_TOL))))
    order = np.lexsort((s, cl))
    o, off, s, e, q, cl = o[order], off[order], s[order], e[order], q[order], cl[order]
    BIG = 1e7
    reach = np.maximum.accumulate(e + cl * BIG)
    brk = np.concatenate(([True], (cl[1:] != cl[:-1]) | (s[1:] + cl[1:] * BIG > reach[:-1] + DENSE_GAP)))
    starts = np.flatnonzero(brk)
    rid = np.cumsum(brk) - 1
    a = np.minimum.reduceat(s, starts)
    b = np.maximum.reduceat(e, starts)
    n = np.diff(np.append(starts, len(s)))
    po = o[starts]
    poff = np.array([np.median(off[x:x + k]) for x, k in zip(starts, n)])
    r = np.full(len(starts), np.nan); late = np.full(len(starts), np.nan)
    A = load_env(cache, m['key'])
    for k in np.flatnonzero(b - a < RULE_DEFAULTS['long_run'] * 4):
        r[k], late[k] = env_corr(A, load_env(cache, metas[po[k]]['key']), a[k], b[k], poff[k])
    return np.stack([po, poff, a, b, n, r, late], axis=1)

def _marks_one(i):
    cache, metas, stamp = _G['cache'], _G['metas'], _G['stamp']
    out = os.path.join(cache, 'marks', metas[i]['key'] + '.npz')
    try:
        runs = find_runs(cache, metas, i)
        tmp = out + '.tmp'
        with open(tmp, 'wb') as f:
            np.savez(f, runs=runs, partners=np.array([metas[int(p)]['key'] for p in runs[:, 0]]), stamp=stamp)
        os.replace(tmp, out)
        return i, len(runs)
    except Exception as e:
        return i, repr(e)

def corpus_stamp(metas):
    return hashlib.sha1(json.dumps(sorted((m['key'], m['stamp']) for m in metas)).encode()).hexdigest()[:16]

def target_set(metas, which):
    if which == 'human':
        return [m['i'] for m in metas if m['role'] in HUMAN]
    if which == 'video':
        return [m['i'] for m in metas if m['role'] in HUMAN or m['video'] or m['has_shift']]
    return [m['i'] for m in metas]

def cmd_marks(args):
    nice(15)
    metas, _ = load_index(args.cache)
    os.makedirs(os.path.join(args.cache, 'marks'), exist_ok=True)
    stamp = corpus_stamp(metas)
    todo = []
    for i in target_set(metas, args.targets):
        p = os.path.join(args.cache, 'marks', metas[i]['key'] + '.npz')
        if os.path.exists(p) and not args.redo:
            with np.load(p) as z:
                if str(z['stamp']) == stamp:
                    continue
        todo.append(i)
    if args.limit:
        todo = todo[:args.limit]
    t0 = time.time()
    _G.update(build_pattern_index(args.cache, metas))
    _G.update(cache=args.cache, metas=metas, stamp=stamp,
              recs=np.array([m['rec'] or m['path'] for m in metas], dtype=object))
    print(f'pattern index: {len(_G["owner"])} patterns from {len(_G["members"])} recordings in {time.time()-t0:.0f}s; {len(todo)} logs to match', flush=True)
    t0 = time.time()
    errs = []
    with pool(args.jobs) as p:
        for k, (i, n) in enumerate(p.imap_unordered(_marks_one, todo, chunksize=2)):
            if not isinstance(n, int):
                errs.append((metas[i]['path'], n))
            if (k + 1) % 200 == 0:
                print(f'  {k+1}/{len(todo)} {time.time()-t0:.0f}s', flush=True)
    dt = time.time() - t0
    print(f'{len(todo)} logs matched in {dt:.0f}s ({dt/max(1,len(todo)):.2f}s each with {args.jobs} jobs)')
    for e in errs:
        print('  failed:', *e)

def load_runs(cache, meta):
    p = os.path.join(cache, 'marks', meta['key'] + '.npz')
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return z['runs'], list(z['partners'])

def _grid(meta):
    """Mark grid: 0.1 s points from floor(first) to ceil(last)."""
    g0 = np.floor(meta['first'])
    return g0, int(np.ceil((meta['last'] - g0) * 10)) + 1

def _cover(intervals, g0, n):
    """Grid points covered by the union of `intervals`."""
    d = np.zeros(n + 1, np.int32)
    for a, b in intervals:
        ia = max(0, int(np.ceil((a - g0) * 10))); ib = min(n, int(np.floor((b - g0) * 10)) + 1)
        if ib > ia:
            d[ia] += 1; d[ib] -= 1
    return np.cumsum(d[:-1]) > 0

def _count(groups, g0, n):
    """Per grid point, how many groups (each a list of intervals) cover it."""
    c = np.zeros(n, np.int32)
    for iv in groups.values():
        c += _cover(iv, g0, n)
    return c

class Marks:
    """Fingerprint ad marks of one log on a 0.1 s grid, in two tiers that are never merged:
    `ad` (first tier: distinct other titles) with `count` of them, and `series` (second tier:
    other episodes of the log's own series count too) with `scount` partners; `shared` is
    inside shared programme content."""
    def __init__(self, g0, ad, count, shared, series, scount):
        self.g0, self.ad, self.count, self.shared, self.series, self.scount = g0, ad, count, shared, series, scount

    def _k(self, times):
        raw = np.round((np.asarray(times) - self.g0) * 10).astype(np.int64)
        return np.clip(raw, 0, len(self.ad) - 1), (raw >= 0) & (raw < len(self.ad))

    def at(self, times):
        """First tier at `times`: (marked, distinct other titles)."""
        k, inside = self._k(times)
        return self.ad[k] & inside, np.where(inside, self.count[k], 0)

    def at2(self, times):
        """Second tier at `times`: (marked, partners)."""
        k, inside = self._k(times)
        return self.series[k] & inside, np.where(inside, self.scount[k], 0)

    def intervals(self):
        return [(self.g0 + a / 10, self.g0 + b / 10, int(self.count[a:b].max())) for a, b in runs_of(self.ad)]

def accepted(runs, rule):
    span = runs[:, 3] - runs[:, 2]
    ok = (span < rule['long_run']) & (runs[:, 4] >= rule['min_patterns'])
    if rule['audio']:
        need = np.where(span < rule['short_span'], max(rule['rmin'], rule['rmin_short']), rule['rmin'])
        ok &= np.nan_to_num(runs[:, 5], nan=-1) >= need
    return ok

def _near(mask, sel, gap, both):
    """Runs of `sel` with a True of `mask` within `gap` points: on both sides, or bridged."""
    out = np.zeros(len(sel), bool)
    pos = np.flatnonzero(mask)
    if not len(pos):
        return out
    for a, b in runs_of(sel):
        l = np.searchsorted(pos, a) - 1
        r = np.searchsorted(pos, b)
        lok = l >= 0 and a - pos[l] <= gap
        rok = r < len(pos) and pos[r] - b <= gap
        if (lok and rok) if both else (lok or rok):
            out[a:b] = True
    return out

def derive_marks(meta, runs, partners, by_key, rule):
    """Marks from a log's runs.

    First tier: accepted runs from at least min_titles distinct normalised titles other than the
    log's own cover the point, outside every long run.
    Second tier (points not in the first): at least series_need partners, other titles and other
    episodes of the log's own title together, of which the episodes are at most series_frac of the
    title's other recordings (at least 2); not in the first series_skip_start seconds; and with
    series_break, within series_gap of first-tier marks (`flanked`: on both sides; `component`:
    on either, gaps of matched points bridged)."""
    g0, n = _grid(meta)
    z = np.zeros(n, bool)
    if runs is None or not len(runs):
        return Marks(g0, z, np.zeros(n, np.int32), z, z, np.zeros(n, np.int32))
    own_t, own_ep = meta['ntitle'], meta['episode']
    ok = accepted(runs, rule)
    span = runs[:, 3] - runs[:, 2]
    shared = _cover([(a, b) for a, b in runs[span >= rule['long_run']][:, 2:4]], g0, n)
    titles, eps = collections.defaultdict(list), collections.defaultdict(list)
    for k in np.flatnonzero(ok):
        pm = by_key[partners[k]]
        if pm['ntitle'] is None or own_t is None:
            continue
        iv = (runs[k, 2], runs[k, 3])
        if pm['ntitle'] != own_t:
            titles[pm['ntitle']].append(iv)
        elif pm['episode'] != own_ep:
            eps[pm['episode']].append(iv)
    tc = _count(titles, g0, n)
    ad = (tc >= rule['min_titles']) & ~shared
    ec = _count(eps, g0, n)
    cap = max(2, rule['series_frac'] * max(0, meta.get('title_recordings', 0) - 1))
    series = ~ad & ~shared & (ec >= 1) & (tc + ec >= rule['series_need']) & (ec <= cap)
    if rule['series_skip_start'] > 0:
        series[:max(0, int((meta['first'] + rule['series_skip_start'] - g0) * 10))] = False
    gap = int(rule['series_gap'] * 10)
    if rule['series_break'] == 'flanked':
        series = _near(ad, series, gap, True)
    elif rule['series_break'] == 'component':
        matched = ad | series
        comp = matched.copy()
        for a, b in runs_of(~matched):
            if 0 < a and b < n and b - a <= gap:
                comp[a:b] = True
        keep = np.zeros(n, bool)
        for a, b in runs_of(comp):
            if ad[a:b].any():
                keep[a:b] = True
        series &= keep
    return Marks(g0, ad, np.where(ad, tc, 0), shared, series, np.where(series, tc + ec, 0))

def marks_for(cache, meta, by_key, rule):
    r = load_runs(cache, meta)
    if r is None:
        return None
    return derive_marks(meta, r[0], r[1], by_key, rule)

def rule_from(args):
    rule = dict(RULE_DEFAULTS)
    for k in rule:
        v = getattr(args, 'fp_' + k, None)
        if v is not None:
            rule[k] = v
    return rule

# --- score-marks -------------------------------------------------------------------------------

def sample_times(meta):
    """One point per second, at its centre, from the first frame to the last."""
    return np.arange(np.ceil(meta['first']) + 0.5, meta['last'], 1.0)

def score_marks(cache, metas, rule, roles=HUMAN, sel=None):
    """Both tiers against the hand labels: counts in T (tier 1 'c'/'s', tier 2 'c2'/'s2' for
    commercial/show seconds marked), the marked show spans of each tier, per-channel counts."""
    by_key = {m['key']: m for m in metas}
    T = collections.Counter()
    resid, chan = {1: [], 2: []}, collections.defaultdict(collections.Counter)
    for m in metas:
        if m['role'] not in roles or m['ntitle'] is None or (sel and not sel(m)):
            continue
        mk = marks_for(cache, m, by_key, rule)
        if mk is None:
            continue
        t = sample_times(m)
        y = labels_at(m['ref_tags'], t)
        v = y >= 0
        T['logs'] += 1
        T['com'] += (v & (y == 1)).sum(); T['show'] += (v & (y == 0)).sum()
        c = chan[m['chanid']]
        c['logs'] += 1; c['com'] += (v & (y == 1)).sum()
        for tier, (ad, cnt) in ((1, mk.at(t)), (2, mk.at2(t))):
            sfx = '' if tier == 1 else '2'
            T['c' + sfx] += (v & ad & (y == 1)).sum(); T['s' + sfx] += (v & ad & (y == 0)).sum()
            c['c' + sfx] += (v & ad & (y == 1)).sum(); c['s' + sfx] += (v & ad & (y == 0)).sum()
            for a, b in runs_of(v & ad & (y == 0)):
                resid[tier].append((m, t[a] - 0.5, t[b - 1] + 0.5, int(cnt[a:b].max())))
            if tier == 2:
                # where the second tier's show marks fall: the first 3 minutes, or later
                early = t < m['first'] + 180
                T['s2_early'] += (v & ad & (y == 0) & early).sum()
        ad1, _ = mk.at(t); ad2, _ = mk.at2(t)
        for tt, (b0, e0) in m['ref_tags']:
            if tt == COM and e0 - b0 >= 120:
                inb = (t >= b0) & (t < e0)
                T['breaks'] += 1; T['breaks_marked'] += bool(ad1[inb].any()); T['breaks_marked2'] += bool((ad1 | ad2)[inb].any())
    return T, resid, chan

def report_score(name, T):
    print(f"{name:<52} tier 1: {100*T['c']/max(1,T['com']):5.1f}% of commercial s, {T['s']:4d} show s, precision "
          f"{100*T['c']/max(1,T['c']+T['s']):.2f}% | tier 2 alone: +{100*T['c2']/max(1,T['com']):4.1f}%, {T['s2']:4d} show s "
          f"({T['s2_early']} in the first 3 min), precision {100*T['c2']/max(1,T['c2']+T['s2']):.2f}% | breaks>=120 s marked "
          f"{100*T['breaks_marked']/max(1,T['breaks']):.1f}% (with tier 2 {100*T['breaks_marked2']/max(1,T['breaks']):.1f}%) of {T['breaks']}")

def single_series(metas, share=0.5):
    """Channels where one title holds at least `share` of the recordings, with that share."""
    by = collections.defaultdict(collections.Counter)
    members = set(corpus(metas))
    for m in metas:
        if m['i'] in members:
            by[m['chanid']][m['ntitle']] += 1
    out = {}
    for ch, c in by.items():
        top = c.most_common(1)[0][1] / sum(c.values())
        out[ch] = top
    return out

def cmd_score_marks(args):
    metas, _ = load_index(args.cache)
    rule = rule_from(args)
    roles = tuple(args.roles.split(','))
    variants = [('rule as given', {})]
    if args.variants:
        variants += [
            ('no audio, 1 pattern', dict(audio=False)),
            ('no audio, >=3 patterns', dict(audio=False, min_patterns=3)),
            ('one other title is enough', dict(min_titles=1)),
            ('long run 240 s', dict(long_run=240.0)),
            ('long run 360 s', dict(long_run=360.0)),
            ('tier 2: no recurrence cap', dict(series_frac=1e9)),
            ('tier 2: skip first 180 s', dict(series_skip_start=180.0)),
            ('tier 2: skip first 300 s', dict(series_skip_start=300.0)),
            ('tier 2: in a break with tier 1 (component)', dict(series_break='component')),
            ('tier 2: flanked by tier 1', dict(series_break='flanked')),
            ('tier 2: component, skip first 180 s', dict(series_break='component', series_skip_start=180.0)),
            ('tier 2: flanked, skip first 180 s', dict(series_break='flanked', series_skip_start=180.0)),
            ('tier 2: cap 0.10, skip 180 s', dict(series_frac=0.10, series_skip_start=180.0)),
            ('tier 2: 4 partners, skip 180 s', dict(series_need=4, series_skip_start=180.0)),
        ]
    for name, over in variants:
        r = dict(rule, **over)
        T, resid, chan = score_marks(args.cache, metas, r, roles)
        report_score(name + f" ({T['logs']} logs)", T)
        if name == 'rule as given' or args.verbose:
            for tier in (1, 2):
                long = [x for x in resid[tier] if x[2] - x[1] >= 8]
                print(f"    tier {tier} show spans marked: {len(resid[tier])}, of 8 s or more: {len(long)} ({sum(x[2]-x[1] for x in long):.0f} s)")
                for mm, a, b, c in sorted(long, key=lambda x: x[1] - x[2])[:args.show]:
                    tags = [(t, round(s), round(e)) for t, (s, e) in mm['tags'] if e > a - 120 and s < b + 120]
                    print(f"      {os.path.basename(mm['path'])[:30]:<30} {a:7.1f}-{b:7.1f} ({a-mm['first']:5.0f} s in) n={c} tags near {tags}")
    if args.channels:
        ss = single_series(metas)
        T, resid, chan = score_marks(args.cache, metas, rule, roles)
        print('channel  logs  top-title share  tier-1 coverage %  +tier 2 %  tier-1 show s  tier-2 show s')
        for ch, c in sorted(chan.items(), key=lambda kv: -kv[1]['com']):
            print(f"  {ch:>6} {c['logs']:4d} {ss.get(ch,0):10.2f} {100*c['c']/max(1,c['com']):12.1f} {100*c['c2']/max(1,c['com']):9.1f} {c['s']:8d} {c['s2']:8d}")
        for name, sel in (('single-series channels (top title >= 50%)', lambda m: ss.get(m['chanid'], 0) >= 0.5),
                          ('other channels', lambda m: ss.get(m['chanid'], 0) < 0.5)):
            T, _, _ = score_marks(args.cache, metas, rule, roles, sel)
            report_score(name + f" ({T['logs']} logs)", T)
    if args.chance:
        chance_report(args.cache, metas)
    if args.lags:
        lag_report(args.cache, metas, rule)

def chance_report(cache, metas):
    """How often a single-pattern match of 3-6 s fails the audio check: the share that is chance."""
    T = collections.Counter()
    for m in metas:
        if m['role'] not in HUMAN:
            continue
        r = load_runs(cache, m)
        if r is None or not len(r[0]):
            continue
        R = r[0]
        span = R[:, 3] - R[:, 2]
        has = ~np.isnan(R[:, 5])
        for name, sel in (('1 pattern, 3-6 s', (R[:, 4] == 1) & (span >= 3) & (span < 6)),
                          ('1 pattern, 6-30 s', (R[:, 4] == 1) & (span >= 6) & (span < 30)),
                          ('>=3 patterns, 6-30 s', (R[:, 4] >= 3) & (span >= 6) & (span < 30))):
            s = sel & has
            T[name, 'n'] += s.sum()
            T[name, 'fail'] += (s & (R[:, 5] < np.where(span < 6, 0.9, 0.8))).sum()
    for name in ('1 pattern, 3-6 s', '1 pattern, 6-30 s', '>=3 patterns, 6-30 s'):
        print(f"  runs of {name}: {T[name,'n']}, failing the audio check {100*T[name,'fail']/max(1,T[name,'n']):.1f}%")

def lag_report(cache, metas, rule):
    """The audio lag at which accepted runs line up: between logs with and without `audio_shift`
    it measures the unrepaired log's lateness; between two repaired logs it should be 0."""
    by_key = {m['key']: m for m in metas}
    rows = []
    for m in metas:
        r = load_runs(cache, m)
        if r is None or not len(r[0]):
            continue
        R, P = r
        ok = accepted(R, rule) & ~np.isnan(R[:, 6])
        rep = np.array([by_key[p]['has_shift'] for p in P], bool)
        L = R[ok & rep, 6]
        if len(L) >= 5:
            q1, med, q3 = np.percentile(L, [25, 50, 75])
            rows.append((m, med, q3 - q1, len(L)))
    def show(name, sel):
        x = [r for r in rows if sel(r[0])]
        if not x:
            return x
        med = np.array([r[1] for r in x]); iqr = np.array([r[2] for r in x])
        print(f"  {name}: {len(x)} logs with >=5 runs against repaired logs; per-log median lag p10/p50/p90 "
              f"{np.percentile(med,10):+.2f}/{np.percentile(med,50):+.2f}/{np.percentile(med,90):+.2f} s; "
              f"within-log IQR median {np.median(iqr):.2f} s")
        return x
    print('audio lag of accepted runs against partners that have audio_shift:')
    show('repaired logs (should be 0)', lambda m: m['has_shift'])
    exact = show('  of those, made by the fixed extractor (audio_shift 0)', lambda m: m['has_shift'] and m['audio_shift'] == 0 and not m.get('audio_shift_from'))
    if exact:
        e = np.abs([r[1] for r in exact])
        print(f"    |median| within 0.05 s: {100*(e<=0.05).mean():.0f}%, 0.1 s: {100*(e<=0.1).mean():.0f}%, 0.2 s: {100*(e<=0.2).mean():.0f}%")
    un = show('unrepaired logs (their lateness)', lambda m: not m['has_shift'])
    both = [(r[1], r[0]['shift_estimate']) for r in un if r[0].get('shift_estimate') is not None]
    if both:
        d = np.abs(np.array([a - b for a, b in both]))
        print(f"    vs upgrade_log's own estimate on {len(both)} of them: within 0.05 s {100*(d<=.05).mean():.0f}%, "
              f"0.1 s {100*(d<=.1).mean():.0f}%, 0.2 s {100*(d<=.2).mean():.0f}%")
        print(f"    and on the {len(un)-len(both)} where that estimate gives no answer, fingerprints still measure one")

# --- ensemble ----------------------------------------------------------------------------------

def model_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def ens_path(cache, meta):
    return os.path.join(cache, 'ens', meta['key'] + '.npz')

def load_ens(cache, meta, models):
    """(times, probs[model, time]) for `models` (paths), or None if any is missing."""
    p = ens_path(cache, meta)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        names = [model_name(x) for x in models]
        if not all(n in z for n in names):
            return None
        return z['times'], np.stack([z[n] for n in names])

def select_logs(metas, which, args):
    """Logs named by --logs: test, train, candidates (machine logs a copy could be written of),
    holdout (machine logs held out for silver), sample (up to 4 recent per channel), or a file
    listing log paths.  Machine logs of channels MythTV has as commercial-free (or flagging off)
    are never candidates, held out or sampled: their old logs still carry false breaks."""
    if which == 'test':
        return [m for m in metas if m['role'] == 'test']
    if which == 'train':
        return [m for m in metas if m['role'] == 'train']
    if which == 'candidates':
        return [m for m in metas if m['role'] == 'machine' and m['has_shift'] and not m['commfree'] and not holdout(m, args.holdout_mod)]
    if which == 'holdout':
        return [m for m in metas if holdout(m, args.holdout_mod) and m['video'] and m['tags'] and not m['commfree']]
    if which == 'sample':
        by = collections.defaultdict(list)
        for m in metas:
            if m['role'] == 'machine' and m['video'] and m['tags'] and not m['edited'] and not m['commfree'] and (m['start'] or '') >= args.since:
                by[m['chanid']].append(m)
        out = []
        for ch, L in sorted(by.items()):
            L.sort(key=lambda m: hashlib.sha1(m['key'].encode()).hexdigest())
            out += L[:4]
        return out
    paths = {os.path.abspath(l.strip()) for l in open(which) if l.strip()}
    return [m for m in metas if m['path'] in paths]

def cmd_ensemble(args):
    limit_threads(args.threads)
    nice(10)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    metas, _ = load_index(args.cache)
    os.makedirs(os.path.join(args.cache, 'ens'), exist_ok=True)
    logs = [m for w in args.logs for m in select_logs(metas, w, args)]
    logs = list({m['key']: m for m in logs}.values())
    if args.limit:
        logs = logs[:args.limit]
    names = [model_name(x) for x in args.model]
    models = {}
    t0 = time.time(); done = 0
    for k, m in enumerate(logs):
        p = ens_path(args.cache, m)
        have = {}
        if os.path.exists(p):
            with np.load(p) as z:
                have = {n: z[n] for n in z.files}
        need = [(n, f) for n, f in zip(names, args.model) if n not in have]
        if not need:
            continue
        try:
            flog = processor.read_feature_log(m['path'])
            flog['tags'] = []   # as when flagging: nothing cropped
            data = neural.load_nonpersistent(flog, False)
            if data is None:
                continue
            win, _, _, times = neural.load_data_sliding_window(data)
            gen = neural.make_data_generator(win)
            for n, f in need:
                if n not in models:
                    models[n] = keras.models.load_model(f)
                have[n] = models[n].predict(gen, verbose=0).flatten().astype(np.float32)
            have['times'] = np.asarray(times, dtype=np.float64)
            tmp = p + '.tmp'
            with open(tmp, 'wb') as fh:
                np.savez(fh, **have)
            os.replace(tmp, p)
            done += 1
        except Exception as e:
            print('  failed', m['path'], repr(e), flush=True)
        if done and done % 20 == 0:
            el = time.time() - t0
            print(f'  {k+1}/{len(logs)} logs, {el:.0f}s, {el/done:.1f}s per log', flush=True)
    el = time.time() - t0
    print(f'{done} logs run in {el:.0f}s ({el/max(1,done):.1f}s each, {len(names)} models); {len(logs)-done} already cached')

def cmd_ens_report(args):
    """The ensemble on the test logs and on machine logs, raw per-second output."""
    metas, _ = load_index(args.cache)
    by_key = {m['key']: m for m in metas}
    rule = rule_from(args)
    T = collections.Counter(); X = collections.Counter(); confl = []
    for m in metas:
        e = load_ens(args.cache, m, args.model)
        if e is None:
            continue
        t, P = e
        mean, lo, hi = P.mean(0), P.min(0), P.max(0)
        unan = (lo > .9) | (hi < .1)
        if m['role'] == 'test':
            y = labels_at(m['ref_tags'], t); v = y >= 0
            T['n'] += v.sum()
            for k in range(len(P)):
                T['single_right'] += (v & ((P[k] > .5) == (y == 1))).sum()
            T['mean_right'] += (v & ((mean > .5) == (y == 1))).sum()
            T['unan'] += (v & unan).sum(); T['unan_right'] += (v & unan & ((lo > .9) == (y == 1))).sum()
        elif m['role'] == 'machine':
            mk = marks_for(args.cache, m, by_key, rule)
            X['logs'] += 1; X['n'] += len(t); X['unan'] += unan.sum(); X['split'] += ((lo < .5) & (hi > .5)).sum()
            if mk is not None:
                ad, _ = mk.at(t)
                c = ad & (mean <= .5)
                rs = [(a, b) for a, b in runs_of(c) if b - a >= 5]
                X['runs'] += len(rs); X['run_logs'] += bool(rs); X['fp'] += ad.sum(); X['fp_show'] += c.sum()
    if T['n']:
        print(f"test: {T['n']} s; one model {100*T['single_right']/T['n']/len(args.model):.2f}% right (mean of the {len(args.model)}), "
              f"ensemble mean {100*T['mean_right']/T['n']:.2f}%; all >0.9 or all <0.1 on {100*T['unan']/T['n']:.1f}% of seconds, "
              f"{100*T['unan_right']/max(1,T['unan']):.2f}% right there")
    if X['n']:
        print(f"machine: {X['logs']} logs, {X['n']/3600:.0f} h; unanimous-confident {100*X['unan']/X['n']:.1f}%, split across 0.5 "
              f"{100*X['split']/X['n']:.2f}%; fingerprint ad seconds the ensemble calls show: {X['fp_show']} of {X['fp']}, "
              f"runs of 5 s or more: {X['runs']} in {X['run_logs']} logs")

# --- label -------------------------------------------------------------------------------------

LABEL_DEFAULTS = dict(com_mean=0.8, show_mean=0.2, veto=0.1, edge_margin=0.0, fp_override=True)

def spans_of(cache, meta):
    with log_npz(cache, meta['key']) as z:
        blank = [(True, (float(b), float(e))) for b, e in z['blank']]
        diff = list(zip(z['diff_t'].tolist(), z['diff_v'].tolist()))
    return {'blank': blank, 'diff': diff}

class Label:
    """One log's labels at the ensemble's times: state 1 commercial, 0 show, -1 undecided."""
    def __init__(self, meta, times, P, mk, lab, lopts, spans):
        self.meta, self.times = meta, times
        mean, lo, hi = P.mean(0), P.min(0), P.max(0)
        self.mean = mean
        if mk is not None:
            fp, cnt = mk.at(times)
        else:
            fp, cnt = np.zeros(len(times), bool), np.zeros(len(times), np.int32)
        self.fp, self.count = fp, cnt
        self.conflict = fp & (hi < lab['veto'])
        st = np.full(len(times), -1, np.int8)
        st[mean > lab['com_mean']] = 1
        st[mean < lab['show_mean']] = 0
        peff = mean.copy()
        if lab['fp_override']:
            st[fp & ~self.conflict] = 1
            peff[fp & ~self.conflict] = 1.0
        st[self.conflict] = -1
        # the length rules the real output gets: a decided second the post-processed tags
        # disagree with is undecided
        self.tags = neural.post_predict({'duration': meta['duration']}, peff, times, lopts, spans=spans)
        r = labels_at(self.tags, times)
        st[(st >= 0) & (st != r)] = -1
        if lab['edge_margin'] > 0:
            d = boundary_distance(self.tags, times)
            st[d < lab['edge_margin']] = -1
        self.state = st

    def decided(self):
        return float((self.state >= 0).mean()) if len(self.state) else 0.0

def labels_for(cache, meta, models, by_key, rule, lab, lopts):
    e = load_ens(cache, meta, models)
    if e is None:
        return None
    mk = marks_for(cache, meta, by_key, rule)
    return Label(meta, e[0], e[1], mk, lab, lopts, spans_of(cache, meta))

def lab_from(args):
    lab = dict(LABEL_DEFAULTS)
    for k in lab:
        v = getattr(args, k, None)
        if v is not None:
            lab[k] = v
    return lab

def score_labels(cache, metas, models, rule, lab, lopts, near=15.0):
    by_key = {m['key']: m for m in metas}
    T = collections.Counter(); per = []
    for m in metas:
        if m['role'] != 'test':
            continue
        L = labels_for(cache, m, models, by_key, rule, lab, lopts)
        if L is None:
            continue
        y = labels_at(m['ref_tags'], L.times); v = y >= 0
        d = boundary_distance(m['ref_tags'], L.times)
        kept = v & (L.state >= 0); right = kept & (L.state == y)
        T['n'] += v.sum(); T['kept'] += kept.sum(); T['right'] += right.sum()
        nb = v & (d <= near)
        T['nb'] += nb.sum(); T['nb_kept'] += (kept & nb).sum(); T['nb_right'] += (right & nb).sum()
        T['com'] += (v & (y == 1)).sum(); T['com_kept'] += (kept & (y == 1)).sum()
        T['err_com_as_show'] += (kept & (y == 1) & (L.state == 0)).sum()
        T['err_show_as_com'] += (kept & (y == 0) & (L.state == 1)).sum()
        T['fp_err'] += (kept & (y == 0) & (L.state == 1) & L.fp).sum()
        per.append((m, kept.sum(), (kept & ~right).sum(), v.sum()))
    return T, per

def report_labels(name, T):
    print(f"{name:<44} kept {100*T['kept']/max(1,T['n']):5.1f}% of {T['n']} s, {100*T['right']/max(1,T['kept']):.3f}% right "
          f"| within 15 s of a boundary: kept {100*T['nb_kept']/max(1,T['nb']):5.1f}%, {100*T['nb_right']/max(1,T['nb_kept']):.2f}% right "
          f"| commercial kept {100*T['com_kept']/max(1,T['com']):.1f}% | wrong: show called commercial {T['err_show_as_com']} "
          f"(by a mark {T['fp_err']}), commercial called show {T['err_com_as_show']}")

def cmd_label(args):
    metas, _ = load_index(args.cache)
    rule, lab, lopts = rule_from(args), lab_from(args), length_opts(args)
    variants = [('rule as given', {})]
    if args.variants:
        variants += [
            ('no fingerprint override', dict(fp_override=False)),
            ('mean 0.9 / 0.1', dict(com_mean=0.9, show_mean=0.1)),
            ('mean 0.95 / 0.05', dict(com_mean=0.95, show_mean=0.05)),
            ('edge margin 2 s', dict(edge_margin=2.0)),
            ('edge margin 5 s', dict(edge_margin=5.0)),
            ('mean 0.9 / 0.1, edge margin 3 s', dict(com_mean=0.9, show_mean=0.1, edge_margin=3.0)),
        ]
    for name, over in variants:
        T, per = score_labels(args.cache, metas, args.model, rule, dict(lab, **over), lopts)
        report_labels(name, T)
        if name == 'rule as given':
            worst = sorted(per, key=lambda x: -x[2])[:args.show]
            for m, k, w, n in worst:
                if w:
                    print(f"    {os.path.basename(m['path'])[:36]:<36} kept {k}/{n}, wrong {w}")

# --- write -------------------------------------------------------------------------------------

def states_to_tags(L, state):
    """Tags for a labelling: commercial runs as COMMERCIAL, undecided ones as TRANSITION, the
    rest untagged (show).  A label change where both sides are decided sits on the
    post-processed tags' edge (snapped to a blank or cut); any other sits half way between
    the two seconds."""
    t = L.times
    edges = np.array(sorted(x for _, (b, e) in L.tags for x in (b, e)))
    out = []
    i = 0
    n = len(state)
    while i < n:
        j = i
        while j + 1 < n and state[j + 1] == state[i]:
            j += 1
        def edge(k, nxt):   # boundary between sample k and k+1
            mid = (t[k] + t[k + 1]) / 2
            if state[k] >= 0 and state[k + 1] >= 0 and len(edges):
                e = edges[np.argmin(np.abs(edges - mid))]
                if abs(e - mid) <= 1.5:
                    return float(e)
            return float(mid)
        b = 0.0 if i == 0 else edge(i - 1, i)
        e = float(L.meta['duration']) if j == n - 1 else edge(j, j + 1)
        if state[i] == 1:
            out.append((COM, (b, e)))
        elif state[i] == -1:
            out.append((UND, (b, e)))
        i = j + 1
    return out

def check_training_view(flog, L, state):
    """What training makes of `flog` (a copy): per labelled second, the answer and weight
    load_nonpersistent gives it, compared with `state`."""
    cond = neural.load_nonpersistent(copy.deepcopy(flog), True)
    if cond is None:
        return None
    cond = cond[round(neural.WINDOW_BEFORE * neural.SUMMARY_RATE):-round(neural.WINDOW_AFTER * neural.SUMMARY_RATE)]
    ct, ans, w = cond[:, neural.TIMESTAMPS], cond[:, neural.ANSWERS], cond[:, neural.WEIGHTS]
    k = np.clip(np.searchsorted(L.times, ct), 1, len(L.times) - 1)
    k = np.where(np.abs(L.times[k - 1] - ct) < np.abs(L.times[k] - ct), k - 1, k)
    s = state[k]
    # every decided-decided label change; an upweight further than a window from all of them
    # comes from an edge next to undecided time, which training should not upweight
    real = [(L.times[i] + L.times[i + 1]) / 2 for i in range(len(state) - 1)
            if state[i] >= 0 and state[i + 1] >= 0 and state[i] != state[i + 1]]
    dreal = np.full(len(ct), np.inf)
    if real:
        real = np.array(real)
        x = np.searchsorted(real, ct)
        dreal = np.minimum(np.abs(ct - real[np.clip(x - 1, 0, len(real) - 1)]), np.abs(real[np.clip(x, 0, len(real) - 1)] - ct))
    return dict(
        rows=len(ct), sample=k,
        leak=(s < 0) & (w > 0),                       # undecided but trained on
        flip=(s >= 0) & (w > 0) & (ans != s),         # trained on the wrong answer
        lost=(s >= 0) & (w == 0),                     # decided but weight 0
        upweight_undecided=(s >= 0) & (w > 1) & (dreal >= max(neural.WINDOW_BEFORE, neural.WINDOW_AFTER)),
        upweight=(w > 1),
    )

def labelled_copy(meta, L, header, max_rounds=6):
    """(the copy to write, its training-view check, rounds) with the source log read fresh;
    seconds training would get wrong become undecided, until none are."""
    state = L.state.copy()
    flog = processor.read_feature_log(meta['path'])
    chk = None
    for rnd in range(max_rounds):
        flog['tags'] = states_to_tags(L, state)
        chk = check_training_view(flog, L, state)
        if chk is None:
            return None, None, rnd, None
        bad = chk['leak'] | chk['flip']
        if not bad.any():
            break
        # undecide the seconds trained on wrongly and those within a snap distance of them, so the
        # edge training snaps lands inside undecided time
        for idx in chk['sample'][bad]:
            lo = np.searchsorted(L.times, L.times[idx] - 6); hi = np.searchsorted(L.times, L.times[idx] + 6)
            state[lo:hi] = -1
    out = processor.set_header_key(flog, 'machine_labels', header, after='filename')
    return out, chk, rnd + 1, state

def cmd_write(args):
    metas, _ = load_index(args.cache)
    by_key = {m['key']: m for m in metas}
    by_path = {m['path']: m for m in metas}
    sources = {os.path.realpath(m['path']) for m in metas}
    source_dirs = {os.path.dirname(p) for p in sources}
    out_dir = os.path.realpath(args.out_dir)
    if out_dir in source_dirs:
        sys.exit(f'{args.out_dir} holds source logs; write into a directory of its own')
    os.makedirs(out_dir, exist_ok=True)
    rule, lab, lopts = rule_from(args), lab_from(args), length_opts(args)
    header = dict(models=[model_name(x) for x in args.model], rule=rule, label=lab,
                  length=vars(lopts), tool='tools/harvest.py', made=time.strftime('%Y-%m-%dT%H:%M:%S'))
    todo = [by_path[os.path.abspath(l.split('\t')[0])] for l in open(args.manifest).read().splitlines()[1:] if l.strip()]
    T = collections.Counter()
    for m in todo:
        if not m['has_shift']:
            print(f"  skip {m['path']}: no audio_shift"); continue
        if m['commfree'] or m['role'] != 'machine':
            print(f"  skip {m['path']}: {'commercial-free channel' if m['commfree'] else 'hand-labelled'}"); continue
        L = labels_for(args.cache, m, args.model, by_key, rule, lab, lopts)
        if L is None:
            print(f"  skip {m['path']}: no ensemble output for every model"); continue
        res = labelled_copy(m, L, dict(header, source=m['path']))
        flog, chk = res[0], res[1]
        if flog is None:
            print(f"  skip {m['path']}: too short for training"); continue
        rounds, state = res[2], res[3]
        name = os.path.basename(m['path'])
        if not name.endswith('.gz'):
            name += '.gz'
        dst = os.path.join(out_dir, name)
        if os.path.realpath(dst) in sources:
            sys.exit(f'refusing to write over a source log: {dst}')
        tmp = os.path.join(out_dir, '.writing-' + name)
        processor.write_feature_log(flog, tmp)
        os.replace(tmp, dst)
        T['logs'] += 1; T['rows'] += chk['rows']
        for k in ('leak', 'flip', 'lost', 'upweight_undecided', 'upweight'):
            T[k] += int(chk[k].sum())
        T['undecided_before'] += int((L.state < 0).sum()); T['undecided_after'] += int((state < 0).sum())
        T['rounds'] += rounds
        print(f"  {name}: decided {100*(state>=0).mean():.1f}% (was {100*L.decided():.1f}%), rounds {rounds}, "
              f"left wrong {int(chk['leak'].sum()+chk['flip'].sum())}, upweighted by undecided edges {int(chk['upweight_undecided'].sum())} of {int(chk['upweight'].sum())}", flush=True)
    if T['logs']:
        print(f"{T['logs']} logs written to {out_dir}; training view of {T['rows']} s: undecided but weighted {T['leak']}, "
              f"wrong answer {T['flip']}, decided but weight 0 {T['lost']}; seconds upweighted {T['upweight']}, of them only by "
              f"edges beside undecided time {T['upweight_undecided']}; seconds made undecided so training agrees: "
              f"{T['undecided_after']-T['undecided_before']}; mean rounds {T['rounds']/T['logs']:.1f}")

# --- select ------------------------------------------------------------------------------------

def breaks_marked(L, mk, min_len=120.0):
    n = k = 0
    for t, (b, e) in L.tags:
        if e - b >= min_len:
            n += 1
            if mk is not None:
                ad, _ = mk.at(L.times[(L.times >= b) & (L.times < e)])
                k += bool(ad.any())
    return n, k

def cmd_select(args):
    metas, _ = load_index(args.cache)
    by_key = {m['key']: m for m in metas}
    rule, lab, lopts = rule_from(args), lab_from(args), length_opts(args)
    human = [m for m in metas if m['role'] in HUMAN]
    human_recs = {m['rec'] for m in human if m['rec']}
    human_files = {os.path.realpath(m['filename']) for m in human if m['filename']}
    test_eps = {(m['ntitle'], norm_title(m.get('subtitle'))) for m in metas if m['role'] == 'test' and m['ntitle']}
    human_per_chan = collections.Counter(m['chanid'] for m in human)
    why = collections.Counter(); cand = []
    unmarked_hours = collections.Counter()
    top = single_series(metas)
    for m in select_logs(metas, 'candidates', args):
        if m['rec'] in human_recs or (m['filename'] and os.path.realpath(m['filename']) in human_files):
            why['same recording as a hand-labelled log'] += 1; continue
        if m['ntitle'] and (m['ntitle'], norm_title(m.get('subtitle'))) in test_eps:
            why['same title and subtitle as a test log'] += 1; continue
        if m['edited']:
            why['edited by a human'] += 1; continue
        L = labels_for(args.cache, m, args.model, by_key, rule, lab, lopts)
        if L is None:
            why['no ensemble output'] += 1; continue
        mk = marks_for(args.cache, m, by_key, rule)
        nb, kb = breaks_marked(L, mk)
        dec = L.decided()
        if dec < args.min_decided:
            why[f'decided under {args.min_decided:.0%}'] += 1; continue
        if nb == 0 and not args.allow_no_breaks:
            why['no break of 120 s or more'] += 1; continue
        if nb and kb < nb * args.min_breaks_marked:
            why[f'under {args.min_breaks_marked:.0%} of long breaks marked'] += 1
            unmarked_hours[m['chanid']] += len(L.times) / 3600
            continue
        hrs = len(L.times) / 3600
        conflicts = len([1 for a, b in runs_of(L.conflict) if b - a >= 3])
        hc = human_per_chan[m['chanid']]
        tier = 0 if hc == 0 else 1 if hc <= 2 else 2 if hc <= 9 else 3
        cand.append(dict(m=m, hrs=hrs, dec=dec, nb=nb, kb=kb, conflicts=conflicts, hc=hc, tier=tier))
    cand.sort(key=lambda c: (c['tier'], -min(c['conflicts'], 1), -(c['m']['file_version'] or 10), -int(c['m']['start'] or 0)))
    picked, hours = [], 0.0
    per_title, per_chan = collections.Counter(), collections.Counter()
    chan_cap = args.channel_cap if args.channel_cap else args.hours
    for c in cand:
        t = c['m']['ntitle'] or c['m']['key']
        if per_title[t] >= args.title_cap or per_chan[c['m']['chanid']] + c['hrs'] > chan_cap or hours + c['hrs'] > args.hours:
            continue
        picked.append(c); hours += c['hrs']; per_title[t] += 1; per_chan[c['m']['chanid']] += c['hrs']
    with open(args.out, 'w') as f:
        f.write('log\tchannel\ttitle\thours\tdecided\tbreaks_120s\tbreaks_marked\tconflicts\tfile_version\thuman_logs_on_channel\n')
        for c in picked:
            m = c['m']
            f.write(f"{m['path']}\t{m['chanid']}\t{m.get('title') or ''}\t{c['hrs']:.2f}\t{c['dec']:.3f}\t{c['nb']}\t{c['kb']}\t"
                    f"{c['conflicts']}\t{m['file_version']}\t{c['hc']}\n")
    print(f"{len(cand)} candidates pass, {len(picked)} picked, {hours:.0f} h -> {args.out}")
    for k, v in why.most_common():
        print(f'  left out, {k}: {v}')
    if unmarked_hours:
        print(f'  hours the long-breaks-marked rule removes, by channel (share of its recordings under its most common title):')
        for k, h in unmarked_hours.most_common():
            print(f'    {k:>6} {h:6.1f} h  ({100*top.get(k,0):.0f}%)')
    print('  by channel (picked logs, hours, hand-labelled logs on it):')
    agg = collections.defaultdict(lambda: [0, 0.0, 0])
    for c in picked:
        a = agg[c['m']['chanid']]; a[0] += 1; a[1] += c['hrs']; a[2] = c['hc']
    for ch, (n, h, hc) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f'    {ch:>6} {n:3d} {h:6.1f} h  {hc}')

# --- review ------------------------------------------------------------------------------------

def cmd_review(args):
    metas, _ = load_index(args.cache)
    by_key = {m['key']: m for m in metas}
    rule, lab, lopts = rule_from(args), lab_from(args), length_opts(args)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for m in metas:
        if m['role'] != 'machine' or m['commfree']:
            continue
        L = labels_for(args.cache, m, args.model, by_key, rule, lab, lopts)
        if L is None:
            continue
        for a, b in runs_of(L.conflict):
            if b - a >= args.min_conflict:
                rows.append((m, L.times[a] - .5, L.times[b - 1] + .5, 'fingerprint_vs_unanimous_models',
                             float(L.mean[a:b].mean()), int(L.count[a:b].max()), 0, (b - a) * int(L.count[a:b].max())))
        for a, b in runs_of(L.state < 0):
            if b - a >= args.min_undecided:
                rows.append((m, L.times[a] - .5, L.times[b - 1] + .5, 'long_undecided',
                             float(L.mean[a:b].mean()), int(L.count[a:b].max()), 1, b - a))
    rows.sort(key=lambda r: (r[6], -r[7]))
    with open(os.path.join(args.out_dir, 'review.tsv'), 'w') as f:
        f.write('log\tstart\tend\treason\tensemble_mean\tpartners\tvideo_exists\n')
        for m, a, b, why, mean, cnt, _, _ in rows:
            f.write(f"{m['path']}\t{a:.1f}\t{b:.1f}\t{why}\t{mean:.3f}\t{cnt}\t{int(bool(m['video']))}\n")
    c = collections.Counter(r[3] for r in rows)
    print(f"review.tsv: {len(rows)} spans: " + ', '.join(f'{k} {v} ({sum(r[2]-r[1] for r in rows if r[3]==k)/3600:.1f} h)' for k, v in c.items()))
    print(f"  in {len(set(r[0]['key'] for r in rows if r[3]=='fingerprint_vs_unanimous_models'))} logs with conflicts")

    # channels whose stored (production) breaks the fingerprints almost never corroborate.  Not a
    # sign of a channel without commercials (those are set so in MythTV and left out here): a
    # channel whose recordings are nearly all one title has few other-title partners to match
    top = single_series(metas)
    ch = collections.defaultdict(collections.Counter)
    for m in metas:
        if m['role'] != 'machine' or m['edited'] or m['commfree']:
            continue
        mk = marks_for(args.cache, m, by_key, rule)
        if mk is None:
            continue
        t = sample_times(m)
        ad, _ = mk.at(t)
        ad2, _ = mk.at2(t)
        y = labels_at(m['tags'], t)
        c = ch[m['chanid']]
        c['logs'] += 1; c['hours'] += len(t) / 3600; c['com'] += (y == 1).sum(); c['ad_in_com'] += (ad & (y == 1)).sum()
        for tt, (b, e) in m['tags']:
            if tt == COM and e - b >= 120:
                inb = (t >= b) & (t < e)
                c['breaks'] += 1; c['breaks_marked'] += bool(ad[inb].any()); c['breaks_marked2'] += bool((ad | ad2)[inb].any())
    chans = sorted(ch.items(), key=lambda kv: kv[1]['breaks_marked'] / max(1, kv[1]['breaks']))
    unseen = []
    with open(os.path.join(args.out_dir, 'channels.tsv'), 'w') as f:
        f.write('channel\tlogs\thours\ttop_title_share\ttagged_commercial_share\tbreaks_120s\tbreaks_marked_share\t'
                'breaks_marked_share_with_tier2\tmarked_share_of_tagged_commercial\tfingerprints_cannot_see\n')
        for k, c in chans:
            share = c['breaks_marked'] / max(1, c['breaks'])
            blind = c['logs'] >= 5 and c['breaks'] >= 10 and share < args.channel_share
            if blind:
                unseen.append((k, c, share))
            f.write(f"{k}\t{c['logs']}\t{c['hours']:.1f}\t{top.get(k, 0):.2f}\t{c['com']/max(1,c['hours']*3600):.3f}\t{c['breaks']}\t"
                    f"{share:.3f}\t{c['breaks_marked2']/max(1,c['breaks']):.3f}\t{c['ad_in_com']/max(1,c['com']):.3f}\t{int(blind)}\n")
    print(f"channels.tsv: {len(chans)} channels (commercial-free ones left out); channels the fingerprints cannot see "
          f"(>=5 logs, >=10 tagged breaks of 120 s+, under {args.channel_share:.0%} with a tier-1 mark):")
    for k, c, share in unseen:
        print(f"  {k}: {c['breaks_marked']}/{c['breaks']} breaks marked ({100*c['breaks_marked2']/max(1,c['breaks']):.0f}% with tier 2), "
              f"most common title on {100*top.get(k,0):.0f}% of its recordings, {c['logs']} logs")

    # hand labels the fingerprints contradict
    out = []
    for m in metas:
        if m['role'] not in HUMAN:
            continue
        mk = marks_for(args.cache, m, by_key, rule)
        if mk is None:
            continue
        t = sample_times(m)
        ad, cnt = mk.at(t)
        y = labels_at(m['ref_tags'], t)
        for a, b in runs_of(ad & (y == 0)):
            if b - a >= args.min_contradiction:
                out.append((m, t[a] - .5, t[b - 1] + .5, int(cnt[a:b].max())))
    with open(os.path.join(args.out_dir, 'label_check.tsv'), 'w') as f:
        f.write('log\tstart\tend\treason\tpartners\tvideo_exists\n')
        for m, a, b, c in sorted(out, key=lambda r: r[1] - r[2]):
            f.write(f"{m['path']}\t{a:.1f}\t{b:.1f}\tmarked_ad_in_labelled_show\t{c}\t{int(bool(m['video']))}\n")
    print(f"label_check.tsv: {len(out)} spans of {args.min_contradiction:.0f} s or more where hand-labelled show is a proven ad")
    for m, a, b, c in out:
        print(f"  {m['path']}\t{a:.0f}-{b:.0f}\t{c} titles")

# --- silver ------------------------------------------------------------------------------------

def cmd_silver(args):
    metas, _ = load_index(args.cache)
    by_key = {m['key']: m for m in metas}
    rule, lopts = rule_from(args), length_opts(args)
    logs = [m for m in select_logs(metas, args.logs, args) if not m['commfree']]
    for mf in args.model:
        T = collections.Counter(); ch = collections.defaultdict(collections.Counter)
        missing = 0
        for m in logs:
            e = load_ens(args.cache, m, [mf])
            mk = marks_for(args.cache, m, by_key, rule)
            if e is None or mk is None:
                missing += 1; continue
            t, P = e
            p = P[0]
            ad, _ = mk.at(t)
            tags = neural.post_predict({'duration': m['duration']}, p, t, lopts, spans=spans_of(args.cache, m))
            y = labels_at(tags, t)
            T['ad'] += ad.sum(); T['ad_raw'] += (ad & (p > .5)).sum(); T['ad_tag'] += (ad & (y == 1)).sum()
            c = ch[m['chanid']]
            for tt, (b, e2) in tags:
                if e2 - b >= 120:
                    c['breaks'] += 1; c['marked'] += bool(ad[(t >= b) & (t < e2)].any())
            c['logs'] += 1; c['ad'] += ad.sum(); c['ad_tag'] += (ad & (y == 1)).sum()
        print(f"{model_name(mf)}: {len(logs)-missing} held-out logs ({missing} without output or marks); proven-ad seconds "
              f"called commercial: raw {100*T['ad_raw']/max(1,T['ad']):.2f}%, after post-processing {100*T['ad_tag']/max(1,T['ad']):.2f}% of {T['ad']}")
        print('  channel  logs  breaks>=120s  with a mark  proven-ad s called commercial')
        for k, c in sorted(ch.items(), key=lambda kv: -kv[1]['breaks']):
            print(f"  {k:>6} {c['logs']:5d} {c['breaks']:8d} {100*c['marked']/max(1,c['breaks']):10.1f}% {100*c['ad_tag']/max(1,c['ad']):10.1f}%")

# --- cli ---------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, models=False):
        p.add_argument('--cache', required=True, help='cache directory')
        for k, v in RULE_DEFAULTS.items():
            if isinstance(v, bool):
                p.add_argument('--fp-' + k.replace('_', '-'), dest='fp_' + k, type=lambda s: s.lower() in ('1', 'yes', 'true'), default=None, help=f'fingerprint rule (default {v})')
            else:
                p.add_argument('--fp-' + k.replace('_', '-'), dest='fp_' + k, type=type(v), default=None, help=f'fingerprint rule (default {v})')
        p.add_argument('--holdout-mod', type=int, default=10, help='machine logs whose recording hashes to 0 mod this are held out for silver (0: none)')
        if models:
            p.add_argument('--model', nargs='+', required=True, help='model files')
            for k, v in LABEL_DEFAULTS.items():
                if isinstance(v, bool):
                    p.add_argument('--' + k.replace('_', '-'), dest=k, type=lambda s: s.lower() in ('1', 'yes', 'true'), default=None, help=f'label rule (default {v})')
                else:
                    p.add_argument('--' + k.replace('_', '-'), dest=k, type=float, default=None, help=f'label rule (default {v})')
            for k in ('show_min_len', 'break_min_len', 'break_max_len'):
                p.add_argument('--' + k.replace('_', '-'), dest=k, type=float, default=None, help='as for flagging (default: its default)')

    p = sub.add_parser('titles', help='dump recording titles from MythTV')
    p.add_argument('out')
    p.set_defaults(fn=cmd_titles)

    p = sub.add_parser('index', help='index the logs')
    p.add_argument('--cache', required=True)
    p.add_argument('--titles', help='TSV from the titles command')
    p.add_argument('--train', nargs='*', default=[], help='directories of hand-labelled training logs')
    p.add_argument('--test', nargs='*', default=[], help='directories of hand-labelled test logs')
    p.add_argument('--machine', nargs='*', default=[], help='directories of machine-flagged logs')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--limit', type=int, default=0, help='only the first N logs, for timing (writes no index)')
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser('marks', help='match cut patterns between logs')
    common(p)
    p.add_argument('--targets', choices=('human', 'video', 'all'), default='video',
                   help='logs to find matches for (all logs are partners): hand-labelled; those plus machine logs with video or audio_shift; all')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--redo', action='store_true')
    p.set_defaults(fn=cmd_marks)

    p = sub.add_parser('score-marks', help='fingerprint marks against the hand labels')
    common(p)
    p.add_argument('--roles', default='train,test')
    p.add_argument('--variants', action='store_true')
    p.add_argument('--channels', action='store_true')
    p.add_argument('--chance', action='store_true')
    p.add_argument('--lags', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--show', type=int, default=20)
    p.set_defaults(fn=cmd_score_marks)

    p = sub.add_parser('ensemble', help='per-second probabilities of each model')
    common(p)
    p.add_argument('--model', nargs='+', required=True)
    p.add_argument('--logs', nargs='+', default=['test'], help='test, train, candidates, holdout, sample, or a file of log paths')
    p.add_argument('--since', default='20260801', help='for --logs sample: recordings from this date on')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--limit', type=int, default=0)
    p.set_defaults(fn=cmd_ensemble)

    p = sub.add_parser('ens-report', help='the ensemble on the test logs and machine logs')
    common(p, models=True)
    p.set_defaults(fn=cmd_ens_report)

    p = sub.add_parser('label', help='score the label rule on the test logs')
    common(p, models=True)
    p.add_argument('--variants', action='store_true')
    p.add_argument('--show', type=int, default=10)
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser('select', help='choose logs to write')
    common(p, models=True)
    p.add_argument('--hours', type=float, default=200.0)
    p.add_argument('--title-cap', type=int, default=3, help='most logs of one title')
    p.add_argument('--channel-cap', type=float, default=30.0, help='most hours of one channel (0: no cap)')
    p.add_argument('--min-decided', type=float, default=0.8)
    p.add_argument('--min-breaks-marked', type=float, default=0.5)
    p.add_argument('--allow-no-breaks', action='store_true')
    p.add_argument('--out', required=True, help='manifest TSV')
    p.set_defaults(fn=cmd_select)

    p = sub.add_parser('write', help='write labelled copies of the manifest logs')
    common(p, models=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out-dir', required=True)
    p.set_defaults(fn=cmd_write)

    p = sub.add_parser('review', help='review lists')
    common(p, models=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--min-conflict', type=float, default=3)
    p.add_argument('--min-undecided', type=float, default=60)
    p.add_argument('--min-contradiction', type=float, default=30)
    p.add_argument('--channel-share', type=float, default=0.1)
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser('silver', help='judge models on held-out machine logs')
    common(p)
    p.add_argument('--model', nargs='+', required=True)
    p.add_argument('--logs', default='holdout')
    p.add_argument('--since', default='20260801')
    for k in ('show_min_len', 'break_min_len', 'break_max_len'):
        p.add_argument('--' + k.replace('_', '-'), dest=k, type=float, default=None)
    p.set_defaults(fn=cmd_silver)

    args = ap.parse_args(argv)
    return args.fn(args)

if __name__ == '__main__':
    sys.exit(main())
