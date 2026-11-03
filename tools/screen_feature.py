#!/usr/bin/env python3
"""Screen a candidate model feature against what a trained model already knows.

A feature's own show/commercial AUC says nothing about whether the model has that information
already, or whether one threshold for it works across recordings (quiet distances screened best of
any feature that way and cost ~0.75 points on blah once trained in).  This asks three narrower
questions, on hand-tagged logs the model was not trained on:

  stack       does the candidate improve the model's own output?  A logistic regression on the
              model's logit is compared with one that also gets the candidate's summaries over the
              window (centre second, mean before, mean after, min, max, and each times the logit),
              both fitted on other recordings (group k-fold by log) and scored on the held-out ones.
              The same candidate circularly shifted within each recording (same values, wrong
              place) gives the gain that fitting more columns buys by chance.  Accuracy here is per
              second at 0.5, not --eval's tag-based score.
  redundancy  how much of the candidate a ridge regression recovers from the existing features'
              window summaries (held-out R^2).  High means the model can already compute it; low
              does not prove it can't, since the model sees the whole sequence, not 4 summaries.
  stability   pooled AUC next to the per-log median, and how many logs it reverses on: per-log AUC
              allows each recording its own threshold, and the model gets one for all of them.

The model's predictions and each candidate's columns are cached per log under --cache (keyed by the
model file and candidate name), so after the first run a new candidate costs one read of each log
and no inference.  Delete a candidate's cache directory after changing its definition.

    tools/screen_feature.py --model models/good-9943-9853.keras -c quiet -c cadence \\
        data/test/* data/blah/*

Logs are grouped into sets by their directory name (test, blah).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import neural, processor

WB = round(neural.WINDOW_BEFORE * neural.SUMMARY_RATE)
WA = round(neural.WINDOW_AFTER * neural.SUMMARY_RATE)
NEAR = 10.0 # seconds from a show/commercial boundary that count as "near" in the fix/break counts

# --- candidates: per-frame columns, computed from the frames condense() is handed (16 columns,
# volume already rescaled, DIFF normalized) and sampled at each second's centre frame, the way
# condense samples BLANK_SINCE/UNTIL

def _since_until(timestamps, mask):
    mtimes = timestamps[mask]
    if not len(mtimes):
        far = np.full(len(timestamps), neural.DIST_CAP)
        return far, far
    j = np.searchsorted(mtimes, timestamps, 'right')
    since = np.where(j > 0, timestamps - mtimes[np.clip(j-1, 0, len(mtimes)-1)], neural.DIST_CAP)
    k = np.searchsorted(mtimes, timestamps, 'left')
    until = np.where(k < len(mtimes), mtimes[np.clip(k, 0, len(mtimes)-1)] - timestamps, neural.DIST_CAP)
    return since, until

def cand_quiet(frames, timestamps, frame_rate):
    """QUIET_SINCE/QUIET_UNTIL as on branch quiet-distance (trained: test same, blah -0.75)."""
    if np.count_nonzero(frames[:, neural.FVOL] > 1e-4) < frame_rate * 60:
        quiet = np.zeros(len(frames), dtype=bool) # no volume rescale happened
    else:
        quiet = (frames[:, neural.FVOL] < 0.01) & (frames[:, neural.RVOL] < 0.01)
    return [neural._scale_dist(d) for d in _since_until(timestamps, quiet)]

def cand_cadence(frames, timestamps, frame_rate):
    """CADENCE_CHANGE as on branch frame-timing (trained: no measurable effect)."""
    span = 3.0
    gap = np.diff(timestamps, append=timestamps[-1] + 1/frame_rate)
    uneven = (np.abs(gap - 1/frame_rate) > 0.25/frame_rate).astype('float64')
    csum = np.concatenate(([0.0], np.cumsum(uneven)))
    i = np.arange(len(timestamps))
    lo = np.searchsorted(timestamps, timestamps - span)
    hi = np.searchsorted(timestamps, timestamps + span)
    before = (csum[i] - csum[lo]) / np.maximum(i - lo, 1)
    after = (csum[hi] - csum[i]) / np.maximum(hi - i, 1)
    before = np.where(i > lo, before, after)
    after = np.where(hi > i, after, before)
    return [np.abs(after - before)]

def cand_logo_until(frames, timestamps, frame_rate):
    """Seconds until the next logo frame (the mirror of LOGO_RUN); never trained."""
    return [neural._scale_dist(_since_until(timestamps, frames[:, neural.LOGO] > 0.5)[1])]

def _audio_label(frames):
    return np.argmax(frames[:, neural.SILENCE:neural.NOISE+1], axis=1)

def _audio_changes(frames):
    """index of every frame whose ina label differs from the frame before it"""
    lab = _audio_label(frames)
    return np.flatnonzero(np.diff(lab) != 0) + 1

def _window_mean(timestamps, values, half):
    csum = np.concatenate(([0.0], np.cumsum(values, dtype='float64')))
    lo = np.searchsorted(timestamps, timestamps - half)
    hi = np.searchsorted(timestamps, timestamps + half, 'right')
    return (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)

def cand_audio_change(frames, timestamps, frame_rate):
    """Seconds since/until the ina label (silence/speech/music/noise) last/next changes."""
    mask = np.zeros(len(frames), dtype=bool)
    mask[_audio_changes(frames)] = True
    return [neural._scale_dist(d) for d in _since_until(timestamps, mask)]

def cand_av_cut(frames, timestamps, frame_rate):
    """Seconds since/until an ina label change within 0.5s of a blank frame or a scene cut:
    picture and sound changing together, where the label changes inside a show mostly don't."""
    cut = (frames[:, neural.BLANK] > 0.5) | (frames[:, neural.DIFF] >= neural.DIFF_THRESHOLD / 40)
    near_cut = _window_mean(timestamps, cut, 0.5) > 0
    mask = np.zeros(len(frames), dtype=bool)
    ch = _audio_changes(frames)
    mask[ch[near_cut[ch]]] = True
    return [neural._scale_dist(d) for d in _since_until(timestamps, mask)]

def cand_audio_seglen(frames, timestamps, frame_rate):
    """Length of the ina label run the frame is in (log-scaled seconds, capped like the distances)."""
    ch = _audio_changes(frames)
    starts = np.concatenate(([0], ch))
    ends = np.concatenate((ch, [len(frames)]))
    t_end = np.append(timestamps, timestamps[-1] + 1/frame_rate)
    lengths = t_end[ends] - timestamps[starts]
    return [neural._scale_dist(np.repeat(lengths, ends - starts))]

def cand_audio_flips(frames, timestamps, frame_rate):
    """ina label changes per 30s around the frame (log1p), which per-second means blur."""
    ct = timestamps[_audio_changes(frames)]
    counts = np.searchsorted(ct, timestamps + 15, 'right') - np.searchsorted(ct, timestamps - 15)
    return [np.log1p(counts) / np.log1p(60)]

def cand_music_wide(frames, timestamps, frame_rate):
    """Share of music frames within +-300s: context five times wider than the model's window."""
    return [_window_mean(timestamps, frames[:, neural.MUSIC], neural.DIST_CAP)]

def cand_rear_change(frames, timestamps, frame_rate):
    """Seconds since/until the rear channels switch on or off (5.1 show vs stereo ads):
    rear active where RVOL >= 0.01 for most of the surrounding 4s."""
    active = _window_mean(timestamps, frames[:, neural.RVOL] >= 0.01, 2.0) > 0.5
    mask = np.zeros(len(frames), dtype=bool)
    mask[np.flatnonzero(np.diff(active.astype('int8')) != 0) + 1] = True
    return [neural._scale_dist(d) for d in _since_until(timestamps, mask)]

CANDIDATES = {
    'quiet': cand_quiet,
    'cadence': cand_cadence,
    'logo_until': cand_logo_until,
    'audio_change': cand_audio_change,
    'av_cut': cand_av_cut,
    'audio_seglen': cand_audio_seglen,
    'audio_flips': cand_audio_flips,
    'music_wide': cand_music_wide,
    'rear_change': cand_rear_change,
}

# --- loading

_stash = {}
_orig_condense = neural.condense

def _condense(frames, timestamps, answers, weights, seconds):
    out = _orig_condense(frames, timestamps, answers, weights, seconds)
    ts = timestamps.reshape(-1)
    b = np.floor((ts - ts[0]) / seconds).astype('int64')
    starts = np.concatenate(([0], np.flatnonzero(np.diff(b)) + 1))
    counts = np.diff(np.append(starts, len(frames)))
    mid = starts + counts//2
    _stash['cands'] = {name: np.stack([np.asarray(c)[mid] for c in CANDIDATES[name](frames, ts, _stash['frame_rate'])], axis=1)
                       for name in _stash['want']}
    return out

neural.condense = _condense

def _pad(a):
    return np.concatenate((np.tile(a[:1], (WB, 1)), a, np.tile(a[-1:], (WA, 1))))

class Model:
    def __init__(self, path):
        self.path = path
        self.model = None

    def logits(self, condensed):
        if self.model is None:
            import keras
            self.model = keras.models.load_model(self.path)
        windows = neural.load_data_sliding_window(condensed)[0]
        p = self.model.predict(neural.make_data_generator(windows), verbose=0).reshape(-1)
        p = np.clip(p.astype('float64'), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

def load(logs, model, cands, cache):
    mdir = os.path.join(cache, 'model-' + os.path.splitext(os.path.basename(os.path.realpath(model.path)))[0])
    os.makedirs(mdir, exist_ok=True)
    for c in cands:
        os.makedirs(os.path.join(cache, 'cand-' + c), exist_ok=True)

    out = []
    for path in logs:
        dset = os.path.basename(os.path.dirname(os.path.abspath(path)))
        stem = dset + '-' + os.path.basename(path).split('.json')[0]
        mfile = os.path.join(mdir, stem + '.npz')
        cfiles = {c: os.path.join(cache, 'cand-' + c, stem + '.npy') for c in cands}
        want = [c for c, f in cfiles.items() if not os.path.exists(f)]
        if want or not os.path.exists(mfile):
            print(f'reading {path}', file=sys.stderr)
            flog = processor.read_feature_log(path)
            _stash.clear()
            _stash.update(frame_rate=flog.get('frame_rate', 29.97), want=want)
            condensed = neural.load_nonpersistent(flog, True)
            if condensed is None:
                print(f'{path}: no usable data, skipped', file=sys.stderr)
                continue
            for c in want:
                np.save(cfiles[c], _pad(_stash['cands'][c]).astype('float32'))
            if not os.path.exists(mfile):
                np.savez(mfile, z=model.logits(condensed),
                         answers=condensed[WB:-WA, neural.ANSWERS], weights=condensed[WB:-WA, neural.WEIGHTS],
                         times=condensed[WB:-WA, neural.TIMESTAMPS], base=condensed[:, :neural.FEATURE_WIDTH])
        m = dict(np.load(mfile))
        m['name'] = os.path.basename(path).split('.json')[0]
        m['set'] = dset
        m['cands'] = {c: np.load(cfiles[c]) for c in cands}
        out.append(m)
    return out

# --- maths

def summaries(padded):
    """centre, mean before, mean after, min and max over each second's window, per column"""
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(padded.astype('float64'), WB+1+WA, axis=0) # (n, cols, window)
    return np.concatenate((w[:, :, WB], w[:, :, :WB].mean(-1), w[:, :, WB+1:].mean(-1),
                           w.min(-1), w.max(-1)), axis=1)

def fit_logistic(X, y, l2=1e-3, iters=30):
    w = np.zeros(X.shape[1])
    reg = np.full(X.shape[1], l2)
    reg[0] = 0 # bias
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) / len(y) + reg * w
        H = (X.T * (p * (1 - p))) @ X / len(y) + np.diag(reg)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w

def stack_predict(rows, folds, make_x):
    """held-out P(commercial) for every row, fitting on the other folds"""
    pred = [None] * len(rows)
    for k in range(folds):
        train = [i for i in range(len(rows)) if i % folds != k]
        test = [i for i in range(len(rows)) if i % folds == k]
        if not test:
            continue
        Xtr = np.concatenate([rows[i]['x'] for i in train])
        mu = Xtr.mean(0)
        sd = Xtr.std(0) + 1e-9
        def design(x, z):
            xs = (x - mu) / sd
            return make_x(xs, z)
        Xd = np.concatenate([design(rows[i]['x'], rows[i]['z']) for i in train])
        w = fit_logistic(Xd, np.concatenate([rows[i]['y'] for i in train]))
        for i in test:
            pred[i] = 1 / (1 + np.exp(-(design(rows[i]['x'], rows[i]['z']) @ w)))
    return pred

def x_base(xs, z):
    return np.column_stack((np.ones(len(z)), z))

def x_cand(xs, z):
    return np.column_stack((np.ones(len(z)), z, xs, xs * z[:, None]))

def auc(y, s):
    npos = int(y.sum())
    nneg = len(y) - npos
    if not npos or not nneg:
        return np.nan
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    start = np.concatenate(([0], np.flatnonzero(np.diff(ss)) + 1))
    end = np.concatenate((start[1:], [len(ss)]))
    ranks = np.empty(len(ss))
    ranks[order] = np.repeat((start + end + 1) / 2.0, end - start)
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# --- the screens

def build_rows(data, cand, shift_rng=None):
    rows = []
    for m in data:
        keep = m['weights'] > 0
        c = m['cands'][cand]
        if shift_rng is not None:
            inner = c[WB:-WA]
            n = len(inner)
            inner = np.roll(inner, int(shift_rng.integers(n // 4, 3 * n // 4 + 1)), axis=0)
            c = _pad(inner)
        y = (m['answers'] >= 0.5).astype('float64')
        t = m['times']
        edges = t[1:][np.diff(y) != 0]
        if len(edges):
            j = np.clip(np.searchsorted(edges, t), 1, len(edges)) - 1
            dist = np.minimum(np.abs(t - edges[j]), np.abs(t - edges[np.minimum(j + 1, len(edges) - 1)]))
        else:
            dist = np.full(len(t), np.inf)
        rows.append(dict(name=m['name'], set=m['set'], x=summaries(c)[keep], z=m['z'][keep],
                         y=y[keep], near=dist[keep] < NEAR))
    return rows

def pct(correct, n):
    return 100.0 * correct / n if n else float('nan')

def screen_stack(data, cand, folds, nulls):
    rows = build_rows(data, cand)
    base = stack_predict(rows, folds, x_base)
    full = stack_predict(rows, folds, x_cand)
    null = [stack_predict(build_rows(data, cand, np.random.default_rng(1000 + r)), folds, x_cand)
            for r in range(nulls)]

    print(f'\n== {cand}: stack (held-out per-second accuracy at 0.5; logloss in millinats/second)')
    print(f'{"set":<6} {"model":>7} {"stack":>7} {"+cand":>7} {"gain":>6} {"null gain":>14}'
          f' {"dLL":>6} {"null dLL":>9}')
    for s in sorted({r['set'] for r in rows}) + ['all']:
        idx = [i for i, r in enumerate(rows) if s == 'all' or r['set'] == s]
        y = np.concatenate([rows[i]['y'] for i in idx])
        raw = np.concatenate([rows[i]['z'] for i in idx]) >= 0
        pb = np.concatenate([base[i] for i in idx])
        pf = np.concatenate([full[i] for i in idx])
        acc = lambda p: pct(np.count_nonzero((p >= 0.5) == (y == 1)), len(y))
        ab, af = acc(pb), acc(pf)
        ngain = [acc(np.concatenate([n[i] for i in idx])) - ab for n in null]
        ndll = [1000 * (logloss(y, pb) - logloss(y, np.concatenate([n[i] for i in idx]))) for n in null]
        dll = 1000 * (logloss(y, pb) - logloss(y, pf))
        print(f'{s:<6} {pct(np.count_nonzero(raw == (y == 1)), len(y)):7.2f} {ab:7.2f} {af:7.2f} {af-ab:+6.2f}'
              f' {min(ngain):+6.2f}..{max(ngain):+6.2f} {dll:+6.2f} {max(ndll):+9.2f}')

    # where the candidate changes the decision, against the stack without it
    print('changed seconds (fixed / broken), by true class and distance to a boundary:')
    for label, cls in (('show', 0), ('commercial', 1)):
        for where, near in ((f'<{NEAR:g}s', True), (f'>={NEAR:g}s', False)):
            fixed = broken = 0
            for i, r in enumerate(rows):
                sel = (r['y'] == cls) & (r['near'] == near)
                okb = (base[i][sel] >= 0.5) == (cls == 1)
                okf = (full[i][sel] >= 0.5) == (cls == 1)
                fixed += np.count_nonzero(okf & ~okb)
                broken += np.count_nonzero(okb & ~okf)
            print(f'  {label:<10} {where:>6}: {fixed:6d} / {broken:6d}')

    deltas = []
    for i, r in enumerate(rows):
        ok = lambda p: np.count_nonzero((p >= 0.5) == (r['y'] == 1))
        deltas.append((pct(ok(full[i]) - ok(base[i]), len(r['y'])), r['set'], r['name']))
    deltas.sort()
    up = sum(d > 0.1 for d, _, _ in deltas)
    down = sum(d < -0.1 for d, _, _ in deltas)
    print(f'per log: {up} better and {down} worse by more than 0.1 points, of {len(deltas)}')
    shown = deltas if len(deltas) <= 10 else deltas[:5] + [None] + deltas[-5:]
    for d in shown:
        print('  ...' if d is None else f'  {d[0]:+6.2f}  {d[1]}/{d[2]}')

def screen_redundancy(data, cand, folds):
    rows = []
    for m in data:
        keep = m['weights'] > 0
        rows.append((summaries(m['base'])[keep], m['cands'][cand][WB:-WA][keep].astype('float64')))
    ncol = rows[0][1].shape[1]
    sse = np.zeros(ncol)
    sst = np.zeros(ncol)
    ally = np.concatenate([r[1] for r in rows])
    for k in range(folds):
        train = [r for i, r in enumerate(rows) if i % folds != k]
        test = [r for i, r in enumerate(rows) if i % folds == k]
        if not test:
            continue
        X = np.concatenate([r[0] for r in train])
        Y = np.concatenate([r[1] for r in train])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = np.column_stack((np.ones(len(X)), (X - mu) / sd))
        w = np.linalg.solve(Xs.T @ Xs + 1e-3 * len(X) * np.diag([0] + [1] * (Xs.shape[1] - 1)), Xs.T @ Y)
        for x, y in test:
            p = np.column_stack((np.ones(len(x)), (x - mu) / sd)) @ w
            sse += ((y - p) ** 2).sum(0)
            sst += ((y - Y.mean(0)) ** 2).sum(0)
    print(f'redundancy: held-out R^2 from the {neural.FEATURE_WIDTH} features\' window summaries: '
          + ', '.join(f'col {j} {1 - sse[j]/sst[j]:.3f}' if sst[j] else f'col {j} constant' for j in range(ncol))
          + f'  (column std {", ".join(f"{s:.3f}" for s in ally.std(0))})')

def screen_stability(data, cand):
    ncol = data[0]['cands'][cand].shape[1]
    for j in range(ncol):
        ys, cs, per = [], [], []
        for m in data:
            keep = m['weights'] > 0
            y = (m['answers'][keep] >= 0.5).astype('float64')
            c = m['cands'][cand][WB:-WA, j][keep].astype('float64')
            ys.append(y)
            cs.append(c)
            per.append(auc(y, c))
        pooled = auc(np.concatenate(ys), np.concatenate(cs))
        per = np.array(per)
        per = per[~np.isnan(per)]
        flip = pooled < 0.5
        if flip:
            pooled, per = 1 - pooled, 1 - per
        q1, med, q3 = np.percentile(per, [25, 50, 75])
        print(f'stability: col {j}{" (inverted)" if flip else ""}: pooled AUC {pooled:.3f}, per-log median {med:.3f}'
              f' (IQR {q1:.3f}-{q3:.3f}), reversed on {np.count_nonzero(per < 0.5)} of {len(per)} logs')

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='the trained model to screen against')
    ap.add_argument('-c', '--candidate', action='append', required=True, choices=sorted(CANDIDATES))
    ap.add_argument('--cache', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.tmp', 'screen-cache'))
    ap.add_argument('--folds', type=int, default=10, help='group k-fold over logs (default 10)')
    ap.add_argument('--nulls', type=int, default=3, help='circularly shifted copies for the chance gain (default 3)')
    ap.add_argument('logs', nargs='+')
    args = ap.parse_args()

    try: os.nice(19)
    except OSError: pass

    logs = sorted(f for f in args.logs if '.json' in os.path.basename(f) and not f.endswith('.npy'))
    data = load(logs, Model(args.model), args.candidate, args.cache)
    print(f'{len(data)} logs: ' + ', '.join(f'{s} {sum(m["set"] == s for m in data)}' for s in sorted({m["set"] for m in data})))
    folds = min(args.folds, len(data))
    for c in args.candidate:
        screen_stack(data, c, folds, args.nulls)
        screen_redundancy(data, c, folds)
        screen_stability(data, c)

if __name__ == '__main__':
    main()
