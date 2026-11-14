#!/usr/bin/env python3
"""Checks tools/harvest.py on synthetic feature logs: the fingerprint marks and the labelled copies.

The logs are made up of segments (show, ads, shared programme), each with its own scene cuts and
front-volume envelope, so a segment placed in two logs repeats both.  Here: an ad planted in
recordings of three titles is marked in each; a cut pattern that recurs with different audio is
not; a log whose audio is 0.7 s late (no audio_shift) still matches, and the lag is measured; a
long stretch of programme shared by three titles is not an ad; a marathon with the same ad slot
twice at one offset keeps both; an ad carried only between episodes of one series is a
second-tier mark when it sits next to a title-diverse one, while the series' title sequence is in
neither tier; and writing a labelled copy leaves the source log as it was,
refuses to write among the sources, and trains with the undecided spans at weight 0.

Run it directly; it exits non-zero if anything fails.  Needs no video and no model.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import harvest
from pycommflag import neural, processor

_fails = []

def check(name, ok, detail=''):
    print(('  pass  ' if ok else '  FAIL  ') + name + (f'   {detail}' if detail else ''))
    if not ok:
        _fails.append(name)

FPS = 29.97

def segment(cut_seed, dur, audio_seed=None, mean_gap=3.0):
    """Cut times and an envelope function for `dur` seconds of content."""
    rng = np.random.default_rng(cut_seed)
    cuts, t = [], rng.uniform(0.5, 2)
    while t < dur - 0.5:
        cuts.append(t)
        t += rng.uniform(0.6, 2 * mean_gap)
    arng = np.random.default_rng(10_000 + (cut_seed if audio_seed is None else audio_seed))
    knots = arng.uniform(0.05, 1.0, int(dur * 4) + 8)   # a new level every 0.25 s
    def env(x):
        return np.interp(x * 4, np.arange(len(knots)), knots)
    return dict(dur=dur, cuts=np.array(cuts), env=env)

def make_log(path, rec, segs, late=0.0, shift_key=True):
    """A feature log of `segs` back to back; audio `late` seconds behind the picture."""
    dur = sum(s['dur'] for s in segs)
    t = np.arange(1, int(dur * FPS)) / FPS
    diff = np.full(len(t), 2.0)
    fvol = np.zeros(len(t))
    blank = np.zeros(len(t))
    start = 0.0
    bounds = []
    for s in segs:
        inside = (t >= start) & (t < start + s['dur'])
        fvol[inside] = s['env'](t[inside] - start)
        for c in s['cuts']:
            diff[np.argmin(np.abs(t - (start + c)))] = 30.0
        bounds.append(start)
        start += s['dur']
    for b in bounds[1:]:
        k = np.argmin(np.abs(t - b))
        blank[k:k + 3] = 1
        diff[k] = 30.0
    if late:
        fvol = np.interp(t - late, t, fvol)
    frames = [[round(float(t[k]), 5), 1, int(blank[k]), float(diff[k]), round(float(fvol[k]), 5), 0.0, 0, 1, 0, 0, 0]
              for k in range(len(t))]
    flog = {'file_version': 11, 'chanid': rec.split('_')[0], 'filename': f'/nonexistent/{rec}.ts',
            'duration': float(dur), 'frame_rate': FPS}
    if shift_key:
        flog['audio_shift'] = 0
    flog.update({'logo': [[0, 0], [10, 10]],
                 'frames_header': ['time', 'logo_present', 'is_blank', 'diff', 'fvol', 'rvol', 'silence', 'speech', 'music', 'noise', 'cc_text'],
                 'frames': frames, 'tags': []})
    processor.write_feature_log(flog, path)

def show(seed, dur):
    return segment(seed, dur, mean_gap=5.0)

tmp = tempfile.mkdtemp()
try:
    logs = os.path.join(tmp, 'logs')
    os.makedirs(logs)
    cache = os.path.join(tmp, 'cache')
    AD_X = segment(1, 30.0)
    AD_Y = segment(2, 30.0)
    AD_Y_OTHER_AUDIO = segment(2, 30.0, audio_seed=99)
    AD_Z = segment(3, 30.0)
    SHARED = segment(4, 300.0, mean_gap=4.0)
    titles = {}
    def put(name, title, segs, **kw):
        rec = f'9000_20260101{len(titles):06d}'
        titles[rec] = (title, name, '')
        make_log(os.path.join(logs, f'cf_{rec}.ts.json'), rec, segs, **kw)
        return rec
    # ad X in three titles (and late audio in a fourth); ad Y's cuts with other audio in a fifth
    recA = put('a', 'Alpha Show', [show(101, 200), AD_X, show(102, 150), AD_Y, show(103, 200)])
    recB = put('b', 'Bravo Show', [show(201, 120), AD_Y, show(202, 140), AD_X, show(203, 220)])
    recC = put('c', 'Charlie Show', [show(301, 260), AD_X, AD_Y, show(302, 250)])
    recL = put('l', 'Late Audio', [show(401, 180), AD_X, show(402, 300)], late=0.7, shift_key=False)
    recE = put('e', 'Echo Show', [show(501, 230), AD_Y_OTHER_AUDIO, show(502, 260)])
    # a long programme stretch shared by three titles
    recH = [put(f'h{k}', f'Hotel {k}', [show(600 + k, 100), SHARED, show(610 + k, 100)]) for k in range(3)]
    # a marathon: three titles with the same ad in the same slot twice, 750 s apart
    recM = [put(f'm{k}', f'Marathon {k}', [show(700 + k, 300), AD_Z, show(710 + k, 720), AD_Z, show(720 + k, 200)]) for k in range(3)]
    # four episodes of one series, each opening with the same title sequence; three carry ad Q
    # (seen under one other title too) right after ad X
    INTRO = segment(5, 40.0)
    AD_Q = segment(6, 30.0)
    recS = [put(f'ep{k}', 'Series Show', [show(800 + k, 30), INTRO, show(810 + k, 250), AD_X] + ([AD_Q] if k < 3 else []) + [show(820 + k, 200)])
            for k in range(4)]
    put('d', 'Delta Show', [show(901, 150), AD_Q, show(902, 200)])
    with open(os.path.join(tmp, 'titles.tsv'), 'w') as f:
        f.write('rec\ttitle\tsubtitle\tprogramid\n')
        for r, (t, s, p) in titles.items():
            f.write(f'{r}\t{t}\t{s}\t{p}\n')

    harvest.main(['index', '--cache', cache, '--titles', os.path.join(tmp, 'titles.tsv'), '--machine', logs, '--jobs', '2'])
    harvest.main(['marks', '--cache', cache, '--targets', 'all', '--jobs', '2'])
    harvest.load_index.cache_clear()
    metas, bad = harvest.load_index(cache)
    by_rec = {m['rec']: m for m in metas}
    by_key = {m['key']: m for m in metas}
    rule = dict(harvest.RULE_DEFAULTS)
    def marked(rec, a, b):
        mk = harvest.marks_for(cache, by_rec[rec], by_key, rule)
        ad, cnt = mk.at(np.arange(a, b) + 0.5)
        return ad.mean(), cnt.max(initial=0)

    check('every log indexed', len(metas) == len(titles) and not bad, f'{len(metas)} of {len(titles)}, unreadable {bad}')
    share, n = marked(recA, 205, 225)
    check('ad X in three titles is marked in the first', share > 0.9 and n >= 2, f'{share:.2f} marked, {n} titles')
    share, n = marked(recC, 265, 285)
    check('...and in the third', share > 0.9, f'{share:.2f}')
    check('show around it is not', marked(recA, 20, 180)[0] == 0)
    share, _ = marked(recE, 235, 255)
    check('a cut pattern repeated with different audio is not marked', share == 0, f'{share:.2f}')
    runs, partners = harvest.load_runs(cache, by_rec[recE])
    pe = {by_key[p]['rec'] for p, r in zip(partners, runs[:, 5]) if r >= 0.8}
    check('...its runs with the real ad fail the audio check', not ({recA, recB, recC} & pe), f'accepted partners {pe}')
    share, n = marked(recL, 185, 205)
    check('audio 0.7 s late still matches', share > 0.9 and n >= 2, f'{share:.2f} marked, {n} titles')
    runs, partners = harvest.load_runs(cache, by_rec[recL])
    lags = [late for p, r, late in zip(partners, runs[:, 5], runs[:, 6]) if r >= 0.8 and by_key[p]['rec'] in (recA, recB, recC)]
    check('...and its lateness is measured', lags and all(abs(x - 0.7) < 0.06 for x in lags), f'{[round(x, 3) for x in lags]}')
    share, _ = marked(recH[0], 110, 390)
    check('a long programme shared by three titles is not an ad', share == 0, f'{share:.2f}')
    mk = harvest.marks_for(cache, by_rec[recH[0]], by_key, rule)
    check('...it is recorded as shared', mk.shared[int((200 - mk.g0) * 10)])
    s1, _ = marked(recM[0], 305, 325)
    s2, _ = marked(recM[0], 1055, 1075)
    check('a marathon ad slot keeps its marks at both airings', s1 > 0.9 and s2 > 0.9, f'{s1:.2f}, {s2:.2f}')

    mk = harvest.marks_for(cache, by_rec[recS[0]], by_key, rule)
    q = np.arange(351, 379) + 0.5
    check('a series ad next to a title-diverse one is a second-tier mark', mk.at2(q)[0].mean() > 0.9 and not mk.at(q)[0].any(),
          f'tier 2 {mk.at2(q)[0].mean():.2f}, tier 1 {mk.at(q)[0].mean():.2f}')
    intro = np.arange(32, 68) + 0.5
    check('the series title sequence is marked in neither tier', not mk.at(intro)[0].any() and not mk.at2(intro)[0].any())
    check('...and stays out of the second tier with the start allowed', not harvest.marks_for(cache, by_rec[recS[0]], by_key, dict(rule, series_skip_start=0.0)).at2(intro)[0].any())

    # a labelled copy, from made-up model output
    m = by_rec[recA]
    flog = processor.read_feature_log(m['path'])
    flog['tags'] = []
    win, _, _, times = neural.load_data_sliding_window(neural.load_nonpersistent(flog, False))
    p = np.full(len(times), 0.02)
    p[(times > 200) & (times < 230)] = 0.97
    p[(times > 380) & (times < 410)] = 0.97
    p[(times > 440) & (times < 470)] = 0.5      # undecided
    os.makedirs(os.path.join(cache, 'ens'), exist_ok=True)
    np.savez(harvest.ens_path(cache, m), times=times, fake1=p.astype(np.float32), fake2=p.astype(np.float32))
    man = os.path.join(tmp, 'manifest.tsv')
    with open(man, 'w') as f:
        f.write('log\n' + m['path'] + '\n')
    def digest(path):
        with open(path, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest(), os.stat(path).st_mtime_ns
    before = {f: digest(os.path.join(logs, f)) for f in os.listdir(logs)}
    out = os.path.join(tmp, 'out')
    models = [os.path.join(tmp, 'fake1.keras'), os.path.join(tmp, 'fake2.keras')]
    common = ['--cache', cache, '--model', *models, '--break-min-len', '20', '--show-min-len', '20']
    harvest.main(['write', *common, '--manifest', man, '--out-dir', out])
    after = {f: digest(os.path.join(logs, f)) for f in os.listdir(logs)}
    check('writing leaves every source log as it was', before == after)
    written = os.path.join(out, os.path.basename(m['path']) + '.gz')
    check('the copy is written', os.path.exists(written))
    if os.path.exists(written):
        w = processor.read_feature_log(written)
        ml = w.get('machine_labels') or {}
        check('...marked as machine-labelled with its models and rule', ml.get('models') == ['fake1', 'fake2'] and 'rule' in ml and list(w).index('machine_labels') == list(w).index('filename') + 1, f'{list(w)[:4]}')
        tags = [(t, round(b), round(e)) for t, (b, e) in w['tags']]
        check('...with the breaks as commercial and the unsure stretch as TRANSITION',
              any(t == 3 and abs(b - 200) <= 3 for t, b, e in tags) and any(t == 2 and b <= 445 and e >= 465 for t, b, e in tags), f'{tags}')
        cond = neural.load_nonpersistent(w, True)
        ct, wt = cond[:, neural.TIMESTAMPS], cond[:, neural.WEIGHTS]
        check('...and the unsure stretch trains at weight 0', (wt[(ct > 445) & (ct < 465)] == 0).all())
    try:
        harvest.main(['write', *common, '--manifest', man, '--out-dir', logs])
        refused = False
    except SystemExit:
        refused = True
    check('writing into a directory of source logs is refused', refused and before == {f: digest(os.path.join(logs, f)) for f in os.listdir(logs)})
finally:
    shutil.rmtree(tmp)

if _fails:
    print(f'{len(_fails)} FAILED')
    sys.exit(1)
print('all passed')
