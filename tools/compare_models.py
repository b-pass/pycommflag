#!/usr/bin/env python3
"""Compare two groups of trained models from saved --eval output: is the difference more than noise?

Reads the per-log lines that --eval and the end of a training run print
("<log> @ <model> -> Acc A% <- FN:x + FP:y = z seconds WRONG") out of any saved output, so it needs
no model, no logs and no TensorFlow.  A training log scores a temporary file; it is given the name
from that log's "Saving as" line.

    tools/compare_models.py -a 1794333185,1794343227 -b 1794024203,1794033405,1794073869 \\
        --logs-in data/test .tmp/eval-*.txt .tmp/train-*.txt

-a is the changed recipe, -b the baseline; each is a comma-separated list of substrings of model
file names (the unix time at the end is enough).  Only logs scored for every selected model are used.  When a (model, log) pair
is in several files the last file given wins.

What it reports, and what each number assumes:

  seed z      the difference of the two groups' mean accuracy over its standard error,
              --seed-sd * sqrt(1/a + 1/b).  The verdict comes from this.  It asks "would retraining
              both recipes give this again, on these logs?" and assumes the seed sd is known and the
              same for both recipes, so a change that makes training less stable looks better or
              worse than it is; the observed spread inside each group is printed to check that.
  runs        exact permutation of which runs belong to which group.  Assumes nothing but
              exchangeable runs; with few runs it cannot get small (2 against 6: 1 in 28 at best).
  logs        sign-flip permutation and bootstrap over logs of the per-log difference.  This treats
              the recording as the unit, which is right for "would it hold on new recordings?" and
              also covers seed noise as far as that is independent from log to log (it is, as far
              as the runs so far can show).  It is not a second confirmation of the seed z: both
              read the same few logs.
  one log     the difference with each log left out in turn.  Most seed noise sits in a handful of
              logs where a whole break appears or vanishes, so a verdict that one log makes or
              breaks is that log's coin flip.

Accuracy is 1 - wrong seconds / scored seconds, pooled over logs, as --eval totals it.  A log's
scored seconds come from its own lines (wrong / (1 - Acc)); a log nothing got wrong has none and
is given the median, which moves the totals by a hundredth at most and both groups alike.
"""
import argparse
import itertools
import math
import os
import re
import sys

import numpy as np

SEED_SD = 0.06  # points of pooled data/test accuracy between two trainings that differ only in seed
BASELINE_MIN = 6 # baseline runs the thresholds below were sized for

LINE = re.compile(r'(\S+) @ (\S+) -> Acc ([-\d.]+)% <- FN:([-\d.e]+) \+ FP:([-\d.e]+) = ')
SAVED = re.compile(r'Saving as (\S+\.(?:keras|h5))')

def log_name(path):
    name = os.path.basename(path)
    for ext in ('.gz', '.json'):
        if name.endswith(ext):
            name = name[:-len(ext)]
    return name

def read_outputs(files):
    """{model: {log: (missed, extra)}} and {log: scored seconds} from saved output files."""
    wrong = {}
    seconds = {} # log -> (most seconds any model got wrong, the duration that line implies)
    for path in files:
        rows = []
        saved = None
        with open(path, errors='replace') as f:
            for line in f:
                if 'Saving as' in line:
                    for m in SAVED.finditer(line):
                        saved = m.group(1)
                if 'seconds WRONG' not in line:
                    continue
                for m in LINE.finditer(line):
                    rows.append((log_name(m.group(1)), m.group(2), float(m.group(3)),
                                 float(m.group(4)), float(m.group(5))))
        models = {r[1] for r in rows}
        for log, model, acc, missed, extra in rows:
            if saved and len(models) == 1:
                model = saved # training scored its temporary file, then saved it under this name
            wrong.setdefault(os.path.basename(model), {})[log] = (missed, extra)
            w = missed + extra
            if acc < 100 and w > seconds.get(log, (0.5, 0))[0]:
                seconds[log] = (w, w / (1 - acc/100))
    return wrong, {log: s for log, (_, s) in seconds.items()}

def pick(wrong, patterns, what):
    out = []
    for p in patterns:
        hits = sorted(m for m in wrong if p in m)
        if not hits:
            sys.exit(f'no scored model matches {p!r} ({what})')
        out += [m for m in hits if m not in out]
    return out

def verdict(z, n_arm):
    """The protocol in CLAUDE.md ("Noise"): what z means after n_arm seeds of the changed recipe."""
    if n_arm < 2:
        return 'no verdict from one seed: add a seed'
    if n_arm < 4:
        if abs(z) >= 2.0:
            return 'BETTER' if z > 0 else 'WORSE'
        if abs(z) <= (0.6 if n_arm == 2 else 0.9):
            return 'NO MEASURABLE EFFECT'
        return 'undecided: add a seed'
    if abs(z) >= 1.8:
        return 'BETTER' if z > 0 else 'WORSE'
    return 'NO MEASURABLE EFFECT'

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    names = lambda s: [x for x in s.split(',') if x]
    ap.add_argument('-a', '--arm', type=names, required=True, metavar='MODEL[,MODEL...]',
                    help='models of the changed recipe (substrings of their names)')
    ap.add_argument('-b', '--baseline', type=names, required=True, metavar='MODEL[,MODEL...]',
                    help='models of the baseline recipe')
    ap.add_argument('--seed-sd', type=float, default=SEED_SD,
                    help=f'seed-to-seed sd of pooled accuracy, in points (default {SEED_SD})')
    ap.add_argument('--logs-in', metavar='DIR', help='only use logs that are in this directory now (e.g. data/test)')
    ap.add_argument('--exclude', type=names, default=[], metavar='LOG[,LOG...]',
                    help='log names to leave out (e.g. ones a group trained on)')
    ap.add_argument('--top', type=int, default=8, help='logs to list by their share of the difference (default 8)')
    ap.add_argument('files', nargs='+', help='saved --eval output or training logs')
    args = ap.parse_args()

    wrong, seconds = read_outputs(args.files)
    arm = pick(wrong, args.arm, 'arm')
    base = pick(wrong, args.baseline, 'baseline')
    both = [m for m in arm if m in base]
    if both:
        sys.exit(f'in both groups: {both}')

    logs = set.intersection(*(set(wrong[m]) for m in arm + base))
    wanted = set.union(*(set(wrong[m]) for m in arm + base))
    if args.logs_in:
        wanted = {log_name(f) for f in os.listdir(args.logs_in) if not f.endswith('.npy')}
        logs &= wanted
    if wanted - logs:
        print(f'not scored for every model, so left out: {sorted(wanted - logs)}')
    logs = sorted(logs - set(args.exclude))
    if len(logs) < 2:
        sys.exit('fewer than two logs are scored for every selected model')

    known = [seconds[l] for l in logs if l in seconds]
    guessed = [l for l in logs if l not in seconds]
    dur = np.array([seconds.get(l, float(np.median(known))) for l in logs])
    total = dur.sum()
    if guessed:
        print(f'no model got these wrong, so their length is a guess (the median): {guessed}')

    def table(models):
        return np.array([[sum(wrong[m][l]) for l in logs] for m in models])
    A, B = table(arm), table(base)
    acc = lambda w: 100 - 100 * w.sum(-1) / total

    print(f'\n{len(logs)} logs, {total/3600:.1f} scored hours; 0.01 points = {total/1e4:.0f} s')
    for name, models, W in (('arm', arm, A), ('baseline', base, B)):
        a = acc(W)
        spread = f', sd {a.std(ddof=1):.3f}' if len(a) > 1 else ''
        print(f'  {name:8} mean {a.mean():.3f}{spread}   ' + ' '.join(f'{x:.3f}' for x in a))
        for m in models:
            print(f'           {m}')

    # --- runs as the unit
    diff = acc(A).mean() - acc(B).mean()
    se = args.seed_sd * math.sqrt(1/len(arm) + 1/len(base))
    z = diff / se
    print(f'\ndifference (arm - baseline): {diff:+.3f} points')
    print(f'  seed z   {z:+.2f}  (se {se:.3f} from seed sd {args.seed_sd}; 95% interval {diff-1.96*se:+.3f} to {diff+1.96*se:+.3f})')
    df = len(arm) + len(base) - 2
    if df:
        within = math.sqrt((((acc(A) - acc(A).mean())**2).sum() + ((acc(B) - acc(B).mean())**2).sum()) / df)
        print(f'           seed sd seen inside these groups: {within:.3f} ({df} df)')
    allacc = np.concatenate([acc(A), acc(B)])
    n = len(allacc)
    if math.comb(n, len(arm)) <= 200000:
        as_big = count = 0
        for idx in itertools.combinations(range(n), len(arm)):
            mask = np.zeros(n, bool)
            mask[list(idx)] = True
            d = allacc[mask].mean() - allacc[~mask].mean()
            as_big += abs(d) >= abs(diff) - 1e-9
            count += 1
        print(f'  runs     p = {as_big/count:.3f}  ({as_big} of {count} ways to split the runs differ this much)')

    # --- logs as the unit
    per_log = B.mean(0) - A.mean(0) # seconds the arm gains on each log
    rng = np.random.default_rng(0)
    flips = rng.choice([-1.0, 1.0], size=(20000, len(logs)))
    p_logs = (np.abs(flips @ per_log) >= abs(per_log.sum()) - 1e-9).mean()
    pick_logs = rng.integers(0, len(logs), size=(20000, len(logs)))
    boot = 100 * per_log[pick_logs].sum(1) / dur[pick_logs].sum(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    print(f'  logs     p = {p_logs:.3f} (sign flip);  95% interval over logs {lo:+.3f} to {hi:+.3f}')
    print(f'           arm ahead by over 5 s on {(per_log > 5).sum()} logs, behind on {(per_log < -5).sum()}, '
          f'within 5 s on {(abs(per_log) <= 5).sum()}')
    without = 100 * (per_log.sum() - per_log) / (total - dur)
    i, j = int(np.argmin(without)), int(np.argmax(without))
    print(f'  one log  left out: {without[i]:+.3f} (without {logs[i]}) to {without[j]:+.3f} (without {logs[j]})')

    print(f'\nverdict: {verdict(z, len(arm))}')
    if len(base) < BASELINE_MIN:
        print(f'  (the baseline has {len(base)} runs; the thresholds were sized for {BASELINE_MIN} or more, '
              f'so expect "undecided" more often)')
    crit = 1.8 if len(arm) >= 4 else 2.0
    print(f'  with these runs a true effect under {(crit + 0.84) * se:.2f} points would usually be missed')
    if (diff > 0) != (without[i] > 0) or (diff > 0) != (without[j] > 0):
        print('  the sign of the difference depends on one log')

    order = np.argsort(-np.abs(per_log))[:args.top]
    print(f'\nlogs that make up most of the difference (points; + is the arm ahead):')
    for k in order:
        print(f'  {100*per_log[k]/total:+.3f}  {logs[k]:42} arm {A[:, k].mean():6.1f} s wrong ({A[:, k].min():.0f}-{A[:, k].max():.0f})'
              f'   baseline {B[:, k].mean():6.1f} ({B[:, k].min():.0f}-{B[:, k].max():.0f})')

if __name__ == '__main__':
    main()
