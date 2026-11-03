#!/usr/bin/env python3
"""Run --eval with some model input columns ablated, to see how much a trained model uses them.

  mean   each column replaced by its mean over the training caches (data/*.data.npy, weighted rows)
  shift  each recording's columns circularly shifted by half its length: the recording's own values,
         in the wrong place

    tools/ablate_eval.py --columns 6-9 --mode shift -- --eval models/good-9943-9853.keras --data data/test/*

Everything after -- is passed to pycommflag unchanged.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import neural, options
from pycommflag import main as pmain

def training_means(cols, pattern):
    total = np.zeros(len(cols))
    n = 0
    for f in sorted(glob.glob(pattern)):
        if f.endswith('.nologo.data.npy'):
            continue
        d = np.load(f, mmap_mode='r')
        if d.shape[-1] != neural.WEIGHTS + 1:
            continue
        keep = d[:, neural.WEIGHTS] > 0
        total += np.asarray(d[keep][:, cols], dtype='float64').sum(0)
        n += int(keep.sum())
    return total / n

def main():
    argv = sys.argv[1:]
    rest = argv[argv.index('--') + 1:] if '--' in argv else []
    argv = argv[:argv.index('--')] if '--' in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--columns', required=True, help='model input columns, e.g. 6-9 or 6,8')
    ap.add_argument('--mode', choices=('mean', 'shift'), required=True)
    ap.add_argument('--train-caches', default='data/*.data.npy')
    args = ap.parse_args(argv)

    cols = []
    for part in args.columns.split(','):
        a, _, b = part.partition('-')
        cols += list(range(int(a), int(b or a) + 1))
    assert all(0 <= c < neural.FEATURE_WIDTH for c in cols)

    if args.mode == 'mean':
        means = training_means(cols, args.train_caches)
        print('ablating columns', cols, 'with training means', np.round(means, 4))
    else:
        print('ablating columns', cols, 'by circular shift')

    orig = neural.load_nonpersistent
    def ablated(*a, **kw):
        d = orig(*a, **kw)
        if d is None:
            return d
        if args.mode == 'mean':
            d[:, cols] = means
        else:
            d[:, cols] = np.roll(d[:, cols], len(d) // 2, axis=0)
        return d
    neural.load_nonpersistent = ablated

    sys.argv = [sys.argv[0]] + rest
    sys.exit(pmain.run(options.parse_argv()))

if __name__ == '__main__':
    main()
