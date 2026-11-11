#!/usr/bin/env python3
"""Checks tools/compare_models.py on made-up --eval output.

The tool reads the per-log lines --eval prints, groups models, and judges the difference in pooled
accuracy against seed noise.  Here: the lines are parsed out of noisy output (progress bars on the
same line), a training log's temporary model takes the name it was saved under, a log's scored
length is recovered from its own line, only logs scored for every model count, and the verdict
follows the thresholds for the number of seeds.

Run it directly; it exits non-zero if anything fails.
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
import compare_models as cm

_fails = []

def check(name, ok, detail=''):
    print(('  pass  ' if ok else '  FAIL  ') + name + (f'   {detail}' if detail and not ok else ''))
    if not ok:
        _fails.append(name)

LENGTH = {'one': 3600.0, 'two': 1800.0, 'three': 7200.0, 'perfect': 3000.0}

def line(log, model, missed, extra, directory='data/test'):
    acc = round(100 - 100 * (missed + extra) / LENGTH[log], 4)
    return (f'{directory}/{log}.json.gz @ {model} -> Acc {acc}% <- FN:{missed} + FP:{extra} = '
            f'{round(missed + extra, 3)} seconds WRONG\n')

tmp = tempfile.mkdtemp()

# wrong seconds per model on (one, two, three); "perfect" is always 0
runs = {
    'models/pycf-0.9900-x-1000000001.keras': (40, 20, 100),
    'models/pycf-0.9900-x-1000000002.keras': (60, 20, 140),
    'models/pycf-0.9900-x-1000000003.keras': (50, 30, 120),
    'models/pycf-0.9900-x-1000000004.keras': (10, 20, 30),
}
evalout = os.path.join(tmp, 'eval.txt')
with open(evalout, 'w') as f:
    f.write('EVALUATE 4 on 4\n')
    for model, (a, b, c) in runs.items():
        f.write('\x1b[1m 1/57\x1b[0m 0s 16ms/step\r\x1b[1m57/57\x1b[0m 0s 6ms/step' + line('one', model, a, 0))
        f.write(line('two', model, 0, b))
        f.write(line('three', model, c / 2, c / 2))
        f.write(line('perfect', model, 0, 0))
# a training log: scores its temporary file, then says what it saved it as
trainout = os.path.join(tmp, 'train.txt')
with open(trainout, 'w') as f:
    for log, w in (('one', 12), ('two', 18), ('three', 30), ('perfect', 0)):
        f.write(line(log, '/tmp/train-abc.pycf.model.keras', w, 0))
    f.write('Saving as /somewhere/models/pycf-0.9950-x-1000000005.keras\n')

wrong, seconds = cm.read_outputs([evalout, trainout])
check('four evaluated models and the trained one are found', len(wrong) == 5, str(sorted(wrong)))
check('the training log\'s model takes its saved name', 'pycf-0.9950-x-1000000005.keras' in wrong)
check('lines after a progress bar are read',
      wrong['pycf-0.9900-x-1000000001.keras'].get('one') == (40.0, 0.0))
check('scored length comes back from the line', all(abs(seconds[l] - LENGTH[l]) < LENGTH[l] * 0.002
                                                    for l in ('one', 'two', 'three')), str(seconds))
check('a log nothing got wrong has no length', 'perfect' not in seconds)

def run(*argv):
    out = io.StringIO()
    old = sys.argv
    sys.argv = ['compare_models.py'] + list(argv)
    try:
        with contextlib.redirect_stdout(out):
            cm.main()
    finally:
        sys.argv = old
    return out.getvalue()

# arm = models 4 and 5 (60 s wrong each), baseline = 1..3 (160, 220, 200 s wrong)
total = 3600 + 1800 + 7200 + 3600 # "perfect" is given the median of the known lengths
out = run('-a', '1000000004,1000000005', '-b', '1000000001,1000000002,1000000003',
          '--seed-sd', '0.5', evalout, trainout)
want = 100 * (160 + 220 + 200) / 3 / total - 100 * 60 / total
check('difference is the gap between the group means', f'difference (arm - baseline): {want:+.3f} points' in out, out)
check('four logs used', out.lstrip().startswith('no model got these wrong') and '\n4 logs,' in out)
check('big gap against a small seed sd is BETTER',
      'verdict: BETTER' in run('-a', '1000000004,1000000005', '-b', '1000000001,1000000002,1000000003',
                               '--seed-sd', '0.05', evalout, trainout))
check('the same gap the other way round is WORSE',
      'verdict: WORSE' in run('-b', '1000000004,1000000005', '-a', '1000000001,1000000002,1000000003',
                              '--seed-sd', '0.05', evalout, trainout))
check('one seed never gets a verdict',
      'no verdict from one seed' in run('-a', '1000000004', '-b', '1000000001,1000000002,1000000003',
                                        '--seed-sd', '0.05', evalout, trainout))
check('exact run permutation: 1 split of 10 is as extreme', 'p = 0.100  (1 of 10 ways' in out, out)

# a log one model lacks is left out for everyone
short = os.path.join(tmp, 'short.txt')
with open(short, 'w') as f:
    for log, w in (('one', 12), ('three', 30)):
        f.write(line(log, 'models/pycf-0.9900-x-1000000006.keras', w, 0))
out = run('-a', '1000000006', '-b', '1000000001', evalout, short)
check('logs missing from one model are dropped and named', "left out: ['perfect', 'two']" in out and '\n2 logs,' in out, out)

# --logs-in keeps only logs that are in the directory; --exclude drops by name
testdir = os.path.join(tmp, 'test')
os.mkdir(testdir)
for name in ('one.json.gz', 'two.json.gz', 'two.data.npy', 'three.json'):
    open(os.path.join(testdir, name), 'w').close()
out = run('-a', '1000000004', '-b', '1000000001', '--logs-in', testdir, '--exclude', 'two', evalout)
check('--logs-in and --exclude choose the logs', '\n2 logs,' in out, out)

# thresholds
check('2 seeds: z 2 decides', cm.verdict(2.0, 2) == 'BETTER' and cm.verdict(-2.0, 2) == 'WORSE')
check('2 seeds: small z stops', cm.verdict(0.6, 2) == 'NO MEASURABLE EFFECT')
check('2 seeds: in between asks for a seed', cm.verdict(1.2, 2).startswith('undecided'))
check('3 seeds: wider stop band', cm.verdict(0.9, 3) == 'NO MEASURABLE EFFECT' and cm.verdict(1.0, 3).startswith('undecided'))
check('4 seeds: always a verdict', cm.verdict(1.8, 4) == 'BETTER' and cm.verdict(1.7, 4) == 'NO MEASURABLE EFFECT'
      and cm.verdict(-1.8, 5) == 'WORSE')

shutil.rmtree(tmp, ignore_errors=True)

if _fails:
    print(f'{len(_fails)} FAILED')
    sys.exit(1)
print('all passed')
