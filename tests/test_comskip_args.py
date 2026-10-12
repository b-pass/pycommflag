#!/usr/bin/env python3
"""Checks for pycommflag-comskip: comskip command-line parsing, and the fallback to the
real comskip when flagging fails.  Needs no video and no model.

Run it directly; it exits non-zero if anything fails.
"""
import os
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from pycommflag.comskip import parse_comskip_args

_fails = []

def check(name, got, want):
    ok = got == want
    print(f'{"ok  " if ok else "FAIL"} {name}' + ('' if ok else f': got {got!r}, want {want!r}'))
    if not ok:
        _fails.append(name)

check('channels-style',
      parse_comskip_args(['--ini=/dvr/Logs/comskip/12/comskip.ini', '--output=/dvr/Logs/comskip/12',
                          '--output-filename=video', '/dvr/TV/Show/Show 2024-01-01-2000.mpg']),
      ('/dvr/TV/Show/Show 2024-01-01-2000.mpg', '/dvr/Logs/comskip/12', 'video'))
check('separate values',
      parse_comskip_args(['--ini', 'x.ini', '--output', '/out', '--threads', '4', '-v', '10', '/r/a.ts']),
      ('/r/a.ts', '/out', 'a'))
check('positional outdir beats --output',
      parse_comskip_args(['--output=/work', '/r/a.ts', '/cut']),
      ('/r/a.ts', '/cut', 'a'))
check('unknown flags ignored',
      parse_comskip_args(['--hwassist', '-t', '--zpcut', '-q', '/r/b.mkv']),
      ('/r/b.mkv', '/r', 'b'))
check('defaults to next to the input',
      parse_comskip_args(['/r/show.name.ts']),
      ('/r/show.name.ts', '/r', 'show.name'))

# fallback: flagging a missing file fails fast (before any model is loaded), so the
# stub "real comskip" must be run with the original arguments
with tempfile.TemporaryDirectory(prefix='pycf_test_') as d:
    stub = os.path.join(d, 'comskip.real')
    record = os.path.join(d, 'args')
    with open(stub, 'w') as f:
        f.write(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{record}"\nexit 1\n')
    os.chmod(stub, 0o755)
    args = ['--ini=/dev/null', f'--output={d}', '--output-filename=video', os.path.join(d, 'missing.mpg')]
    env = dict(os.environ, PYCOMMFLAG_COMSKIP_REAL=stub)
    env.pop('PYCOMMFLAG_LOG_DIR', None)
    p = subprocess.run([sys.executable, '-m', 'pycommflag.comskip'] + args, cwd=_ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    check('fallback exit status', p.returncode, 1)
    got = open(record).read().splitlines() if os.path.exists(record) else None
    check('fallback args', got, args)
    check('no edl left behind', os.path.exists(os.path.join(d, 'video.edl')), False)

    # and with no real comskip to fall back on, it fails with 2
    env['PYCOMMFLAG_COMSKIP_REAL'] = os.path.join(d, 'nonexistent')
    p = subprocess.run([sys.executable, '-m', 'pycommflag.comskip'] + args, cwd=_ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    check('no fallback exit status', p.returncode, 2)

if _fails:
    print(f'\n{len(_fails)} failed')
    sys.exit(1)
print('\nall passed')
