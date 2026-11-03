#!/usr/bin/env python3
"""Move feature logs back out of LOGDIR/old/ when their video still exists.

-r archives a log into old/ when its video is gone.  Before it learned that a deleted .ts may
have been transcoded to a .mkv of the same name, it archived the logs of every transcoded
recording too.  This finds those (the video the log names, or its .mkv, exists) and moves them
back beside the others, unless a log of the same name is already there.

    tools/restore_archived_logs.py /path/to/logs           # dry run: what would move
    tools/restore_archived_logs.py /path/to/logs --apply
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag.main import _find_video
from pycommflag.picker import read_header

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logdir')
    ap.add_argument('--apply', action='store_true', help='move them (default: only report)')
    args = ap.parse_args()

    old = os.path.join(args.logdir, 'old')
    (moved, gone, clash, bad) = (0, 0, 0, 0)
    for name in sorted(os.listdir(old)):
        if not name.startswith('cf_') or '.json' not in name:
            continue
        src = os.path.join(old, name)
        try:
            vf = read_header(src).get('filename') or ''
        except Exception as e:
            print(f'{name}: unreadable ({e})')
            bad += 1
            continue
        found = _find_video(vf)
        if not found or not os.path.exists(found):
            gone += 1
            continue
        dst = os.path.join(args.logdir, name)
        if os.path.exists(dst):
            print(f'{name}: {found} exists, but so does {dst}; left in old/')
            clash += 1
            continue
        print(f'{name}: {found} exists' + (', moved back' if args.apply else ''))
        if args.apply:
            os.rename(src, dst)
        moved += 1
    print(f'{"moved" if args.apply else "would move"} {moved}; video gone {gone}; '
          f'name already taken {clash}; unreadable {bad}')

if __name__ == '__main__':
    main()
