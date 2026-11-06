#!/usr/bin/env python3
"""Checks tools/add_cc_text.py: caption counts from raw video packets, and the log rewrite.

Captions are found in MPEG-2 user data and H.264/HEVC SEI (behind emulation prevention bytes)
and counted by processor._cc_text's rules (field 1 only, no control codes); a stray "GA94" in
picture data doesn't count.  From the log's own video a packet counts only for the frame with
its timestamp; from a transcode, for the frame shown at its time.  The rewrite appends cc_text,
sets file_version 11, keeps everything else, and leaves a log that has cc_text alone.  Also
the editor's log_fixes, which tells the user which tools a log needs and how to run them.

Run it directly; it exits non-zero if anything fails.  Needs no video.
"""
import os
import shlex
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import add_cc_text
from pycommflag import gui, processor

_fails = []

def check(name, ok, detail=''):
    print(('  pass  ' if ok else '  FAIL  ') + name + (f'   {detail}' if detail else ''))
    if not ok:
        _fails.append(name)

def odd(c):
    """A 7-bit character with its odd-parity bit, as broadcast."""
    return c | (0x80 if bin(c).count('1') % 2 == 0 else 0)

def cc_payload(triples):
    return b'GA94\x03' + bytes([0x40 | len(triples), 0xff]) + b''.join(bytes(t) for t in triples)

def epb(rbsp):
    """NAL emulation prevention: 0x03 after any 00 00 that is followed by 00-03."""
    out, zeros = bytearray(), 0
    for b in rbsp:
        if zeros >= 2 and b <= 3:
            out.append(3)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)

HI = (0xfc, odd(ord('H')), odd(ord('i')))      # field 1, two characters
CTRL = (0xfc, odd(0x14), odd(0x2c))            # field 1 control code (erase displayed memory)
F2 = (0xfd, odd(ord('X')), odd(ord('Y')))      # field 2: not counted
INVALID = (0xf8, odd(ord('Z')), odd(ord('Z'))) # cc_valid clear
PAD = (0xfc, 0x80, 0x80)                       # null padding
ZEROS = (0x00, 0x00, 0x01)                     # not valid, and makes an emulation prevention byte

check('cc_chars counts field-1 characters only', add_cc_text.cc_chars(b''.join(bytes(t) for t in (HI, CTRL, F2, INVALID, PAD))) == 2)

mpeg2 = (b'\x00\x00\x01\x00' + b'\x11' * 40 + b'\x00\x00\x01\xb2' + cc_payload([HI, F2, HI])
         + b'\x00\x00\x01\x01' + b'\x22' * 40 + b'GA94\x03\x43\xff' + bytes(HI) * 3)  # stray, in slice data
check('MPEG-2 user data', add_cc_text.packet_cc(mpeg2, False) == 4, add_cc_text.packet_cc(mpeg2, False))

sei = b'\x06\x04\x20' + b'\xb5\x00\x31' + cc_payload([ZEROS, HI, CTRL, HI, PAD]) + b'\xff\x80'
h264 = b'\x00\x00\x00\x01\x09\xf0' + b'\x00\x00\x00\x01' + epb(sei) + b'\x00\x00\x00\x01\x65' + b'\x33' * 40
check('H.264 SEI survives emulation prevention', b'\x00\x00\x03\x01' in h264 and add_cc_text.packet_cc(h264, True) == 4,
      add_cc_text.packet_cc(h264, True))
check('a T.35 prefix is required in a NAL', add_cc_text.packet_cc(h264.replace(b'\xb5\x00\x31', b'\xb5\x00\x32'), True) == 0)
check('no captions, no count', add_cc_text.packet_cc(b'\x00\x00\x01\xb2' + b'\x00' * 20, False) == 0)

rows = np.array([0.0, 1/30, 2/30, 3/30, 5/30])  # one frame missing from the log
times = np.array([-1/30, 0.0, 1/30, 2/30, 3/30, 4/30, 5/30])
counts = np.array([7, 1, 2, 3, 4, 5, 6])
cc, matched = add_cc_text.assign(np.round(rows, 5), times, counts, True, 29.97)
check('own video: only same-timestamp packets count', cc.tolist() == [1, 2, 3, 4, 6] and matched == 1.0, (cc.tolist(), matched))
t60 = np.arange(-1, 12) / 60  # a 59.94 transcode of the same frames
cc, matched = add_cc_text.assign(rows, t60, np.ones(len(t60), dtype=np.int64), False, 29.97)
check('transcode: packets count for the frame shown then', cc.tolist() == [2, 2, 2, 4, 2] and matched == 1.0, (cc.tolist(), matched))

tmp = tempfile.mkdtemp(prefix='pycf-addcc-')
video = os.path.join(tmp, 'rec dir', '1234_20260101000000.ts')
os.makedirs(os.path.dirname(video))
open(video, 'wb').close()
path = os.path.join(tmp, 'rec dir', 'cf_1234_20260101000000.ts.json.gz')  # a space, for the quoting
flog = {'file_version': 10, 'filename': video, 'frame_rate': 29.97, 'audio_shift': 0.5, 'logo': None,
        'frames_header': ['time', 'logo_present', 'is_blank', 'diff', 'fvol', 'rvol', 'silence', 'speech', 'music', 'noise'],
        'frames': [None] + [[round(i/29.97, 5), 1, 0, 0.5, 0.1, 0.0, 0, 1, 0, 0] for i in range(100)],
        'tags': [[3, [1.0, 2.0]]]}
processor.write_feature_log(flog, path)
os.chmod(path, 0o640)

problems, cmd = gui.log_fixes(processor.read_feature_log(path), path)
argv = shlex.split(cmd or '')
check('log_fixes asks for captions only', len(problems) == 1 and 'cc_text' in problems[0] and '&&' not in argv, cmd)
check('log_fixes runs the tool with this python', argv[:3] == [sys.executable, os.path.join(ROOT, 'tools', 'add_cc_text.py'), path], argv)
noaudio = dict(flog)
del noaudio['audio_shift']
problems, cmd = gui.log_fixes(noaudio, path)
check('log_fixes asks for both, audio first', len(problems) == 2 and 'repair_audio_shift.py' in cmd.split('&&')[0] and 'add_cc_text.py' in cmd.split('&&')[1], cmd)

add_cc_text.read_packets = lambda v: (np.array([round(i/29.97, 5) for i in range(100)]), np.arange(100) % 3)
msg = add_cc_text.add_cc(path, False)
new = processor.read_feature_log(path)
body = new['frames'][1:]
check('rewrite appends cc_text', new['frames_header'][-1] == 'cc_text' and [r[-1] for r in body] == (np.arange(100) % 3).tolist(), msg)
check('rewrite sets file_version 11 and keeps the rest',
      new['file_version'] == 11 and [r[:-1] for r in body] == flog['frames'][1:] and new['tags'] == flog['tags']
      and list(new) == list(flog) and new['audio_shift'] == 0.5)
check('rewrite keeps the permissions', os.stat(path).st_mode & 0o777 == 0o640)
check('a log with cc_text is skipped', add_cc_text.add_cc(path, False).startswith('skip'))
check('log_fixes is satisfied', gui.log_fixes(new, path)[1] is None)

shutil.rmtree(tmp, ignore_errors=True)

if _fails:
    print(f'{len(_fails)} FAILED')
    sys.exit(1)
print('all passed')
