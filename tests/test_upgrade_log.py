#!/usr/bin/env python3
"""Checks tools/upgrade_log.py and pycommflag/timeline.py: the .mkv lag, captions from raw video
packets, the audio probe through an .mkv, and the log rewrite.

Captions are found in MPEG-2 user data and H.264/HEVC SEI (behind emulation prevention bytes)
and kept as processor.cc_keep keeps them (all but padding and invalid constructs); a stray "GA94"
in picture data doesn't count.  From the log's own video a packet counts only for the frame with
its timestamp; from a transcode, for the frame shown at its time once the transcode's lag is
added back.  The lag comes from the stream start times when the log has them, else from
lining up cuts, and a vote without enough agreeing cuts gives no answer.  A probe of the .mkv is
used for the audio only less that lag.  The rewrite appends cc_text and cc_data, sets
file_version 12, keeps everything else, adds cc_data to a log that only has cc_text without
touching its count, and leaves a log that has both alone unless asked to redo them.  Also
the editor's log_fixes, which tells the user what a log needs and how to run the tool.

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
import upgrade_log
from pycommflag import gui, processor, timeline

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

def cc(*triples):
    return b''.join(bytes(t) for t in triples)

mpeg2 = (b'\x00\x00\x01\x00' + b'\x11' * 40 + b'\x00\x00\x01\xb2' + cc_payload([HI, F2, HI])
         + b'\x00\x00\x01\x01' + b'\x22' * 40 + b'GA94\x03\x43\xff' + bytes(HI) * 3)  # stray, in slice data
check('MPEG-2 user data', upgrade_log.packet_cc(mpeg2, False) == cc(HI, F2, HI), upgrade_log.packet_cc(mpeg2, False))

sei = b'\x06\x04\x20' + b'\xb5\x00\x31' + cc_payload([ZEROS, HI, CTRL, HI, PAD]) + b'\xff\x80'
h264 = b'\x00\x00\x00\x01\x09\xf0' + b'\x00\x00\x00\x01' + epb(sei) + b'\x00\x00\x00\x01\x65' + b'\x33' * 40
check('H.264 SEI survives emulation prevention', b'\x00\x00\x03\x01' in h264 and upgrade_log.packet_cc(h264, True) == cc(HI, CTRL, HI),
      upgrade_log.packet_cc(h264, True))
check('a T.35 prefix is required in a NAL', upgrade_log.packet_cc(h264.replace(b'\xb5\x00\x31', b'\xb5\x00\x32'), True) == b'')
check('no captions, no bytes', upgrade_log.packet_cc(b'\x00\x00\x01\xb2' + b'\x00' * 20, False) == b'')

rows = np.array([0.0, 1/30, 2/30, 3/30, 5/30])  # one frame missing from the log
times = np.array([-1/30, 0.0, 1/30, 2/30, 3/30, 4/30, 5/30])
data = [bytes([n]) for n in (7, 1, 2, 3, 4, 5, 6)]
got, matched = upgrade_log.assign(np.round(rows, 5), times, data, True, 29.97)
check('own video: only same-timestamp packets count', got == [b'\x01', b'\x02', b'\x03', b'\x04', b'\x06'] and matched == 1.0, (got, matched))
t60 = np.arange(-1, 12) / 60  # a 59.94 transcode of the same frames
got, matched = upgrade_log.assign(rows, t60, [bytes([n]) for n in range(len(t60))], False, 29.97)
check('transcode: packets count for the frame shown then, in order',
      got == [b'\x01\x02', b'\x03\x04', b'\x05\x06', b'\x07\x08\x09\x0a', b'\x0b\x0c'] and matched == 1.0, (got, matched))

# --- the lag ---

rng = np.random.default_rng(1)
step = 1 / 29.97
log_cuts = [np.sort(rng.uniform(100, 140, 12)), np.sort(rng.uniform(900, 940, 10))]
# the copy's clock is 0.734s behind; its cuts land a frame early or late now and then, and it
# finds cuts the log didn't (and misses some)
video_cuts = [np.sort(np.concatenate((lc[1:] - 0.734 + rng.choice([-step, 0, 0, 0, step], len(lc) - 1),
                                      rng.uniform(lc[0] - 3, lc[-1], 6)))) for lc in log_cuts]
off = timeline.match_cuts(log_cuts, video_cuts, 29.97)
check('cuts: the lag that lines them up', off is not None and abs(off - 0.734) < step / 2, off)
noise = [np.sort(rng.uniform(lc[0] - 3, lc[-1] + 3, len(lc))) for lc in log_cuts]
check('cuts: no answer from cuts that don\'t line up', timeline.match_cuts(log_cuts, noise, 29.97) is None)
check('cuts: no answer from one window', timeline.match_cuts(log_cuts[:1] + [np.zeros(0)], video_cuts[:1] + [np.zeros(0)], 29.97) is None)

real_starts = timeline.stream_starts
timeline.stream_starts = lambda v: (1.502, 0.0)
fl = {'filename': '/nowhere/1_2.ts', 'video_start': 2362.2014, 'audio_start': 2361.755733}
check('starts: the copy\'s video moved away from its audio', timeline.lag_from_starts(fl, '/nowhere/1_2.mkv') == round(1.502 - 0.445667, 4))
check('own video: no lag', timeline.video_lag(fl, '/nowhere/1_2.ts') == (0.0, 'own'))
check('a recorded lag wins', timeline.video_lag(dict(fl, mkv_lag=0.7, mkv_lag_from='cuts'), '/nowhere/1_2.mkv') == (0.7, 'cuts'))
check('no starts, not measured: unknown', timeline.video_lag({'filename': '/nowhere/1_2.ts'}, '/nowhere/1_2.mkv', measure=False) == (None, 'unknown'))
timeline.stream_starts = real_starts
check('tags move back by the lag, none before 0',
      timeline.shift_tags([(3, (0.0, 7.5)), (3, (100.0, 160.0))], 0.734) == [(3, (0.0, 6.766)), (3, (99.266, 159.266))])

# --- the audio probe through an .mkv ---

real_probe = upgrade_log.probe
upgrade_log.probe = lambda f: 1.034
quiet = np.zeros((300, 10))
quiet[:, 0] = np.arange(300) / 29.97  # no blanks, so no estimate
mkvlog = {'filename': '/nowhere/1_2.ts', 'frame_rate': 29.97}
shift, source, note = upgrade_log.choose_shift(mkvlog, quiet, '/nowhere/1_2.mkv', 0.734)
check('probe-mkv: less the lag', (round(shift, 4), source) == (0.3, 'probe-mkv'), (shift, source, note))
shift, source, note = upgrade_log.choose_shift(mkvlog, quiet, '/nowhere/1_2.mkv', None)
check('probe-mkv: not used without the lag', source != 'probe-mkv', (shift, source, note))
upgrade_log.probe = real_probe

# --- the log ---

tmp = tempfile.mkdtemp(prefix='pycf-upgrade-')
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
check('log_fixes asks for captions only', len(problems) == 1 and 'cc_text' in problems[0], problems)
check('log_fixes runs the tool with this python', argv == [sys.executable, os.path.join(ROOT, 'tools', 'upgrade_log.py'), path], argv)
noaudio = dict(flog)
del noaudio['audio_shift']
problems, cmd = gui.log_fixes(noaudio, path)
check('log_fixes names both, audio first', len(problems) == 2 and 'audio_shift' in problems[0] and '&&' not in cmd, (problems, cmd))

T = [round(i/29.97, 5) for i in range(100)]
upgrade_log.read_packets = lambda v: (np.array(T), [cc(HI) * (i % 3) for i in range(100)])
msgs = upgrade_log.upgrade(path)
new = processor.read_feature_log(path)
body = new['frames'][1:]
check('rewrite appends cc_text', new['frames_header'][-1] == 'cc_text' and [r[-1] for r in body] == [2 * (i % 3) for i in range(100)], msgs)
check('rewrite adds cc_data after the frames, by frame time, frames without captions left out',
      new['cc_data'] == [[T[i], (cc(HI) * (i % 3)).hex()] for i in range(100) if i % 3] and list(new)[-1] == 'cc_data', msgs)
check('rewrite sets file_version 12 and keeps the rest',
      new['file_version'] == 12 and [r[:-1] for r in body] == flog['frames'][1:] and new['tags'] == flog['tags']
      and [k for k in new if k not in ('cc_text_from', 'cc_data_from', 'cc_data')] == list(flog) and new['audio_shift'] == 0.5)
check('own video: both from packets, no mkv_lag',
      (new.get('cc_text_from'), new.get('cc_data_from')) == ('packets', 'packets') and 'mkv_lag' not in new)
check('rewrite keeps the permissions', os.stat(path).st_mode & 0o777 == 0o640)
check('a log with cc_text and cc_data is skipped', any(m.startswith('cc: skip') for m in upgrade_log.upgrade(path)))
check('log_fixes is satisfied', gui.log_fixes(new, path)[1] is None)

# a file_version 11 log: counted by its extraction, no bytes
v11 = dict(new, file_version=11)
for k in ('cc_data', 'cc_data_from', 'cc_text_from'):
    del v11[k]
problems, cmd = gui.log_fixes(v11, path)
check('log_fixes asks for the upgrade again on a log with counts only',
      cmd is not None and len(problems) == 1 and 'cc_data' in problems[0] and 'again' in problems[0], problems)
processor.write_feature_log(v11, path)
upgrade_log.read_packets = lambda v: (np.array(T), [cc(HI, F2)] * 100)
msgs = upgrade_log.upgrade(path)
new = processor.read_feature_log(path)
check('cc_data added to a log with counts, its counts untouched',
      [r[-1] for r in new['frames'][1:]] == [2 * (i % 3) for i in range(100)] and 'cc_text_from' not in new
      and new['cc_data'] == [[t, cc(HI, F2).hex()] for t in T] and new['file_version'] == 12
      and new.get('cc_data_from') == 'packets', msgs)

# the same recording, transcoded: the .ts is gone, and the .mkv's clock is 0.5s behind
os.remove(video)
open(video[:-3] + '.mkv', 'wb').close()
timeline.stream_starts = lambda v: (0.9, 0.0)
old = processor.read_feature_log(path)
old.update(video_start=10.0, audio_start=9.6)
processor.write_feature_log(old, path)
upgrade_log.read_packets = lambda v: (np.array([t - 0.5 for t in T]), [cc(HI, CTRL)] * 100)
old = processor.read_feature_log(path)
lost = upgrade_log.read_packets
upgrade_log.read_packets = lambda v: (np.array([t - 0.5 for t in T]), [b''] * 100)
msgs = upgrade_log.upgrade(path, redo_cc=True)
check('mkv: a transcode without the captions the log counted is skipped',
      any(m.startswith('cc: skip') and 'lost captions' in m for m in msgs)
      and processor.read_feature_log(path)['cc_data'] == old['cc_data'], msgs)
upgrade_log.read_packets = lost
msgs = upgrade_log.upgrade(path, redo_cc=True)
new = processor.read_feature_log(path)
body = new['frames'][1:]
timeline.stream_starts = real_starts
check('mkv: the lag is recorded, from the starts', (new.get('mkv_lag'), new.get('mkv_lag_from')) == (0.5, 'starts'), msgs)
check('mkv: --redo-cc replaces the column, counted at the log\'s times',
      new['frames_header'].count('cc_text') == 1 and len(body[0]) == 11 and all(r[-1] == 2 for r in body)
      and new.get('cc_text_from') == 'packets-mkv', msgs)
check('mkv: --redo-cc replaces cc_data, at the log\'s times',
      new['cc_data'] == [[t, cc(HI, CTRL).hex()] for t in T] and new.get('cc_data_from') == 'packets-mkv'
      and list(new)[-1] == 'cc_data', msgs)
check('mkv: the timing keys sit before the frames', list(new).index('mkv_lag') < list(new).index('frames'))

shutil.rmtree(tmp, ignore_errors=True)

if _fails:
    print(f'{len(_fails)} FAILED')
    sys.exit(1)
print('all passed')
