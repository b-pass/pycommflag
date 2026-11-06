#!/usr/bin/env python3
"""Add the cc_text column to feature logs made before file_version 11, from the video's packets.

Broadcast captions (ATSC A/53 "GA94" user data: MPEG-2 picture user data, or an H.264/HEVC
SEI message) ride in each video packet as plain bytes, so they can be read without decoding:
a few seconds per hour of video, where re-extracting the log takes ~60 CPU-minutes.  Each
frame gets the number of CEA-608 field-1 characters it carries, counted as processor._cc_text
counts them.

The video is the log's own `filename`, else the .mkv it was transcoded to.  From the log's own
video every packet's timestamp is a frame's (checked identical to an extraction's cc_text on
H.264 and MPEG-2, 29.97 and 59.94), and a packet with no frame (one the decoder dropped)
counts for nothing.  From a transcode, a packet counts for the frame shown at its time: 10s
totals then match an extraction's within ~1%, also through a 29.97 -> 59.94 transcode.

The log is rewritten in place (same compression and permissions) with cc_text appended to
frames_header and every row, and file_version 11.  Nothing else changes, so the neural caches
beside it stay valid.  Logs that already have cc_text are skipped.

    tools/add_cc_text.py [--dry-run] LOG...
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import processor

GA94 = re.compile(rb'GA94\x03')
NAL_CODECS = ('h264', 'hevc')   # caption bytes sit in a NAL unit, behind emulation prevention
MIN_MATCHED = 0.99              # share of the log's frames that must line up with a packet

def cc_chars(cc:bytes) -> int:
    """CEA-608 field-1 characters in a run of 3-byte cc_data constructs (processor._cc_text's rules)."""
    n = 0
    for i in range(0, len(cc) - 2, 3):
        if not (cc[i] & 0x04) or (cc[i] & 3) != 0:
            continue  # not cc_valid, or not 608 field 1
        c1, c2 = cc[i+1] & 0x7f, cc[i+2] & 0x7f
        if 0x10 <= c1 <= 0x1f:
            continue  # a control code (both bytes)
        n += (c1 >= 0x20) + (c2 >= 0x20)
    return n

def packet_cc(data:bytes, nal:bool) -> int:
    """Caption characters in one video packet."""
    if b'GA94' not in data:
        return 0
    if nal:
        data = data.replace(b'\x00\x00\x03', b'\x00\x00')
    n = 0
    for m in GA94.finditer(data):
        p = m.start()
        # what precedes it: the T.35 country/provider code (0xB5, 0x0031) in an SEI, or the
        # user_data start code in MPEG-2; anything else is a chance match in picture data
        if data[p-4 + nal:p] != (b'\xb5\x00\x31' if nal else b'\x00\x00\x01\xb2'):
            continue
        q = m.end()
        if q + 2 > len(data) or not (data[q] & 0x40):
            continue  # no process_cc_data_flag
        count = data[q] & 0x1f
        n += cc_chars(data[q+2:q+2+count*3])  # skip the flags byte and em_data
    return n

def read_packets(video:str) -> tuple[np.ndarray, np.ndarray]:
    """(times from the stream's start, caption characters) per packet of the first video stream, the one Player decodes."""
    import av
    times, counts = [], []
    with av.open(video) as c:
        vs = c.streams.video[0]
        nal = vs.codec_context.name in NAL_CODECS
        if not nal and vs.codec_context.name != 'mpeg2video':
            raise ValueError(f"don't know where {vs.codec_context.name} keeps captions")
        start, tb = vs.start_time or 0, vs.time_base
        for pkt in c.demux(vs):
            if pkt.pts is None or pkt.size == 0:
                continue
            times.append(float((pkt.pts - start) * tb))
            counts.append(packet_cc(bytes(pkt), nal))
    order = np.argsort(times, kind='stable')
    return np.asarray(times)[order], np.asarray(counts, dtype=np.int64)[order]

def assign(row_times:np.ndarray, times:np.ndarray, counts:np.ndarray, exact:bool, frame_rate:float) -> tuple[np.ndarray, float]:
    """Caption characters per row, and the share of rows that a packet lined up with.

    exact: packets count only for the row with their own timestamp (within 1ms; the log
    rounds times to 5 places).  Otherwise each counts for the row shown at its time, unless
    that row is more than 1.5 frames older (a gap in the log)."""
    out = np.zeros(len(row_times), dtype=np.int64)
    idx = np.searchsorted(row_times, times + 0.001, side='right') - 1
    ok = idx >= 0
    lag = times - row_times[np.maximum(idx, 0)]
    ok &= (np.abs(lag) <= 0.001) if exact else (lag <= 1.5 / frame_rate)
    np.add.at(out, idx[ok], counts[ok])
    hit = np.zeros(len(row_times), dtype=bool)
    hit[idx[ok]] = True
    return out, float(hit.mean()) if len(hit) else 0.0

def add_cc(path:str, dry_run:bool) -> str:
    flog = processor.read_feature_log(path)
    header = flog.get('frames_header', [])
    if 'cc_text' in header:
        return 'skip (already has cc_text)'
    if header[:1] != ['time']:
        return f'skip (unexpected frames_header {header})'
    fn = flog.get('filename') or ''
    video = fn if os.path.exists(fn) else fn[:-3] + '.mkv' if fn.endswith('.ts') and os.path.isfile(fn[:-3] + '.mkv') else None
    if video is None:
        return f'skip (no video: {fn})'
    exact = video == fn

    rows = flog['frames']
    body = rows[1:] if rows and rows[0] is None else rows
    row_times = np.array([r[0] for r in body], dtype='float64')
    times, counts = read_packets(video)
    cc, matched = assign(row_times, times, counts, exact, flog.get('frame_rate', 29.97))
    how = 'own video' if exact else os.path.basename(video)
    msg = f'{cc.sum()} chars on {np.count_nonzero(cc)} of {len(cc)} frames, {matched:.1%} matched ({how})'
    if matched < MIN_MATCHED:
        return f'skip (only {msg}; not this log\'s video?)'
    if dry_run:
        return msg

    for r, n in zip(body, cc.tolist()):
        r.append(n)
    flog['frames_header'] = header + ['cc_text']
    flog['file_version'] = 11

    mode = os.stat(path).st_mode & 0o777
    d, base = os.path.split(path)
    tmp = os.path.join(d, '.adding-cc-' + base)  # same suffix, so .gz still means gzip
    processor.write_feature_log(flog, tmp)
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    return msg

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dry-run', action='store_true', help='report what would be added, change nothing')
    ap.add_argument('logs', nargs='+')
    args = ap.parse_args()
    failed = 0
    for path in args.logs:
        try:
            print(f'{path}: {add_cc(path, args.dry_run)}', flush=True)
        except Exception as e:
            failed += 1
            print(f'{path}: FAILED {e!r}', flush=True)
    return 1 if failed else 0

if __name__ == '__main__':
    sys.exit(main())
