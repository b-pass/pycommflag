#!/usr/bin/env python3
"""Checks how -g finds a recording's feature log, the "edited" marker, and -r's archiving guard.

-g takes a log, a video or a link to one (mythlink's), or a <chanid>_<starttime>, and finds the
log as cf_<video name>.json[.gz] in the log dirs and their old/, falling back between .ts and
.mkv names (recordings are transcoded after flagging).  Saving from the editor writes "edited"
near the top of the log, where the picker reads it without parsing the rest, and -r skips such
logs.  -r must not archive a log just because the whole recordings directory is missing.

Run it directly; it exits non-zero if anything fails.
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import main, picker, processor

_fails = []

def check(name, ok, detail=''):
    print(('  pass  ' if ok else '  FAIL  ') + name + (f'   {detail}' if detail else ''))
    if not ok:
        _fails.append(name)

def write_log(path, video, **extra):
    flog = {'file_version': 11, 'chanid': '1234', 'starttime': '2026-01-01 00:00:00', 'filename': video,
            'duration': 10.0, 'frame_rate': 29.97, 'logo': None, 'frames_header': ['time'],
            'frames': [[i/30] for i in range(300)], 'tags': []}
    flog.update(extra)
    processor.write_feature_log(flog, path)
    return flog

tmp = tempfile.mkdtemp(prefix='pycf-lookup-')
logs = os.path.join(tmp, 'logs')
recs = os.path.join(tmp, 'recordings')
links = os.path.join(tmp, 'links')
for d in (logs, os.path.join(logs, 'old'), recs, links):
    os.makedirs(d)

# a recording transcoded to .mkv whose log was archived, and one still a .ts with a gzipped log
open(os.path.join(recs, '1234_20260101000000.mkv'), 'w').close()
write_log(os.path.join(logs, 'old', 'cf_1234_20260101000000.ts.json'), os.path.join(recs, '1234_20260101000000.ts'))
open(os.path.join(recs, '5678_20260102000000.ts'), 'w').close()
write_log(os.path.join(logs, 'cf_5678_20260102000000.ts.json.gz'), os.path.join(recs, '5678_20260102000000.ts'))
os.symlink(os.path.join(recs, '1234_20260101000000.mkv'), os.path.join(links, 'Show - 20260101 - Pilot.mkv'))
os.symlink(os.path.join(recs, '5678_20260102000000.ts'), os.path.join(links, 'Show - 20260102 - Two.ts'))
os.symlink(os.path.join(recs, '9999_20260103000000.ts'), os.path.join(links, 'Show - 20260103 - Gone.ts'))

dirs = picker.log_dirs(SimpleNamespace(log_dir=[logs]))
check('log dirs include old/', dirs == [logs, os.path.join(logs, 'old')], str(dirs))

(log, video, search) = picker.resolve_gui_arg(os.path.join(links, 'Show - 20260101 - Pilot.mkv'), dirs)
check('a link to a .mkv finds the .ts log in old/', log == os.path.join(logs, 'old', 'cf_1234_20260101000000.ts.json'), str(log))
check('... and the video it links to', video == os.path.join(recs, '1234_20260101000000.mkv'), str(video))
(log, video, search) = picker.resolve_gui_arg(os.path.join(links, 'Show - 20260102 - Two.ts'), dirs)
check('a link to a .ts finds its .json.gz', log == os.path.join(logs, 'cf_5678_20260102000000.ts.json.gz'), str(log))
(log, _, _) = picker.resolve_gui_arg('5678_20260102000000', dirs)
check('<chanid>_<starttime> finds its log', log == os.path.join(logs, 'cf_5678_20260102000000.ts.json.gz'), str(log))
check('search text is search text', picker.resolve_gui_arg('some title', dirs) == (None, None, 'some title'))
check('no argument opens the picker', picker.resolve_gui_arg('', dirs) == (None, None, ''))
for (name, arg) in (('a broken link without a log', os.path.join(links, 'Show - 20260103 - Gone.ts')),
                    ('a missing path', os.path.join(tmp, 'nope.ts')),
                    ('an unknown recording', '1111_20260101000000')):
    try:
        picker.resolve_gui_arg(arg, dirs)
        check(name + ' is an error', False)
    except LookupError:
        check(name + ' is an error', True)

idx = picker.log_index(dirs)
check('log index', sorted(idx) == ['1234_20260101000000', '5678_20260102000000'], str(sorted(idx)))

# the editor's save: "edited" goes right after "filename", within read_header's reach
path = os.path.join(logs, 'cf_5678_20260102000000.ts.json.gz')
flog = processor.read_feature_log(path)
check('not edited yet', picker.read_header(path)['edited'] is None)
flog = processor.set_header_key(flog, 'edited', '2026-11-02T10:00:00')
check('edited follows filename', list(flog)[:5] == ['file_version', 'chanid', 'starttime', 'filename', 'edited'], str(list(flog)[:5]))
processor.write_feature_log(flog, path)
h = picker.read_header(path, size=200)
check('read_header sees it', h == {'filename': os.path.join(recs, '5678_20260102000000.ts'), 'edited': '2026-11-02T10:00:00'}, str(h))
check('the log still reads whole', processor.read_feature_log(path)['edited'] == '2026-11-02T10:00:00')

# -r leaves an edited log alone, and won't archive when the recordings directory is missing
opts = SimpleNamespace(chanid=None, starttime=None)
out = io.StringIO()
with contextlib.redirect_stdout(out):
    archived = main._reprocess_one(opts, path, allow_archive=True)
check('-r skips an edited log', not archived and 'hand-edited' in out.getvalue(), out.getvalue().strip())

lost = os.path.join(logs, 'cf_7777_20260104000000.ts.json')
write_log(lost, os.path.join(tmp, 'unmounted', '7777_20260104000000.ts'))
out = io.StringIO()
with contextlib.redirect_stdout(out):
    archived = main._reprocess_one(opts, lost, allow_archive=True)
check('-r keeps a log whose recordings dir is missing', not archived and os.path.exists(lost), out.getvalue().strip())

gone = os.path.join(logs, 'cf_8888_20260105000000.ts.json')
write_log(gone, os.path.join(recs, '8888_20260105000000.ts'))
with contextlib.redirect_stdout(io.StringIO()):
    archived = main._reprocess_one(opts, gone, allow_archive=True)
check('-r archives a log whose video is gone', archived and os.path.exists(os.path.join(logs, 'old', os.path.basename(gone))))

shutil.rmtree(tmp, ignore_errors=True)

if _fails:
    print(f'{len(_fails)} FAILED')
    sys.exit(1)
print('all passed')
