"""Push breaks to a Channels DVR server (-o channelsdvr).

This uses the API behind Channels' own "edit commercials" dialog, which is not
officially documented: GET /dvr/files lists the recordings (with ID and Path), and
POST /dvr/files/<ID>/comskip/edit replaces one recording's markers.
"""
import json
import logging as log
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from .feature_span import SceneType

TIMEOUT = 30

def _parts(path:str) -> list[str]:
    return [p for p in re.split(r'[\\/]+', path) if p]

def find_file_id(base_url:str, video:str) -> None|str:
    """The ID of the recording whose Path best matches video, or None.

    Channels may store paths relative to its recordings directory, and pycommflag may see
    the recordings through a different mount, so this matches the most trailing path
    components (at least the filename) rather than the whole path.
    """
    with urllib.request.urlopen(base_url.rstrip('/') + '/dvr/files', timeout=TIMEOUT) as r:
        files = json.load(r)
    if isinstance(files, dict):
        files = list(files.values())

    want = _parts(os.path.realpath(video))
    best, best_ids = 0, []
    for f in files:
        if not isinstance(f, dict) or not f.get('ID') or not f.get('Path'):
            continue
        have = _parts(f['Path'])
        n = 0
        while n < min(len(have), len(want)) and have[-1-n] == want[-1-n]:
            n += 1
        if n > best:
            best, best_ids = n, [f['ID']]
        elif n == best and n > 0:
            best_ids.append(f['ID'])

    if len(best_ids) != 1:
        if best_ids:
            log.error(f'Channels DVR has {len(best_ids)} recordings matching {video}: {best_ids}')
        return None
    return str(best_ids[0])

def set_breaks(opts, result:list, video:str) -> bool:
    base = opts.channels_url
    if not video:
        log.error('No video file to look up in Channels DVR')
        return False

    marks = []
    for (t,(s,e)) in result:
        if t in [SceneType.COMMERCIAL, SceneType.COMMERCIAL.value]:
            marks += [round(s, 3), round(e, 3)]

    try:
        fid = find_file_id(base, video)
        if fid is None:
            log.error(f'Recording {video} was not found in Channels DVR at {base}')
            return False

        url = f"{base.rstrip('/')}/dvr/files/{urllib.parse.quote(fid)}/comskip/edit?source=local"
        # posted like `curl --data-raw`, which is how this API is known to be used
        with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(marks).encode(), method='POST'),
                                    timeout=TIMEOUT):
            pass
    except (urllib.error.URLError, OSError, ValueError) as e:
        log.error(f'Channels DVR at {base}: {e}')
        return False

    log.info(f'Set {len(marks)//2} breaks on Channels DVR file {fid}')
    return True
