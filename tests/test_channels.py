#!/usr/bin/env python3
"""Checks for -o channelsdvr against a fake Channels DVR server.  Needs no video and no model.

Run it directly; it exits non-zero if anything fails.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import channels
from pycommflag.feature_span import SceneType

FILES = [
    {'ID': '101', 'Path': 'TV/Show A/Show A S01E01 2024-01-01-2000.mpg'},
    {'ID': '102', 'Path': 'TV/Show B/Show B S01E01 2024-01-02-2000.mpg'},
    {'ID': '103', 'Path': '/shares/dvr/TV/Show C/Show C 2024-01-03-2000.mpg'},
    {'ID': '104', 'Path': 'TV/Dup One/same.mpg'},
    {'ID': '105', 'Path': 'TV/Dup Two/same.mpg'},
]
posts = []

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _reply(self, code, body=b''):
        self.send_response(code)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/dvr/files':
            self._reply(200, json.dumps(FILES).encode())
        else:
            self._reply(404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        posts.append((self.path, json.loads(body)))
        self._reply(200)

server = HTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f'http://127.0.0.1:{server.server_port}'

_fails = []

def check(name, got, want):
    ok = got == want
    print(f'{"ok  " if ok else "FAIL"} {name}' + ('' if ok else f': got {got!r}, want {want!r}'))
    if not ok:
        _fails.append(name)

check('relative path', channels.find_file_id(url, '/dvr/TV/Show A/Show A S01E01 2024-01-01-2000.mpg'), '101')
check('absolute path, other mount', channels.find_file_id(url, '/mnt/dvr/TV/Show C/Show C 2024-01-03-2000.mpg'), '103')
check('not found', channels.find_file_id(url, '/dvr/TV/Show A/other.mpg'), None)
check('ambiguous', channels.find_file_id(url, '/elsewhere/same.mpg'), None)
check('ambiguity broken by directory', channels.find_file_id(url, '/x/TV/Dup Two/same.mpg'), '105')

opts = SimpleNamespace(channels_url=url + '/')
result = [(SceneType.COMMERCIAL.value, (10.0, 70.12345)), (SceneType.INTRO.value, (80.0, 90.0)),
          (SceneType.COMMERCIAL, (600.5, 780.0))]
check('set_breaks', channels.set_breaks(opts, result, '/dvr/TV/Show B/Show B S01E01 2024-01-02-2000.mpg'), True)
check('posted', posts, [('/dvr/files/102/comskip/edit?source=local', [10.0, 70.123, 600.5, 780.0])])
check('set_breaks no breaks', channels.set_breaks(opts, [], '/dvr/TV/Show B/Show B S01E01 2024-01-02-2000.mpg'), True)
check('posted empty', posts[-1][1], [])
check('set_breaks not found', channels.set_breaks(opts, result, '/dvr/nope.mpg'), False)

server.shutdown()
server.server_close()
check('server down', channels.set_breaks(opts, result, '/dvr/TV/Show B/Show B S01E01 2024-01-02-2000.mpg'), False)

if _fails:
    print(f'\n{len(_fails)} failed')
    sys.exit(1)
print('\nall passed')
