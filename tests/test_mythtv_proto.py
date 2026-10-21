#!/usr/bin/env python3
"""Checks for the COMMFLAG_UPDATE messages sent to mythbackend, against a fake backend.
Needs no video, no database and no model.

Run it directly; it exits non-zero if anything fails.
"""
import datetime
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import mythtv

SEP = '[]:[]'
_fails = []

def check(name, got, want):
    ok = got == want
    print(f'{"ok  " if ok else "FAIL"} {name}' + ('' if ok else f': got {got!r}, want {want!r}'))
    if not ok:
        _fails.append(name)

class FakeBackend(threading.Thread):
    """Speaks one protocol version, the way mainserver.cpp answers it."""
    def __init__(self, version, hang_up_after_message=False):
        super().__init__(daemon=True)
        self.version = version
        self.hang_up = hang_up_after_message
        self.srv = socket.create_server(('127.0.0.1', 0))
        self.port = self.srv.getsockname()[1]
        self.received = []   # every command, split on the separator
        self.connections = 0
        self.start()

    @staticmethod
    def _recv(conn, n):
        buf = b''
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise EOFError()
            buf += chunk
        return buf

    def _send(self, conn, *parts):
        data = SEP.join(parts).encode('utf-8')
        conn.sendall(b'%-8d' % len(data) + data)

    def run(self):
        while True:
            conn, _ = self.srv.accept()
            self.connections += 1
            try:
                while True:
                    size = int(self._recv(conn, 8))
                    cmd = self._recv(conn, size).decode('utf-8').split(SEP)
                    self.received.append(cmd)
                    words = cmd[0].split(' ')
                    if words[0] == 'MYTH_PROTO_VERSION':
                        if words[1] != self.version or ' '.join(words[2:]) != mythtv.PROTO_TOKENS.get(self.version):
                            self._send(conn, 'REJECT', self.version)
                            break
                        self._send(conn, 'ACCEPT', self.version)
                    elif words[0] == 'ANN':
                        self._send(conn, 'OK')
                    elif words[0] == 'MESSAGE':
                        self._send(conn, 'OK')
                        if self.hang_up:
                            self.hang_up = False
                            break
                    else:
                        self._send(conn, 'ERROR', 'unknown command')
            except EOFError:
                pass
            finally:
                conn.close()

def reset(port):
    mythtv.g_backend = None
    mythtv.g_backend_off = False
    mythtv._backend_address = lambda cursor: ('127.0.0.1', port)

KEY = '1051_2026-10-21T01:00:00Z'
MSG = ['MESSAGE', f'COMMFLAG_UPDATE {KEY} 100:4,2000:5,5000:4,7000:5']
MARKS = [(100,4), (2000,5), (5000,4), (7000,5)]

check('key from filename time', mythtv._unique_key(1051, '20261021010000'), KEY)
check('key from datetime', mythtv._unique_key('1051', datetime.datetime(2026, 10, 21, 1, 0, 0)), KEY)
check('key from iso', mythtv._unique_key(1051, '2026-10-21 01:00:00'), KEY)

be = FakeBackend('91')
reset(be.port)
check('send', mythtv.send_commflag_update(None, 1051, '20261021010000', MARKS), True)
check('handshake', [c[0].split(' ')[0] for c in be.received[:2]], ['MYTH_PROTO_VERSION', 'ANN'])
check('message', be.received[-1], MSG)
check('send again', mythtv.send_commflag_update(None, 1051, '20261021010000', []), True)
check('one connection', be.connections, 1)
check('no breaks', be.received[-1], ['MESSAGE', f'COMMFLAG_UPDATE {KEY}'])

be = FakeBackend('88')
reset(be.port)
check('older backend', mythtv.send_commflag_update(None, 1051, '20261021010000', MARKS), True)
check('older backend retried with its token', be.received[1][0], 'MYTH_PROTO_VERSION 88 XmasGift')
check('older backend message', be.received[-1], MSG)

be = FakeBackend('87')
reset(be.port)
check('multibyte token', mythtv.send_commflag_update(None, 1051, '20261021010000', MARKS), True)

be = FakeBackend('99')
reset(be.port)
check('unknown version', mythtv.send_commflag_update(None, 1051, '20261021010000', MARKS), False)
check('gives up after that', mythtv.g_backend_off, True)
n = len(be.received)
check('stays quiet', (mythtv.send_commflag_update(None, 1051, '20261021010000', MARKS), len(be.received)), (False, n))

be = FakeBackend('91', hang_up_after_message=True)
reset(be.port)
check('before restart', mythtv.send_commflag_update(None, 1051, '20261021010000', MARKS), True)
check('after restart', mythtv.send_commflag_update(None, 1051, '20261021010000', MARKS), True)
check('reconnected', be.connections, 2)

reset(1)  # nothing listens there
check('no backend', mythtv.send_commflag_update(None, 1051, '20261021010000', MARKS), False)

if _fails:
    print(f'\n{len(_fails)} failed')
    sys.exit(1)
