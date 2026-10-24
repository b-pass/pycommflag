#!/usr/bin/env python3
"""Checks that feature extraction fails cleanly instead of hanging.

process_video feeds two worker threads through bounded queues.  A worker that died used to
leave the producer blocked on a full queue forever, and an exception in the producer left
the workers waiting on theirs, so the process never exited.  Each case here runs
process_video against a fake player (no video, no TensorFlow) under a timeout.

Run it directly; it exits non-zero if anything fails.
"""
import os
import sys
import tempfile
import threading
import types

import numpy as np

# AudioProc.__init__ imports tensorflow only to limit its threads
_tf = types.ModuleType('tensorflow')
_tf.config = types.SimpleNamespace(threading=types.SimpleNamespace(
    set_intra_op_parallelism_threads=lambda n: None,
    set_inter_op_parallelism_threads=lambda n: None))
sys.modules.setdefault('tensorflow', _tf)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycommflag import options, processor

class _Seg:
    def __call__(self, samples):
        return [('speech', 0.0, len(samples)/16000.0)]
processor.ina_foss.Segmenter = _Seg

FPS = 30.0
TIMEOUT = 60
_fails = []

def check(name, ok, detail=''):
    print(('  pass  ' if ok else '  FAIL  ') + name + (f'   {detail}' if detail else ''))
    if not ok:
        _fails.append(name)

class FakeFrame:
    def __init__(self, n, shape):
        self.time = n / FPS
        self.pts = n
        self.height, self.width = shape
        self.format = types.SimpleNamespace(is_planar=False)
        self._n = n
    def to_ndarray(self, format=None):
        x = np.full((self.height, self.width, 3), 60 + (self._n % 50), 'uint8')
        x[4:20, 4:20] = 255  # a bright square where the test logo is
        return x

class FakePlayer:
    """What process_video uses of Player: frames of `shapes[i]` until `count`, raising
    `fail` at frame `fail_at` if given."""
    count, shapes, fail_at, fail = 1500, None, None, None
    def __init__(self, filename, no_deinterlace=False):
        self.duration = self.count / FPS
        self.frame_rate = FPS
        self.vt_start = 0.0
        self.shape = self.shapes(0)
    def enable_audio(self):
        self.at_start = 0.0
    def seek(self, seconds):
        pass
    def move_audio(self):
        return []
    def frames(self):
        for n in range(self.count):
            if n == self.fail_at:
                raise self.fail
            yield FakeFrame(n, self.shapes(n))
processor.Player = FakePlayer

def workers_alive():
    return [t.name for t in threading.enumerate() if t.name in ('videoProc', 'audioProc') and t.is_alive()]

def run(log_path, no_logo=True):
    """process_video in a thread: (finished in time, exception or None)"""
    argv = ['-q', '-f', '/nonexistent/video.ts', '-l', log_path] + (['--no-logo'] if no_logo else [])
    opts = options.get_options().parse_args(argv)
    result = {}
    def go():
        try:
            processor.process_video(opts.filename, log_path, opts)
        except BaseException as e:
            result['error'] = e
    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(TIMEOUT)
    return not t.is_alive(), result.get('error')

tmp = tempfile.mkdtemp(prefix='pycf-test-')
small = lambda n: (72, 128)

# a video worker that dies must stop the run, not leave it blocked on a full queue
orig_mean = processor.mean_axis1
calls = [0]
def failing_mean(*a, **k):
    calls[0] += 1
    if calls[0] == 50:
        raise ValueError('boom in the video worker')
    return orig_mean(*a, **k)
processor.mean_axis1 = failing_mean
FakePlayer.shapes = staticmethod(small)
path = os.path.join(tmp, 'cf_video_fail.json')
done, err = run(path)
processor.mean_axis1 = orig_mean
check('video worker failure ends the run', done, 'still running after %ds' % TIMEOUT if not done else '')
check('...and is reported', isinstance(err, RuntimeError) and isinstance(err.__cause__, ValueError), repr(err))
check('...and leaves no unreadable log behind', not os.path.exists(path))
check('...and no worker threads', not workers_alive(), str(workers_alive()))

# the same for the audio worker
orig_run = processor.AudioProc._run
def failing_run(self):
    raise ValueError('boom in the audio worker')
processor.AudioProc._run = failing_run
path = os.path.join(tmp, 'cf_audio_fail.json')
done, err = run(path)
processor.AudioProc._run = orig_run
check('audio worker failure ends the run', done)
check('...and is reported', isinstance(err, RuntimeError) and isinstance(err.__cause__, ValueError), repr(err))
check('...and leaves no unreadable log behind', not os.path.exists(path))
check('...and no worker threads', not workers_alive(), str(workers_alive()))

# an exception in the producer must still stop the workers
FakePlayer.fail_at, FakePlayer.fail = 700, OSError('decode exploded')
path = os.path.join(tmp, 'cf_main_fail.json')
done, err = run(path)
FakePlayer.fail_at = FakePlayer.fail = None
check('a producer exception propagates', done and isinstance(err, OSError), repr(err))
check('...and leaves no unreadable log behind', not os.path.exists(path))
check('...and no worker threads', not workers_alive(), str(workers_alive()))

# a resolution change mid-recording, with a logo, is survivable
mask = np.zeros((24, 24), bool)
mask[4, 4:20] = mask[19, 4:20] = mask[4:20, 4] = mask[4:20, 19] = True
logo = ((0, 0), (24, 24), mask, int(mask.sum() * 2 / 3))
path = os.path.join(tmp, 'cf_resize.json')
processor.write_feature_log({'logo': logo}, path)  # an existing log's logo is re-used
FakePlayer.shapes = staticmethod(lambda n: (72, 128) if n < 600 else (90, 160))
done, err = run(path, no_logo=False)
check('a resolution change does not stop the run', done and err is None, repr(err))
if done and err is None:
    flog = processor.read_feature_log(path)
    f = np.array(flog['frames'])
    check('...all frames logged', len(f) == FakePlayer.count, f'{len(f)} frames')
    check('...logo seen before the change', f[:600, 1].all())
    check('...and not claimed after it', not f[600:, 1].any())
check('...and no worker threads', not workers_alive(), str(workers_alive()))

if _fails:
    print(f'{len(_fails)} failed')
    sys.exit(1)
print('all passed')
