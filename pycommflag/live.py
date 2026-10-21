"""Following a recording that is still being written.

av.open() on a growing file reads to wherever the recorder has got to and stops there.
GrowingFile is a file object for av.open() that instead waits at the end of the file
until the recording is over, the way mythcommflag's RingBuffer does.
"""
import logging as log
import os
import time

HEADSTART = 600   # seconds of recording to wait for before the logo search, which only sees those
INTERVAL = 120    # seconds of new video between interim predictions
IDLE_SECS = 120   # without MythTV, a recording is over when its file hasn't grown for this long
SETTLE_SECS = 5   # after the recorder says it is done, wait for the file to stop growing
POLL_SECS = 1.0
HEADSTART_POLL_SECS = 15

class Recording:
    """A recording in progress. Calling it opens a GrowingFile on it (Player's `source`)."""
    def __init__(self, path:str, still_recording):
        self.path = path
        self.still_recording = still_recording  # still_recording(Recording) -> bool
        self.follow = False   # off: files behave like plain files and stop at the current end

    def __call__(self) -> 'GrowingFile':
        return GrowingFile(self)

    def idle_for(self) -> float:
        """Seconds since the file was last written."""
        return time.time() - os.stat(self.path).st_mtime

class GrowingFile:
    def __init__(self, rec:Recording):
        self.rec = rec
        self.f = open(rec.path, 'rb')

    def read(self, n:int=-1) -> bytes:
        data = self.f.read(n)
        if data or not self.rec.follow:
            return data

        last_data = time.time()
        while True:
            time.sleep(POLL_SECS)
            data = self.f.read(n)
            if data:
                return data
            if not self.rec.still_recording(self.rec) and time.time() - last_data >= SETTLE_SECS:
                return b''

    def seek(self, offset:int, whence:int=0) -> int:
        return self.f.seek(offset, whence)

    def tell(self) -> int:
        return self.f.tell()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def close(self):
        self.f.close()

def recording_check(opts):
    """A still_recording(Recording) callable for this recording.

    MythTV knows when it stops recording; anything else is over once the file goes idle.
    """
    if opts.chanid and opts.starttime:
        from . import mythtv
        if mythtv.recording_in_progress(opts.chanid, opts.starttime) is not None:
            last = [0.0, True]
            def myth(rec:Recording) -> bool:
                # one query every few seconds is plenty
                if time.time() - last[0] >= 10:
                    last[0] = time.time()
                    r = mythtv.recording_in_progress(opts.chanid, opts.starttime)
                    last[1] = r if r is not None else rec.idle_for() < IDLE_SECS
                return last[1]
            return myth
    return lambda rec: rec.idle_for() < IDLE_SECS

def is_live(opts) -> bool:
    """Whether opts.filename is a recording that is still being made."""
    if getattr(opts, 'live', False):
        return True
    if opts.chanid and opts.starttime:
        from . import mythtv
        return bool(mythtv.recording_in_progress(opts.chanid, opts.starttime))
    return False

def wait_for_headstart(rec:Recording, opts) -> None:
    """Wait until the recording holds HEADSTART seconds, or has ended sooner."""
    import av
    while True:
        have = 0.0
        try:
            # for a .ts this is libav's estimate from the file size and bitrate, which is
            # plenty for deciding when to start
            with av.open(rec.path) as c:
                have = (c.duration or 0) / av.time_base
        except Exception as e:
            log.debug(f'Could not measure {rec.path} yet: {e}')
        if have >= HEADSTART:
            return
        if not rec.still_recording(rec):
            log.info(f'Recording ended after {have:.0f}s, before the {HEADSTART}s head start')
            return
        if not opts.quiet:
            print(f'Waiting for a head start, have {have:.0f} of {HEADSTART} seconds', end='\r')
        time.sleep(HEADSTART_POLL_SECS)
