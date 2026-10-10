import math
import av
import errno
import logging as log
import av.container
import numpy as np
import os

class Player:
    def __init__(self, filename:str, no_deinterlace:bool=False):
        self.filename = filename
        self.graph = None
        self.trouble = False
        self.streams = {'video':0}
        self.aq = None
        self._audio_res = []
        self.shape = (-1,-1)

        self._resync(None)
        
        self.duration = self.container.duration / av.time_base
        self.frame_rate = round(self.container.streams.video[0].guessed_rate,3)
        self.vt_start = self.container.streams.video[0].start_time * self.container.streams.video[0].time_base
        self.vpts = self.container.streams.video[0].start_time
        
        inter = 0
        ninter = 0
        for f in self.frames():
            self.shape = (f.height, f.width)
            if f.interlaced_frame:
                inter += 1
            else:
                ninter += 1
            if inter+ninter >= 360 or no_deinterlace:
                break
        
        if self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError(f"Could not decode any video frames from {filename}")

        self.seek(0)
        
        if not no_deinterlace:
            if inter*10 > ninter:
                self.interlaced = True
                log.info(f"{inter} interlaced frames (and {ninter} not), means we will deinterlace.")
                self._create_graph()
            else:
                log.debug(f"We will NOT deinterlace (had {inter} interlaced frames and {ninter} non-interlaced frames)")
                self.interlaced = False
        else:
            log.debug(f"Deinterlace is disabled, so we will NOT deinterlace (had {inter} interlaced frames and {ninter} non-interlaced frames)")
            self.interlaced = False
        log.debug(f"Video {filename} is {self.shape} at {float(self.frame_rate)} fps for {self.duration} seconds totalling {int(self.duration * self.frame_rate)} frames)")
    
    def _video_stream(self):
        return self.container.streams.video[self.streams.get('video', 0)]

    def seek(self, seconds:float):
        if seconds > self.duration:
            seconds = self.duration
        vs = self._video_stream()
        if seconds <= 0.1:
            self.vpts = vs.start_time
            self.container.seek(vs.start_time, stream=vs, any_frame=True)
        else:
            pts = int(seconds / vs.time_base) + vs.start_time
            self.vpts = pts
            self.container.seek(pts, stream=vs)
        self._flush()

    def seek_exact(self, seconds:float)->av.VideoFrame:
        if seconds > self.duration:
            seconds = self.duration
        orig_ask = seconds
        # f.time and vt_start are floats, and a bare >= against orig_ask loses to ~1e-16 of
        # rounding on some frames, silently handing back the NEXT frame. A thousandth of a
        # frame is far above that noise and far below a real frame interval.
        eps = 1.0 / (float(self.frame_rate) * 1000.0)
        reseek = 0.5
        tries = 0
        while True:
            # re-fetch every pass; a decode failure inside frames() can _resync() and
            # replace the container out from under us
            vs = self._video_stream()
            self._flush()
            if seconds <= 0.1:
                self.vpts = vs.start_time
                self.container.seek(vs.start_time, stream=vs)
                break
            else:
                pts = int(seconds / vs.time_base) + vs.start_time
                self.vpts = pts
                # NO any_frame here: we need to land on a keyframe so the probe frame
                # (and everything we decode forward from it) has its references
                self.container.seek(pts, stream=vs)
                probe = self.frames()
                try:
                    vf = next(probe, None)
                finally:
                    probe.close()
                if vf and ((vf.time + 1/self.frame_rate) - self.vt_start) <= orig_ask + eps:
                    break
                #print(f"trying to get to {orig_ask} at {seconds} got {vf.time-self.vt_start}")
                tries += 1
                if tries >= 20:
                    # nothing decodable anywhere near there, give up and scan from the start
                    log.debug(f"seek_exact({orig_ask}) probing failed {tries} times, scanning from the beginning")
                    seconds = 0
                    continue
                seconds -= reseek
                reseek += 0.5
        
        # the probing above pushed frames through the filter graph and queued audio from
        # the wrong positions, throw all of that away before the real scan
        self._flush()
        for f in self.frames():
            if (f.time - self.vt_start) >= orig_ask - eps:
                return f
        return None
    
    def enable_audio(self, stream=0):
        self.streams['audio'] = stream
        self._flush()
        self.vt_start = self.container.streams.video[self.streams['video']].start_time * self.container.streams.video[self.streams['video']].time_base
        self.at_start = self.container.streams.audio[self.streams['audio']].start_time * self.container.streams.audio[self.streams['audio']].time_base
    
    def disable_audio(self):
        if 'audio' in self.streams:
            del self.streams['audio']
        self.aq = None
        self._audio_res = []

    def _queue_audio(self, af:av.AudioFrame):
        if self.aq is None or af is None or af.time is None:
            return
        d = af.to_ndarray()
        # av.AudioResampler doesn't actually work, so we have to do it manually:
        if d.dtype.kind == 'u':
            # unsigned sample formats (u8) sit around the midpoint, not zero
            bias = float(np.iinfo(d.dtype).max + 1) / 2.0
            d = (d.astype('float32') - bias) / bias
        elif d.dtype.kind != 'f':
            d = d.astype('float32') / float(np.iinfo(d.dtype).max)
        if d.shape[0] == 1:
            # packed/interleaved layout: to_ndarray() gives us (1, samples*channels),
            # so de-interleave along the samples, not along the (length 1) first axis
            nc = len(af.layout.channels)
            if nc > 1:
                flat = d[0]
                usable = (len(flat) // nc) * nc
                d = np.vstack([flat[c:usable:nc] for c in range(nc)])
        if d.shape[0] < 1:
            return
        
        n = 3 if d.shape[0] >= 3 and af.layout.channels[2].name.endswith('C') else 2
        n = min(n, d.shape[0])
        main = np.sum(d[:n], 0) / float(n)
        #d = ((d[0,...] + d[1,...]) / 2.0 + d[2,...]) / 1.4142

        surr = np.sum(d[3:,...], 0) / (d.shape[0]-3) if d.shape[0] >= 4 else None

        when = af.time - self.at_start

        if self.aq and \
            self.aq[-1][3] == af.sample_rate and \
            (self.aq[-1][2] + len(self.aq[-1][0]) / af.sample_rate) >= when and \
            ((surr is None and self.aq[-1][1] is None) or \
                 (surr is not None and self.aq[-1][1] is not None and len(self.aq[-1][0]) == len(self.aq[-1][1]))):

            #print(f'Append {len(main)} to {len(self.aq[-1][0])} at {when} sr {af.sample_rate}')
            self.aq[-1] = (
                np.append(self.aq[-1][0], main),
                np.append(self.aq[-1][1], surr) if surr is not None else None,
                self.aq[-1][2],
                af.sample_rate
            )
        else:
            #print("AQ:",d.shape,af.time-self.at_start,af.sample_rate)
            self.aq.append((main,surr,when,af.sample_rate))

    def _flush(self):
        if self.graph:
            self._create_graph()
        self.aq = [] if 'audio' in self.streams else None
    
    def _create_graph(self):
        self.graph = av.filter.Graph()
        buffer = self.graph.add_buffer(template=self._video_stream())
        yadif = self.graph.add("yadif", "")
        buffersink = self.graph.add("buffersink")
        buffer.link_to(yadif)
        yadif.link_to(buffersink)
        self.graph.configure()

    def _resync(self, pts):
        old = getattr(self, 'container', None)
        self.container:av.container.InputContainer = av.open(self.filename)#, options={'probesize':'10000000'})
        if old is not None:
            # av.open() again without this leaks the file descriptor every resync
            try:
                old.close()
            except Exception:
                log.debug("could not close the previous container", exc_info=True)
        try: 
            self.container.gen_pts = True
            #if self.trouble:
            #    self.container.discard_corrupt = True
        except:
            pass
        try:
            f:int = 0
            f |= av.container.Flags.gen_pts
            #if self.trouble:
            #    f |= av.container.Flags.discard_corrupt
            self.container.flags = f
        except:
            pass
        vs = self._video_stream()
        vs.thread_type = "AUTO"
        vs.thread_count = 2
        # stash this so we can still log a position after a resync has closed the old stream
        self.vtime_base = vs.time_base
        if len(self.container.streams.audio) > 0:
            astream = self.container.streams.audio[min(self.streams.get('audio', 0), len(self.container.streams.audio)-1)]
            astream.thread_type = "AUTO"
            astream.thread_count = 2
        if pts is not None:
            self.container.seek(pts, stream=vs, any_frame=True)

    # Video frames must never go backwards, recovering or not: seeking into a corrupt patch
    # makes the decoder hand back a frame from ahead of the target and then replay a span
    # from behind it.  Letting the replay through drags vpts back to the pts that just failed,
    # so the next error seeks to the same place and recovery cycles forever.
    #
    # But a single corrupt packet can also carry a timestamp hours ahead.  Taking it as the
    # high-water mark would make that rule drop every real frame after it (seen losing the
    # last 15 minutes of a recording), so a jump this far ahead is skipped unless enough
    # frames in a row agree with it, which means the clock really did move.
    MAX_PTS_JUMP = 15*60 # seconds
    PTS_JUMP_CONFIRM = 5 # frames

    def _check_pts(self, pts, last_pts, ahead):
        """Returns (use this frame, new last_pts, new ahead count)."""
        if pts is None:
            return True, last_pts, ahead
        if last_pts is not None:
            if pts <= last_pts:
                return False, last_pts, ahead # require monotonic pts
            if (pts - last_pts) * self.vtime_base > self.MAX_PTS_JUMP:
                ahead += 1
                if ahead < self.PTS_JUMP_CONFIRM:
                    if ahead == 1:
                        log.info(f"Skipping a frame whose timestamp jumps {float((pts - last_pts) * self.vtime_base):.0f}s ahead")
                    return False, last_pts, ahead
        return True, pts, 0

    def move_audio(self)->list[tuple[np.ndarray,np.ndarray|None,float,int]]|None:
        x = self.aq
        if x is not None:
            self.aq = []
        return x

    def frames(self) -> iter:
        rec = None
        last_pts = None
        ahead = 0
        good = 0
        packets = self.container.decode(**self.streams)
        while True:
            try:
                frame = next(packets)
                if frame is None: continue
            except Exception as e:
                rec, action = _Recovery.handle(rec, self, e, good, max_fail=2000, drop_audio_at=500)
                if action is None:
                    raise
                if action is _Recovery.STOP:
                    break
                packets = self.container.decode(**self.streams)
                continue

            if type(frame) is av.AudioFrame:
                self._queue_audio(frame)
            elif type(frame) is av.VideoFrame:
                use, last_pts, ahead = self._check_pts(frame.pts, last_pts, ahead)
                if not use:
                    continue

                if rec is not None:
                    if rec.restore_audio():
                        packets = self.container.decode(**self.streams)
                    if self.graph:
                        self._create_graph()
                    rec.done()
                    rec = None

                if self.graph:
                    self.graph.push(frame)
                    try:
                        frame = self.graph.pull()
                    except av.FFmpegError as e:
                        if e.errno != errno.EAGAIN:
                            raise
                        continue
                if frame.pts is not None:
                    self.vpts = frame.pts
                good += 1
                yield frame

        return #raise StopIteration()

    def frames_stride(self, skip) -> iter:
        if self.graph or 'audio' in self.streams:
            count = 0
            for frame in self.frames():
                if (count%skip) == 0:
                    yield frame
                count += 1
            return

        rec = None
        last_pts = None
        ahead = 0
        packets = self.container.demux(**self.streams)
        to_skip = 0
        while True:
            try:
                pkt = next(packets)
                if pkt is None:
                    continue
                if to_skip > 0:
                    # don't decode packets just to throw away the frames
                    to_skip -= len(pkt.decode()) if pkt.is_keyframe else 1
                    continue
                frames = pkt.decode()
                if not frames:
                    continue
                # countdown rather than a modulo so a multi-frame packet can't step over
                # the next sample point and make the stride uneven
                to_skip = skip - len(frames)
                frame = frames[0]
            except Exception as e:
                rec, action = _Recovery.handle(rec, self, e, max_fail=500)
                if action is None:
                    raise
                if action is _Recovery.STOP:
                    break
                packets = self.container.demux(**self.streams)
                continue

            use, last_pts, ahead = self._check_pts(frame.pts, last_pts, ahead)
            if not use:
                to_skip = 0
                continue

            if rec is not None:
                rec.done()
                rec = None

            if type(frame) is av.VideoFrame:
                if frame.pts is not None:
                    self.vpts = frame.pts
                yield frame

        return #raise StopIteration()

class _Recovery:
    """
    Shared decode/demux error recovery for Player.frames() and Player.frames_stride().

    One of these exists only while decoding is broken. The callers keep it in a variable that
    is None the rest of the time, so `rec is not None` IS the "we are in the failure path"
    flag -- there is no separate counter to consult. handle() creates it on the first failure
    and clears it on give-up; the caller clears it once a usable item comes back.
    """

    STOP = 'stop'          # iteration is over, break out
    REBUILD = 'rebuild'    # repositioned; rebuild the iterator and keep going

    def __init__(self, player, max_fail, drop_audio_at=None):
        self.p = player
        self.max_fail = max_fail           # give up after this many consecutive failures
        self.drop_audio_at = drop_audio_at # try without audio past this many (None = never)
        self.count = 0                     # failures in THIS episode; drives the escalation
        self.start_vpts = player.vpts
        self._audio_stream = None

    @classmethod
    def handle(cls, rec, player, e, good=None, **opts):
        """Classify an exception off a libav iterator and act on it.

        `rec` is the caller's current recovery or None. Returns (rec, action):
          action is None    -> not ours, the caller should bare-`raise`
          action is STOP    -> iteration is over, break (returned rec is None)
          action is REBUILD -> repositioned, rebuild the iterator and continue
        """
        if isinstance(e, (StopIteration, av.error.EOFError)):
            action = cls.STOP
        elif isinstance(e, IndexError):
            log.exception(f"IndexError during decode, ending iteration early")
            action = cls.STOP
        elif isinstance(e, av.error.PatchWelcomeError):
            log.exception("unrecoverable AV error")
            os._exit(134)
        elif isinstance(e, (av.error.InvalidDataError, av.error.UndefinedError)):
            if rec is None:
                rec = cls(player, **opts)
            action = rec._reposition(good)
        else:
            return rec, None
        if action is cls.STOP and rec is not None:
            # we are leaving mid-recovery, put audio back for whoever uses this Player next
            rec.restore_audio()
            rec = None
        return rec, action

    def _reposition(self, good):
        """Step past one frame of damage, escalating if this keeps happening."""
        p = self.p
        self.count += 1
        vs = p._video_stream()
        p.vpts += math.ceil( (1.0/p.frame_rate)/vs.time_base )
        if self.count % 100 == 0:
            log.debug(f"InvalidDataError during decode -- seeking ahead #{self.count}, "
                      f"from {self.start_vpts} to {p.vpts}")
            # most InvalidDataErrors seem to come from audio, so try without it for a while
            if self.drop_audio_at is not None and self.count >= self.drop_audio_at \
                    and self._audio_stream is None and 'audio' in p.streams:
                p.trouble = True
                self._audio_stream = p.streams.pop('audio')
            if self.count >= self.max_fail:
                log.error(f"Repeated InvalidDataError, skipped {self.count} things but found nothing good")
                # good is None when the caller has no frame count to judge by (frames_stride):
                # give up quietly rather than taking the whole process down
                if good is not None and good < p.frame_rate * 300:
                    os._exit(134)
                return self.STOP
            p._resync(p.vpts)
        else:
            try:
                p.container.seek(p.vpts, stream=vs, any_frame=True, backward=(self.count % 2 == 0))
            except av.error.PermissionError:
                log.exception("seek permission error, bailing")
                return self.STOP
        return self.REBUILD

    def restore_audio(self):
        """Put the audio stream back if we dropped it. True means rebuild the iterator."""
        if self._audio_stream is None:
            return False
        self.p.streams['audio'] = self._audio_stream
        self._audio_stream = None
        return True

    def done(self):
        """A usable item finally arrived; log it. The caller then drops this object."""
        log.info(f"Resync'd to {float(self.p.vpts*self.p.vtime_base)} after {self.count} skipped/dropped/corrupt/whatever frames/packets")
