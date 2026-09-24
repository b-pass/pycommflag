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

    def move_audio(self)->list[tuple[np.ndarray,np.ndarray|None,float,int]]|None:
        x = self.aq
        if x is not None:
            self.aq = []
        return x

    def frames(self) -> iter:
        good = 0
        fail = 0
        stuck = 0
        last_pts = None
        ovtp = self.vpts
        packets = self.container.decode(**self.streams)
        fix_audio = False
        audio_stream = None
        while True:
            try:
                frame = next(packets)
                if frame is None: continue
            except StopIteration:
                break
            except IndexError:
                log.exception("IndexError during decode, ending iteration early")
                break
            except av.error.EOFError:
                break
            except av.error.PatchWelcomeError as wtf:
                log.exception("unrecoverable AV error")
                os._exit(134)
            except (av.error.InvalidDataError,av.error.UndefinedError) as e:
                fail += 1
                vs = self._video_stream()
                self.vpts += math.ceil( (1.0/self.frame_rate)/vs.time_base )
                if fail%100 == 0:
                    log.debug(f"InvalidDataError during decode -- seeking ahead #{fail}, from {ovtp} to {self.vpts}")
                    if fail >= 500 and not fix_audio and 'audio' in self.streams:
                        self.trouble = True
                        fix_audio = True
                        audio_stream = self.streams.pop('audio')
                    if fail >= 2000:
                        log.exception(f"Repeated InvalidDataError, skipped {fail} frames but found nothing good")
                        if good >= self.frame_rate * 600:
                            # put audio back before we go, the caller will keep using this Player
                            if audio_stream is not None:
                                self.streams['audio'] = audio_stream
                            return
                        else:
                            os._exit(134)
                    self._resync(self.vpts)
                else:
                    try:
                        self.container.seek(self.vpts, stream=vs, any_frame=True, backward=(fail%2 == 0))
                    except av.error.PermissionError as e:
                        log.exception("seek permission error, bailing")
                        break
                packets = self.container.decode(**self.streams)
                continue
            
            if type(frame) is av.AudioFrame:
                self._queue_audio(frame)
            elif type(frame) is av.VideoFrame:
                # NEVER go backwards, whether or not we are currently recovering. Seeking into
                # a corrupt patch makes the decoder hand back one frame from AHEAD of the seek
                # target and then replay a span from behind it. Letting that replay through
                # drags self.vpts back to the pts that just failed, so the next error seeks to
                # the same place, gets the same replay, and the whole thing cycles forever at
                # zero net progress -- while feeding the caller duplicate frames. Dropping the
                # replay keeps vpts ahead of the bad patch, so the next seek clears it.
                src_pts = frame.pts
                if src_pts is not None and last_pts is not None and src_pts <= last_pts:
                    stuck += 1
                    if stuck % 500 == 0:
                        log.warning(f"dropped {stuck} out-of-order/replayed frames around pts {src_pts}")
                    continue
                if src_pts is not None:
                    last_pts = src_pts
                if fail:
                    if fix_audio:
                        fix_audio = False
                        if audio_stream is not None:
                            self.streams['audio'] = audio_stream
                            audio_stream = None
                            packets = self.container.decode(**self.streams)
                    if self.graph:
                        self._create_graph()
                    log.info(f"Resync'd to {float(self.vpts*self.vtime_base)} after {fail} skipped/dropped/corrupt/whatever frames")
                    fail = 0
                    stuck = 0
                
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
        
        fail = 0
        stuck = 0
        last_pts = None
        ovtp = self.vpts
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
                #print("decoded",len(frames),to_skip,skip)
                if not frames:
                    continue
                # countdown rather than a modulo so a multi-frame packet can't step over
                # the next sample point and make the stride uneven
                to_skip = skip - len(frames)
                frame = frames[0]
            except StopIteration:
                break
            except av.error.EOFError:
                break
            except av.error.PatchWelcomeError as wtf:
                log.exception("unrecoverable AV error")
                os._exit(134)
            except (av.error.InvalidDataError,av.error.UndefinedError) as e:
                fail += 1
                vs = self._video_stream()
                self.vpts += math.ceil( (1.0/self.frame_rate)/vs.time_base )
                if fail%100 == 0:
                    #log.debug(f"InvalidDataError during decode -- seeking ahead #{fail}, from {ovtp} to {self.vpts}")
                    if fail >= 500:
                        log.exception(f"Repeated InvalidDataError, skipped {fail} frames but found nothing good")
                        return
                    self._resync(self.vpts)
                else:
                    try:
                        self.container.seek(self.vpts, stream=vs, any_frame=True, backward=(fail%2 == 0))
                    except av.error.PermissionError as e:
                        log.exception("seek permission error, bailing")
                        break
                packets = self.container.demux(**self.streams)
                continue
            
            # same no-progress trap as frames(), and same unconditional fix
            if frame.pts is not None and last_pts is not None and frame.pts <= last_pts:
                stuck += 1
                if stuck % 500 == 0:
                    log.warning(f"dropped {stuck} out-of-order/replayed packets around pts {frame.pts}")
                continue
            if fail:
                log.info(f"Resync'd to {float(self.vpts*self.vtime_base)} after {fail} skipped/dropped/corrupt/whatever packets")
                fail = 0
                stuck = 0
        
            if type(frame) is av.VideoFrame:
                if frame.pts is not None:
                    self.vpts = frame.pts
                    last_pts = frame.pts
                yield frame
        
        return #raise StopIteration()
