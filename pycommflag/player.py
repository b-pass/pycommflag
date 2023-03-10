import av
import errno
import logging as log
import numpy as np

class Player:
    def __init__(self, filename:str, no_deinterlace:bool=False):
        self.filename = filename
        self.container = av.open(self.filename)
        self.container.gen_pts = True
        #self.container.discard_corrupt = True
        self.container.streams.video[0].thread_type = "AUTO"
        #self.container.streams.video[0].thread_count = 4
        self.container.streams.audio[0].thread_type = "AUTO"

        self.duration = self.container.duration / av.time_base
        self.streams = {'video':0}
        self.aq = None
        self.vq = []

        inter = 0
        ninter = 0
        f = None
        try:
            for f in self.container.decode(video=0):
                if f.interlaced_frame:
                    inter += 1
                else:
                    ninter += 1
                if inter+ninter >= 360:
                    break
        except Exception as e:
            log.warn("Decode failure: " + str(e))
        
        self.container.seek(self.container.streams.video[0].start_time, stream=self.container.streams.video[0])
        
        self.shape = (f.height, f.width)
        self.frame_rate = self.container.streams.video[0].guessed_rate

        if inter*4 > ninter and not no_deinterlace:
            self.interlaced = True
            log.debug(f"{inter} interlaced frames (and {ninter} not), means we will deinterlace.")
            self._create_graph()
        else:
            log.debug(f"We will NOT deinterlace (had {inter} interlaced frames and {ninter} non-interlaced frames)")
            self.interlaced = False
            self.graph = None
        self.vt_start = self.container.streams.video[0].start_time * self.container.streams.video[0].time_base
        self.vt_pos = self.vt_start
        log.debug(f"Video {filename} is {self.shape} at {float(self.frame_rate)} fps")
    
    def seek(self, seconds:float):
        if seconds > self.duration:
            seconds = self.duration
        vs = self.container.streams.video[self.streams.get('video', 0)]
        if seconds <= 0.1:
            self.vt_pos = vs.start_time
            self.container.seek(vs.start_time, stream=vs, any_frame=True)
        else:
            time = int(seconds / vs.time_base) + vs.start_time
            self.vt_pos = time
            self.container.seek(time, stream=vs)
        self._flush()

    def seek_exact(self, seconds:float)->tuple|None:
        if seconds > self.duration:
            seconds = self.duration
        orig_ask = seconds
        vs = self.container.streams.video[self.streams.get('video', 0)]
        while True:
            if seconds <= 0.1:
                self.vt_pos = vs.start_time
                self.container.seek(vs.start_time, stream=vs, any_frame=True, backward=True)
                self._flush()
                break
            else:
                time = int(seconds / vs.time_base) + vs.start_time
                self.vt_pos = time
                self.container.seek(time, stream=vs, backward=True)
                self._flush()
                try:
                    f = next(self.frames())
                except StopIteration:
                    seconds -= 0.25
                    continue
                if (f[0].time - self.vt_start) > orig_ask:
                    seconds -= 0.25
                elif (f[0].time + 1.75/self.frame_rate - self.vt_start) < orig_ask:
                    break
                else:
                    return f

        for f in self.frames():
            if (f[0].time - self.vt_start) >= orig_ask:
                return f
        return (None,None)
    
    def enable_audio(self, stream=0):
        self.streams['audio'] = stream
        self._flush()
        self.vt_start = self.container.streams.video[self.streams['video']].start_time * self.container.streams.video[self.streams['video']].time_base
        self.at_start = self.container.streams.audio[self.streams['audio']].start_time * self.container.streams.video[self.streams['audio']].time_base
    
    def disable_audio(self):
        if 'audio' in self.streams:
            del self.streams['audio']
        self.aq = None

    def _queue_audio(self, af):
        if self.aq is None:
            return
        d = af.to_ndarray()
        # av.AudioResampler doesn't actually work, so we have to do it manually:
        if d.dtype.kind != 'f':
            d = d.astype('float32') / 32767.0
        if d.shape[0] == 1:
            x = []
            nc = self.aq[0].channels
            for c in range(nc):
                x.append(d[c::nc])
            for c in range(6-nc):
                x.append([])
            d = np.vstack(x)
        if d.shape[0] < 6:
            d = np.pad(d, [(0,6-d.shape[0]),(0,0)])
        elif d.shape[0] > 6:
            d = d[:6,...]
        #print("AQ:",d.shape,af.time-self.at_start,af.sample_rate)
        self.aq.append((d,af.time-self.at_start,af.sample_rate))
        if len(self.aq) > 1 and self.aq[-2][1] > self.aq[-1][1]:
            # out of order PTS, should never happen...
            self.aq = sorted(self.aq, key=lambda x:x[1])

    def _get_audio_peaks(self, end):
        if not self.aq:
            return None
        
        if end and self.aq[-1][1] < end:
            return None
        
        #print(len(self.aq), end, end - self.aq[0][1], "t=", self.aq[0][1], self.aq[0][2])
        need = (end - self.aq[0][1]) if end else 1.0
        if need <= 0:
            return np.zeros((6,1))
        
        peaks = []
        while self.aq and need > 0:
            (d,t,sr) = self.aq[0]
            nsamp = int(need * sr)
            if nsamp <= 0:
                break
            #print(len(self.aq),need, end, "t=", t,sr,nsamp,nsamp/sr,d.shape,t+nsamp/sr)
            if nsamp >= d.shape[1]:
                nsamp = d.shape[1]
                self.aq.pop(0)
            else:
                self.aq[0] = (d[..., nsamp:], t+nsamp/sr, sr)
                d = d[..., 0:nsamp]
            need -= nsamp/sr
            m = np.max(np.abs(d), axis=1)
            m.resize((6,),refcheck=False)
            peaks.append(m)
        
        return np.max(peaks, axis=0) if peaks else np.zeros((6,1))

    def _flush(self):
        if self.graph:
            self._create_graph()
        self.vq = []
        self.aq = [] if 'audio' in self.streams else None
    
    def _create_graph(self):
        self.graph = av.filter.Graph()
        buffer = self.graph.add_buffer(template=self.container.streams.video[0])
        bwdif = self.graph.add("yadif", "")
        buffersink = self.graph.add("buffersink")
        buffer.link_to(bwdif)
        bwdif.link_to(buffersink)
        self.graph.configure()

    def _resync(self, pts):
        self.container = av.open(self.filename)
        self.container.gen_pts = True
        self.container.streams.video[0].thread_type = "AUTO"
        #self.container.streams.video[0].thread_count = 4
        self.container.streams.audio[0].thread_type = "AUTO"
        self.container.seek(pts, stream=self.container.streams.video[0], any_frame=True, backward=False)
        self._flush()

    def frames(self) -> iter:
        fail = 0
        vs = self.container.streams.video[self.streams.get('video', 0)]
        iter = self.container.decode(**self.streams)
        while True:
            try:
                frame = next(iter)
                if fail:
                    log.debug(f"Resync'd after {fail} skipped/dropped/corrupt/whatever frames")
                    fail = 0
                    self._resync(frame.pts)
                    continue
            except StopIteration:
                break
            except av.error.InvalidDataError as e:
                if fail%100 == 0:
                    if fail >= 10000:
                        log.critical(f"Repeated InvalidDataError, skipped {fail} frames but found nothing good")
                        break
                    log.debug(f"InvalidDataError during decode -- seeking ahead #{fail}")
                fail += 1
                self.vt_pos = int(self.vt_pos + (1/self.frame_rate)/vs.time_base)
                self.container.seek(self.vt_pos, stream=vs, any_frame=True, backward=False)
                self._flush()
                iter = self.container.decode(**self.streams)
                continue
            
            if type(frame) is av.AudioFrame:
                self._queue_audio(frame)
            elif type(frame) is av.VideoFrame:
                if self.graph:
                    self.graph.push(frame)
                    try:
                        frame = self.graph.pull()
                    except av.AVError as e:
                        if e.errno != errno.EAGAIN:
                            raise
                        continue
                self.vt_pos = frame.time
                if self.aq is not None:
                    self.vq.append(frame)
                else:
                    yield (frame,None)
            if len(self.vq) > 1:
                af = self._get_audio_peaks(self.vq[1].time - self.vt_start)
                if af is None:
                    continue
                vf = self.vq.pop(0)
                yield (vf,af)
        while self.vq:
            af = self._get_audio_peaks(self.vq[1].time - self.vt_start if len(self.vq) > 1 else None)
            vf = self.vq.pop(0)
            yield (vf,af)
        return #raise StopIteration()
