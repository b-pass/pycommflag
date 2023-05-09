import av
import errno
import logging as log
import numpy as np
import scipy.signal as spsig

class Player:
    def __init__(self, filename:str, no_deinterlace:bool=False):
        self.filename = filename
        self.container = av.open(self.filename)
        self.container.gen_pts = True
        #self.container.discard_corrupt = True
        self.container.streams.video[0].thread_type = "AUTO"
        self.container.streams.video[0].thread_count = 2
        self.container.streams.audio[0].thread_type = "AUTO"
        self.container.streams.audio[0].thread_count = 2

        self.duration = self.container.duration / av.time_base
        self.streams = {'video':0}
        self.aq = None
        self._audio_res = []

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
        self.graph = None
        if inter*10 > ninter and not no_deinterlace:
            self.interlaced = True
            log.debug(f"{inter} interlaced frames (and {ninter} not), means we will deinterlace.")
            self._create_graph()
        else:
            log.debug(f"We will NOT deinterlace (had {inter} interlaced frames and {ninter} non-interlaced frames)")
            self.interlaced = False
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

    def seek_exact(self, seconds:float)->av.VideoFrame:
        if seconds > self.duration:
            seconds = self.duration
        orig_ask = seconds
        vs = self.container.streams.video[self.streams.get('video', 0)]
        self._flush()
        while True:
            if seconds <= 0.1:
                self.vt_pos = vs.start_time
                self.container.seek(vs.start_time, stream=vs, any_frame=True, backward=True)
                break
            else:
                time = int(seconds / vs.time_base) + vs.start_time
                self.vt_pos = time
                self.container.seek(time, stream=vs, backward=True)
                try:
                    vf = next(self.container.decode(video=0))
                except StopIteration:
                    seconds -= 1
                    continue
                if vf.time + 2/self.frame_rate - self.vt_start < orig_ask:
                    break
                seconds -= 0.5
        
        if self.graph:
            for vf in self.container.decode(video=0):
                if vf.time + 2/self.frame_rate - self.vt_start >= orig_ask:
                    self.graph.push(vf)
                    break
        for f in self.frames():
            if (f.time - self.vt_start) >= orig_ask:
                return f
        return None
    
    def enable_audio(self, stream=0):
        self.streams['audio'] = stream
        self._flush()
        self.vt_start = self.container.streams.video[self.streams['video']].start_time * self.container.streams.video[self.streams['video']].time_base
        self.at_start = self.container.streams.audio[self.streams['audio']].start_time * self.container.streams.video[self.streams['audio']].time_base
    
    def disable_audio(self):
        if 'audio' in self.streams:
            del self.streams['audio']
        self.aq = None
        self._audio_res = []

    def _queue_audio(self, af):
        if self.aq is None:
            return
        d = af.to_ndarray()
        # av.AudioResampler doesn't actually work, so we have to do it manually:
        if d.dtype.kind != 'f':
            d = d.astype('float32') / 32767.0
        if d.shape[0] == 1:
            x = []
            nc = len(af.layout.channels)
            for c in range(nc):
                x.append(d[c::nc])
            d = np.vstack(x)
        
        if d.shape[0] >= 4 or (d.shape[0] == 3 and af.layout.channels[2].name.endswith('C')):
            d = (d[0,...] + d[1,...] + d[2,...]) / 3 #d = ((d[0,...] + d[1,...]) / 2.0 + d[2,...]) / 1.4142
        elif d.shape[0] >= 2:
            d = (d[0,...] + d[1,...]) / 2.0
        else:
            return
        
        #print("AQ:",d.shape,af.time-self.at_start,af.sample_rate)
        self.aq.append((d,af.time-self.at_start,af.sample_rate))
    
    def _resample_audio(self):
        # resample to 16 kHz

        if not self.aq:
            return
        
        # out of order PTS, should never happen...?
        self.aq = sorted(self.aq, key=lambda x:x[1])
        
        start = self.aq[0][1]
        samples = None
        psr = None
        while self.aq:
            (d,t,sr) = self.aq.pop(0)
            #print(t,sr,len(d),t+len(d)/sr)
            if psr != sr and samples is not None:
                self._audio_res.append((spsig.resample(samples, int(len(samples)*16000/psr))), start)
                start = self.aq[0][1] if self.aq else -1
                samples = None
            psr = sr
            if samples is not None:
                samples = np.append(samples, d)
            else:
                samples = d
        
        if samples is not None:
            self._audio_res.append((spsig.resample(samples, int(len(samples)*16000/psr)), start))

    def _flush(self):
        if self.graph:
            self._create_graph()
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
        self.container.streams.video[0].thread_count = 2
        self.container.streams.audio[0].thread_type = "AUTO"
        self.container.streams.audio[0].thread_count = 2
        self.container.seek(pts, stream=self.container.streams.video[0], any_frame=True, backward=False)
        self._flush()

    def move_audio(self)->list[tuple[np.ndarray,float]]:
        self._resample_audio()
        x = self._audio_res
        self._audio_res = []
        return x

    def frames(self) -> iter:
        fail = 0
        iter = self.container.decode(**self.streams)
        while True:
            try:
                frame = next(iter)
            except StopIteration:
                break
            except av.error.InvalidDataError as e:
                fail += 1
                vs = self.container.streams.video[self.streams.get('video', 0)]
                self.vt_pos = int(self.vt_pos + (1/self.frame_rate)/vs.time_base)
                if fail%100 == 0:
                    if fail >= 10000:
                        log.critical(f"Repeated InvalidDataError, skipped {fail} frames but found nothing good")
                        break
                    log.debug(f"InvalidDataError during decode -- seeking ahead #{fail}")
                    self._resync(self.vt_pos)
                else:
                    self.container.seek(self.vt_pos, stream=vs, any_frame=True, backward=False)
                iter = self.container.decode(**self.streams)
                continue
            
            if fail:
                log.debug(f"Resync'd after {fail} skipped/dropped/corrupt/whatever frames")
                self._flush()
                fail = 0
            
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
                self.vt_pos = frame.pts
                yield frame
        
        self._resample_audio()
        return #raise StopIteration()
