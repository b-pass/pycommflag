import av
import errno
import logging as log
import numpy as np

class Player:
    def __init__(self, filename:str, no_deinterlace:bool=False):
        self.container = av.open(filename)
        self.container.flags |= av.container.core.Flags.GENPTS
        self.container.flags |= av.container.core.Flags.DISCARD_CORRUPT
        self.container.streams.video[0].thread_type = "AUTO"
        #self.container.streams.video[0].thread_count = 4
        self.container.streams.audio[0].thread_type = "AUTO"

        self.duration = self.container.duration / av.time_base
        self.streams = {'video':0}
        self.aq = None
        self.vq = []

        inter = 0
        ninter = 0
        for f in self.container.decode(video=0):
            if f.interlaced_frame:
                inter += 1
            else:
                ninter += 1
            if inter+ninter >= 360:
                break
        self.container.seek(self.container.streams.video[0].start_time, stream=self.container.streams.video[0])
        
        #f = next(self.container.decode(video=0))
        self.shape = (f.height, f.width)

        if inter*4 > ninter and not no_deinterlace:
            self.interlaced = True
            log.debug(f"{inter} interlaced frames (and {ninter} not), means we will deinterlace.")
            self.graph = av.filter.Graph()
            buffer = self.graph.add_buffer(template=self.container.streams.video[0])
            bwdif = self.graph.add("yadif", "")
            buffersink = self.graph.add("buffersink")
            buffer.link_to(bwdif)
            bwdif.link_to(buffersink)
            self.graph.configure()
        else:
            log.debug(f"We will NOT deinterlace (had {inter} interlaced frames and {ninter} non-interlaced frames)")
            self.interlaced = False
            self.graph = None
        self.frame_rate = float(self.container.streams.video[0].average_rate)
        self.vt_start = self.container.streams.video[0].start_time * self.container.streams.video[0].time_base
        log.debug(f"Video {filename} is {self.shape} at {self.frame_rate} fps")
    
    def seek(self, seconds:float):
        vs = self.container.streams.video[self.streams.get('video', 0)]
        if seconds <= 0.1:
            self.container.seek(vs.start_time, stream=vs, any_frame=True)
        else:
            self.container.seek(int(seconds / vs.time_base) + vs.start_time, stream=vs)
        self._flush()

    def seek_exact(self, seconds:float)->tuple|None:
        self.seek(seconds)
        for f in self.frames():
            if f[0].time + 1/self.frame_rate > seconds:
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

    def _queue_audio(self, af):
        if self.aq is None:
            return
        d = af.to_ndarray()
        if d.dtype.kind != 'f':
            d = d.astype('float32') / 32767.0
        if d.shape[0] == 1:
            x = []
            nc = self.aq[0].channels
            for c in range(nc):
                x.append(d[c::nc])
            for c in range(8-nc):
                x.append([])
            d = np.vstack(x)
        if d.shape[0] < 6:
            d = np.pad(d, [(0,8-d.shape[0]),(0,0)])
        elif d.shape[0] > 6:
            d = d[:6,...]
        #print("AQ:",d.shape,af.time-self.at_start,af.sample_rate)
        self.aq.append((d,af.time-self.at_start,af.sample_rate))
        assert(self.aq[0][1] <= self.aq[-1][1])

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
            peaks.append(np.max(np.abs(d), axis=1))
        
        return np.max(peaks, axis=0) if peaks else np.zeros((6,1))

    def _flush(self):
        if self.graph:
            try:
                while True:
                    self.graph.pull()
            except av.AVError as e:
                if e.errno != errno.EAGAIN:
                    raise
        self.vq = []
        self.aq = [] if 'audio' in self.streams else None

    def frames(self) -> iter:
        for frame in self.container.decode(**self.streams):
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