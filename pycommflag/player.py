import math
import av
import errno
import logging as log
import numpy as np

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
        
        self.seek(0)
        
        if not no_deinterlace:
            if inter*10 > ninter:
                self.interlaced = True
                log.debug(f"{inter} interlaced frames (and {ninter} not), means we will deinterlace.")
                self._create_graph()
            else:
                log.debug(f"We will NOT deinterlace (had {inter} interlaced frames and {ninter} non-interlaced frames)")
                self.interlaced = False
        else:
            self.interlaced = False
        log.debug(f"Video {filename} is {self.shape} at {float(self.frame_rate)} fps")
    
    def seek(self, seconds:float):
        if seconds > self.duration:
            seconds = self.duration
        vs = self.container.streams.video[self.streams.get('video', 0)]
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
        vs = self.container.streams.video[self.streams.get('video', 0)]
        self._flush()
        reseek = 0.5
        while True:
            if seconds <= 0.1:
                self.vpts = vs.start_time
                self.container.seek(vs.start_time, stream=vs)
                break
            else:
                pts = int(seconds / vs.time_base) + vs.start_time
                self.vpts = pts
                self.container.seek(pts, stream=vs, any_frame=True)
                vf = next(self.frames(), None)
                if vf and ((vf.time + 1/self.frame_rate) - self.vt_start) <= orig_ask:
                    break
                #print(f"trying to get to {orig_ask} at {seconds} got {vf.time-self.vt_start}")
                seconds -= reseek
                reseek += 0.5
        
        for f in self.frames():
            if (f.time - self.vt_start) >= orig_ask:
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
        if d.dtype.kind != 'f':
            d = d.astype('float32') / 32767.0
        if d.shape[0] == 1:
            x = []
            nc = len(af.layout.channels)
            for c in range(nc):
                x.append(d[c::nc])
            d = np.vstack(x)
        if d.shape[0] < 1:
            return
        
        n = 3 if d.shape[0] >= 3 and af.layout.channels[2].name.endswith('C') else 2
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
            self.aq.append((main,surr,af.time-self.at_start,af.sample_rate))

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
        self.container = av.open(self.filename)#, options={'probesize':'10000000'})
        self.container.gen_pts = True
        if self.trouble:
            self.container.discard_corrupt = True
        self.container.streams.video[0].thread_type = "AUTO"
        self.container.streams.video[0].thread_count = 2
        self.container.streams.audio[0].thread_type = "AUTO"
        self.container.streams.audio[0].thread_count = 2
        if pts is not None:
            self.container.seek(pts, stream=self.container.streams.video[0], any_frame=True, backward=False)

    def move_audio(self)->list[tuple[np.ndarray,float,list[float]]]:
        x = self.aq
        if x is not None:
            self.aq = []
        return x

    def frames(self) -> iter:
        fail = 0
        ovtp = self.vpts
        iter = self.container.decode(**self.streams)
        while True:
            try:
                frame = next(iter)
                if frame is None: continue
            except StopIteration:
                break
            except (av.error.InvalidDataError,av.error.UndefinedError) as e:
                fail += 1
                vs = self.container.streams.video[self.streams.get('video', 0)]
                self.vpts += math.ceil( (1.0/self.frame_rate)/vs.time_base )
                if fail%100 == 0:
                    log.debug(f"InvalidDataError during decode -- seeking ahead #{fail}, from {ovtp} to {self.vpts}")
                    if fail >= 1000:
                        log.critical(f"Repeated InvalidDataError, skipped {fail} frames but found nothing good")
                        import os
                        os._exit(134)
                        raise
                    if fail >= 500:
                        self.trouble = True
                    self._resync(self.vpts)
                else:
                    self.container.seek(self.vpts, stream=vs, any_frame=True, backward=False)
                iter = self.container.decode(**self.streams)
                continue
            
            if fail:
                if self.graph:
                    self._create_graph()
                log.info(f"Resync'd after {fail} skipped/dropped/corrupt/whatever frames")
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
                self.vpts = frame.pts
                yield frame
        
        return #raise StopIteration()
