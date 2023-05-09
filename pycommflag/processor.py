import logging as log
from queue import Queue
import struct
import sys
import os
import re
from threading import Thread
import numpy as np
import math
import time
import gc

from typing import Any,BinaryIO

from . import logo_finder, mythtv, segmenter
from .player import Player
from .scene import Scene, SceneType
from .extern import ina_foss

FLOG_VERSION = 3

def process_video(video_filename:str, feature_log:str|BinaryIO, opts:Any=None) -> None:
    logo = None
    if type(feature_log) is str:
        if os.path.exists(feature_log) and not opts.no_logo:
            with open(feature_log, 'r+b') as tfl:
                tfl.seek(28)
                logo = logo_finder.read(tfl)
                if logo and not opts.quiet:
                    print(f"{feature_log} exists, re-using logo ", logo[0], logo[1])
        feature_log = open(feature_log, 'w+b')
    
    player = Player(video_filename, no_deinterlace=opts.no_deinterlace)

    if opts.no_logo:
        logo = None
    elif logo is not None:
        pass
    else:
       logo = logo_finder.search(player, opts=opts)
    
    player.seek(0)
    player.enable_audio()
    player.seek(0)

    # not doing format/aspect because everything is widescreen all the time now (that was very early '00s)
    # except ultra wide screen movies, and sometimes ultra-wide commercials?

    fcount = 0
    ftotal = int(player.duration * player.frame_rate)

    percent = ftotal/100.0
    report = math.ceil(ftotal/1000) if not opts.quiet else ftotal*10
    rt = time.perf_counter()
    p = 0

    fcolor = None
    feature_log.seek(0)
    feature_log.write(struct.pack('@Iff', FLOG_VERSION, player.duration, player.frame_rate))
    feature_log.write(struct.pack('IIII', 0, 0, 0, 0))

    logo_finder.write(feature_log, logo)
    
    pos = feature_log.tell()
    feature_log.seek(12, 0)
    feature_log.write(struct.pack('I', pos))
    feature_log.seek(pos)

    audio_interval = int(player.frame_rate * 60)

    audioProc = AudioProc()
    audioProc.start()

    videoProc = VideoProc(feature_log, player.vt_start, logo, opts)
    videoProc.start()

    if not opts.quiet: print('\nExtracting features...', end='\r') 

    try:
      for frame in player.frames():
        p += 1
        fcount += 1
        if p >= report:
            ro = rt
            rt = time.perf_counter()
            print("Extracting, %5.1f%% (%5.1f fps)           " % (min(fcount/percent,100.0), p/(rt - ro)), end='\r')
            #gc.collect()
            p = 0
        if fcount%audio_interval == 0:
            audioProc.add_audio(player.move_audio())
        videoProc.add_frame(frame)
    except KeyboardInterrupt:    
        videoProc.stop()
        audioProc.stop()
        raise

    videoProc.stop()

    audioProc.add_audio(player.move_audio())
    audioProc.stop()

    videoProc.join()
    
    feature_log.write(struct.pack('I', 0xFFFFFFFF))

    pos = feature_log.tell()
    feature_log.seek(16, 0)
    feature_log.write(struct.pack('I', pos))
    feature_log.seek(pos)

    audioProc.join()
    feature_log.write(struct.pack('I', len(audioProc.audio)))
    for (lab,start,stop) in audioProc.audio:
        feature_log.write(struct.pack('Iff', ina_foss.AudioSegmentLabel[lab].value, start, stop))

    pos = feature_log.tell()
    feature_log.seek(20, 0)
    feature_log.write(struct.pack('@I', pos))
    feature_log.seek(pos)
    
    if not opts.quiet:
        print('Extraction complete           ')

class VideoProc(Thread):
    def __init__(self, feature_log, vt_start, logo, opts):
        super().__init__()
        self.queue = Queue(100)
        self.go = True
        self.feature_log = feature_log
        self.vt_start = vt_start
        self.fcount = 0
        self.logo = logo
        self.opts = opts
        self.prev_col = None

    def add_frame(self,frame):
        self.queue.put(frame)
    
    def stop(self):
        self.go = False
        self.queue.put(None)
        self.go = False
    
    def run(self):
        while True:
            frame = self.queue.get()
            if frame is not None:
                self._proc(frame)
            else:
                if self.queue.empty():
                    if not self.go:
                        break

    def _proc(self, frame):
        self.fcount += 1
        
        logo_present = logo_finder.check_frame(frame, self.logo)

        fcolor = frame.to_ndarray(format="rgb24", height=720, width=frame.width*(720/frame.height))
        column = mean_axis1_float_uint8(fcolor)
        
        # trying to be fast, just look at the middle 1/4 for blank-ish-ness and then verify with the full frame
        x = np.max(fcolor[int(fcolor.shape[0]*3/8):int(fcolor.shape[0]*5/8)])
        if x < 64:
            m = np.median(fcolor, (0,1))
            frame_blank = max(m) < 24 and np.std(m) < 3 and np.std(fcolor) < 6
        else:
            frame_blank = False

        if not self.opts.delay_diff:
            column = column.astype('int16')
            if self.prev_col is not None:
                diff = column - self.prev_col
                scm = np.mean(np.std(np.abs(diff), (0)))
                scene_change = scm >= self.opts.scene_threshold 
            else:
                scene_change = False
            self.prev_col = column

        self.feature_log.write(struct.pack(
            'If???BII',
            self.fcount,
            frame.time-self.vt_start,
            logo_present, frame_blank, scene_change,
            column.shape[2] if self.opts.delay_diff else 0, 
            column.shape[0], column.shape[1]
        ))
        
        if self.opts.delay_diff:
            column.astype('uint8').tofile(self.feature_log)

def mean_axis1_float_uint8(fcolor:np.ndarray)->np.ndarray:
    # the below code is equivalent to:
    #   return fcolor.mean(axis=(1),dtype='float32').astype('uint8')
    # but is almost TEN TIMES faster!
    
    # pick out the individual color channels by skipping by 3, and then average them
    cr = fcolor[...,0::3].mean(axis=(1), dtype='float32')
    cg = fcolor[...,1::3].mean(axis=(1), dtype='float32')
    cb = fcolor[...,2::3].mean(axis=(1), dtype='float32')
    
    # and now convert those stacks back into a 720x3
    return np.stack((cb,cg,cr), axis=1).astype('uint8')

def columnize_frame(frame)->np.ndarray:
    return mean_axis1_float_uint8(frame.to_ndarray(format="rgb24", height=720, width=frame.width*(720/frame.height)))

class AudioProc(Thread):
    def __init__(self):
        super().__init__()
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        self.seg = ina_foss.Segmenter(energy_ratio=0.05)
        self.queue = Queue(10)
        self.go = True
        self.audio = []
        self.prev = None

    def add_audio(self,segments):
        self.queue.put(segments)
    
    def stop(self):
        self.go = False
        self.queue.put([])
        self.go = False
    
    def run(self):
        while True:
            segments = self.queue.get()
            if not segments:
                if self.queue.empty():
                    if not self.go:
                        break
                continue

            isstart = False
            if self.prev is None:
                self.prev = segments.pop(0)
                isstart = True
            
            all = self.prev[0]
            start = self.prev[1]
            time = self.prev[1] + len(self.prev[0])/16000
            for s in segments:
                missing = int(round((s[1] - time)*16000))
                if missing > 0:
                    all = np.append(all, np.zeros(missing, 'float32'))
                all = np.append(all, s[0])
                time = s[1] + len(s[0])/16000
            
            self.prev = (all[-8000:], start+(len(all)-8000)/16000)

            startbound = 0 if isstart else 0.5
            for (lab, sb, se) in self.seg(all):
                if se <= startbound:
                    continue
                old = self.audio[-1] if self.audio else (0,0,-1)
                if old[1] <= (start+sb) and (start+sb) < old[2]:
                    self.audio[-1] = (old[0],old[1],(start+sb)) # overlapping, truncate old
                self.audio.append((lab, start+sb, start+se))


def read_logo(log_in:str|BinaryIO) -> None|tuple:
    if type(log_in) is str:
        with open(log_in, 'r+b') as fd:
            return read_logo(fd)
    
    (ver,duration,frame_rate) = struct.unpack('@Iff', log_in.read(12))
    if ver != FLOG_VERSION:
        raise RuntimeError("Unsupported feature log version: "+str(ver))
    log_in.read(16) # poses

    return logo_finder.read(log_in)

def segment_scenes(log_f:str|BinaryIO, opts:Any) -> list[Scene]:
    if type(log_f) is str:
        with open(log_f, 'r+b') as fd:
            return segment_scenes(fd, opts=opts)

    if not opts.quiet: print("Segmenting feature log into scenes...")
    log_f.seek(0)

    fseg = segmenter.parse(opts.segmeth)
    
    (ver,duration,frame_rate) = struct.unpack('@Iff', log_f.read(12))
    if ver != FLOG_VERSION:
        raise RuntimeError("Unsupported feature log version: "+str(ver))

    (frame_pos,audio_pos,scene_pos,x_pos) = struct.unpack('IIII', log_f.read(16))
    
    logo_finder.read(log_f)

    audio = read_audio(log_f)
    if not audio:
        audio = [(ina_foss.AudioSegmentLabel.SILENCE, 0, duration)]
    audio_idx = 0

    log_f.seek(frame_pos)

    fprev = 0
    column = None
    scenes = []
    scene = None
    temp_set = []
    blanks = []
    prev_check = None
    while True:
        d = log_f.read(4)
        if len(d) < 4:
            break
        (fnum,) = struct.unpack('I', d)
        if fnum-1 != fprev:
            log.error(f'Bad frame number {fnum} (expected {fprev+1}) at {log_f.tell()}')
        if fnum == 0xFFFFFFFF:
            break
        
        fprev = fnum
        prev_col = column

        (ftime, logo_present, frame_blank, is_diff, depth, h, w) = struct.unpack('f???BII', log_f.read(16))
        if depth > 0:
            column = np.fromfile(log_f, 'uint8', depth*h*w, '').astype('int16')
            column.shape = (h,w,depth)
        else:
            column = None

        while audio_idx+1 < len(audio) and audio[audio_idx][2] <= ftime:
            audio_idx += 1
        if ftime >= audio[audio_idx][1] and ftime < audio[audio_idx][2]:
            faudio = audio[audio_idx][0]
        else:
            faudio = ina_foss.AudioSegmentLabel.SILENCE

        if prev_col is None:
            pass # use is_diff from file
        else:
            diff = column - prev_col
            scm = np.mean(np.std(np.abs(diff), (0)))
            is_diff = scm >= opts.scene_threshold 
        
        # When the segmenter returns True we end the scene and start another
        # But if it returns consecutive Trues then we combine those into one Scene
        # The subsequent Scene starts with the last True
        # this also tends to keep the blanks together in a scene when using the 'blank' segmenter
        # Also, Every scene should be at least 2 frames
        if fseg.check(ftime=ftime, faudio=faudio, logo_present=logo_present, is_blank=frame_blank, is_diff=is_diff):
            if scene is not None:
                scene.finish()
                scenes.append(scene)
                scene = None
            temp_set.append((ftime, faudio, logo_present, frame_blank, is_diff))
        elif scene is None:
            if len(temp_set) > 2:
                scene = Scene(*temp_set[0])
                for x in temp_set[1:-1]:
                    scene.add(*x)
                scene.finish()
                scenes.append(scene)
                scene = None
                temp_set = temp_set[-1:]
            if temp_set:
                scene = Scene(*temp_set[0])
                for x in temp_set[1:]:
                    scene.add(*x)
                temp_set = []
                scene.add(ftime, faudio, logo_present, frame_blank, is_diff)
            else:
                scene = Scene(ftime, faudio, logo_present, frame_blank, is_diff)
        else:
            scene.add(ftime, faudio, logo_present, frame_blank, is_diff)
    
    if scene is None:
         scene = Scene(*temp_set[0])
         for x in temp_set[1:]:
            scene.add(*x)
    scene.finish()
    scenes.append(scene)

    if not opts.quiet: print("Done, got",len(scenes),"scenes in",fprev,"frames")

    #for s in scenes: print(s)
    
    if scene_pos:
        log_f.seek(scene_pos)
        log_f.truncate(scene_pos)
        rewrite_scenes(scenes, log_f, opts=opts)
    
    return scenes

def read_scenes(log_f:str|BinaryIO) -> None:
    if type(log_f) is str:
        with open(log_f, 'r+b') as fd:
            return read_scenes(fd)
    
    log_f.seek(20)
    (scene_pos,) = struct.unpack('I', log_f.read(4))
    if scene_pos < 20:
        return []
    
    log_f.seek(scene_pos)
    b = log_f.read(4)
    if len(b) == 0:
        return []
    scenes = []
    (count,) = struct.unpack('I', b)
    for _ in range(count):
        scenes.append(Scene(infile=log_f))
    
    # Special, short scenes at the start are basically truncated so don't use them for training
    if len(scenes) > 2:
        if scenes[0].duration < 5:
            scenes[0].type = SceneType.DO_NOT_USE
        
        # Special, blank segments at show/commercial boundaries are always counted as show for ML
        # When we write out results for playback we choose a good break point in the middle of the blank scene
        for i in range(1,len(scenes)-1):
            if scenes[i].blank >= .975 and scenes[i].type == SceneType.COMMERCIAL:
                if scenes[i-1].type == SceneType.SHOW or scenes[i+1].type == SceneType.SHOW:
                    scenes[i].type = SceneType.SHOW
        
        # Special, short scenes at the end are basically truncated so don't use them for training
        if scenes[-1].duration < 5:
            scenes[-1].type = SceneType.DO_NOT_USE

    return scenes

def read_audio(log_f:str|BinaryIO) -> None:
    if type(log_f) is str:
        with open(log_f, 'r+b') as fd:
            return read_audio(fd)
    
    log_f.seek(16)
    (audio_pos,) = struct.unpack('I', log_f.read(4))
    if audio_pos < 20:
        return []
    
    log_f.seek(audio_pos)
    (n,) = struct.unpack('I', log_f.read(4))
    audio = []
    for i in range(n):
        (t,b,e) = struct.unpack('Iff', log_f.read(12))
        if audio and audio[-1][1] <= b and audio[-1][2] > b:
            old = audio.pop()
            audio.append((old[0],old[1],b)) # overlapping, truncate old
        audio.append((t,b,e))
    
    return audio

def rewrite_scenes(scenes:list[Scene], log_f:str|BinaryIO, opts=None) -> None:
    # TODO should we write the segmenter into the scenes? Training with two different segmenters might not make sense
    if type(log_f) is str:
        with open(log_f, 'r+b') as fd:
            return rewrite_scenes(scenes, fd)
    
    log_f.seek(20)
    (scene_pos,) = struct.unpack('I', log_f.read(4))
    if scene_pos < 20:
        if not scenes:
            return
        log_f.seek(0, 2)
        scene_pos = log_f.tell()
        log_f.seek(20)
        log_f.write(struct.pack('I', scene_pos))
    log_f.seek(scene_pos)
    
    log_f.write(struct.pack('I', len(scenes)))
    if len(scenes) > 2:
        if scenes[0].duration < 5:
            scenes[0].type = scenes[0].newtype = SceneType.DO_NOT_USE
        if scenes[-1].duration < 5:
            scenes[-1].type = scenes[-1].newtype = SceneType.DO_NOT_USE
    for s in scenes:
        s.write_bin(log_f)

def guess_external_breaks(opts:Any):
    if not opts:
        return None
    
    if opts.chanid and opts.starttime:
        return mythtv.get_breaks(opts.chanid, opts.starttime)

    if opts.filename:
        if m:=re.match(r'(?:.*/)?(\d{4,6})_(\d{12,})\.[a-zA-Z0-9]{2,5}', opts.filename):
            mtb = mythtv.get_breaks(m[1], m[2])
            if mtb:
                return mtb
    
    if opts.feature_log:
        if m:=re.match(r'(?:.*/)?cf_(\d{4,6})_(\d{12,})\.[a-zA-Z0-9]{2,5}\.feat', opts.feature_log):
            mtb = mythtv.get_breaks(m[1], m[2])
            if mtb:
                return mtb
    
    if opts.comm_file:
        if m:=re.search(r'\D(\d{4,6})[_.-](\d{12,})\D', opts.comm_file):
            mtb = mythtv.get_breaks(m[1], m[2])
            if mtb:
                return mtb

    if opts.comm_file and os.path.exists(opts.comm_file):
        with open(opts.comm_file, 'r') as cf:
            # TODO: more formats?
            magic = cf.read(2)
            marks = []
            fps = None
            if magic == '# ':
                markre = re.compile(r'\s*framenum:\s*(\d+)\s*marktype:\s*(\d+)\s*')
                for line in cf.readlines():
                    if fps is None and line[0] == 'F':
                        if m := re.match(r'FPS\s*=\s*(\d*\.\d*)\s*', line):
                            fps = float(m[1])
                    if m := markre.match(line):
                        mv = int(m[1])
                        mt = int(m[2])
                        if mt == 4:
                            marks.append((mv,None))
                        elif mt == 5:
                            marks[-1] = (marks[-1][0], mv)
            elif magic == 'FI':
                if m := re.match(r'^(?:FI)?LE\s*PROCESSING\s*COMPLETE\s*\d+\s*FRAMES\s*(?:AT\s*)?\s*(\d+).*?$', cf.readline()):
                    if m[1]:
                        fps = float(m[1])/100
                markre = re.compile('(\d+)\s*(\d+)\s*')
                for line in cf.readlines():
                    if line[0:4] in ('FILE','----'):
                        continue
                    if m:=markre.match(line):
                        marks.append((int(m[1]),int(m[2])))
            else:
                log.info(f"Unlnowmn comm break text file format '{magic}'")
                return []
            if fps is None:
                fps = 29.97
            ret = []
            for (a,b) in marks:
                if a is not None and b is not None and a < b:
                    ret.append((a/fps, b/fps))
            return ret
    
    return None

def external_scene_tags(scenes:list[Scene],marks:list[tuple[float,float]]=None,opts:Any=None)->None:
    if marks is None:
        if opts is not None:
            marks = guess_external_breaks(opts)
        if marks is None:
            log.info('No external source of marks/breaks available')
            return
    
    for (start,stop) in marks:
        first = True
        for s in scenes:
            if s.start_time >= stop:
                break
            elif s.duration >= 1 and s.duration < 10 and (stop - s.start_time) < s.duration/2:
                # if it ends just inside this scene then don't count it
                break
            if s.start_time <= stop and s.stop_time > start:
                if first and s.duration >= 1 and s.duration < 10 and (s.stop_time - start) < s.duration/2:
                    # if the tag is near the end of the scene then don't count it
                    continue
                first = False
                s.is_break = True
