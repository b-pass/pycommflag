import json
import logging as log
import numpy as np
import math
import os
from queue import Queue
import re
from threading import Thread
import time
from typing import Any,TextIO, TextIO

from . import logo_finder, mythtv
from .player import Player
from .extern import ina_foss
from .feature_span import *

def read_feature_log(feature_log_file:str|TextIO|dict) -> dict:
    if type(feature_log_file) is dict:
        return feature_log_file
    elif type(feature_log_file) is str:
        with open(feature_log_file, 'r') as fl:
            return read_feature_log(fl)
    else:
        feature_log_file.seek(0)
        try:
            return json.load(feature_log_file)
        except json.JSONDecodeError:
            # partial file, probably closed in the middle of a frame array; try to recover....
            feature_log_file.seek(0)
            return json.loads(feature_log_file.read() + ']}')

def write_feature_log(flog:dict, log_file:str|TextIO):
    if type(log_file) is str:
        with open(log_file, 'w+') as fd:
            return write_feature_log(flog, fd)
    log_file.seek(0)
    log_file.truncate()
    log_file.write('{\n')
    first = True
    for (k,v) in flog.items():
        if first:
            first = False
        else:
            log_file.write(',\n')
        log_file.write(f'"{k}" : {json.dumps(v)}')
    log_file.write("\n}\n")

def process_video(video_filename:str, feature_log:str|TextIO, opts:Any=None) -> None:
    logo = None
    if type(feature_log) is str:
        if os.path.exists(feature_log) and not opts.no_logo:
            logo = logo_finder.from_json(read_feature_log(feature_log).get('logo', None))
            if logo and not opts.quiet:
                print(f"{feature_log} exists, re-using logo ", logo[0], logo[1])
        feature_log = open(feature_log, 'w+')
    else:
        feature_log.seek(0)
        feature_log.truncate()

    feature_log.write('{ "file_version":10')

    if opts.chanid: feature_log.write(f',\n"chanid":"{opts.chanid}"')
    if opts.starttime: feature_log.write(f',\n"starttime":"{opts.starttime}"')
    try:
        feature_log.write(f',\n"filename":"{os.path.realpath(opts.filename)}"')
    except:
        pass

    player = Player(video_filename, no_deinterlace=opts.no_deinterlace)

    if opts.no_logo:
        logo = None
    elif logo is not None:
        pass
    else:
        logo = logo_finder.search(player, opts=opts)
        player.seek(0)
    
    player.enable_audio()

    feature_log.write(f',\n"duration":{float(player.duration)}')
    feature_log.write(f',\n"frame_rate":{round(float(player.frame_rate),4)}')

    feature_log.write(',\n"logo":')
    feature_log.write(logo_finder.to_json(logo))

    fcount = 0
    ftotal = int(player.duration * player.frame_rate)

    percent = ftotal/100.0
    report = math.ceil(ftotal/1000) if not opts.quiet else ftotal*10
    rt = time.perf_counter()
    p = 0
    
    audio_interval = int(player.frame_rate * 60)

    audioProc = AudioProc()
    audioProc.start()

    videoProc = VideoProc(feature_log, player.vt_start, logo, opts)
    videoProc.start()

    feature_log.write(',\n"frames_header":' + videoProc.frame_header())
    feature_log.write(',\n"frames":[null\n')
    
    if not opts.quiet: print('\nExtracting features...', end='\r') 

    # not doing format/aspect because everything is widescreen all the time now (that was very early '00s)
    # except ultra wide screen movies, and sometimes ultra-wide commercials?

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
        videoProc.join()
        audioProc.join()
        feature_log.write("\n]}")
        raise

    videoProc.stop()

    audioProc.add_audio(player.move_audio())
    audioProc.stop()

    videoProc.join()
    
    feature_log.write("]")
    
    feature_log.write(',\n"logo_span":')
    feature_log.write(videoProc.logof.to_json())
    
    feature_log.write(',\n"blank_span":')
    feature_log.write(videoProc.blankf.to_json())
    
    feature_log.write(',\n"diff_span":')
    feature_log.write(videoProc.difff.to_json())

    audioProc.join()
    audioProc.fspan.end(player.duration)

    feature_log.write(',\n"audio":')
    feature_log.write(audioProc.fspan.to_json())

    feature_log.write('\n}\n')
    feature_log.flush()
    
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
        
        self.lasttime = 0
        self.logof = FeatureSpan()
        self.logof.start(0,False)
        self.blankf = FeatureSpan()
        self.blankf.start(0,True)
        self.difff = SeparatorFeatureSpan()
        self.difff.start(0,True)

    def stop(self):
        self.go = False
        self.queue.put(None)
        self.go = False
    
    def run(self):
        self.name = 'videoProc'
        while True:
            x = self.queue.get()
            if x is not None:
                self._proc(*x)
            else:
                if self.queue.empty():
                    if not self.go:
                        break
        
        self.logof.end(self.lasttime)
        self.blankf.end(self.lasttime)
        self.difff.end(self.lasttime)

    def add_frame(self, frame):
        fcolor = frame.to_ndarray(format="rgb24", height=720, width=frame.width*(720/frame.height))
        self.queue.put((frame,fcolor))
    
    def _proc(self, frame, fcolor):
        self.fcount += 1
        
        logo_present = logo_finder.check_frame(frame, self.logo)

        # trying to be fast, just look at the middle 1/4 for blank-ish-ness and then verify with the full frame
        x = np.max(fcolor[int(fcolor.shape[0]*3/8):int(fcolor.shape[0]*5/8)])
        if x < 64:
            m = np.median(fcolor, (0,1))
            frame_blank = max(m) < 24 and np.std(m) < 3 and np.std(fcolor) < 6
        else:
            frame_blank = False

        column = mean_axis1_float_uint8(fcolor).astype('int16')
        if self.prev_col is not None:
            diff = column - self.prev_col
            scm = np.mean(np.std(np.abs(diff), (0)))
            is_diff = scm >= self.opts.diff_threshold 
        else:
            is_diff = False
        self.prev_col = column

        self.lasttime = round(frame.time-self.vt_start,5)
        self.logof.add(self.lasttime, logo_present)
        self.blankf.add(self.lasttime, frame_blank)
        self.difff.add(self.lasttime, is_diff)
        self.feature_log.write(
            f",[{self.lasttime},{int(logo_present)},{int(frame_blank)},{int(is_diff)}]\n"
        )
    
    def frame_header(self):
        return '["time","logo_present","is_blank","is_diff"]'

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
        self.fspan = AudioFeatureSpan()
        self.fspan.start()

    def add_audio(self,segments):
        self.queue.put(segments)
    
    def stop(self):
        self.go = False
        self.queue.put([])
        self.go = False
    
    def run(self):
        self.name = 'audioProc'
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
                if se > startbound:
                    self.fspan.add(round(start+sb,5), round(start+se,5), AudioSegmentLabel[lab])

def reprocess(feature_log_filename:str, opts:Any=None) -> dict[str, FeatureSpan]:
    flog = read_feature_log(feature_log_filename)

    lasttime = 0
    logof = FeatureSpan()
    logof.start(0,False)
    blankf = FeatureSpan()
    blankf.start(0,True)
    difff = SeparatorFeatureSpan()
    difff.start(0,True)

    for f in flog['frames']:
        if f is None:
            continue

        lasttime = f[0]
        logof.add(lasttime,f[1])
        blankf.add(lasttime, f[2])
        difff.add(lasttime, f[3])
    
    logof.end(lasttime)
    blankf.end(lasttime)
    difff.end(lasttime)

    audiof = AudioFeatureSpan()
    audiof.start()
    for (t,b,e) in read_audio(flog):
        audiof.add(b,e,t)
    audiof.end(lasttime)
    
    flog['logo_span'] = logof.to_list()
    flog['blank_span'] = blankf.to_list()
    flog['diff_span'] = difff.to_list()
    flog['audio'] = audiof.to_list(serializable=True)

    write_feature_log(flog, feature_log_filename)

    return {
        'logo':logof.to_list(),
        'blank':blankf.to_list(),
        'diff':difff.to_list(),
        'audio':audiof.to_list(),
    }

def read_logo(log_in:str|TextIO|dict) -> None|tuple:
    return logo_finder.from_json(read_feature_log(log_in).get('logo', None))

def read_audio(log_f:str|TextIO|dict) -> list[tuple]:
    audiof = AudioFeatureSpan()
    audiof.from_json(read_feature_log(log_f)['audio'], AudioSegmentLabel)

    return [(x,y,z) for (x,(y,z)) in audiof.to_list()]

def read_tags(log_f:str|TextIO|dict):
    return read_feature_log(log_f).get('tags', [])

def read_feature_spans(log:str|TextIO|dict) -> dict[str, FeatureSpan]:
    log = read_feature_log(log)

    if 'audio' not in log:
        return {}

    audiof = AudioFeatureSpan()
    audiof.from_json(log['audio'], AudioSegmentLabel)

    logof = FeatureSpan()
    logof.from_json(log['logo_span'], bool)
    
    blankf = FeatureSpan()
    blankf.from_json(log['blank_span'], bool)
    
    difff = SeparatorFeatureSpan()
    difff.from_json(log['diff_span'], bool)

    return {
        'logo':logof.to_list(),
        'blank':blankf.to_list(),
        'diff':difff.to_list(),
        'audio':audiof.to_list(),
    }

def guess_external_breaks(opts:Any)->list:
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
        if m:=re.match(r'(?:.*/)?cf_(\d{4,6})_(\d{12,})(?:\.[a-zA-Z0-9]{2,5})+', opts.feature_log):
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

def external_tags(marks:list[tuple[float,float]]=None,opts:Any=None,duration:float|None=None)->FeatureSpan:
    if marks is None:
        if opts is not None:
            marks = guess_external_breaks(opts)
        if marks is None:
            log.info('No external source of marks/breaks available')
            return None
    
    markf = FeatureSpan()
    markf.start(0, SceneType.SHOW)
    for (start,stop) in marks:
        markf.add(start, SceneType.COMMERCIAL)
        markf.add(stop, SceneType.SHOW)
    if duration:
        markf.end(duration)
    return markf
