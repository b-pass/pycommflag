import json
import logging as log
import numpy as np
import math
import os
from queue import Queue
import re
from threading import Thread
import time
import scipy.signal as spsig
from typing import Any,TextIO, TextIO

from . import logo_finder, mythtv
from .player import Player
from .extern import ina_foss
from .feature_span import *

def read_feature_log(feature_log_file:str|TextIO|dict) -> dict:
    if type(feature_log_file) is dict:
        return feature_log_file
    elif type(feature_log_file) is str:
        if feature_log_file.endswith('.gz'):
            import gzip
            feature_log_file = gzip.open(feature_log_file, 'r')
        else:
            feature_log_file = open(feature_log_file, 'r')
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
        if log_file.endswith('.gz'):
            import gzip
            log_file = gzip.open(log_file, 'w')
        else:
            log_file = open(log_file, 'w+')
    else:
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
    if np.empty(0, 'int'): raise Exception('antigravity') # get the stupid warning out early from numpy 1.13

    logo = None
    if type(feature_log) is str:
        if os.path.exists(feature_log) and not opts.no_logo:
            logo = logo_finder.from_json(read_feature_log(feature_log).get('logo', None))
            if logo and not opts.quiet:
                print(f"{feature_log} exists, re-using logo ", logo[0], logo[1])
        
        if feature_log.endswith('.gz'):
            import gzip
            feature_log = gzip.open(feature_log, 'w')
        else:
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
    
    audio_interval = round(player.frame_rate)

    audioProc = AudioProc(player.frame_rate)
    audioProc.start()

    videoProc = VideoProc(player.vt_start, logo, opts)
    videoProc.start()

    if not opts.quiet: print('\nExtracting features...', end='\r') 

    # not doing format/aspect because everything is widescreen all the time now
    # that was very early '00s... except ultra wide screen movies, and sometimes ultra-wide commercials?

    die = False
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
        print('\nInterrupt!!\n\n')
        die = True

    videoProc.stop()

    audioProc.add_audio(player.move_audio())
    audioProc.stop()

    videoProc.join()
    audioProc.join()

    frames = videoProc.frames
    if len(frames) == 0: frames = [[0,0,0,0]]
    
    audioProc.fspan.end(frames[-1][0])

    # often the video doesn't start a PTS 0 because of avsync issues, back-fill the first video frame
    # note, assumes time is at frame[0]
    if frames[0][0] > 0:
        while True:
            frames[0:0] = [list(frames[0])] # insert
            frames[0][0] = round(frames[0][0] - 1/player.frame_rate, 5)
            if frames[0][0] <= 0.0:
                frames[0][0] = 0.0
                break
    
    header = videoProc.frame_header()
    
    header += ['fvol', 'rvol']

    assert(AudioSegmentLabel.count() == 4)
    header += ['silence', 'speech', 'music', 'noise']

    sentinel = frames[-1][0]+1
    
    # normalize with the max volume of the whole recording
    vscale = np.max(np.array(audioProc.rms)[..., 1:3])
    vit = iter(audioProc.rms)
    volume = next(vit, (sentinel,0,0))

    ait = iter(audioProc.fspan.to_list())
    audio = next(ait, (AudioSegmentLabel.SILENCE,(0,sentinel)))
    
    for frame in frames:
        ft = frame[0]
        
        # mark up each frame with the calculated audio type and volume level
        while ft > volume[0]:
            volume = next(vit, (sentinel,0,0))
        frame += [volume[1]/vscale, volume[2]/vscale]
    
        # convert audio to one-hot and then add it
        while ft >= audio[1][1]:
            audio = next(ait, (AudioSegmentLabel.SILENCE,(0,sentinel)))
        x = [0] * AudioSegmentLabel.count()
        x[audio[0].value if ft >= audio[1][0] else AudioSegmentLabel.SILENCE.value] = 1
        frame += x

    feature_log.write(',\n"frames_header":')
    json.dump(header, feature_log)
    feature_log.write(',\n"frames":')
    json.dump(frames, feature_log)
    
    if die:
        feature_log.write("\n\n,\"EARLY_STOP_TRUNCATION\":true")

    feature_log.write('\n}\n')
    feature_log.flush()
    feature_log.close()

    if die:
        raise KeyboardInterrupt()
    
    if not opts.quiet:
        print('Extraction complete           ')

class VideoProc(Thread):
    def __init__(self, vt_start, logo, opts):
        super().__init__(name="videoProc")
        self.queue = Queue(150)
        self.go = True
        self.vt_start = vt_start
        self.fcount = 0
        self.logo = logo
        self.opts = opts
        self.prev_col = None
        self.lasttime = 0
        self.frames = []

    def stop(self):
        self.go = False
        self.queue.put(None)
        self.go = False
    
    def run(self):
        while True:
            x = self.queue.get()
            if x is not None:
                self._proc(*x)
            else:
                if self.queue.empty():
                    if not self.go:
                        break

    def add_frame(self, frame):
        fcolor = frame.to_ndarray(format="rgb24")#, height=720, width=frame.width*(720/frame.height))
        self.queue.put((frame,fcolor))
    
    def _proc(self, frame, fcolor):
        self.fcount += 1
        
        logo_present = logo_finder.check_frame(frame, self.logo)

        column = mean_axis1_float_uint8(fcolor).astype('int16')
        if self.prev_col is not None:
            diff = column - self.prev_col
            scm = np.mean(np.std(np.abs(diff), (0)))
            is_diff = scm >= self.opts.diff_threshold 
        else:
            is_diff = False
        self.prev_col = column

        # trying to be fast, just look at the middle 1/4 for blank-ish-ness and then verify with the full frame
        x = np.max(fcolor[int(fcolor.shape[0]*3/8):int(fcolor.shape[0]*5/8),int(fcolor.shape[1]*3/8):int(fcolor.shape[1]*5/8)])
        #print("at",frame.time-self.vt_start)
        #print("max=",x)
        if x < 64:
            bchk = logo_finder.subtract(fcolor, self.logo)
            m = np.median(bchk, (0,1))
            #print("median=",m,"maxmediam=",max(m),"stdmedian=",np.std(m),"allstd=",np.std(bchk))
            frame_blank = max(m) < 24 and np.std(m) < 3 and np.std(bchk) < 6
        else:
            frame_blank = False

        self.lasttime = round(frame.time-self.vt_start,5)
        self.frames.append([self.lasttime, int(logo_present), int(frame_blank), int(is_diff)])
        #f",[{self.lasttime},{int(logo_present)},{int(frame_blank)},{int(is_diff)}]\n"
    
    def frame_header(self):
        return ["time","logo_present","is_blank","is_diff"]

def mean_axis1_float_uint8(fcolor:np.ndarray)->np.ndarray:
    # the below code is equivalent to:
    #    return fcolor.mean(axis=(1),dtype='float32').astype('uint8')
    # but is almost TEN TIMES faster!
    
    # pick out the individual color channels by skipping by 3, and then average them
    cr = fcolor[...,0::3].mean(axis=(1), dtype='uint32')
    cg = fcolor[...,1::3].mean(axis=(1), dtype='uint32')
    cb = fcolor[...,2::3].mean(axis=(1), dtype='uint32')
    
    # and now convert those stacks back into a 720x3
    return np.stack((cb,cg,cr), axis=1).astype('uint8')

def columnize_frame(frame)->np.ndarray:
    return mean_axis1_float_uint8(frame.to_ndarray(format="rgb24", height=720, width=frame.width*(720/frame.height)))

class AudioProc(Thread):
    def __init__(self, frame_rate, volume_window=.1, work_rate=30.0):
        super().__init__(name="audioProc")
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        self.frame_rate = frame_rate
        self.volume_window = max(1/frame_rate, volume_window)
        self.work_rate = max(1.0, work_rate)
        self.seg = ina_foss.Segmenter()
        self.aq = []
        self.go = True
        self.fspan = AudioFeatureSpan()
        self.fspan.start()
        self.rms = [(0,0,0)]

    def add_audio(self, audio):
        self.aq += audio
    
    def stop(self):
        self.go = False
    
    def _resample_queue(self) -> list[tuple[float,np.ndarray,np.ndarray|None]]:
        # resample to 16 kHz

        if not self.aq:
            return []

        # out of order PTS, should never happen...?
        self.aq = sorted(self.aq, key=lambda x:x[2])
        result = []
        
        start = self.aq[0][2]
        main_samp = None
        surr_samp = None
        psr = None
        while self.aq:
            (main,surr,t,sr) = self.aq.pop(0)
            #print(t,sr,len(main),t+len(main)/sr)
            if psr != sr and main_samp is not None:
                result.append((
                    start, 
                    spsig.resample(main_samp, int(len(main_samp)*16000/psr)), 
                    spsig.resample(surr_samp, int(len(surr_samp)*16000/psr)) if surr_samp is not None else None,
                ))
                start = self.aq[0][2] if self.aq else -1 # peek next timestamp
                main_samp = None
                surr_samp = None
            psr = sr
            
            if surr is not None:
                if main_samp is not None:
                    # pad left with zeros as needed
                    if surr_samp is None:
                        surr_samp = np.zeros(int(len(main_samp)), 'float32')
                    elif len(main_samp) > len(surr_samp):
                        surr_samp = np.append(surr_samp, np.zeros(len(main_samp) - len(surr_samp), 'float32'))
                surr_samp = np.append(surr_samp, surr) if surr_samp is not None else surr
            main_samp = np.append(main_samp, main) if main_samp is not None else main
        
        if main_samp is not None:
            result.append((
                start, 
                spsig.resample(main_samp, int(len(main_samp)*16000/psr)), 
                spsig.resample(surr_samp, int(len(surr_samp)*16000/psr)) if surr_samp is not None else None,
            ))
        
        return result

    def run(self):
        main = np.empty(0, 'float32')
        surr = np.empty(0, 'float32')
        cur = 0.0
        nexttime = 0.0
        work_unit = round(16000 * self.work_rate) + 8000
        vwnd = round(16000*self.volume_window)
        done = False
        while not done:
            # first, resample the summarized audio packets into 16khz
            segments = self._resample_queue()
            if not segments:
                time.sleep(0.001)
                if self.go or self.aq:
                    continue
                else:
                    done = True
            
            # merge samples into a contiguous array
            for (st,sm,ss) in segments:
                # check for a hole, fill with zeros if needed
                missing = int(round((st - nexttime)*16000))
                nexttime = st + len(sm)/16000
                
                if missing > 0:
                    sm = np.append(sm, np.zeros(missing, 'float32'))

                # resample code might leave trailing holes in ss; so pad it if needed
                if ss is None:
                    ss = np.zeros(len(sm), 'float32')
                elif len(sm) > len(ss):
                    ss = np.append(ss, np.zeros(len(sm) - len(ss), 'float32'))
                
                main = np.append(main, sm)
                surr = np.append(surr, ss)
            
            # now chunk into "work_rate" sized pieces and work on them individually
            while len(main) >= work_unit or (done and len(main) > 8000):
                assert(len(main) == len(surr))

                # slice the time
                mwork = main[0:work_unit]
                swork = surr[0:work_unit]
                
                # calculate the volume via RMS for both the main and surround in small rolling slices
                rt = round(cur + self.volume_window, 5)
                x = vwnd
                while x <= len(mwork):
                    if rt > self.rms[-1][0]:
                        mrms = math.sqrt(np.mean(np.square(mwork[x-vwnd:x])))
                        srms = math.sqrt(np.mean(np.square(swork[x-vwnd:x])))
                        self.rms.append((round(rt,5), round(mrms,5), round(srms,5)))
                    x += vwnd//2
                    rt += self.volume_window/2
                
                # classify the main channel
                for (lab, sb, se) in self.seg(mwork):
                    if cur == 0 or se > 0.5:
                        self.fspan.add(round(cur+sb,5), round(cur+se,5), AudioSegmentLabel[lab])
                
                # done with this time slice
                cur += self.work_rate-0.5
                main = main[work_unit-8000:]
                surr = surr[work_unit-8000:]

def reprocess(feature_log_filename:str, opts:Any=None) -> dict:
    if feature_log_filename is None:
        raise Exception('missing feature_log_filename')
    # deprecated?
    return read_feature_log(feature_log_filename)

def read_logo(log_in:str|TextIO|dict) -> None|tuple:
    return logo_finder.from_json(read_feature_log(log_in).get('logo', None))

def read_tags(log_f:str|TextIO|dict):
    return read_feature_log(log_f).get('tags', [])

def read_feature_spans(log:str|TextIO|dict) -> dict[str, FeatureSpan]:
    log = read_feature_log(log)

    audiof = AudioFeatureSpan()
    audiof.start()
    logof = FeatureSpan()
    logof.start(0,False)
    blankf = FeatureSpan()
    blankf.start(0,True)
    difff = SeparatorFeatureSpan()
    difff.start(0,True)
    volume = []
    
    lasttime = 0
    lab = AudioSegmentLabel.SILENCE

    for f in log['frames']:
        audiof.add(lasttime, f[0], lab)
        
        lasttime = f[0]
        logof.add(lasttime,f[1])
        blankf.add(lasttime, f[2])
        difff.add(lasttime, f[3])
        volume.append((lasttime, f[4], f[5]))
        for i in range(AudioSegmentLabel.count()):
            if f[6+i]:
                lab = AudioSegmentLabel(i)
    
    logof.end(lasttime)
    blankf.end(lasttime)
    difff.end(lasttime)
    audiof.end(lasttime)

    return {
        'logo':logof.to_list(),
        'blank':blankf.to_list(),
        'diff':difff.to_list(),
        'audio':audiof.to_list(),
        'volume':volume,
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
