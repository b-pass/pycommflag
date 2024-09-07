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
    # if we ever implement "live" processing then don't do this...
    
    # yield CPU time to useful tasks, this is a background thing.
    os.nice(10)

    logo = None
    if type(feature_log) is str:
        if os.path.exists(feature_log) and not opts.no_logo:
            logo = logo_finder.from_json(read_feature_log(feature_log).get('logo', None))
            if logo and not opts.quiet:
                log.info(f"{feature_log} exists, re-using logo {logo[0]},{logo[1]}")

    player = Player(video_filename, no_deinterlace=opts.no_deinterlace)

    if opts.no_logo:
        logo = None
    elif logo is not None:
        pass
    else:
        logo = logo_finder.search(player, opts=opts)
        player.seek(0)
    
    if type(feature_log) is str:
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

    feature_log.write(f',\n"duration":{float(player.duration)}')
    feature_log.write(f',\n"frame_rate":{round(float(player.frame_rate),4)}')

    feature_log.write(',\n"logo":')
    feature_log.write(logo_finder.to_json(logo))

    start = time.time()
    
    fcount = 0
    ftotal = int(player.duration * player.frame_rate)+1

    percent = ftotal/100.0
    report = math.ceil(ftotal/1000) if not opts.quiet else ftotal*10
    rt = time.perf_counter()
    p = 0
    
    player.enable_audio()
    audio_interval = round(player.frame_rate)
    audioProc = AudioProc()
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
            perc = min(fcount/percent,100.0)
            timeleft = round( (100.0 - perc) * ((time.time() - start) / perc) )+1
            print("Extracting, %5.1f%% @%5.1f fps, %4d seconds left               " % (perc, p/(rt - ro), timeleft), end='\r')
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

    # often the video doesn't start a PTS 0 because of avsync issues, back-fill the first video frame
    #if frames[0][0] > 0:
    #    while True:
    #        frames[0:0] = [list(frames[0])] # insert
    #        frames[0][0] = round(frames[0][0] - 1/player.frame_rate, 5)
    #        if frames[0][0] <= 0.0:
    #            frames[0][0] = 0.0
    #            break
    
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
        frame += [round(float(volume[1]/vscale),6), round(float(volume[2]/vscale),6)]
    
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
        self.queue = Queue(300)
        self.vt_start = vt_start
        self.fcount = 0
        self.logo = logo
        self.opts = opts
        self.prev_col = None
        self.lasttime = 0
        self.frames = []

    def stop(self):
        self.queue.put(None)
    
    def run(self):
        while True:
            x = self.queue.get()
            if x is not None:
                self._proc(*x)
            else:
                break

    def add_frame(self, frame):
        fcolor = frame.to_ndarray(format="rgb24")#, height=720, width=frame.width*(720/frame.height))
        logo_present = logo_finder.check_frame(frame, self.logo)
        self.queue.put((frame.time,fcolor,logo_present))
    
    def _proc(self, ftime, fcolor, logo_present):
        self.fcount += 1

        column = mean_axis1_float_uint8(fcolor).astype('int16')
        if self.prev_col is not None:
            diff = column - self.prev_col
            diff = np.mean(np.std(np.abs(diff), (0)))
            #is_diff = diff >= self.opts.diff_threshold 
        else:
            diff = 0
        self.prev_col = column

        # trying to be fast, just look at the middle 1/4 for blank-ish-ness and then verify with the full frame
        x = np.max(fcolor[int(fcolor.shape[0]*3/8):int(fcolor.shape[0]*5/8),int(fcolor.shape[1]*3/8):int(fcolor.shape[1]*5/8)])
        #print("at",frame.time-self.vt_start)
        #print("max=",x)
        if x < 32:
            fcolor = logo_finder.subtract(fcolor, self.logo)
            m = np.median(fcolor, (0,1))
            #print("median=",m,"maxmediam=",max(m),"stdmedian=",np.std(m),"allstd=",np.std(fcolor))
            frame_blank = max(m) < 24 and np.std(m) < 3 and np.std(fcolor) < 6
            fcolor = None
        else:
            frame_blank = False

        self.lasttime = round(ftime-self.vt_start,5)
        self.frames.append([self.lasttime, int(logo_present), int(frame_blank), round(diff, 6)])
        #f",[{self.lasttime},{int(logo_present)},{int(frame_blank)},{int(is_diff)}]\n"
    
    def frame_header(self):
        return ["time","logo_present","is_blank","diff"]

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

class AudioProc(Thread):
    def __init__(self, volume_window=.05, work_rate=60.0):
        super().__init__(name="audioProc")
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        self.volume_window = max(2/29.97, volume_window)
        self.work_rate = max(1.0, work_rate)
        self.seg = ina_foss.Segmenter()
        self.queue = Queue(300)
        self.fspan = AudioFeatureSpan()
        self.fspan.start()
        self.rms = [(0,0,0)]

    def add_audio(self, audio):
        for a in audio:
            if a:
                self.queue.put(a)
    
    def stop(self):
        self.queue.put(None)
    
    def _resample(self,main,surr,t,sr) -> tuple[float,np.ndarray,np.ndarray]:
        # resample to 16 kHz
        return (
            t, 
            spsig.resample_poly(main, 16000, sr, padtype='mean'), 
            spsig.resample_poly(surr, 16000, sr, padtype='mean') if surr is not None else None,
        )

    def run(self):
        main = np.empty(0, 'float32')
        surr = np.empty(0, 'float32')
        cur = 0.0
        nexttime = 0.0
        self.rms = [(0,0,0)]
        work_unit = round(16000 * self.work_rate)
        min_work = work_unit + 8000
        vwnd = round(16000*self.volume_window)
        done = False
        while not done:
            # merge samples into a contiguous array
            if segment := self.queue.get(True):
                # check for a hole, fill with zeros if needed
                (st,sm,ss) = self._resample(*segment)

                missing = int(round((st - nexttime)*16000))
                nexttime = st + len(sm)/16000
                
                if missing > 0:
                    # these are LEADING missing values, the last frame actually didnt have enough but we only tell by the NEXT pts
                    log.info(f'Missing {missing} audio samples before time {st} (got {len(sm)})')
                    sm = np.append(np.zeros(missing, 'float32'), sm)
                elif missing < -1:
                    # these are LEADING extra samples
                    # sometimes we get bad PTS for audio where drops 512 and then the next one has the dropped samples
                    # BUT the PTS was wrong, and the hole isn't in the same place... a 1/32 hole was created and we can't fill it
                    extra = -missing
                    log.info(f'Extra audio samples at time {st}? Got {len(sm)}, dropping {extra} of them...')
                    if extra == len(sm):
                        continue
                    sm = sm[extra:]
                    if len(ss) >= extra:
                        ss = ss[extra:]

                # resample code might leave TRAILING holes in ss; so pad it if needed
                if ss is None:
                    ss = np.zeros(len(sm), 'float32')
                elif len(sm) > len(ss):
                    ss = np.append(ss, np.zeros(len(sm) - len(ss), 'float32'))
                
                main = np.append(main, sm)
                surr = np.append(surr, ss)
            else:
                done = True
                min_work = 8000
                #print('FINISHED at',cur,'have',len(main),'audio samples left')
            
            # now chunk into "work_rate" sized pieces and work on them individually
            while len(main) >= min_work:
                assert(len(main) == len(surr))

                # slice the time
                mwork = main[0:work_unit]
                swork = surr[0:work_unit]
                
                # calculate the volume via RMS for both the main and surround 
                # in small rolling and overlapping slices
                rt = round(cur + self.volume_window, 5)
                x = vwnd
                while x <= len(mwork):
                    # avoid dupes by checking the previous timestamp
                    if rt > self.rms[-1][0]:
                        mrms = math.sqrt(np.mean(np.square(mwork[x-vwnd:x])))
                        srms = math.sqrt(np.mean(np.square(swork[x-vwnd:x])))
                        self.rms.append((round(rt,5), mrms, srms))
                    x += vwnd//2
                    rt += self.volume_window/2
                
                # classify the main channel
                #print('Running audio segmenter on',len(mwork),'samples at',cur)
                for (lab, sb, se) in self.seg(mwork):
                    # we put an extra half second at the beginning so there is overlap in the data that
                    # we see on successive runs (so it isn't starting cold on important data). But,
                    # we don't actually care about that time...
                    if cur == 0 or se > 0.5:
                        self.fspan.add(round(cur+sb,5), round(cur+se,5), AudioSegmentLabel[lab])
                
                # done with this time slice, leaving the half second in the buffer to repeat it next time.
                main = main[work_unit-8000:]
                surr = surr[work_unit-8000:]
                if cur == 0:
                    # all the subsequent runs will overlap by half a second with the previous one.
                    work_unit += 8000
                cur += (len(mwork) - 8000)/16000.0
        self.fspan.end(cur)

def reprocess(feature_log_filename:str, opts:Any=None) -> dict:
    if feature_log_filename is None:
        raise Exception('missing feature_log_filename')
    # deprecated?
    flog = read_feature_log(feature_log_filename)
    read_logo(flog)
    return flog

def read_logo(log_in:str|TextIO|dict) -> None|tuple:
    return logo_finder.from_json(read_feature_log(log_in).get('logo', None))

def read_tags(log_f:str|TextIO|dict):
    return read_feature_log(log_f).get('tags', [])

def read_feature_spans(log:str|TextIO|dict, *spans) -> dict[str, list]:
    log = read_feature_log(log)

    if len(spans) == 0:
        spans = ['logo','blank','diff','audio','volume']

    if 'audio' in spans:
        audiof = AudioFeatureSpan()
        audiof.start()
    else:
        audiof = None
    
    if 'volume' in spans:
        volume = []
    else:
        volume = None
    
    if 'logo' in spans:
        logof = FeatureSpan()
        logof.start(0,False)
    else:
        logof = None
    
    if 'blank' in spans:
        blankf = FeatureSpan()
        blankf.start(0,True)
    else:
        blankf = None
    
    if 'diff' in spans:
        difff = FeatureSpan()
        difff.start(0,True)
    else:
        difff = None
    
    lasttime = 0
    lab = AudioSegmentLabel.SILENCE

    if log['frames'] and log['frames'][0] is None:
        log['frames'].pop(0)

    for f in log['frames']:
        if audiof is not None:
            audiof.add(lasttime, f[0], lab)

        lasttime = f[0]
        if logof is not None:
            logof.add(lasttime,f[1])
        if blankf is not None:
            blankf.add(lasttime, f[2])
        if difff is not None:
            difff.add(lasttime, f[3] >= 15.0 or f[2])
        if volume is not None:
            volume.append((lasttime, f[4], f[5]))
        
        if audiof is not None:
            for i in range(AudioSegmentLabel.count()):
                if f[6+i]:
                    lab = AudioSegmentLabel(i)
    
    result = {}
    if logof is not None:
        logof.end(lasttime)
        result['logo'] = logof.to_list()
    if blankf is not None:
        blankf.end(lasttime)
        result['blank'] = blankf.to_list()
    if difff is not None:
        difff.end(lasttime)
        result['diff'] = difff.to_list()
    if audiof is not None:
        audiof.end(lasttime)
        result['audio'] = audiof.to_list()
    if volume is not None:
        result['volume'] = volume
    
    return result

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
