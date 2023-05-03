import logging as log
import struct
import sys
import os
import re
import numpy as np
import math
import time
import gc

from typing import Any,BinaryIO

from . import logo_finder, mythtv, segmenter
from .player import Player
from .scene import Scene
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
    if not opts.quiet: print('Extracting features...', end='\r') 

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
    aseg = ina_foss.Segmenter(energy_ratio=0.05)
    aprev = None
    audio = []

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
            (ares, aprev) = _process_audio(aseg, aprev, player.move_audio())
            audio += ares

        fcolor = frame.to_ndarray(format="rgb24", height=720, width=frame.width*(720/frame.height))
        
        # trying to be fast, just look at the middle 1/4 for blank-ish-ness and then verify with the full frame
        x = np.max(fcolor[int(fcolor.shape[0]*3/8):int(fcolor.shape[0]*5/8)])
        if x < 64:
            m = np.median(fcolor, (0,1))
            frame_blank = max(m) < 24 and np.std(m) < 3 and np.std(fcolor) < 6
        else:
            frame_blank = False

        column = mean_axis1_float_uint8(fcolor)
        
        if frame_blank:
            logo_present = False
            scene_change = True
        else:
            logo_present = logo_finder.check_frame(frame, logo)
            scene_change = False

            # do scene detection here instead? make sure to change the output type of the stack to int16!
        
        feature_log.write(struct.pack(
            'If???BII',
            fcount,frame.time-player.vt_start,
            logo_present,frame_blank,scene_change,
            column.shape[2], column.shape[0], column.shape[1]
        ))
        column.astype('uint8').tofile(feature_log)
    
    audio += _process_audio(aseg, aprev, player.move_audio())[0]

    feature_log.write(struct.pack('I', 0xFFFFFFFF))

    pos = feature_log.tell()
    feature_log.seek(16, 0)
    feature_log.write(struct.pack('I', pos))
    feature_log.seek(pos)

    feature_log.write(struct.pack('I', len(audio)))
    for (lab,start,stop) in audio:
        feature_log.write(struct.pack('Iff', ina_foss.AudioSegmentLabel[lab].value, start, stop))

    pos = feature_log.tell()
    feature_log.seek(20, 0)
    feature_log.write(struct.pack('@I', pos))
    feature_log.seek(pos)
    
    if not opts.quiet:
        print('Extraction complete           ')

def mean_axis1_float_uint8(fcolor:np.ndarray)->np.ndarray:
    # the below code is equivalent to:
    #   column = fcolor.mean(axis=(1),dtype='float32').astype('uint8')
    # but is almost TEN TIMES faster!
    
    # pick out the individual color channels by skipping by 3, and then average them
    #cr = fcolor[...,0::3].astype('float32').mean(axis=(1))
    cr = fcolor[...,0::3].mean(axis=(1), dtype='float32')
    cg = fcolor[...,1::3].mean(axis=(1), dtype='float32')
    cb = fcolor[...,2::3].mean(axis=(1), dtype='float32')
    
    # and now convert those stacks back into a 720x3
    return np.stack((cb,cg,cr), axis=1).astype('uint8')

def columnize_frame(frame)->np.ndarray:
    return mean_axis1_float_uint8(frame.to_ndarray(format="rgb24", height=720, width=frame.width*(720/frame.height)))

def _process_audio(aseg, prev, segments):
    if not segments:
        return ([],prev)
    
    #print(prev,segments)
    isstart = False
    if prev is None:
        prev = segments.pop(0)
        isstart = True
    
    all = prev[0]
    start = prev[1]
    time = prev[1] + len(prev[0])/16000
    for s in segments:
        missing = int(round((s[1] - time)*16000))
        if missing > 0:
            all = np.append(all, np.zeros(missing, 'float32'))
        all = np.append(all, s[0])
        time = s[1] + len(s[0])/16000
    
    prev = (all[-8000:], start+(len(all)-8000)/16000)
    res = []

    startbound = 0 if isstart else 0.5
    for (lab, sb, se) in aseg(all):
        if se <= startbound:
            continue
        res.append((lab, start+sb, start+se))
    
    print(len(res))
    print(res)
    return (res, prev)

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
    prev_check = None
    while True:
        d = log_f.read(4)
        if len(d) < 4:
            break
        (fnum,) = struct.unpack('I', d)
        if fnum <= fprev:
            log.error(f'Bad frame number {fnum} (expected > {fprev})')
        if fnum == 0xFFFFFFFF:
            break
        
        fprev = fnum
        prev_col = column

        (ftime, logo_present, frame_blank, is_diff, depth, h, w) = struct.unpack('f???BII', log_f.read(16))
        column = np.fromfile(log_f, 'uint8', depth*h*w, '').astype('int16')
        column.shape = (h,w,depth)

        while audio_idx+1 < len(audio) and audio[audio_idx][2] <= ftime:
            audio_idx += 1
        if ftime >= audio[audio_idx][1] and ftime < audio[audio_idx][2]:
            faudio = audio[audio_idx][0]
        else:
            faudio = ina_foss.AudioSegmentLabel.SILENCE

        if prev_col is None:
            is_diff = False
        else:
            diff = column - prev_col
            scm = np.mean(np.std(np.abs(diff), (0)))
            is_diff = scm >= opts.scene_threshold 
        
        segbreak = fseg.check(ftime=ftime, faudio=faudio, logo_present=logo_present, is_blank=frame_blank, is_diff=is_diff)
        if segbreak != prev_check:
            if scene is not None:
                scene.finish()
                scenes.append(scene)
                scene = None
            scene = Scene(ftime, column, faudio, logo_present, frame_blank, is_diff)
        else:
            scene += (ftime, column, faudio, logo_present, frame_blank, is_diff)
        prev_check = segbreak
    
    scene.finish()
    scenes.append(scene)

    if not opts.quiet: print("Done, got",len(scenes),"scenes in",fprev,"frames")
    
    if scene_pos:
        log_f.seek(scene_pos)
        log_f.truncate(scene_pos)
        rewrite_scenes(scenes, log_f)
    
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
        audio.append(struct.unpack('Iff', log_f.read(12)))
    
    return audio

def rewrite_scenes(scenes:list[Scene], log_f:str|BinaryIO) -> None:
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
