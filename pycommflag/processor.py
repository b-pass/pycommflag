import logging as log
import struct
import sys
import os
import re
import numpy as np
import math
import time

from typing import Any, Union, BinaryIO

from . import logo_finder, mythtv
from .player import Player
from .scene import Scene

def process_video(video_filename:str, feature_log:Union[str,BinaryIO], opts:Any=None) -> None:
    logo = None
    if type(feature_log) is str:
        if os.path.exists(feature_log) and not opts.no_logo:
            with open(feature_log, 'r+b') as tfl:
                tfl.seek(12)
                if tfl.tell() == 12:
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
       logo = logo_finder.search(player, search_seconds=600 if player.duration <= 3700 else 900, opts=opts)
    
    player.seek(0)
    player.enable_audio()
    player.seek(0)

    # not doing format/aspect because everything is widescreen all the time now (that was very early '00s)
    # except ultra wide screen movies, and sometimes ultra-wide commercials?

    prev_col = None
    fcount = 0
    ftotal = int(player.duration * player.frame_rate)

    percent = ftotal/100.0
    report = math.ceil(ftotal/1000) if not opts.quiet else ftotal*10
    rt = time.perf_counter()
    p = 0
    if not opts.quiet: print('Extracting features...', end='\r') 

    fcolor = None
    feature_log.seek(0)
    feature_log.write(struct.pack('@Iff', 1, player.duration, player.frame_rate))
    logo_finder.write(feature_log, logo)
    
    for (frame,audio) in player.frames():
        p += 1
        fcount += 1
        if p >= report:
            ro = rt
            rt = time.perf_counter()
            print("Extracting, %5.1f%% (%5.1f fps)           " % (min(fcount/percent,100.0), p/(rt - ro)), end='\r')
            p = 0

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
        
        if audio is None:
            audio = np.zeros((6,1), dtype='float32')

        feature_log.write(struct.pack(
            'If???BIII',
            fcount,frame.time-player.vt_start,
            logo_present,frame_blank,scene_change,
            column.shape[2], column.shape[0], column.shape[1], audio.shape[0]
        ))
        column.astype('uint8').tofile(feature_log)
        audio.astype('float32').tofile(feature_log)
    
    feature_log.write(struct.pack('I', 0xFFFFFFFF))
    
    if not opts.quiet: print('Extraction complete           ')

def mean_axis1_float_uint8(fcolor:np.ndarray)->np.ndarray:
    # the below code is equivalent to:
    #   column = fcolor.mean(axis=(1),dtype='float32').astype('uint8')
    # but is almost TEN TIMES faster!
    
    # flatten 3rd dimension (color) into contiguous 2nd dimension
    fcolor.reshape(-1, fcolor.shape[1]*fcolor.shape[2])

    # pick out the individual color channels by skipping by 3, and then average them
    cr = fcolor[...,0::3].astype('float32').mean(axis=(1))
    cg = fcolor[...,1::3].astype('float32').mean(axis=(1))
    cb = fcolor[...,2::3].astype('float32').mean(axis=(1))
    
    # and now convert those stacks back into a 720x3
    return np.stack((cb,cg,cr), axis=1).astype('uint8')

def columnize_frame(frame)->np.ndarray:
    return mean_axis1_float_uint8(frame.to_ndarray(format="rgb24", height=720, width=frame.width*(720/frame.height)))

def read_logo(log_in:Union[str,BinaryIO]) -> None|tuple:
    if type(log_in) is str:
        with open(log_in, 'r+b') as fd:
            (ver,duration,frame_rate) = struct.unpack('@Iff', fd.read(12))
            return read_logo(fd)
    return logo_finder.read(log_in)

def process_scenes(log_in:Union[str,BinaryIO], out=None, opts:Any=None) -> list[tuple[float,float]]:
    if type(log_in) is str:
        with open(log_in, 'r+b') as fd:
            return process_scenes(fd, out=out, opts=opts)

    # TODO re-write log to output...?
    scene_thresh = opts.scene_threshold if opts else 15.0

    print("Reading feature log...")
    log_in.seek(0)

    (ver,duration,frame_rate) = struct.unpack('@Iff', log_in.read(12))
    if ver > 256:
        raise RuntimeError("Wrong endianness in feature log data.")
    
    logo_finder.read(log_in)
    
    fprev = 0
    column = None
    scenes = []
    scene = None
    blanks = []
    while True:
        d = log_in.read(4)
        if len(d) < 4:
            break
        (fnum,) = struct.unpack('I', d)
        if fnum <= fprev:
            log.error(f'Bad frame number {fnum} (expected > {fprev})')
        if fnum == 0xFFFFFFFF:
            break
        
        fprev = fnum
        prev_col = column

        (ftime, logo_present, frame_blank, scene_change, depth, h, w, alen) = struct.unpack('f???BIII', log_in.read(20))
        column = np.fromfile(log_in, 'uint8', depth*h*w, '').astype('int16')
        peaks = np.fromfile(log_in, 'float32', alen, '')

        column.shape = (h,w,depth)

        if frame_blank:
            if scene is not None:
                scene.finish()
                scenes.append(scene)
                scene = None
            scene_change = True
            prev_sc = False
            blanks.append((ftime, column, peaks, logo_present))
            continue

        if blanks:
            assert(scene is None)
            scene = Scene(*blanks[0], is_blank=True)
            for bf in blanks[1:]:
                scene += bf
            scene.finish()
            scenes.append(scene)
            scene = Scene(ftime, column, peaks, logo_present)
            blanks = []
            prev_sc = False
            scene_change = False
            continue
        
        if prev_col is None:
            scene_change = True
        else:
            diff = column - prev_col
            scm = np.mean(np.std(np.abs(diff), (0)))
            if scm >= scene_thresh or (scm >= scene_thresh*2 and prev_sc):
                scene_change = True
            else:
                scene_change = False
        
        #print(len(scene) if scene else '-',ftime, column, peaks, logo_present)
        if scene is not None and (prev_sc or not scene_change):
            scene += (ftime, column, peaks, logo_present)
        else:
            if scene is not None:
                scene.finish(ftime)
                scenes.append(scene)
            scene = Scene(ftime, column, peaks, logo_present)
        prev_sc = scene_change
    
    scene.finish()
    scenes.append(scene)

    print("Done reading, got",len(scenes),"scenes in",fprev,"frames")

    return scenes

def read_breaks(opts:Any):
    if not opts:
        return None
    
    if opts.chanid and opts.starttime:
        return mythtv.get_breaks(opts.chanid, opts.starttime)

    if opts.filename:
        if m:=re.match(r'(?:.*/|)(\d{4,6})_(\d{12,})\.[a-zA-Z0-9]{2,5}', opts.filename):
            mtb = mythtv.get_breaks(m[1], m[2])
            if mtb:
                return mtb
    
    if opts.feature_log:
        if m:=re.match(r'(?:.*/|)cf_(\d{4,6})_(\d{12,})\.[a-zA-Z0-9]{2,5}\.feat', opts.feature_log):
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
                if m := re.match(r'^(?:FI)LE\s*PROCESSING\s*COMPLETE\s*\d+\s*FRAMES\s*(?:AT\s*)?\s*(\d+)\s*', cf.readline()):
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

def update_scene_tags(scenes:list[Scene],marks:list[tuple[float,float]]=None,opts:Any=None)->None:
    if marks is None:
        if opts is not None:
            marks = read_breaks(opts)
        else:
            return
    
    for (start,stop) in marks:
        first = True
        for s in scenes:
            if s.start_time >= stop:
                break
            elif s.duration > 1 and (stop - s.start_time) < s.duration/2:
                # if it ends just inside this scene then don't count it
                break
            if s.start_time <= stop and s.stop_time >= start:
                if first and s.duration >= 1 and s.duration < 10 and (s.stop_time - start) < s.duration/2:
                    # if the tag is near the end of the scene then don't count it
                    continue
                first = False
                s.is_break = True
