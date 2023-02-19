import logging as log
import struct
import os
import numpy as np
import math
import time

from typing import Any, Union, BinaryIO

from .player import Player
from . import logo_finder

def process_video(video_filename:str, feature_log:Union[str,BinaryIO], opts:Any=None) -> None:
    logo = None
    if type(feature_log) is str:
        if os.path.exists(feature_log):
            with open(feature_log, 'r+b') as tfl:
                tfl.seek(12)
                if tfl.tell() > 0:
                    if not opts.quiet: print(f"{opts.feature_log} exists, re-using logo")
                    logo = logo_finder.read(tfl)
        feature_log = open(feature_log, 'w+b')
    
    player = Player(video_filename)

    if opts.no_logo:
        logo = None
    elif logo is not None:
        print(logo)
        pass
    else:
       logo = logo_finder.search(player, search_seconds=600 if player.duration <= 3700 else 900, opts=opts)
    
    player.seek(0)
    player.enable_audio()
    player.seek(0, any_frame=False)

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
            print("Extracting, %5.1f%% (%5.1f fps)          " % (min(fcount/percent,100.0), p/(rt - ro)), end='\r')
            p = 0

        fcolor = frame.to_ndarray(format="rgb24", height=720, width=frame.width*(720/frame.height))
        
        # trying to be fast, just look at the middle 1/4 for the blank checking
        x = np.max(fcolor[int(fcolor.shape[0]*3/8):int(fcolor.shape[0]*5/8)])
        if x < 96:
            m = np.median(fcolor, (0,1))
            frame_blank = max(m) < 32 and np.std(m) < 3 and np.std(fcolor) < 10
        else:
            frame_blank = False

        #s = np.std(fcolor, (0,1,2))
        #m = np.mean(fcolor, (0,1))
        #d = np.median(fcolor, (0,1))
        #x = np.max(fcolor)
        #print(s,m,d,x)

        # the below code is equivalent to:
        #   column = fcolor.mean(axis=(1),dtype='float32').astype('int16')
        # but is almost TEN TIMES faster!
        
        # flatten 3rd dimension (color) into contiguous 2nd dimension
        fcolor.reshape(-1, fcolor.shape[1]*fcolor.shape[2])

        # pick out the individual color channels by skipping by 3, and then average them
        cr = fcolor[...,0::3].astype('float32').mean(axis=(1))
        cg = fcolor[...,1::3].astype('float32').mean(axis=(1))
        cb = fcolor[...,2::3].astype('float32').mean(axis=(1))
        
        # and now convert those stacks back into a 720x3
        column = np.stack((cb,cg,cr), axis=1).astype('uint8')
        
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

def process_features(log_in:Union[str,BinaryIO], log_out:Union[str,BinaryIO], opts:Any=None) -> None:
    if type(log_in) is str:
        if log_in == log_out:
            with open(log_in, 'r+b') as fd:
                return process_features(fd, fd)
        else:
            with open(log_in, 'rb') as fd:
                return process_features(fd, log_out)

    if type(log_out) is str:
        with open(log_out, 'w+b') as fd:
            return process_features(log_in, fd)
    
    log_in.seek(0)

    (ver,duration,frame_rate) = struct.unpack('@Iff', log_in.read(12))
    if ver > 256:
        raise RuntimeError("Wrong endianness in feature log data.")
    
    logo = logo_finder.read(log_in)
    
    fprev = 0
    prev_blank = True
    prev_col = None
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

        (ftime, logo_present, frame_blank, scene_change, depth, h, w, alen) = struct.unpack('f???BIII', log_in.read(20))
        column = np.fromfile(log_in, 'uint8', depth*h*w, '').astype('int16')
        peaks = np.fromfile(log_in, 'float32', alen, '')

        column.shape = (h,w,depth)
        
        if prev_col is None or frame_blank or prev_blank:
            scene_change = True
        else:
            diff = column - prev_col
            s = np.std(diff, (0))
            if max(s) >= 8:
                scene_change = True
            else:
                scene_change = False
        prev_col = column
        prev_blank = frame_blank

        #if not quiet: print('B' if frame_blank else 'S' if scene_change else '_', end='')
    
    if log_in != log_out:
        n = log_in.tell()
        log_in.seek(0)
        while n > 0:
            w = min(n, 1048576)
            log_out.write(log_in.read(w))
            n -= w
    log_out.seek(0, 2)
    
    print("Done reading")
