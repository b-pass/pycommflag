import logging as log
import struct

import numpy as np
import math
import time

from typing import Any, BinaryIO

from .player import Player
from . import logo_finder

def process_video(filename:str, frame_log:Any[str|BinaryIO], opts:Any=None) -> None:
    if type(frame_log) is str:
        with open(frame_log, 'w+b') as fd:
            return process_video(filename, fd, opts)
    
    player = Player(filename)

    if not opts.no_logo:
        logo = logo_finder.search(player, search_seconds=600 if player.duration <= 3700 else 900, opts=opts)
    else:
        logo = None
    
    player.seek(0)
    player.enable_audio()
    player.seek(0)

    # not doing format/aspect because everything is widescreen all the time now (that was very early '00s)
    # except ultra wide screen movies, and sometimes ultra-wide commercials?

    prev_bar = None
    fcount = 0
    ftotal = int(player.duration * player.frame_rate)

    percent = ftotal/100.0
    report = math.ceil(ftotal/1000) if not opts.quiet else ftotal*10
    rt = time.perf_counter()
    p = 0
    if not opts.quiet: print('Processing...', end='\r') 

    fcolor = None
    frame_log.write(struct.pack('@Iff',1, player.duration, player.frame_rate))
    if logo:
        frame_log.write(struct.pack('IIIII', int(logo[3]), logo[0][0], logo[0][1], logo[1][0], logo[1][1]))
        frame_log.write(logo[2].astype('uint8').tobytes())
    else:
        frame_log.write(struct.pack('IIIII', 0, 0, 0, 0, 0))
    
    for (frame,audio) in player.frames():
        p += 1
        fcount += 1
        if p >= report:
            ro = rt
            rt = time.perf_counter()
            print("Processing, %5.1f%% (%5.1f fps)          " % (min(fcount/percent,100.0), p/(rt - ro)), end='\r')
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
        
        # and now convert those stacks back into a Hx3
        column = np.stack((cb,cg,cr), axis=1).astype('int16')
        
        if frame_blank:
            logo_present = False
            scene_change = True
            prev_bar = None
        else:
            logo_present = logo_finder.check_frame(frame, logo)
        
            scene_change = False
            if prev_bar is not None:
                diff = column - prev_bar
                prev_bar = column
                s = np.std(diff, (0))
                if max(s) >= 8:
                    scene_change = True
            else:
                prev_bar = column
        
        if audio is None:
            audio = np.zeros((6,1), dtype='float32')

        frame_log.write(struct.pack(
            'If???BIII',
            fcount,frame.time-player.vt_start,
            logo_present,frame_blank,scene_change,
            column.shape[2], column.shape[0], column.shape[1], audio.shape[0]
        ))
        frame_log.write(column.astype('uint8').tobytes())
        frame_log.write(audio.astype('float32').tobytes())
    
    if not opts.quiet: print('Processing complete           ')

def process_log(log_in:Any[str | BinaryIO], log_out:Any[str|BinaryIO]) -> None:
    if type(log_in) is str:
        if log_in == log_out:
            with open(log_in, 'r+b') as fd:
                return process_log(fd, fd)
        else:
            with open(log_in, 'rb') as fd:
                return process_log(fd, log_out)

    if type(log_out) is str:
        with open(log_out, 'ab') as fd:
            return process_log(log_in, fd)
    
    log_in.seek(0)

    (ver,) = struct.unpack_from('@I', log_in)
    if ver > 256:
        raise RuntimeError("wrong endianness in frame log data.  unsupported... todo?")
    
