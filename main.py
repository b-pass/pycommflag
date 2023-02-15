#!/usr/bin/env python3
import json
import logging as log
import optparse
import os
import sys
import struct

import time
now = time.time
import numpy as np
import math

from .player import Player
from . import logo_finder

def run():
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', dest="filename", 
                        help="Input video file")
    parser.add_option('-s', '--skip', dest="skip", type="int", default=4,
                        help="Frame skipping for logo search phase (higher number, faster processing)")
    #opts.add_option('--file', dest="filename", help="Input video file", metavar="FILE")
    (opts, args) = parser.parse_args()

    print(opts.filename)
    player = Player(opts.filename)

    if os.path.exists('/tmp/logo'):
        logo = json.load(open('/tmp/logo'))
        logo[2] = np.array(logo[2])
    else:
        logo = logo_finder.search(player, skip=opts.skip, search_seconds=600 if player.duration <= 3700 else 900)
        if logo:
            temp = list(logo)
            temp[2] = logo[2].tolist()
            json.dump(temp, open('/tmp/logo','w'))

    player.seek(0)
    player.enable_audio()
    player.seek(0)

    # not doing format/aspect because everything is widescreen all the time now (that was very early '00s)
    # except ultra wide screen movies, and sometimes ultra-wide commercials?

    prev_bar = None
    fcount = 0
    ftotal = int(player.duration * player.frame_rate)

    percent = ftotal/100.0
    report = math.ceil(ftotal/1000)
    rt = time.perf_counter()
    p = 0
    print('Processing...', end='\r') 

    fcolor = None
    fdata = open('/tmp/frames', 'wb')
    fdata.write(struct.pack('@Iff',1, player.duration, player.frame_rate))
    if logo:
        fdata.write(struct.pack('IIIII', int(logo[3]), logo[0][0], logo[0][1], logo[1][0], logo[1][1]))
        fdata.write(logo[2].astype('uint8').tobytes())
    else:
        fdata.write(struct.pack('IIIII', 0, 0, 0, 0, 0))
    
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

        fdata.write(struct.pack(
            'If???BIII',
            fcount,frame.time-player.vt_start,
            logo_present,frame_blank,scene_change,
            column.shape[2], column.shape[0], column.shape[1], audio.shape[0]
        ))
        fdata.write(column.astype('uint8').tobytes())
        fdata.write(audio.astype('float32').tobytes())
    
    fdata.close()
    print("done!")
