import logging as log
import struct
import sys
import os
import numpy as np
import math
import time

from typing import Any, Union, BinaryIO

from . import logo_finder
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
    
    player = Player(video_filename)

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
            print("Extracting, %5.1f%% (%5.1f fps)          " % (min(fcount/percent,100.0), p/(rt - ro)), end='\r')
            p = 0

        fcolor = frame.to_ndarray(format="rgb24", height=720, width=frame.width*(720/frame.height))
        
        # trying to be fast, just look at the middle 1/4 for blank-ish-ness and then verify with the full frame
        x = np.max(fcolor[int(fcolor.shape[0]*3/8):int(fcolor.shape[0]*5/8)])
        if x < 64:
            blank = fcolor
            if opts.blank_no_logo:
                blank = logo_finder.subtract(blank, logo)
            m = np.median(blank, (0,1))
            frame_blank = max(m) < 24 and np.std(m) < 3 and np.std(blank) < 5
            blank = None
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
            scene_change = True
            prev_sc = False
            blanks.append((ftime, column, peaks, logo_present))
            continue

        if blanks:
            if scene is not None:
                piv = min(math.ceil(len(blanks)/2), 60)
                for x in blanks[:piv]:
                    scene += x
                scene.finish(end_blank=True)
                scenes.append(scene)
            else:
                piv = 0
            if piv < len(blanks):
                scene = Scene(*blanks[piv], start_blank=True)
                for x in blanks[piv+1:]:
                    scene += x
                scene += (ftime, column, peaks, logo_present)
            else:
                scene = Scene(ftime, column, peaks, logo_present, start_blank=True)
            blanks = []
            prev_sc = False
            scene_change = False
            continue
        
        if prev_col is None:
            scene_change = True
        else:
            diff = column - prev_col
            scm = np.mean(np.std(np.abs(diff), (0)))
            if scm >= 12 or (scm >= 10 and prev_sc):
                scene_change = True
            else:
                scene_change = False
        
        if scene is not None and (not scene_change or prev_sc or len(scene) < 4):
            scene += (ftime, column, peaks, logo_present)
        else:
            if scene is not None:
                scene.finish(ftime)
                scenes.append(scene)
            scene = Scene(ftime, column, peaks, logo_present)
        prev_sc = scene_change
    
    scene.finish(end_blank=True)
    scenes.append(scene)

    if log_in != log_out and log_out:
        n = log_in.tell()
        log_in.seek(0)
        while n > 0:
            w = min(n, 1048576)
            log_out.write(log_in.read(w))
            n -= w
    if log_out:
        log_out.seek(0, 2)
    
    temp = open('/tmp/scenes', 'w')
    temp.write(f'{len(scenes)}')
    for s in scenes:
        s.write_txt(temp)
    
    vbc = []
    temp.write('\n\n')
    m = []
    for ai in range(4,len(scenes)):
        a = scenes[ai]
        r = []
        for bi in range(ai-4,ai):
            b = scenes[bi]
            
            diff = a.barcode.astype('int16') - b.barcode
            x = np.mean(np.std(diff, (0)))
            r.append(x)
            #temp.write("%-7.03f "%(np.max(s)))
        temp.write(str(ai) + ": ")
        temp.write(str(r))
        temp.write('\n')

        x = a.barcode.astype('uint8').reshape(-1, 1, 3)
        if a.start_blank:
            vbc.append(np.zeros(x.shape, dtype='uint8'))
        vbc.append(x)
        if a.end_blank:
            vbc.append(np.zeros(x.shape, dtype='uint8'))

    temp.write('\n\n')
    temp.close()

    vbc = np.hstack(vbc)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.gray()
    ax1 = fig.add_subplot() 
    ax1.imshow(vbc)
    plt.show()
    
    print("Done reading")

def process_scenes(log_in:Union[str,BinaryIO], opts:Any=None) -> list[tuple[float,float]]:
    if type(log_in) is str:
        with open(log_in, 'r+b') as fd:
            return process_scenes(fd)

    print("Reading feature log...")
    log_in.seek(0)

    (ver,duration,frame_rate) = struct.unpack('@Iff', log_in.read(12))
    if ver > 256:
        raise RuntimeError("Wrong endianness in feature log data.")
    
    logo = logo_finder.read(log_in)
    
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
            scene_change = True
            prev_sc = False
            blanks.append((ftime, column, peaks, logo_present))
            continue

        if blanks:
            if scene is not None:
                piv = min(math.ceil(len(blanks)/2), 60)
                for x in blanks[:piv]:
                    scene += x
                scene.finish(end_blank=True)
                scenes.append(scene)
            else:
                piv = 0
            if piv < len(blanks):
                scene = Scene(*blanks[piv], start_blank=True)
                for x in blanks[piv+1:]:
                    scene += x
                scene += (ftime, column, peaks, logo_present)
            else:
                scene = Scene(ftime, column, peaks, logo_present, start_blank=True)
            blanks = []
            prev_sc = False
            scene_change = False
            continue
        
        if prev_col is None:
            scene_change = True
        else:
            diff = column - prev_col
            scm = np.mean(np.std(np.abs(diff), (0)))
            if scm >= 12 or (scm >= 10 and prev_sc):
                scene_change = True
            else:
                scene_change = False
        
        if scene is not None and (not scene_change or prev_sc or len(scene) < 4):
            scene += (ftime, column, peaks, logo_present)
        else:
            if scene is not None:
                scene.finish(ftime)
                scenes.append(scene)
            scene = Scene(ftime, column, peaks, logo_present)
        prev_sc = scene_change
    
    scene.finish(end_blank=True)
    scenes.append(scene)

    print("Done reading, got",len(scenes),"scenes in",fprev,"frames")

    return scenes
