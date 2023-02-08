#!/usr/bin/env python3
import json
import logging
import optparse
import os
import sys

from time import time as now
import numpy as np
import av
import scipy.ndimage
import math

def logdbg(*a,**kw):
    print('DBG:',*a,**kw)

def logerr(*a,**kw):
    print('ERROR:',*a,**kw)

def loginfo(*a,**kw):
    print('INFO:',*a,**kw)

def nscale(a,n):
    s = np.zeros((int(a.shape[0]/n),int(a.shape[1]/n)), np.uint16)
    for x in range(n):
        for y in range(n):
            s += a[x::n, y::n]
    return (s / n).astype(np.uint8)

def grayscale_plane(frame):
    return frame.to_ndarray()[:frame.height]

def grayscale_rgb(frame):
    array = frame.to_ndarray(format="rgb24")[:frame.height]
    return np.dot(array[...,:3], [0.2989, 0.5870, 0.1140])

grayscale_frame = grayscale_rgb

def find_logo_mask(container, scale=1, threshold=.33, search_seconds=600, search_beginning=False, log=print):
    dur = float(container.duration / av.time_base)
    container.seek(0)

    get_frame = lambda: grayscale_frame(next(container.decode(video=0)))
    f = get_frame()
    orig_height = f.height
    width = f.width
    height = f.height
    while width%scale or height%scale:
        scale -= 1
    width = int(width/scale)
    height = int(height/scale)
    get_frame = lambda gf=get_frame: nscale(gf(),scale)
    
    logdbg(f"Logo Search Scale /{scale} = {height}x{width}")
    logo_sum = np.ndarray((height,width), np.uint32)
    
    # search around 1/3 of the way through the show, there should be a lot of show there
    if dur > search_seconds and not search_beginning:
        container.seek(int((dur/3 - search_seconds/2) * av.time_base))
    
    thresh = int(255*threshold)
    fcount = 0
    ftotal = int(search_seconds * container.streams.video[0].average_rate)
    
    report = max(math.ceil(ftotal/250),30)
    percent = int(ftotal/100)
    p = report
    for _ in range(ftotal):
        try:
            frame = get_frame()
        except StopIteration:
            break
        logo_sum += scipy.ndimage.sobel(frame, mode='constant') > thresh
        fcount += 1
        p += 1
        if p >= report:
            p = 0
            print("Searching, %3.1f%%" % (min(fcount/percent,100.0)), end='\r')
    print("Searching is complete.")

    # overscan, ignore 3.5% on each side
    for n in range(math.ceil(height/100*3.5)):
        logo_sum[n] = 0
        logo_sum[-n-1] = 0
    for n in range(math.ceil(width/100*3.5)):
        logo_sum[..., n] = 0
        logo_sum[..., -n-1] = 0
    
    # no logos in the middle 1/3 of the screen
    logo_sum[int(height/3):int(height/3*2),int(width/3):int(width/3*2)] = 0

    best = np.max(logo_sum)
    #if best >= fcount*.95:
    #    logo_sum = np.where(logo_sum <= fcount*.90, logo_sum, 0)
    #    best = np.max(logo_sum)
    if best <= fcount / 3:
        loginfo("No logo found (insufficient edges)")
        return None
    
    logo_mask = logo_sum >= (best - fcount/10)
    if logo_mask.count_nonzero() <= 40/scale:
        loginfo("No logo found (not enough edges)")
        return None
    
    hh = int(logo_mask.shape[0]/2)
    hw = int(logo_mask.shape[1]/2)
    tl = logo_mask[:hh, :hw].count_nonzero()
    tr = logo_mask[:hh, hw:].count_nonzero()
    bl = logo_mask[hh:, :hw].count_nonzero()
    br = logo_mask[hh:, hw:].count_nonzero()
    c = [tl,tr,bl,br]
    w = np.argmax(c)
    if c[w] < 50/scale:
        loginfo("No single-quadrant logo found")
        return None
    
    if w == 0: return (0,logo_mask[:hh, :hw])
    if w == 1: return (1,logo_mask[:hh, hw:])
    if w == 2: return (2,logo_mask[hh:, :hw])
    if w == 3: return (3,logo_mask[hh:, hw:])
    return None # unreachable

    if w != 0: logo_mask[:hh, :hw] = 0
    if w != 1: logo_mask[:hh, hw:] = 0
    if w != 2: logo_mask[hh:, :hw] = 0
    if w != 3: logo_mask[hh:, hw:] = 0
    return logo_mask

def main(argv):
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', dest="filename", 
                        help="Input video file")
    parser.add_option('-s', '--scale', dest="scale", type="int", default=1,
                        help="Downscaling factorfor image processing (higher number, faster processing)")
    #opts.add_option('--file', dest="filename", help="Input video file", metavar="FILE")
    (opts, args) = parser.parse_args()

    print(opts.filename)
    container = av.open(opts.filename)
    container.flags |= av.container.core.Flags.GENPTS
    container.flags |= av.container.core.Flags.DISCARD_CORRUPT
    container.streams.video[0].thread_type = "AUTO"
    container.streams.audio[0].thread_type = "AUTO"
    container.streams.video[0].thread_count = 2
    container.streams.audio[0].thread_count = 2
    
    frame0 = next(container.decode(video=0))
    grayscale_frame = grayscale_plane if frame0.format.is_planar else grayscale_rgb

    duration = float(container.duration / av.time_base)

    (logoq,logo_mask) = find_logo_mask(container, scale=opts.scale, search_seconds=600 if duration <= 3700 else 900)

    

    frame = next(container.decode(video=0)).to_ndarray()
    
    #sys.exit(1)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.gray()
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ax1.imshow(frame)
    ax2.imshow(logo_mask)
    plt.show()

    sys.exit(1)
    if False:
        print(s)
        # Collapse down to a column.
        column = array.mean(axis=1)

        # Convert to bytes, as the `mean` turned our array into floats.
        column = column.clip(0, 255).astype("uint8")

        # Get us in the right shape for the `hstack` below.
        column = column.reshape(-1, 1, 3)

        columns.append(column)

        print(column)
        print(columns)
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv)
