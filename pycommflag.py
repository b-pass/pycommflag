#!/usr/bin/env python3
import errno
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

def fplanar_to_gray(frame):
    return frame.to_ndarray()[:frame.planes[0].height, ...]

def frgb_to_gray(frame):
    array = frame.to_ndarray(format="rgb24")
    return np.dot(array[...,:3], [0.2989, 0.5870, 0.1140])
    
#def nscale(a,n):
#    s = np.zeros((int(a.shape[0]/n),int(a.shape[1]/n)), np.uint16)
#    for x in range(n):
#        for y in range(n):
#            s += a[x::n, y::n]
#    return (s / n).astype(np.uint8)

class Player:
    def __init__(self, filename, nscale=1):
        self.container = av.open(filename)
        self.container.flags |= av.container.core.Flags.GENPTS
        self.container.flags |= av.container.core.Flags.DISCARD_CORRUPT
        self.container.streams.video[0].thread_type = "AUTO"
        self.container.streams.audio[0].thread_type = "AUTO"
        #self.container.streams.video[0].thread_count = 2
        self.container.streams.audio[0].thread_count = 2

        self.get_frame = lambda: self.next_raw_frame()

        self.duration = self.container.duration / av.time_base
        
        inter = 0
        ninter = 0
        for f in self.container.decode(video=0):
            if f.interlaced_frame:
                inter += 1
            else:
                ninter += 1
            if inter+ninter >= 100:
                break
        
        self.frame_format = f.format
        self.shape = (f.height, f.width)

        if inter*2 > ninter:
            self.interlaced = True
            logdbg(f"{inter} interlaced frame and {ninter} not, means we will deinterlace.")
            self.graph = av.filter.Graph()
            buffer = self.graph.add_buffer(template=self.container.streams.video[0])
            bwdif = self.graph.add("yadif", "")
            buffersink = self.graph.add("buffersink")
            buffer.link_to(bwdif)
            bwdif.link_to(buffersink)
            self.graph.configure()
            self.get_frame = lambda: self.next_deinterlaced_frame()
        else:
            self.interlaced = False
        self.frame_rate = self.container.streams.video[0].average_rate

        pass
    
    def seek(self, seconds):
        self.container.seek(int(seconds / av.time_base))
        # drain the graph
        try:
            while True:
                self.graph.pull()
        except av.AVError as e:
            if e.errno != errno.EAGAIN:
                raise

    def next_raw_frame(self):
        return next(self.container.decode(video=0))
    
    def next_deinterlaced_frame(self):
        try:
            return self.graph.pull()
        except av.AVError as e:
            if e.errno != errno.EAGAIN:
                raise
            self.graph.push(self.next_raw_frame())
            return self.next_deinterlaced_frame()

def find_logo_mask(player, skip=1, threshold=.33, search_seconds=600, search_beginning=False):
    make_gray = lambda f: frgb_to_gray(f)
    if player.frame_format.is_planar:
        make_gray = lambda f: fplanar_to_gray(f)
    
    logo_sum = np.ndarray(player.shape, np.uint16)
    
    # search around 1/3 of the way through the show, there should be a lot of show there
    if not search_beginning and player.duration >= search_seconds*2:
        player.seek(player.duration/3 - search_seconds/2)
    else:
        player.seek(0)
    
    thresh = int(255*threshold)
    fcount = 0
    ftotal = min(int(search_seconds * player.frame_rate), 65000*skip)
    
    percent = ftotal/skip/100.0
    report = math.ceil(ftotal/250)
    p = report
    for _ in range(ftotal):
        try:
            frame = player.get_frame()
        except StopIteration:
            break
        
        p += 1
        if p%skip != 0:
            continue
        data = make_gray(frame)
        logo_sum += scipy.ndimage.sobel(data, mode='constant') > thresh
        fcount += 1
        if p >= report:
            p = 0
            print("Searching, %3.1f%%" % (min(fcount/percent,100.0)), end='\r')
    print("Searching is complete.")

    # overscan, ignore 3.5% on each side
    for n in range(math.ceil(player.shape[0]*.035)):
        logo_sum[n] = 0
        logo_sum[-n-1] = 0
    for n in range(math.ceil(player.shape[1]*.035)):
        logo_sum[..., n] = 0
        logo_sum[..., -n-1] = 0
    
    # no logos in the middle 1/3 of the screen
    logo_sum[int(player.shape[0]/3):int(player.shape[0]*2/3),int(player.shape[1]/3):int(player.shape[1]*2/3)] = 0

    # in case we found something stuck on the screen, try to look beyond that
    best = np.max(logo_sum)
    if best > fcount*.95:
        logo_sum = np.where(logo_sum <= fcount*.85, logo_sum, 0)
        best = np.max(logo_sum)

    if best <= fcount / 3:
        loginfo("No logo found (insufficient edges)")
        return (None,None,None)
    
    logo_mask = logo_sum >= (best - fcount/10)
    if np.count_nonzero(logo_mask) < 50:
        loginfo("No logo found (not enough edges)")
        return (None,None,None)
    
    nz = np.nonzero(logo_mask)
    top = min(nz[0])
    left = min(nz[1])
    bottom = max(nz[0])
    right = max(nz[1])
    if right - left < 5 or bottom - top < 5:
        loginfo("No logo found (bounding box too narrow)")
        return (None,None,None)
    
    top -= 5
    left -= 5
    bottom += 5
    right += 5

    # if the bound is more than half the image then clip it
    if bottom-top >= player.shape[0]/2:
        if bottom >= player.shape[0]*.75:
            top = int(player.shape[0]/2)
        else:
            bottom = int(player.shape[0]/2)
    if right-left >= player.shape[1]/2:
        if right >= player.shape[1]*.75:
            left = int(player.shape[1]/2)
        else:
            right = int(player.shape[1]/2)
    
    logdbg(f"Logo bounding box: {top},{left} to {bottom},{right}")

    logo_mask = logo_mask[top:bottom,left:right]
    if np.count_nonzero(logo_mask) < 20:
        loginfo("No logo found (not enough edges within bounding box)")
        return (None,None,None)

    return ((top,left),(bottom,right), logo_mask)

def main(argv):
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', dest="filename", 
                        help="Input video file")
    parser.add_option('-s', '--skip', dest="skip", type="int", default=4,
                        help="Frame skipping for logo search phase (higher number, faster processing)")
    #opts.add_option('--file', dest="filename", help="Input video file", metavar="FILE")
    (opts, args) = parser.parse_args()

    print(opts.filename)
    player = Player(opts.filename)

    (lmin,lmax,lmask) = find_logo_mask(player, skip=opts.skip, search_seconds=600 if player.duration <= 3700 else 900)
    if lmask is None:
        lmin = lmax = (0,0)
    else:
        for y in range(lmask.shape[0]):
            for x in range(lmask.shape[1]):
                print('#' if lmask[y][x] else ' ', end='')
            print()

    #sys.exit(1)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.gray()
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ax1.imshow(player.get_frame().to_ndarray(format="rgb24"))
    ax2.imshow(lmask)
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
