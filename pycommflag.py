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

LOGO_EDGE_THRESHOLD = 85 # how strong an edge is strong enough?

def logdbg(*a,**kw):
    print('DBG:',*a,**kw)

def logerr(*a,**kw):
    print('ERROR:',*a,**kw)

def loginfo(*a,**kw):
    print('INFO:',*a,**kw)

def fplanar_to_gray(frame,box):
    return frame.to_ndarray()[:frame.planes[0].height, ...]

def frgb_to_gray(frame):
    array = frame.to_ndarray(format="rgb24")
    return np.dot(array[...,:3], [0.2989, 0.5870, 0.1140])

def gray(frame,box=None):
    if False and frame.format.is_planar:
        x = frame.to_ndarray()
        if box:
            return x[box[0]:box[1],box[2]:box[3]]
        else:
            return x[:frame.planes[0].height]
    else:
        x = frame.to_ndarray(format="rgb24")
        if box:
            x = x[box[0]:box[1],box[2]:box[3]]
        x = np.dot(x[...,:3], [0.2989, 0.5870, 0.1140])
        return x

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

        self.duration = self.container.duration / av.time_base
        self.streams = {'video':0}
        self.aq = None
        self.vq = []

        inter = 0
        ninter = 0
        for f in self.container.decode(video=0):
            if f.interlaced_frame:
                inter += 1
            else:
                ninter += 1
            if inter+ninter >= 100:
                break
        
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
            self.frames = lambda: self.next_deinterlaced_frame()
        else:
            self.interlaced = False
        self.frame_rate = self.container.streams.video[0].average_rate
    
    def seek(self, seconds):
        self.container.seek(int(seconds / av.time_base))
        self.flush_buffers()
    
    def enable_audio(self, stream=0):
        self.streams['audio'] = stream
        self.flush_buffers()
        
    def disable_audio(self):
        if 'audio' in self.streams:
            del self.streams['audio']
        self.aq = None

    def queue_audio(self, af):
        # todo, convert to s16p
        self.aq.append(af.to_ndarray())

    def flush_buffers(self):
        if self.graph:
            try:
                while True:
                    self.graph.pull()
            except av.AVError as e:
                if e.errno != errno.EAGAIN:
                    raise
        self.vq = []
        self.aq = [] if 'audio' in self.streams else None

    def fill(self):
        if self.aq:
            af = self.aq[-1]
            anext = af.time + af.samples * (af.time_base / af.sample_rate)
        else:
            anext = None

        for frame in self.container.decode(**self.streams):
            if type(frame) is av.AudioFrame:
                af = frame
                anext = af.time + af.samples * (af.time_base / af.sample_rate)
                self.queue_audio(af)    
            elif type(frame) is av.VideoFrame:
                if self.graph:
                    self.graph.push(frame)
                    try:
                        frame = self.graph.pull()
                    except av.AVError as e:
                        if e.errno != errno.EAGAIN:
                            raise
                        continue
                self.vq.append(frame)
                if self.aq is None:
                    break
            if len(self.vq) > 1 and anext:
                if self.vq[1].time <= anext:
                    break
        
    def frames(self):
        self.fill()
        if not self.vq:
            raise StopIteration()
        
        vf = self.vq.pop()
        # grab audio samples matching vf
        yield (vf,audio)

def find_logo_mask(player, skip=1, search_seconds=600, search_beginning=False):
    global LOGO_EDGE_THRESHOLD
    
    logo_sum = np.ndarray(player.shape, np.uint16)
    
    # search around 1/3 of the way through the show, there should be a lot of show there
    if not search_beginning and player.duration >= search_seconds*2:
        player.seek(player.duration/3 - search_seconds/2)
    else:
        player.seek(0)
    
    fcount = 0
    ftotal = min(int(search_seconds * player.frame_rate), 65000*skip)
    
    percent = ftotal/skip/100.0
    report = math.ceil(ftotal/250)
    p = report
    for _ in range(ftotal):
        frame = next(player.frames())
        
        p += 1
        if p%skip != 0:
            continue
        data = gray(frame)
        logo_sum += scipy.ndimage.sobel(data) > LOGO_EDGE_THRESHOLD
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
        return None
    
    logo_mask = logo_sum >= (best - fcount/10)
    if np.count_nonzero(logo_mask) < 50:
        loginfo("No logo found (not enough edges)")
        return None
    
    nz = np.nonzero(logo_mask)
    top = int(min(nz[0]))
    left = int(min(nz[1]))
    bottom = int(max(nz[0]))
    right = int(max(nz[1]))
    if right - left < 5 or bottom - top < 5:
        loginfo("No logo found (bounding box too narrow)")
        return None
    
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
    lmc = np.count_nonzero(logo_mask)
    if lmc < 20:
        loginfo("No logo found (not enough edges within bounding box)")
        return None
    thresh = lmc * .75
    return ((top,left),(bottom,right), logo_mask, thresh)

def check_for_logo(frame, logo):
    global LOGO_EDGE_THRESHOLD
    if not logo:
        return False
    
    ((top,left),(bottom,right),lmask,thresh) = logo
    c = gray(frame, [top,bottom,left,right])
    c = scipy.ndimage.sobel(c) > LOGO_EDGE_THRESHOLD
    c = np.where(lmask, c, False)
    #print('\n!',np.count_nonzero(c),'of',np.count_nonzero(lmask),'!')
    return np.count_nonzero(c) >= thresh
    
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

    if os.path.exists('/tmp/logo'):
        logo = json.load(open('/tmp/logo'))
        logo[2] = np.array(logo[2])
    else:
        logo = find_logo_mask(player, skip=opts.skip, search_seconds=600 if player.duration <= 3700 else 900)
        if logo:
            temp = list(logo)
            temp[2] = logo[2].tolist()
            json.dump(temp, open('/tmp/logo','w'))

    player.seek(0)
    player.enable_audio()

    # not doing format/aspect because everything is widescreen all the time now (that was very early '00s)
    # for each frame: 
    #    is it blank? 
    #    does it have logo?
    #    scene change -> video barcode?
    #    also do a horizontal barcode? video qrcode?
    #
    prev_bar = None
    fcount = 0
    ftotal = int(player.duration * player.frame_rate)
    percent = ftotal/100.0
    report = math.ceil(ftotal/1000)
    p = report
    
    def plot(l,r):
        #sys.exit(1)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        #plt.gray()
        ax1 = fig.add_subplot(121)  # left side
        ax2 = fig.add_subplot(122)  # right side
        ax1.imshow(l)
        ax2.imshow(r)
        plt.show()

    fcolor = None
    for frame in player.frames():
        print(type(frame))
        p += 1
        fcount += 1
        if p >= report:
            p = 0
            #print("Processing, %3.1f%%" % (min(fcount/percent,100.0)), end='\r')

        prev_frame = fcolor
        fcolor = frame.to_ndarray(format="rgb24")
        x = np.max(fcolor)
        if x < 128:
            m = np.median(fcolor, (0,1))
            frame_blank = max(m) < 25 and np.std(m) < 3 and np.std(fcolor) < 10
        else:
            frame_blank = False

        if frame_blank:
            logo_present = False
            scene_change = True
            prev_bar = None
            print("BLANK!")
        else:
            logo_present = check_for_logo(frame, logo)
        
            #s = np.std(fcolor, (0,1,2))
            #m = np.mean(fcolor, (0,1))
            #d = np.median(fcolor, (0,1))
            #x = np.max(fcolor)
            #print(s,m,d,x)

            column = fcolor.mean(axis=1)
            if prev_bar is not None:
                diff = column - prev_bar
                s = np.std(diff, (0))
                print(s)
                if max(s) >= 8:
                    scene_change = True
            prev_bar = column
        
        #breakpoint()

        if logo_present:
            break


    sys.exit(1)
    if False:
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
