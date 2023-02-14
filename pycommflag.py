#!/usr/bin/env python3
import errno
import json
import logging
import optparse
import os
import sys
import struct

import time
now = time.time
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
    if frame.format.is_planar:
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
    def __init__(self, filename, nscale=1, deinterlace=False):
        self.container = av.open(filename)
        self.container.flags |= av.container.core.Flags.GENPTS
        self.container.flags |= av.container.core.Flags.DISCARD_CORRUPT
        self.container.streams.video[0].thread_type = "AUTO"
        #self.container.streams.video[0].thread_count = 4
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
            if inter+ninter >= 360:
                break
        self.container.seek(0)
        
        #f = next(self.container.decode(video=0))
        self.shape = (f.height, f.width)

        if inter*4 > ninter:
            self.interlaced = True
            logdbg(f"{inter} interlaced frames and {ninter}, means we will deinterlace.")
            self.graph = av.filter.Graph()
            buffer = self.graph.add_buffer(template=self.container.streams.video[0])
            bwdif = self.graph.add("yadif", "")
            buffersink = self.graph.add("buffersink")
            buffer.link_to(bwdif)
            bwdif.link_to(buffersink)
            self.graph.configure()
        else:
            logdbg(f"We will NOT deinterlace (had {inter} interlaced frames and {ninter} non-interlaced frames)")
            self.interlaced = False
            self.graph = None
        self.frame_rate = self.container.streams.video[0].average_rate
    
    def seek(self, seconds):
        self.container.seek(int(seconds / av.time_base))
        self.flush_buffers()
    
    def enable_audio(self, stream=0):
        self.streams['audio'] = stream
        self.flush_buffers()
        self.vt_start = self.container.streams.video[self.streams['video']].start_time * self.container.streams.video[self.streams['video']].time_base
        self.at_start = self.container.streams.audio[self.streams['audio']].start_time * self.container.streams.video[self.streams['audio']].time_base
        
    def disable_audio(self):
        if 'audio' in self.streams:
            del self.streams['audio']
        self.aq = None

    def queue_audio(self, af):
        d = af.to_ndarray()
        if d.dtype.kind != 'f':
            d = d.astype('float32') / 32767.0
        if d.shape[0] == 1:
            x = []
            nc = self.aq[0].channels
            for c in range(nc):
                x.append(d[c::nc])
            for c in range(8-nc):
                x.append([])
            d = np.vstack(x)
        if d.shape[0] < 6:
            d = np.pad(d, [(0,8-d.shape[0]),(0,0)])
        elif d.shape[0] > 6:
            d = d[:6,...]
        #print("AQ:",d.shape,af.time-self.at_start,af.sample_rate)
        self.aq.append((d,af.time-self.at_start,af.sample_rate))

    def get_audio_peaks(self, end):
        if not self.aq:
            return None
        
        qe = self.aq[-1][1]
        if qe < end and end:
            return None
        
        #print(len(self.aq), end, end - self.aq[0][1], "t=", self.aq[0][1], self.aq[0][2])
        need = end - self.aq[0][1] if end else 1.0
        if need < 0:
            sys.exit(1)
        assert(need >= 0)
        
        peaks = []
        while self.aq and need > 0:
            (d,t,sr) = self.aq[0]
            nsamp = int(need * sr)
            if nsamp == 0:
                break
            #print(len(self.aq),need, end, "t=", t,sr,nsamp,nsamp/sr,d.shape,t+nsamp/sr)
            if nsamp >= d.shape[1]:
                nsamp = d.shape[1]
                self.aq.pop(0)
            else:
                self.aq[0] = (d[..., nsamp:], t+nsamp/sr, sr)
                d = d[..., 0:nsamp]
            need -= nsamp/sr
            peaks.append(np.max(np.abs(d), axis=1))
        
        return np.max(peaks, axis=0) if peaks else None

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

    def frames(self):
        for frame in self.container.decode(**self.streams):
            if type(frame) is av.AudioFrame:
                self.queue_audio(frame)
            elif type(frame) is av.VideoFrame:
                if self.graph:
                    self.graph.push(frame)
                    try:
                        frame = self.graph.pull()
                    except av.AVError as e:
                        if e.errno != errno.EAGAIN:
                            raise
                        continue
                if self.aq is not None:
                    self.vq.append(frame)
                else:
                    yield (frame,None)
            if len(self.vq) > 1:
                af = self.get_audio_peaks(self.vq[1].time - self.vt_start)
                if af is None:
                    continue
                vf = self.vq.pop(0)
                yield (vf,af)
        while self.vq:
            af = self.get_audio_peaks(None)
            vf = self.vq.pop(0)
            yield (vf,af)
        return #raise StopIteration()

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
    
    get_frame = player.frames()
    percent = ftotal/skip/100.0
    report = math.ceil(ftotal/250)
    p = report
    for _ in range(ftotal):
        (frame,audio) = next(get_frame)
        
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

    # overscan, ignore 3% on each side
    logo_sum[:math.ceil(player.shape[0]*.03)] = 0
    logo_sum[-math.ceil(player.shape[0]*.03)-1:] = 0
    logo_sum[..., 0:math.ceil(player.shape[1]*.03)] = 0
    logo_sum[..., -math.ceil(player.shape[1]*.03)-1:] = 0
    
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
    thresh = int(lmc * .75)
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
    # except ultra wide screen movies, and sometimes ultra-wide commercials?

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
    rt = time.perf_counter()
    p = 0
    print('Processing...', end='\r') 

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

        if frame_blank:
            logo_present = False
            scene_change = True
            prev_bar = None
        else:
            logo_present = check_for_logo(frame, logo)
        
            #s = np.std(fcolor, (0,1,2))
            #m = np.mean(fcolor, (0,1))
            #d = np.median(fcolor, (0,1))
            #x = np.max(fcolor)
            #print(s,m,d,x)

            # the below code is equivalent to:
            #   column = fcolor.mean(axis=(1),dtype='float32').astype('int16')
            # but is almost TEN TIMES faster!
            
            # change 1080x1920x3 into 1080x5760
            fcolor.reshape(-1, fcolor.shape[1]*fcolor.shape[2])

            # pick out the individual color channels by skipping by 3, and then average them
            cr = fcolor[...,0::3].astype('float32').mean(axis=(1))
            cg = fcolor[...,1::3].astype('float32').mean(axis=(1))
            cb = fcolor[...,2::3].astype('float32').mean(axis=(1))
            
            # and now convert back into a 1080x3
            column = np.stack((cb,cg,cr), axis=1).astype('int16')
            
            scene_change = False
            if prev_bar is not None:
                diff = column - prev_bar
                prev_bar = column
                s = np.std(diff, (0))
                if max(s) >= 8:
                    scene_change = True
            else:
                prev_bar = column
        
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

if __name__ == '__main__':
    main(sys.argv)
