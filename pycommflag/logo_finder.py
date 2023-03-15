from typing import Any, BinaryIO
import numpy as np
import scipy.ndimage
import math
import logging as log
from av.video import VideoFrame

from .player import Player
_LOGO_EDGE_THRESHOLD = 85 # how strong an edge is strong enough?

def search(player : Player, search_beginning :bool =False, opts:Any=None) -> tuple|None:
    global _LOGO_EDGE_THRESHOLD

    skip = opts.logo_skip
    
    # search around 1/3 of the way through the show, there should be a lot of show there
    if opts.logo_search_all:
        search_seconds = player.duration
        player.seek(0)
    else:
        search_seconds = 600 if player.duration <= 3700 else 900
        if not search_beginning and player.duration >= search_seconds*2:
            player.seek(player.duration/3 - search_seconds/2)
        else:
            player.seek(0)
    
    fcount = 0
    ftotal = int(search_seconds * player.frame_rate)
    if ftotal <= 65000:
        logo_sum = np.ndarray(player.shape, np.uint16)
    else:
        logo_sum = np.ndarray(player.shape, np.uint32)

    get_frame = player.frames()
    percent = ftotal/skip/100.0
    report = math.ceil(ftotal/250) if not opts.quiet else ftotal * 10
    p = 0
    i = 0
    if not opts.quiet: print("Searching          ", end='\r')
    for (frame,audio) in player.frames():
        i += 1
        if i > ftotal:
            break
        p += 1
        if p%skip != 0:
            continue
        elif p >= report:
            p = 0
            print("Searching, %3.1f%%    " % (min(fcount/percent,100.0)), end='\r')
        data = _gray(frame)
        logo_sum += scipy.ndimage.sobel(data) > _LOGO_EDGE_THRESHOLD
        fcount += 1
    
    if not opts.quiet: print("Searching is complete.\n")

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
        log.info("No logo found (insufficient edge strength)")
        return None
    
    logo_mask = logo_sum >= (best - fcount*.15)
    if np.count_nonzero(logo_mask) < 50:
        log.info("No logo found (not enough edges)")
        return None
    
    nz = np.nonzero(logo_mask)
    top = int(min(nz[0]))
    left = int(min(nz[1]))
    bottom = int(max(nz[0]))
    right = int(max(nz[1]))
    if right - left < 5 or bottom - top < 5:
        log.info("No logo found (bounding box too narrow)")
        return None

    # if the bound is more than half the image then clip it
    if bottom-top >= player.shape[0]/2 or right-left >= player.shape[1]/2:
        pos = np.argwhere(logo_sum >= best)[-1]
        if pos[0] >= player.shape[0]/2:
            top = int(player.shape[0]/2)
        else:
            bottom = int(player.shape[0]/2)
        if pos[1] >= player.shape[1]/2:
            left = int(player.shape[1]/2)
        else:
            right = int(player.shape[1]/2)
        
        log.debug(f"Pre-trunc bounding box: {top},{left}->{bottom},{right}. Max ele={pos}")
        # recalculate after we truncated to shrink down on the real area as best we can
        logo_mask = logo_mask[top:bottom,left:right]
        nz = np.nonzero(logo_mask)
        bottom = top+int(max(nz[0]))
        right = left+int(max(nz[1]))
        top = top+int(min(nz[0]))
        left = left+int(min(nz[1]))
        if right - left < 5 or bottom - top < 5:
            log.info("No logo found (truncated bounding box too narrow)")
            return None
    
    top -= 5
    left -= 5
    bottom += 5
    right += 5
    
    logo_mask = logo_mask[top:bottom,left:right]
    
    log.debug(f"Logo bounding box: {top},{left} to {bottom},{right}")

    lmc = np.count_nonzero(logo_mask)
    if lmc < 20:
        log.info(f"No logo found (not enough edges within bounding box, got {lmc})")
        return None
    thresh = int(lmc * .75)
    return ((top,left), (bottom,right), logo_mask, thresh)

def check_frame(frame :VideoFrame, logo :tuple) -> bool:
    global _LOGO_EDGE_THRESHOLD

    if not logo:
        return False
    
    ((top,left),(bottom,right),lmask,thresh) = logo
    c = _gray(frame, [top,bottom,left,right])
    c = scipy.ndimage.sobel(c) > _LOGO_EDGE_THRESHOLD
    c = np.where(lmask, c, False)
    #print('\n!',np.count_nonzero(c),'of',np.count_nonzero(lmask),'!')
    return np.count_nonzero(c) >= thresh

def _gray(frame:VideoFrame,box:tuple=None) -> np.ndarray:
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

def write(frame_log:BinaryIO, logo:tuple) -> None:
    import struct
    if logo:
        frame_log.write(struct.pack('IIIII', logo[3], logo[0][0], logo[0][1], logo[1][0], logo[1][1]))
        logo[2].astype('uint8').tofile(frame_log)
    else:
        frame_log.write(struct.pack('IIIII', 0, 0, 0, 0, 0))

def read(log_in:BinaryIO) -> tuple|None:
    import struct
    buf = log_in.read(20)
    if len(buf) == 0:
        return None
    info = struct.unpack('IIIII', buf)
    if not info[0]:
        return None
    shape = ((info[3] - info[1]), (info[4] - info[2]))
    data = np.fromfile(log_in, 'uint8', shape[0]*shape[1], '')
    data.shape = shape
    return ((info[1], info[2]), (info[3], info[4]), data.astype('bool'), info[0])

def toimage(logo):
    from PIL import Image
    if logo is None:
        return None
    return Image.fromarray(np.where(logo[2], 255, 0).astype('uint8'), mode="L")

def subtract(data:np.ndarray, logo:tuple)->np.ndarray:
    if logo is None:
        return data
    
    data = np.copy(data)
    ((top,left),(bottom,right),lmask,thresh) = logo
    data[top:bottom,left:right] = 0
    return data
