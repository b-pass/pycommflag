from typing import Any, BinaryIO
import numpy as np
import scipy.ndimage
import math
import logging as log
from av.video import VideoFrame

from .player import Player
_LOGO_EDGE_THRESHOLD = 85 # how strong an edge is strong enough?

def search(player:Player, search_beginning:bool=False, opts:Any=None) -> tuple|None:
    global _LOGO_EDGE_THRESHOLD

    skip = opts.logo_skip
    
    if opts.logo_search_all:
        search_seconds = player.duration
        player.seek(0)
    else:
        search_seconds = 600 if player.duration <= 3700 else 900
        if not search_beginning and player.duration >= search_seconds*2:
            player.seek(player.duration/2-search_seconds/2)
        else:
            player.seek(0)
    
    fcount = 0
    ftotal = int(search_seconds * player.frame_rate)
    logo_sum = np.zeros(player.shape, np.uint32)
    percent = ftotal/skip/100.0
    report = math.ceil(ftotal/250) if not opts.quiet else ftotal * 10
    p = 0
    i = 0
    if not opts.quiet: print("Searching          ", end='\r')
    for frame in player.frames():
        i += 1
        if i > ftotal:
            break
        p += 1
        if p%skip != 0:
            continue
        elif p >= report:
            p = 0
            print("Logo Searching, %3.1f%%    " % (min(fcount/percent,100.0)), end='\r')
        data = _gray(frame)
        logo_sum += scipy.ndimage.sobel(data) > _LOGO_EDGE_THRESHOLD
        fcount += 1
    
    if not opts.quiet: print("Logo Searching is complete.\n")

    # overscan, ignore 3% on each side -- sometimes there are signal artifacts here (which the edge det sees)
    logo_sum[:math.ceil(player.shape[0]*.03)] = 0
    logo_sum[-math.ceil(player.shape[0]*.03)-1:] = 0
    logo_sum[..., 0:math.ceil(player.shape[1]*.03)] = 0
    logo_sum[..., -math.ceil(player.shape[1]*.03)-1:] = 0
    
    # no logos in the middle 1/3 of the screen
    logo_sum[int(player.shape[0]/3):int(player.shape[0]*2/3),int(player.shape[1]/3):int(player.shape[1]*2/3)] = 0

    # in case we found something stuck on the screen, try to look beyond that
    stuck = []
    best = np.max(logo_sum)
    while best > fcount*.95:
        stuck_mask = logo_sum >= fcount*.8
        
        ((top,left), (bottom,right)) = refine_logo(player, stuck_mask, allow_failure=False)
        stuck.append( ((top,left), (bottom,right)) )
        
        pos = (top + (bottom-top)//2, left + (right-left)//2)
        log.info(f'Stuck logo ({best*100/fcount}%) at {pos[0]},{pos[1]} -- nuke quarter of the screen')
        if pos[0] >= player.shape[0]/2:
            if pos[1] >= player.shape[1]/2:
                logo_sum[player.shape[0]//2:,player.shape[1]//2:] = 0
            else:
                logo_sum[player.shape[0]//2:, :player.shape[1]//2] = 0
        else:
            if pos[1] >= player.shape[1]/2:
                logo_sum[:player.shape[0]//2, player.shape[1]//2:] = 0
            else:
                logo_sum[:player.shape[0]//2, :player.shape[1]//2] = 0
        
        best = np.max(logo_sum)
        log.debug(f'New best = {best*100/fcount}% of {fcount}')

    if best <= fcount / 3:
        log.info(f"No logo found (insufficient edge strength, best={best*100/fcount}%)")
        return None
    
    logo_mask = logo_sum >= (best - fcount*.15)

    res = refine_logo(player, logo_mask)
    if res is None: return None
    ((top,left), (bottom,right)) = res
    
    log.debug(f"Final logo bounding box: {top},{left}->{bottom},{right}")

    logo_mask = logo_mask[top:bottom,left:right]
    filt = 0
    for y in range(5,logo_mask.shape[0]-10):
        for x in range(5,logo_mask.shape[1]-10):
            if logo_mask[y,x]:
                ok = False
                for yo in range(-2,3):
                    for xo in range(-2,3):
                        if (xo or yo) and logo_mask[y+yo,x+xo]:
                            ok = True
                            break
                if not ok:
                    logo_mask[y,x] = False
                    filt += 1
    if filt:
        log.debug(f"Filtered {filt} logo mask elements that were isolated from others")
    
    lmc = np.count_nonzero(logo_mask)
    if lmc < 20:
        log.info(f"No logo found (not enough edges within bounding box, got {lmc})")
        return None
    thresh = round(lmc * 2 / 3)

    return ((top,left), (bottom,right), logo_mask, thresh, *stuck)

def refine_logo(player, logo_mask, allow_failure=True):
    if allow_failure and np.count_nonzero(logo_mask) < 50:
        log.info("No logo found (not enough edges)")
        return None
    
    nz = np.nonzero(logo_mask)
    top = int(np.min(nz[0]))
    left = int(np.min(nz[1]))
    bottom = int(np.max(nz[0]))
    right = int(np.max(nz[1]))
    if allow_failure and right - left < 5 or bottom - top < 5:
        log.info(f"No logo found (bounding box {top},{left}->{bottom},{right} is too small)")
        return None

    # if the bound is more than half the image then clip it
    if bottom-top >= player.shape[0]/2 or right-left >= player.shape[1]/2:
        pos = (top + (bottom-top)//2, left + (right-left)//2)
        
        if pos[0] >= player.shape[0]/2:
            top = int(player.shape[0]/2)
        else:
            bottom = int(player.shape[0]/2)
        if pos[1] >= player.shape[1]/2:
            left = int(player.shape[1]/2)
        else:
            right = int(player.shape[1]/2)
        
        # recalculate after we truncated to shrink down on the real area as best we can
        nz = np.nonzero(logo_mask[top:bottom,left:right])
        bottom = top+int(max(nz[0]))
        right = left+int(max(nz[1]))
        top = top+int(min(nz[0]))
        left = left+int(min(nz[1]))
        if allow_failure and right - left < 5 or bottom - top < 5:
            log.info(f"No logo found (truncated bounding box at {top},{left}->{bottom},{right} too small)")
            return None
    
    top -= 5
    left -= 5
    bottom += 5
    right += 5

    return ((top,left), (bottom,right))

def logo_in_frame(frame :VideoFrame, logo :tuple) -> tuple[int, int]:
    if not logo:
        return (0,1)
    global _LOGO_EDGE_THRESHOLD
    
    ((top,left),(bottom,right),lmask,thresh,*_) = logo
    c = _gray(frame, [top,bottom,left,right])
    c = scipy.ndimage.sobel(c) > _LOGO_EDGE_THRESHOLD
    c = np.where(lmask, c, False)
    #print('\n!',np.count_nonzero(c),'of',np.count_nonzero(lmask),'!')
    n = np.count_nonzero(c)
    return n, thresh

def check_frame(frame :VideoFrame, logo :tuple) -> bool:
    (n,t) = logo_in_frame(frame, logo)
    return n >= t

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

import json
def from_json(js:list|str)->tuple|None:
    if type(js) is str:
        js = json.loads(js)
    if js is None:
        return None
    if type(js) != list:
        raise Exception("Expected list")
    assert(len(js) >= 4)
    js = list(js)
    js[2] = np.array(js[2], 'bool')
    return js

def to_json(logo:tuple)->str:
    if logo is None:
        return json.dumps(None)
    
    simplified = list(logo)
    simplified[2] = logo[2].astype('uint8').tolist()
    
    return json.dumps(simplified)

def toimage(logo):
    from PIL import Image
    if logo is None:
        return None
    return Image.fromarray(np.where(logo[2], 255, 0).astype('uint8'), mode="L")

def subtract(data:np.ndarray, logo:tuple)->np.ndarray:
    if logo is not None:
        ((top,left),(bottom,right),lmask,thresh,*stuck) = logo
        data[top:bottom,left:right] = 0
        if stuck is not None and len(stuck) > 0:
            for ((top,left),(bottom,right)) in stuck:
                data[top:bottom,left:right] = 0    
    return data
