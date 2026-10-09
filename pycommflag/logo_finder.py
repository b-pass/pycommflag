from av.video import VideoFrame
import logging as log
import math
import numpy as np
from scipy import ndimage
from typing import Any, BinaryIO

from .player import Player

_LOGO_EDGE_THRESHOLD = 12 # minimum contrast step, in gray levels, to count as an edge

_OVERSCAN = .03   # ignore this fraction of each side; signal artifacts live there

def _edges(data:np.ndarray) -> np.ndarray:
    # re-cast so that sobel doesn't wrap the u8
    data = data.astype(np.float32, copy=False)
    # The /4 is the 3x3 kernel's gain for a step edge, which puts the result back on the input's scale -- a step of N gray levels comes out as N.
    mag = np.hypot(ndimage.sobel(data, 0), ndimage.sobel(data, 1)) / 4
    return mag > _LOGO_EDGE_THRESHOLD

def _blank_margins(a:np.ndarray, shape:tuple) -> np.ndarray:
    """Zero the places neither a logo nor an alert banner can be.  In place.

    The frame edges carry signal artifacts that the edge detector happily reports, and
    they are persistent, so they have to go before anything else looks at the map.
    """
    # overscan, ignore 3% on each side -- sometimes there are signal artifacts here (which the edge det sees)
    a[:math.ceil(shape[0]*_OVERSCAN)] = 0
    a[-math.ceil(shape[0]*_OVERSCAN)-1:] = 0
    a[..., 0:math.ceil(shape[1]*_OVERSCAN)] = 0
    a[..., -math.ceil(shape[1]*_OVERSCAN)-1:] = 0

    # no logos in the middle 1/3 of the screen -- and no alert banners either, they always
    # sit near the top or the bottom
    a[int(shape[0]/3):int(shape[0]*2/3),int(shape[1]/3):int(shape[1]*2/3)] = 0
    return a

def _find_stuck(persist:np.ndarray, shape:tuple) -> tuple[list, list]:
    """Locate always-on overlays (weather alerts and the like) in a persistence map.

    These are not solid graphics: typically a colored band with scrolling text, sometimes
    with a static radar map on it.  Summed over the recording that leaves high-persistence
    islands -- the band's edges, the radar -- separated by the moving-text gap, which is
    only patchily persistent.  So finding them is two steps: grow each island to pick up
    the soft skirt around it, then merge nearby islands to recover the band as a whole.

    Returns (remove, report).  `remove` is what to erase before looking for a logo, where
    being too generous is harmless.  `report` is what the blank-frame check should ignore,
    where being too generous means throwing away real picture, so an implausibly large
    region is dropped from it.
    """
    # Something on screen nearly all the time cannot tell show from commercial, so it is
    # useless as a feature and worse than nothing -- it feeds a constant signal the model
    # reads as "show".  That covers an alert banner, a logo on a commercial-free recording,
    # and a station bug that is simply malfunctioning and never goes off.  All three are
    # removed, and no attempt is made to tell them apart.
    #
    # Note this compares *per-pixel* persistence, which runs ~.08-.16 above the fraction of
    # frames the logo is actually detected in, because background content keeps re-lighting
    # the same pixels while the logo is off.  Measured over ten recordings, per-pixel
    # persistence also ORDERS the cases wrongly -- 2755 (a working logo, off for 23% of its
    # runtime) peaks at .909 while 33201 (off for 15%) peaks at .894 -- whereas the duty
    # cycle orders them correctly.  Duty cycle would need a second sampling pass once the
    # mask is known.  Until then .85 is the value that gets all ten right, but only together
    # with _STUCK_CORE_PX, and the margin is thin.
    _STUCK_PERSIST   = .85  # at/above this fraction of frames, it isn't a usable logo
    _STUCK_CORE_PX   = 100  # only bound a core where that many connected pixels agree
    _STUCK_GROW      = .60  # follow each core's skirt down to here, where connected to it
    _STUCK_GROW_PX   = 15   # ...and only this far, so growth can't run across the frame
    _STUCK_MERGE_GAP = .05  # join boxes within this fraction of frame height of each other
    _STUCK_PAD       = 10    # px of slack on the final boxes
    _STUCK_MAX_AREA  = .35  # a merged box larger than this is a detection failure, not an overlay

    core = persist >= _STUCK_PERSIST
    if not core.any():
        return [], []

    # Throw away cores too small to be an overlay before growing them.  A logo's strongest
    # few pixels can poke over the line on their own: on real recordings whose peak sits
    # just under _STUCK_PERSIST the largest connected group above it is 4-10px, and growing
    # one of those would drag the whole logo into a "stuck" box and delete it.  Every part
    # of a real alert -- a band's edges, its labels, a radar map outline -- runs to several
    # hundred pixels, so the two don't overlap.  This also covers the specks (dead pixels,
    # encoder artifacts) that the old post-growth size check was there for -- that check ran
    # after the dilation, by which point a 4px seed had already grown into thousands.
    labels, n = ndimage.label(core, structure=np.ones((3,3), bool))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    keep = sizes >= _STUCK_CORE_PX
    if not keep.any():
        log.debug(f"Ignoring {n} persistent core(s), none over {_STUCK_CORE_PX}px "
                  f"(largest {sizes.max()}px); too small to be an overlay")
        return [], []
    if not keep[1:].all():
        log.debug(f"Ignoring {n - int(keep.sum())} of {n} persistent cores, under {_STUCK_CORE_PX}px")
    core = keep[labels]

    # Grow each core into its own skirt.  The skirt is contiguous with the core, so a
    # geodesic dilation picks it up while a logo elsewhere is left alone; the iteration
    # limit is what keeps it from crawling across a frame full of static scenery.
    region = ndimage.binary_dilation(core, structure=np.ones((3,3), bool),
                                     iterations=_STUCK_GROW_PX, mask=persist >= _STUCK_GROW)

    labels, n = ndimage.label(region, structure=np.ones((3,3), bool))
    boxes = []
    for sl in ndimage.find_objects(labels):
        if sl is None:
            continue
        ys, xs = sl
        boxes.append([ys.start, xs.start, ys.stop, xs.stop])
    if not boxes:
        return [], []

    # Merge the islands.  Two edges of one band are separated by the text gap, so anything
    # within a fraction of the frame height belongs to the same overlay.
    gap = max(1, int(shape[0] * _STUCK_MERGE_GAP))
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(len(boxes)-1, i, -1):
                a, b = boxes[i], boxes[j]
                if (a[0]-gap < b[2] and b[0]-gap < a[2] and
                    a[1]-gap < b[3] and b[1]-gap < a[3]):
                    boxes[i] = [min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3])]
                    del boxes[j]
                    merged = True

    over_y = math.ceil(shape[0]*_OVERSCAN)
    over_x = math.ceil(shape[1]*_OVERSCAN)
    remove, report = [], []
    for t, l, b, r in boxes:
        t, l = max(0, t-_STUCK_PAD), max(0, l-_STUCK_PAD)
        b, r = min(shape[0], b+_STUCK_PAD), min(shape[1], r+_STUCK_PAD)
        # These bands often run right off the top or bottom of the picture, but the
        # overscan margin was blanked before we ever saw it, so the outermost edge is
        # missing from the map.  Anything that reaches the margin gets extended to meet it.
        if t <= over_y: t = 0
        if l <= over_x: l = 0
        if b >= shape[0]-over_y: b = shape[0]
        if r >= shape[1]-over_x: r = shape[1]

        box = ((t,l),(b,r))
        remove.append(box)
        area = (b-t)*(r-l) / float(shape[0]*shape[1])
        if area > _STUCK_MAX_AREA:
            log.warning(f"Stuck region {box} covers {area*100:.0f}% of the frame; removing it "
                        f"from the logo search but not from blank detection")
        else:
            report.append(box)
        log.info(f"Stuck overlay at {box} ({area*100:.1f}% of frame), peak "
                 f"{persist[t:b,l:r].max()*100:.0f}% persistent")
    return remove, report

def search(player:Player, opts:Any=None) -> tuple|None:
    player.disable_audio()
    player.seek(0)

    logo_sum = np.zeros(player.shape, np.uint32)

    skip = round(float(player.frame_rate / opts.logo_samples))
    if skip < 1: skip = 1
    ftotal = int(min(3600, player.duration) * player.frame_rate / skip)
    percent = ftotal/100.0
    report = math.ceil(percent/4) if not opts.quiet else ftotal * 2
    
    fcount = 0
    r = 0
    if not opts.quiet: print("Logo Searching          ", end='\r')
    for frame in player.frames_stride(skip):
        r += 1
        if r >= report:
            r = 0
            print("Logo Searching, %3.1f%%    " % (min(fcount/percent,100.0)), end='\r')
        data = _gray(frame)
        logo_sum += _edges(data)
        fcount += 1
        if fcount >= ftotal: 
            break
    
    if not opts.quiet: print("Logo Searching is complete.\n")

    return _analyze(logo_sum, fcount, player.shape)

def _analyze(logo_sum:np.ndarray, fcount:int, shape:tuple) -> tuple|None:
    """Turn a per-pixel count of "was an edge in this frame" into a logo, or None.

    Split out of search() so the same decision logic can be re-run over an accumulation
    gathered at a different edge threshold, without decoding the video again.
    """
    if fcount < 1:
        log.info("No logo found (no frames were sampled)")
        return None

    _blank_margins(logo_sum, shape)

    # Anything that never goes away is an overlay rather than a logo, so cut it out and
    # look for the logo in what is left.
    #
    # Note this does NOT bound the peak that survives.  Zeroing every pixel over
    # _STUCK_PERSIST was tried and reverted: on a light-commercial recording with a real
    # logo (2755, duty .766, which separates true show from true commercial 87.8% to 0.3%)
    # it deleted 340px -- 14% of the mask, and the strongest edges in it -- then dragged the
    # mask band down to admit weaker ones in their place.  A logo's per-pixel persistence
    # legitimately exceeds _STUCK_PERSIST; only its duty cycle says whether it is usable,
    # and that is not knowable from this map.  The abutment check below is what keeps an
    # unbounded fragment from standing in for a logo.
    remove, stuck = _find_stuck(logo_sum / fcount, shape)
    for (t,l),(b,r) in remove:
        logo_sum[t:b, l:r] = 0

    best = np.max(logo_sum)
    log.debug(f"Logo detection result: {best} ({round(best*100/fcount)}%)")

    if best <= fcount*.55:
        log.info(f"No logo found (insufficient edge strength, best={best*100/fcount}%)")
        return None
    
    logo_mask = logo_sum >= (best - fcount*.15)

    if np.count_nonzero(logo_mask) < 50:
        log.info("No logo found (not enough edges)")
        return None
    
    nz = np.nonzero(logo_mask)
    top = int(np.min(nz[0]))
    left = int(np.min(nz[1]))
    bottom = int(np.max(nz[0]))
    right = int(np.max(nz[1]))
    
    # if the bound is more than half the image then clip it
    if bottom-top >= shape[0]/2 or right-left >= shape[1]/2:
        log.debug(f"Need to clip logo bounding box {top},{left}->{bottom},{right}, it is too large")

        h = shape[0]//2
        w = shape[1]//2
        count = 0
        top = left = 0
        bottom = h
        right = w

        for y in (0, h):
            for x in (0, w):
                c = np.count_nonzero(logo_mask[y:y+h, x:x+w])
                if c >= count:
                    count = c
                    top = y
                    left = x
        bottom = top + h
        right = left + w

        # recalculate after we truncated to shrink down on the real area as best we can
        nz = np.nonzero(logo_mask[top:bottom,left:right])
        bottom = top + int(np.max(nz[0]))
        right = left + int(np.max(nz[1]))
        top += int(np.min(nz[0])) 
        left += int(np.min(nz[1]))

        log.debug(f"Clipped to bounding box {top},{left}->{bottom},{right} with count={count}")
    
    if right - left < 5 or bottom - top < 5:
        log.info(f"No logo found (bounding box {top},{left}->{bottom},{right} is too small)")
        return None

    filt = 0
    for y in range(top+5,bottom-5+1):
        for x in range(left+5,right-5+1):
            if logo_mask[y,x]:
                ok = False
                for yo in range(-2,3):
                    for xo in range(-2,3):
                        if (xo or yo) and logo_mask[y+yo,x+xo]:
                            ok = True
                if not ok:
                    logo_mask[y,x] = False
                    filt += 1
    if filt:
        log.debug(f"Filtered {filt} logo mask elements that were isolated from others")
        nz = np.nonzero(logo_mask[top:bottom+1,left:right+1])
        bottom = top + int(np.max(nz[0]))
        right = left + int(np.max(nz[1]))
        top += int(np.min(nz[0])) 
        left += int(np.min(nz[1]))

    if right - left < 5 or bottom - top < 5:
        log.info(f"No logo found (truncated bounding box at {top},{left}->{bottom},{right} too small)")
        return None
    
    top -= 2
    left -= 2
    bottom += 2
    right += 2

    # An overlay's skirt is not always connected to its core.  On one measured recording the
    # leftover sat a single pixel below a removed banner with nothing above .50 joining the
    # two, so no amount of geodesic growth reaches it -- and what got returned was a sparse
    # scatter of edges, not a logo.  Anything found this close to a region we just deleted
    # belongs to that region; a real logo that near an overlay was already swallowed by the
    # box's own padding, and would be unusable anyway.
    _STUCK_ABUT = 12  # a candidate this close to a removed overlay is part of it
    for (bt,bl),(bb,br) in remove:
        if (bt - _STUCK_ABUT < bottom and top < bb + _STUCK_ABUT and
            bl - _STUCK_ABUT < right  and left < br + _STUCK_ABUT):
            log.info(f"No logo found (candidate {top},{left}->{bottom},{right} abuts the stuck "
                     f"region {bt},{bl}->{bb},{br}, so it is part of it)")
            return None

    logo_mask = logo_mask[top:bottom,left:right]
    
    lmc = np.count_nonzero(logo_mask)
    if lmc < 20:
        log.info(f"No logo found (not enough edges within bounding box, got {lmc})")
        return None
    thresh = round(lmc * 2 / 3)
    
    log.debug(f"Final logo bounding box: {top},{left}->{bottom},{right} count_threshold={thresh}")

    return ((top,left), (bottom,right), logo_mask, thresh, *stuck)

def logo_in_frame(frame :VideoFrame, logo :tuple) -> tuple[int, int]:
    if not logo:
        return (0,1)

    ((top,left),(bottom,right),lmask,thresh,*_) = logo
    c = _gray(frame, [top,bottom,left,right])
    c = _edges(c)
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
def from_json(js:list|str)->list|None:
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

def keep_mask(shape:tuple, logo:tuple)->np.ndarray|None:
    """True where a pixel should count toward whole-frame statistics, or None for all of them.

    Blank detection has to ignore the logo and any stuck overlay: during a break the
    picture goes black but those stay lit, and the frame only reads as blank once they are
    out of the sample.  Excluding the pixels rather than blacking them out keeps the
    median and standard deviation honest -- forced zeros would drag both.
    """
    if logo is None:
        return None
    ((top,left),(bottom,right),lmask,thresh,*stuck) = logo
    keep = np.ones(shape[:2], bool)
    keep[top:bottom,left:right] = False
    for ((top,left),(bottom,right)) in stuck:
        keep[top:bottom,left:right] = False
    return keep
