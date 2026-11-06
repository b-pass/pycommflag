import tkinter as tk
import tkinter.font as tkfont
import bisect
import logging as log
import math
import queue
import threading
import time
import numpy as np
from PIL import ImageTk, Image
from .player import Player
from . import logo_finder
from . import processor
from . import neural
from . import timeline
from .feature_span import *

INFO_LINES = 16 # in the info box, which must not change height
LOGO_DEBOUNCE = 3 # log frames a new logo state must last before Logo |</>| count it as a change

# the review queue (see review_items)
REVIEW_UNSURE = (0.15, 0.85) # model probabilities it asks about wherever they are
REVIEW_EDGE_GAP = 10 # seconds; other reasons to stop this close to an edge are left to the edge's own stop
REVIEW_MERGE = 30 # seconds; reasons to stop this close together (away from edges) make one stop
REVIEW_LOGO_OFF = 10 # seconds the logo must be gone, inside show, to ask about it
REVIEW_LOGO_ON = 20 # seconds the logo must be up, inside a break, to ask about it
REVIEW_DOUBT_LEN = 3 # seconds the model must be unsure for, when it doesn't disagree with the tags outright
REVIEW_BLANK_SAME = 5 # seconds; blanks this close together are one fade (or flicker), not the ends of ads
REVIEW_BLANK_GAP = 65 # seconds; blanks inside show this close together are spaced like the ends of ads
REVIEW_BLANKS = 3 # blanks spaced like that before it asks (a missed break of two ads has three)
REVIEW_TOL = 0.5 # seconds; a stop this close to the position (or to an edge just flagged) has been seen

# the contact sheet
SHEET_STEP = 10 # seconds per thumbnail
SHEET_COLS = 12 # thumbnails per row, so a row is two minutes
SHEET_THUMB = (96, 54)

def log_fixes(flog:dict, path:str) -> tuple[list[str], str|None]:
    """What the log at `path` is missing that tools/upgrade_log.py can add, and a shell command (with
    the python running this, so it works from the same venv) that runs it on the log, or None."""
    import os, shlex, sys
    tool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools', 'upgrade_log.py')
    header = flog.get('frames_header', ())
    problems = []
    # extraction before 2026-10-23 timed the audio from the audio stream's own start, so in a log
    # without audio_shift every audio column is late (0.2-1s); training on it as-is teaches that
    if 'audio_shift' not in flog and 'fvol' in header:
        problems.append("its audio is late (no audio_shift): repair it before training on it")
    # logs before file_version 11 have no captions; they can be added from the video in seconds
    if header and 'cc_text' not in header:
        problems.append("it has no caption data (no cc_text)")
    if not problems:
        return problems, None
    q = shlex.quote
    return problems, f'{q(sys.executable)} {q(tool)} {q(os.path.abspath(path))}'

def _review_kind(t):
    # intro and credits train as show (see load_nonpersistent), so the queue counts them as show
    if isinstance(t, int):
        t = SceneType(t)
    return t if t in (SceneType.COMMERCIAL, SceneType.DO_NOT_USE) else SceneType.SHOW

def review_items(tags, raw, logo_spans, blank_mids, opts) -> list:
    """Where the editor's review queue stops, as (time, why, is_edge) in time order.

    It stops at the start and end of the recording, at every edge between show, break and
    ignored, and, away from those, wherever the tags might be missing a break (or a piece of show
    inside one): the model unsure of or disagreeing with the tags, the logo gone from show or up in
    a break, and blanks clustered inside show the way they fall between ads.

    `tags` cover the whole recording, as Window.tags do; `raw` is raw_predict's (time, probability)
    list, `logo_spans` the debounced (on, (b,e)) logo runs (empty without a logo) and `blank_mids`
    the sorted middles of the blanks.
    """
    names = Window.TAG_NAMES
    runs = []
    for (t,(b,e)) in tags:
        k = _review_kind(t)
        if runs and runs[-1][0] == k:
            runs[-1] = (k, (runs[-1][1][0], e))
        else:
            runs.append((k, (b, e)))
    if not runs:
        return []
    starts = np.array([b for (_,(b,_)) in runs], dtype='float64')
    edges = starts[1:]
    # the recording's first and last second are always stops, and count as edges for what is near one
    (first, last) = (float(starts[0]), max(float(starts[0]), runs[-1][1][1] - 1))
    bounds = np.concatenate(([first], edges, [last]))
    def run_at(t):
        return max(0, int(np.searchsorted(starts, t, 'right')) - 1)
    def far(t):
        return float(np.min(np.abs(bounds - t))) > REVIEW_EDGE_GAP

    # lengths post_predict wouldn't produce go with the edge that starts them (or ends the first)
    notes = [[] for _ in edges]
    for (j,(k,(b,e))) in enumerate(runs):
        n = None
        if k == SceneType.COMMERCIAL and e - b < opts.break_min_len:
            n = f'short break {e-b:.0f}s'
        elif k == SceneType.COMMERCIAL and e - b > opts.break_max_len:
            n = f'long break {e-b:.0f}s'
        elif (k == SceneType.SHOW and 0 < j < len(runs)-1 and e - b < opts.show_min_len
              and runs[j-1][0] == runs[j+1][0] == SceneType.COMMERCIAL):
            n = f'short show {e-b:.0f}s'
        if n and len(edges):
            notes[j-1 if j > 0 else 0].append(n)
    items = [(float(edges[i]), '; '.join([f'{names[runs[i][0]]} -> {names[runs[i+1][0]]}'] + notes[i]), True)
             for i in range(len(edges))]
    items += [(first, 'start of recording', True), (last, 'end of recording', True)]

    # (start, end, reason, value): value is the model's probability (worst one wins when merged),
    # or seconds of logo, or a count of blanks (both added up)
    other = []
    if raw:
        rt = np.array([t for (t,_) in raw], dtype='float64')
        p = np.array([v for (_,v) in raw], dtype='float64')
        idx = np.clip(np.searchsorted(starts, rt, 'right') - 1, 0, len(runs)-1)
        brk = np.array([runs[i][0] == SceneType.COMMERCIAL for i in idx])
        dnu = np.array([runs[i][0] == SceneType.DO_NOT_USE for i in idx])
        doubt = (((p > REVIEW_UNSURE[0]) & (p < REVIEW_UNSURE[1])) | ((p >= 0.5) != brk)) & ~dnu
        doubt &= np.abs(rt[:,None] - bounds[None,:]).min(axis=1) > REVIEW_EDGE_GAP
        i = 0
        while i < len(rt):
            if not doubt[i]:
                i += 1
                continue
            j = i
            while j+1 < len(rt) and doubt[j+1] and idx[j+1] == idx[i]:
                j += 1
            seg = p[i:j+1]
            # a second or two of mild doubt is everywhere; disagreeing with the tags is not
            if j - i >= REVIEW_DOUBT_LEN - 1 or np.any((seg >= 0.5) != brk[i]):
                if brk[i]:
                    other.append((float(rt[i]), float(rt[j]) + 1, 'down', float(seg.min())))
                else:
                    other.append((float(rt[i]), float(rt[j]) + 1, 'up', float(seg.max())))
            i = j + 1

    for (on,(b,e)) in logo_spans:
        k = runs[run_at(b)][0]
        if run_at(b) != run_at(e) or not far(b):
            continue
        if not on and k == SceneType.SHOW and e - b >= REVIEW_LOGO_OFF:
            other.append((float(b), float(e), 'logo off', float(e - b)))
        elif on and k == SceneType.COMMERCIAL and e - b >= REVIEW_LOGO_ON:
            other.append((float(b), float(e), 'logo on', float(e - b)))

    # blanks inside show, away from edges; one fade can read as several blanks a few frames apart
    events = []
    for m in blank_mids:
        if runs[run_at(m)][0] != SceneType.SHOW or not far(m):
            continue
        if events and m - events[-1] <= REVIEW_BLANK_SAME and run_at(m) == run_at(events[-1]):
            continue
        events.append(float(m))
    group = []
    for m in events + [None]:
        if m is not None and (not group or (m - group[-1] <= REVIEW_BLANK_GAP and run_at(m) == run_at(group[0]))):
            group.append(m)
            continue
        if len(group) >= REVIEW_BLANKS:
            other.append((group[0], group[-1], 'blanks', float(len(group))))
        group = [m]

    merged = [] # [start, end, {reason: value}]
    for (b, e, why, v) in sorted(other):
        if merged and b - merged[-1][1] <= REVIEW_MERGE:
            m = merged[-1]
            m[1] = max(m[1], e)
            if why not in m[2]:
                m[2][why] = v
            elif why == 'up':
                m[2][why] = max(m[2][why], v)
            elif why == 'down':
                m[2][why] = min(m[2][why], v)
            else:
                m[2][why] += v
        else:
            merged.append([b, e, {why: v}])
    fmt = {'up': lambda v: f'model up to {v*100:.0f}% in Show',
           'down': lambda v: f'model down to {v*100:.0f}% in Break',
           'logo off': lambda v: f'logo off {v:.0f}s in Show',
           'logo on': lambda v: f'logo on {v:.0f}s in Break',
           'blanks': lambda v: f'{v:.0f} blanks in Show'}
    for (b, e, whys) in merged:
        items.append((b, '; '.join(fmt[w](v) for (w,v) in whys.items()) + f' (over {e-b:.0f}s)', False))
    return sorted(items)

class Window(tk.Tk):
    def __init__(self, opts, video, flog):
        tk.Tk.__init__(self)
        self.title("pycommflag editor")
        # Tk's own close would destroy the window without going through destroy() below
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        self.player = Player(video, no_deinterlace=True)
        # log time - the video's: not 0 when the video is a transcode that moved its clock
        (offset, source) = timeline.video_offset(flog, video, measure=False)
        if offset is None:
            log.warning(f"Where the log falls in {video} isn't known; showing it as if it lined up")
        self.video_offset = offset or 0.0
        # the main frames and the -5s/+5s thumbnails each get their own decoder, so they never
        # seek each other's position away (which is what makes small steps cheap)
        self.cursor = _Cursor(self.player, offset=self.video_offset)
        self.thumb_cursors = [_Cursor(Player(video, no_deinterlace=True), keep=int(1.5*self.player.frame_rate),
                                      offset=self.video_offset)
                              for _ in range(2)]
        self.thumb_job = None

        self.duration = self.player.duration
        self.frame_rate = self.player.frame_rate
        self.result = None
        self.position = 0
        self.prev_frame_time = 0
        self.next_frame_time = 1/self.frame_rate
        self.settype = None
        self.setpos = 0
        self.opts = opts
        self.reviewed = [] # times of review stops already seen
        self.review_at = None # the stop the review buttons last went to
        self.review_cache = None
        
        self.logo = processor.read_logo(flog)
        self.spans = processor.read_feature_spans(flog)
        # 'diff' arrives as per-frame magnitudes; the map row and the |< Diff / Diff >| buttons
        # want scene change marks, so threshold it into that shape here
        self.spans['diff'] = [(True, (t, t)) for (t, v) in self.spans.get('diff', [])
                              if v >= neural.DIFF_THRESHOLD]
        # the log's per-frame rows, for the info panel
        self.frames = [f for f in flog['frames'] if f is not None]
        self.ftimes = np.array([f[0] for f in self.frames])
        self.fstep = float(np.median(np.diff(self.ftimes[:1000]))) if len(self.ftimes) > 1 else 1.0
        # the logo detection flickers, so for skipping (and the info box, which shows where the
        # skips go) a change only counts once the new state has held for LOGO_DEBOUNCE frames
        self.logo_spans = []
        state = None
        for f in self.frames:
            v = bool(f[1])
            if state is None:
                (state, start, run) = (v, f[0], 0)
            elif v == state:
                run = 0
            else:
                if run == 0:
                    run_start = f[0]
                run += 1
                if run >= LOGO_DEBOUNCE:
                    self.logo_spans.append((state, (start, run_start)))
                    (state, start, run) = (v, run_start, 0)
        if state is not None:
            self.logo_spans.append((state, (start, self.ftimes[-1])))
        # what training's _adjust_tags snaps a tag edge onto
        self.blank_mids = np.array([b + (e-b)/2 for (v,(b,e)) in self.spans.get('blank', []) if v])
        self.diff_mags = np.array([f[3] for f in self.frames], dtype='float32')
        # load_nonpersistent crops a leading/trailing DO_NOT_USE out of the log it is given (deleting
        # those tags, frames and shortening duration), which would then be lost when this saves
        self.raw = neural.raw_predict(dict(flog, tags=list(flog.get('tags', [])), frames=list(flog['frames'])), opts)
        self.raw_times = np.array([t for (t,_) in self.raw])
        tags = processor.read_tags(flog)
        if not tags or opts.reprocess:
            (times,preds) = zip(*self.raw)
            tags = neural.post_predict(flog, list(preds), list(times), opts)
        
        p = 0
        self.tags = []
        if type(tags) is FeatureSpan:
            tags = tags.to_list()
        if tags and len(tags[-1][1]) < 2:
            tags[-1] = (tags[-1][0], (tags[-1][1][0], self.duration))
        for (t,(b,e)) in tags:
            if b is None or t is None or e is None: continue
            if b > p:
                self.tags.append((SceneType.SHOW,(p,b)))
            if type(t) is int:
                t = SceneType(t)
            self.tags.append((t,(b,e)))
            p = e
        tags = None
        if p < self.duration:
            self.tags.append((SceneType.SHOW,(p,self.duration)))

        self.misc = []
        self.video_labels = []
        self.images = []
        for x in range(5):
            v = tk.Label(self)
            if x == 2:
                v.grid(row=1,column=1, columnspan=3)
            else:
                v.grid(row=2, column=x)
            self.video_labels.append(v)

        a = tk.Label(self, text="-5s")
        a.grid(row=3, column=0, sticky="nswe")
        self.misc.append(a)
            
        a = tk.Label(self, text="-1f")
        a.grid(row=3, column=1, sticky="nswe")
        self.misc.append(a)
        
        self.pos_label = tk.Label(self, text="00:00.000")
        self.pos_label.grid(row=2, column=2, sticky="nswe")

        a = tk.Label(self, text="+1f")
        a.grid(row=3, column=3, sticky="nswe")
        self.misc.append(a)
        
        a = tk.Label(self, text="+5s")
        a.grid(row=3, column=4, sticky="nswe")
        self.misc.append(a)

        skipf = tk.Frame(self)
        skipf.grid(row=4, column=0, columnspan=5)
        self.misc.append(skipf)

        b = tk.Button(skipf, text="<<<< 30s", command=lambda:self.move(seconds=-30))
        b.grid(row=0, column=0, padx=5)
        self.misc.append(b)
        
        b = tk.Button(skipf, text="<<< 5s", command=lambda:self.move(seconds=-5))
        b.grid(row=0, column=1, padx=5)
        self.misc.append(b)

        b = tk.Button(skipf, text="<< 1s", command=lambda:self.move(seconds=-1))
        b.grid(row=0, column=3, padx=5)
        self.misc.append(b)
        
        b = tk.Button(skipf, text="< 1f", command=lambda:self.move(abs=self.prev_frame_time))
        b.grid(row=0, column=4, padx=(5,20))
        self.misc.append(b)

        b = tk.Button(skipf, text="1f >", command=lambda:self.move(abs=self.next_frame_time))
        b.grid(row=0, column=5, padx=(20,5))
        self.misc.append(b)
        
        b = tk.Button(skipf, text="1s >>", command=lambda:self.move(seconds=1))
        b.grid(row=0, column=6, padx=5)
        self.misc.append(b)
        
        b = tk.Button(skipf, text="5s >>>", command=lambda:self.move(seconds=5))
        b.grid(row=0, column=8, padx=5)
        self.misc.append(b)

        b = tk.Button(skipf, text="30s >>>>", command=lambda:self.move(seconds=30))
        b.grid(row=0, column=9, padx=5)
        self.misc.append(b)

        skips = tk.Frame(self)
        skips.grid(row=5, column=0, columnspan=5)
        self.misc.append(skips)

        # packed rather than gridded, so the review line below can't stretch the buttons apart
        line = tk.Frame(skips)
        line.pack()
        for (text, cmd, pad) in (("|< Break", lambda:self.prev('break'), 5),
                                 ("|< Blank", lambda:self.prev('blank'), 5),
                                 ("|< Logo", lambda:self.prev('logo'), 5),
                                 ("|< Diff", lambda:self.prev('diff'), (5,20)),
                                 ("Diff >|", lambda:self.next('diff'), (20,5)),
                                 ("Logo >|", lambda:self.next('logo'), 5),
                                 ("Blank >|", lambda:self.next('blank'), 5),
                                 ("Break >|", lambda:self.next('break'), 5)):
            b = tk.Button(line, text=text, command=cmd)
            b.pack(side=tk.LEFT, padx=pad)
            self.misc.append(b)

        line = tk.Frame(skips)
        line.pack(pady=(5,0))
        for (text, cmd) in (("|< Review", lambda:self.review(-1)),
                            ("Contact sheet", lambda:self.sheet.show()),
                            ("Review >|", lambda:self.review(1))):
            b = tk.Button(line, text=text, command=cmd)
            b.pack(side=tk.LEFT, padx=5)
            self.misc.append(b)
        # a fixed width, so the changing text can't widen the window
        self.review_label = tk.Label(skips, width=100)
        self.review_label.pack()

        self.bind('n', lambda e:self.review(1))
        self.bind('p', lambda e:self.review(-1))
        
        tags = tk.Frame(self)
        tags.grid(row=6, column=0, columnspan=5)
        self.misc.append(tags)

        self.taggers = []
        
        self.tag_cancel = tk.Button(tags, text='Cancel This Flag')

        self.taggers.append((tk.Button(tags), SceneType.COMMERCIAL, 'Break'))
        self.taggers.append((tk.Button(tags), SceneType.SHOW, 'Show'))
        self.taggers.append((tk.Button(tags), SceneType.DO_NOT_USE, 'Ignore'))

        c = 0
        for (b,t,l) in self.taggers:
            b.grid(row=0, column=c, padx=5)
            b.configure(command=lambda x=c:self.do_tag(x), text=f"Flag {l}")
            c += 1        

        #septags = tk.Label(tags, text=" ")
        #septags.grid(row=0, column=c, padx=10)
        #self.misc.append(septags)
        #c += 1

        eof = tk.Button(tags, text="Flag End/Truncate & Save & Exit", command=lambda:self.truncate_now())
        eof.grid(row=0, column=c, padx=5)
        self.misc.append(eof)
        c += 1
        
        save = tk.Button(tags, text="Save & Exit", command=lambda:self.save_and_close())
        save.grid(row=0, column=c, padx=5)
        self.misc.append(save)
        c += 1
        
        self.map_height = 180
        self.map_width = 1280
        self.mapCanvas = tk.Canvas(self, width=self.map_width, height=self.map_height)
        self.mapCanvas.grid(row=7, column=0, columnspan=5)
        self.mapCanvas.bind("<Button-1>", lambda e:self.move(abs=float(e.x)/self.map_width*self.duration))
        
        self.scale_pos = tk.DoubleVar()
        self.scroller = tk.Scale(self, 
                                from_=0, to_=self.duration/60, resolution=1/60, tickinterval=5, showvalue=False,
                                length=self.map_width+25, orient=tk.HORIZONTAL, sliderlength=25,
                                variable=self.scale_pos,
                                command=lambda n: self.move(abs=float(n)*60))
        self.scroller.grid(row=9, column=0, columnspan=5)

        shifted = f'; {self.video_offset:+.3f}s behind the log ({source})' if self.video_offset else ''
        self.info = tk.Label(self, text=f'File: {video}{shifted}; Length:{self.duration/60.0:0.1f} mins; {float(self.frame_rate)} fps')
        self.info.grid(row=10, column=0, sticky="se", columnspan=5)

        # the tools rewrite the log, and saving here would write back the copy loaded before they ran,
        # so they have to wait until this closes
        problems, cmd = log_fixes(flog, opts.feature_log or '')
        if cmd:
            msg = f"This log: {'; '.join(problems)}. Close this, then run (the tags are kept):"
            log.warning(f'{msg}\n{cmd}')
            bg = 'red' if 'audio_shift' in problems[0] else 'dark orange'
            banner = tk.Frame(self, bg=bg)
            banner.grid(row=0, column=0, columnspan=5, sticky='we')
            tk.Label(banner, text=msg, bg=bg, fg='white', wraplength=self.map_width, pady=2).pack(fill='x')
            # an Entry rather than a Label, so the command can be selected and copied
            cmdvar = tk.StringVar(value=cmd)
            self.misc.append(cmdvar)
            tk.Entry(banner, textvariable=cmdvar, state='readonly', relief=tk.FLAT).pack(fill='x', padx=4, pady=(0, 4))

        limg = logo_finder.toimage(self.logo)
        if limg:
            self.misc.append(limg)
            limg = ImageTk.PhotoImage(limg)
            self.misc.append(limg)
        v = tk.Label(self, relief=tk.SUNKEN)
        v.grid(row=1, column=0)
        self.misc.append(v)
        if limg:
            v.configure(image=limg)
        else:
            v.configure(text='[No logo]')
        
        # a fixed-size box the size of the +5s thumbnail below it, so the text can't widen that column
        infof = tk.Frame(self, width=320, height=360)
        infof.grid(row=1, column=4, sticky='nw')
        infof.grid_propagate(False)
        # sized once for ~30 chars x INFO_LINES, so it doesn't jump around while stepping
        self.info_font = tkfont.nametofont('TkFixedFont').copy()
        size = self.info_font.cget('size')
        while abs(size) > 6 and (self.info_font.measure('0'*30) > 314 or self.info_font.metrics('linespace')*INFO_LINES > 358):
            size += 1 if size < 0 else -1
            self.info_font.configure(size=size)
        self.vinfo = tk.Label(infof, font=self.info_font, justify=tk.LEFT, anchor='nw')
        self.vinfo.place(x=0, y=0, relwidth=1, relheight=1)

        self.images = [ImageTk.PhotoImage(Image.new("RGB", (320,180))), ImageTk.PhotoImage(Image.new("RGB", (640,360)))]
        for v in range(len(self.video_labels)):
            self.video_labels[v].configure(image=self.images[0 if v != 2 else 1])
        
        self.vMapPos = None
        self.vMaybe = None
        self.tag_canvas_items = []
        self.sheet = _ContactSheet(self, video, self.video_offset)
        self.drawMap()

        self.move(abs=0)

    def _review_items(self):
        # recomputed only when the tags change
        if self.review_cache is None:
            self.review_cache = review_items(self.tags, self.raw, self.logo_spans if self.logo else [],
                                             self.blank_mids, self.opts)
        return self.review_cache

    def _is_reviewed(self, t):
        return any(math.fabs(r - t) <= REVIEW_TOL for r in self.reviewed)

    def _mark_reviewed(self, t):
        if not self._is_reviewed(t):
            self.reviewed.append(t)

    def review(self, step):
        """Go to the next review stop not yet seen (step 1), or to the previous stop (step -1)."""
        items = self._review_items()
        if step > 0:
            # leaving a stop is what marks it seen: the one the buttons went to, unless we've
            # wandered well away from it since, and any we happen to be sitting on
            if self.review_at is not None and math.fabs(self.review_at - self.position) <= 60:
                self._mark_reviewed(self.review_at)
            for (t,_,_) in items:
                if math.fabs(t - self.position) <= REVIEW_TOL:
                    self._mark_reviewed(t)
            todo = [t for (t,_,_) in items if not self._is_reviewed(t)]
            ahead = [t for t in todo if t > self.position + REVIEW_TOL]
            # past the last one, go back round for any skipped over
            target = ahead[0] if ahead else todo[0] if todo else None
        else:
            behind = [t for (t,_,_) in items if t < self.position - REVIEW_TOL]
            target = behind[-1] if behind else None
        self.review_at = target
        if target is not None:
            self.move(abs=target)
        else:
            self.updatePosIndicators()
        self.drawReviewMarks()

    def updateReview(self):
        items = self._review_items()
        done = sum(1 for (t,_,_) in items if self._is_reviewed(t))
        here = [why for (t,why,_) in items if math.fabs(t - self.position) <= REVIEW_TOL]
        text = f'Review: {done} of {len(items)} seen'
        if done == len(items):
            text += ' (all done)'
        if here:
            text += ' -- ' + '; '.join(here)
        self.review_label.configure(text=text)
    
    def _targets(self, key):
        """Where the |< and >| buttons for `key` jump to, in order."""
        out = []
        span = self.tags if key == 'break' else self.logo_spans if key == 'logo' else self.spans.get(key, [])
        for (t,(b,e)) in span:
            if key == 'logo':
                # every change, the logo going away as much as it coming back
                if b > 0:
                    out.append(b)
                continue
            if not t: continue
            # a blank of a few frames is a better landing spot in its middle
            out.append(b+(e-b)/2 if key == 'blank' and (e-b) >= 3/self.frame_rate else b)
        return out

    def _around(self, key):
        """The (prev, next) targets for `key`, and whether we are on one; the same ones the buttons use."""
        # landing on a target can be off by a frame, so a target this close counts as here
        tol = 2/self.frame_rate
        targets = self._targets(key)
        prev = [p for p in targets if p - self.position < -tol]
        nxt = [p for p in targets if p - self.position > tol]
        here = any(math.fabs(p - self.position) <= tol for p in targets)
        return (prev[-1] if prev else None, nxt[0] if nxt else None, here)

    def prev(self,key='diff'):
        (p, _, _) = self._around(key)
        if p is not None:
            self.move(abs=p)

    def next(self,key='diff'):
        (_, n, _) = self._around(key)
        if n is not None:
            self.move(abs=n)

    def move_prev_frame(self):
        self.move(abs=self.prev_frame_time)
        
    def move_next_frame(self):
        self.move(abs=self.next_frame_time)

    def move(self, frames=0, seconds=0, abs=None):
        seconds += frames/self.frame_rate
        if abs is not None:
            seconds += abs
        else:
            seconds += self.position
        seconds = max(0, min(seconds, self.duration))

        (prev, cur, nxt) = self.cursor.window(seconds)
        t = self.cursor.time
        self.position = t(cur) if cur is not None else seconds
        self.prev_frame_time = t(prev) if prev is not None else self.position - 1/self.frame_rate
        self.next_frame_time = t(nxt) if nxt is not None else self.position + 1/self.frame_rate

        if len(self.images) != 5:
            self.images = [None]*5
        for (n, f, size) in ((1, prev, (320,180)), (2, cur, (640,360)), (3, nxt, (320,180))):
            self.images[n] = ImageTk.PhotoImage(f.to_image(width=size[0], height=size[1])) if f is not None else None
            # tkinter drops None options, which would leave the old picture up
            self.video_labels[n].configure(image=self.images[n] if self.images[n] is not None else '')

        # the -5s/+5s thumbnails are drawn just after the main frames have shown up, and only for
        # the last of several quick moves
        if self.thumb_job is not None:
            self.after_cancel(self.thumb_job)
        self.thumb_job = self.after(10, self.updateThumbs)

        self.updatePosIndicators()

    def updateThumbs(self):
        self.thumb_job = None
        for (i, n, d) in ((0, 0, -5), (1, 4, 5)):
            at = self.position + d
            f = self.thumb_cursors[i].window(at)[1] if 0 <= at <= self.duration else None
            self.images[n] = ImageTk.PhotoImage(f.to_image(width=320, height=180)) if f is not None else None
            self.video_labels[n].configure(image=self.images[n] if self.images[n] is not None else '')

    def updatePosIndicators(self):
        self.pos_label.configure(text=f'{int(self.position/60):02}:{self.position%60:06.03f}')
        self.updateInfo()
        self.updateReview()
        self.sheet.update_position()
        
        self.scale_pos.set(self.position/60) #self.scroller.set(self.position/60)
        x = math.ceil(self.position / (self.duration / self.map_width))
        self.mapCanvas.coords(self.vMapPos, x, 0, x, self.map_height)
        
        if self.vMaybe is not None:
            startx = math.floor(self.setpos / (self.duration / self.map_width))
            stopx = x
            if startx > stopx:
                (startx, stopx) = (stopx, startx)
            elif startx == stopx:
                stopx += 1
            self.mapCanvas.coords(self.vMaybe, startx, 0, stopx, self.map_height)
            #print(self.settype, self.vMaybe, self.setpos, self.position, startx, stopx)

    TAG_NAMES = {SceneType.SHOW:'Show', SceneType.INTRO:'Intro', SceneType.TRANSITION:'Trans',
                 SceneType.COMMERCIAL:'Break', SceneType.CREDITS:'Credits', SceneType.DO_NOT_USE:'Ignore'}
    AUDIO_NAMES = ['silent', 'speech', 'music', 'noise']

    @staticmethod
    def _span_at(spans, pos):
        # the (value,(b,e)) span containing pos, or None
        i = bisect.bisect_right([b for (_,(b,_)) in spans], pos) - 1
        if 0 <= i < len(spans) and spans[i][1][0] <= pos <= spans[i][1][1]:
            return spans[i]
        return None

    def _around_text(self, key):
        # 'prev -1.23s  next +4.56s' for where the |< and >| buttons would go
        (p, n, here) = self._around(key)
        p = f'{p-self.position:+.2f}s' if p is not None else 'none'
        n = f'{n-self.position:+.2f}s' if n is not None else 'none'
        return f'{p:>8} {n:>8}' + (' HERE' if here else '')

    def updateInfo(self):
        pos = self.position
        lines = []

        if self.settype is not None:
            lines.append(f'FLAG {self.TAG_NAMES.get(self.settype)} from {self.setpos-pos:+.2f}s')
        else:
            lines.append('')

        if len(self.ftimes):
            # the log's rows nearest the video's frames; they differ when the video was transcoded to
            # another frame rate (from 29.97 to 59.94, two video frames can share a row)
            def nearest(t):
                i = int(np.searchsorted(self.ftimes, t))
                if i > 0 and (i >= len(self.ftimes) or t - self.ftimes[i-1] < self.ftimes[i] - t):
                    i -= 1
                return self.frames[i] if math.fabs(self.ftimes[i] - t) <= self.fstep else None
            rows = [nearest(t) for t in (self.prev_frame_time, pos, self.next_frame_time)]
            def row(name, vals):
                return f'{name:<5}' + ''.join(f'{v:>7}' for v in vals)
            def frow(name, fmt):
                return row(name, [fmt(r) if r is not None else '' for r in rows])
            def audio(r):
                for a in range(4):
                    if r[6+a]:
                        return self.AUDIO_NAMES[a]
                return '?'
            lines.append(row('', ['-1f', 'this', '+1f']))
            # the model's commercial probability; it is per second, so the sides are -1s/+1s
            j = int(np.searchsorted(self.raw_times, pos))
            if j > 0 and (j >= len(self.raw_times) or pos - self.raw_times[j-1] < self.raw_times[j] - pos):
                j -= 1
            lines.append(row('model', [f'{self.raw[k][1]*100:.0f}%' if 0 <= k < len(self.raw) else ''
                                       for k in (j-1, j, j+1)]))
            lines.append(frow('diff', lambda r: ('*' if r[3] >= neural.DIFF_THRESHOLD else '') + f'{r[3]:.2f}'))
            lines.append(frow('logo', lambda r: 'yes' if r[1] else '-'))
            lines.append(frow('blank', lambda r: 'BLANK' if r[2] else '-'))
            lines.append(frow('fvol', lambda r: f'{r[4]:.3f}'))
            lines.append(frow('rvol', lambda r: f'{r[5]:.3f}'))
            lines.append(frow('audio', audio))

        lines.append('')
        lines.append(f'{"":<5}{"prev":>9}{"next":>9}')
        lines.append(f'{"blank":<5} ' + self._around_text('blank'))
        lines.append(f'{"diff":<5} ' + self._around_text('diff'))
        for key in ('logo', 'audio'):
            # the frame times shown are rounded to the ms, which can put a frame that starts a span
            # a hair before it
            sp = self._span_at(self.logo_spans if key == 'logo' else self.spans.get(key, []), pos + 0.5/self.frame_rate)
            if sp is not None:
                (v,(b,e)) = sp
                v = ('on' if v else 'off') if key == 'logo' else self.AUDIO_NAMES[v.value]
                lines.append(f'{key:<5} {v} {b-pos:+.1f}..{e-pos:+.1f}s')
            else:
                lines.append('')

        # where training would move a break/show edge put here
        snap = neural._align_edge(pos, neural._snap_distance(SceneType.COMMERCIAL), self.blank_mids, self.ftimes, self.diff_mags)
        k = int(np.searchsorted(self.ftimes, snap))
        if len(self.blank_mids) and np.min(np.abs(self.blank_mids - snap)) < 1e-6:
            lines.append(f'Break snaps {snap-pos:+.2f}s blank')
        elif k < len(self.ftimes) and self.ftimes[k] == snap and self.diff_mags[k] >= neural.DIFF_THRESHOLD:
            lines.append(f'Break snaps {snap-pos:+.2f}s cut')
        else:
            lines.append('Break snaps here')

        # a fixed line count, since a taller box would shift the grid
        lines = (lines + ['']*INFO_LINES)[:INFO_LINES]
        self.vinfo.configure(text='\n'.join(lines))

    def drawSpan(self, span, colorMap, top, bottom, force_width=None, name="span"):
        items = []
        sec_per_pix = self.duration / self.map_width
        for (t,(b,e)) in span:
            color = colorMap.get(t, None)
            if color is None:
                continue
            startx = math.floor(b / sec_per_pix)
            if force_width is not None:
                stopx = startx + force_width
            else:
                stopx = max(startx+1,math.ceil(e / sec_per_pix))
            x = self.mapCanvas.create_rectangle(startx, top, stopx, bottom, width=0, fill=color, tags=(name,))
            items.append(x)
            #print(t,startx,stopx,color)
        return items
    
    def drawVolume(self, span, top, bottom, height, fcolor, rcolor):
        scale = np.max(np.array(span)[...,1:3])
        sec_per_pix = self.duration / self.map_width
        prev = None
        for (time,value,_) in span:
            x = math.floor(time / sec_per_pix)
            y = bottom - (value/scale) * height
            if prev is not None:
                self.mapCanvas.create_line(prev[0], prev[1], x, y, fill=fcolor, width=0.05)
            prev = (x,y)
        prev = None
        for (time,_,value) in span:
            x = math.floor(time / sec_per_pix)
            y = bottom - (value/scale) * height
            if prev is not None:
                self.mapCanvas.create_line(prev[0], prev[1], x, y, fill=rcolor, width=0.1)
            prev = (x,y)
    
    def drawRaw(self, span, top, bottom, height, color):
        sec_per_pix = self.duration / self.map_width
        prev = (0, math.floor(bottom - .5*height))
        for (time,value) in span:
            x = math.floor(time / sec_per_pix)
            y = bottom - value * height
            self.mapCanvas.create_line(prev[0], prev[1], x, y, fill=color, width=0.1)
            prev = (x,y)
    
    def redrawTags(self):
        for x in self.tag_canvas_items:
            self.mapCanvas.delete(x)
        if self.vMaybe is not None:
            self.mapCanvas.delete(self.vMaybe)
            self.vMaybe = None
            
        row = self.map_height/6
        pos = 0
        self.tag_canvas_items = self.drawSpan(self.tags, top=pos, bottom=pos+row, colorMap=SceneType.color_map())
        self.mapCanvas.tag_lower("span", "blank")
        
        if self.settype is not None:
            color = SceneType.color_map()[self.settype]
            if color is not None:
                self.vMaybe = self.mapCanvas.create_rectangle(0, 0, 0, self.map_height, width=0, fill=color, stipple='gray50')

        self.review_cache = None
        self.drawReviewMarks()
        self.sheet.recolor()

    def drawReviewMarks(self):
        # a tick at the top of the tags row for each review stop, orange until it has been seen
        self.mapCanvas.delete('review')
        sec_per_pix = self.duration / self.map_width
        for (t,_,_) in self._review_items():
            x = math.floor(t / sec_per_pix)
            self.mapCanvas.create_line(x, 0, x, self.map_height/12, width=2, tags=('review',),
                                       fill='grey40' if self._is_reviewed(t) else 'orange')
        if self.vMapPos is not None:
            self.mapCanvas.tag_raise(self.vMapPos)
        self.sheet.mark_review()

    def drawMap(self):
        for x in self.mapCanvas.find_all():
            self.mapCanvas.delete(x)
        self.vMaybe = None
        self.vMapPos = None
        self.tag_canvas_items = []

        row = self.map_height/6
        pos = 0
        pos += row
        self.drawSpan(self.spans.get('logo',[]), top=pos, bottom=pos+row, colorMap={True:'blue'})
        pos += row
        
        self.drawSpan(self.spans.get('blank',[]), top=int(row*.3), bottom=pos-int(row*.3), colorMap={True:'black'}, name="blank")

        self.drawSpan(self.spans.get('diff',[]), top=pos, bottom=pos+row, colorMap={True:'purple'}, force_width=1)
        pos += row
        self.drawSpan(self.spans.get('audio',[]), top=pos, bottom=pos+row, colorMap=AudioSegmentLabel.color_map())
        pos += row

        self.drawRaw(self.raw, top=pos, bottom=pos+row, height=row, color='darkred')
        pos += row
        
        if 'volume' in self.spans:
            self.drawVolume(self.spans.get('volume'), top=pos, bottom=pos+row, height=row, fcolor='darkblue', rcolor='lightblue')
            pos += row
        
        self.redrawTags()

        # lastly, add the positional indicator
        self.vMapPos = self.mapCanvas.create_line(0,0,0,self.map_height,arrow=tk.BOTH,fill='orange',width=1.5)
        self.updatePosIndicators()

    def do_tag(self, btnIdx):
        for (b,t,l) in self.taggers:
            b.grid_forget()
        
        self.tag_cancel.configure(state='normal', command=lambda x=btnIdx:self.cancel_tag(x))
        self.tag_cancel.grid(row=0, column=0, padx=5) # show

        (b,self.settype,label) = self.taggers[btnIdx]
        b.configure(state='normal', text=f'Stop {label}', command=lambda x=btnIdx:self.end_tag(x))
        b.grid(row=0, column=1, padx=5)

        self.setpos = self.position
        
        #self.drawMap()
        self.redrawTags()
        self.updatePosIndicators()
    
    def cancel_tag(self, btnIdx):
        self.settype = None
        self.end_tag(btnIdx)

    def end_tag(self, btnIdx):
        if self.settype is not None and self.setpos != self.position:
            settype = self.settype
            startpos = self.setpos
            endpos = self.position

            if endpos < startpos:
                (endpos, startpos) = (startpos, endpos)
            
            # seek to the first tag overlapping us
            b = 0
            while b < len(self.tags) and self.tags[b][1][1] <= startpos:
                b += 1
            if b < len(self.tags) and self.tags[b][1][0] < startpos:
                # split the tag so our start lines up with the start of a tag
                x = self.tags[b][1][1]
                tt = self.tags[b][0]
                self.tags[b] = (tt, (self.tags[b][1][0], startpos))
                b += 1
                self.tags[b:b] = [(tt, (startpos,x))]
            # seek to the first tag after us
            e = b
            while e < len(self.tags) and self.tags[e][1][1] <= endpos:
                e += 1
            if e < len(self.tags) and self.tags[e][1][0] < endpos:
                # split the tag so our end exactly lines up with the start of a tag
                x = self.tags[e][1][1]
                tt = self.tags[e][0]
                self.tags[e] = (tt, (self.tags[e][1][0], endpos))
                e += 1
                self.tags[e:e] = [(tt, (endpos,x))]
            
            # if the one before is the same type, merge with that
            if b > 0 and b-1 < len(self.tags) and self.tags[b-1][0] == settype and (startpos-self.tags[b-1][1][1]) < 0.05:
                b -= 1
                startpos = self.tags[b][1][0]
            # if the one after is the same type, merge with that
            if e < len(self.tags) and self.tags[e][0] == settype and (self.tags[e][1][0] - endpos) < 0.05:
                endpos = self.tags[e][1][1]
                e += 1
            
            # and now replace all of that stuff with this new tag
            self.tags[b:e] = [(settype, (startpos, endpos))]

            # the edges just placed by hand need no review
            self._mark_reviewed(self.setpos)
            self._mark_reviewed(self.position)
            
        self.settype = None
        self.setpos = 0
        
        if btnIdx is not None:
            self.tag_cancel.grid_forget() 
            c = 0
            for (b,t,l) in self.taggers:
                b.configure(state='normal')
                b.grid(row=0, column=c, padx=5)
                c += 1
            (b,t,label) = self.taggers[btnIdx]
            b.configure(text=f'Flag {label}', command=lambda x=btnIdx:self.do_tag(x))

        self.redrawTags()
        self.updatePosIndicators()
    
    def truncate_now(self):
        if self.settype is not None:
            savepos = self.position
            self.end_tag(None)
            self.position = savepos
        self.setpos = self.duration
        self.settype = SceneType.DO_NOT_USE
        self.end_tag(None)
        self.settype = None
        self.save_and_close()
    
    def save_and_close(self):
        if self.settype is not None:
            self.end_tag(None)
        
        self.result = []
        for (t,(b,e)) in self.tags:
            if t != SceneType.SHOW:
                self.result.append((t.value,(b,e)))
        print('RESULT', self.result)
        self.destroy()

    def destroy(self):
        # however the window goes (Save, the window manager's close), nothing may be left pending
        if getattr(self, 'sheet', None) is not None:
            self.sheet.close()
        if getattr(self, 'thumb_job', None) is not None:
            self.after_cancel(self.thumb_job)
            self.thumb_job = None
        tk.Tk.destroy(self)

    def run(self):
        tk.mainloop()
        return self.result

    def tag_at(self, t):
        """The tag type at time `t`."""
        i = bisect.bisect_right([b for (_,(b,_)) in self.tags], t) - 1
        return self.tags[max(0, i)][0] if self.tags else SceneType.SHOW

class _ContactSheet:
    """A window of one thumbnail every SHEET_STEP seconds of the whole recording, each framed in the
    colour of its tag with the model's highest commercial probability in it as a bar underneath;
    clicking one moves the editor there.  A run of ads stands out from the show at a glance, which
    is the quick way to find a break the tags miss altogether."""

    LABEL_W = 50
    CELL_W = SHEET_THUMB[0] + 8
    CELL_H = SHEET_THUMB[1] + 14

    def __init__(self, win, video, offset=0.0):
        self.win = win
        self.video = video
        self.offset = offset # log time - the video's
        self.n = max(1, int(math.ceil(win.duration / SHEET_STEP)))
        self.times = [k*SHEET_STEP for k in range(self.n)] # each thumbnail's time, once it is decoded
        self.photos = [None]*self.n
        self.prob = [0.0]*self.n
        for (t,v) in win.raw:
            k = int(t // SHEET_STEP)
            if 0 <= k < self.n:
                self.prob[k] = max(self.prob[k], float(v))
        self.top = None
        self.canvas = None
        self.frames = {} # canvas item ids, per slot
        self.cursor = None
        self.done = queue.Queue()
        self.stop = threading.Event()
        self.poll_job = None

    def _start(self):
        # decoding waits for the window to be asked for
        threading.Thread(target=self._decode, daemon=True).start()
        self.poll_job = self.win.after(100, self._poll)

    def _decode(self):
        # its own Player, on this thread; seeking to each slot and decoding just the keyframe
        # there is far cheaper than decoding the recording through
        try:
            player = Player(self.video, no_deinterlace=True)
            player._skip_frames('NONKEY')
            vt_start = float(player.vt_start)
            for k in range(self.n):
                if self.stop.is_set():
                    return
                # in a .ts this lands on the first keyframe after the slot's start, a GOP into it
                player.seek(max(0.0, k*SHEET_STEP - self.offset))
                frames = player.frames()
                try:
                    f = next(frames, None)
                finally:
                    frames.close()
                if f is None or f.time is None:
                    continue
                self.done.put((k, f.time - vt_start + self.offset, f.to_image(width=SHEET_THUMB[0], height=SHEET_THUMB[1])))
        except Exception:
            log.exception('contact sheet decoding failed')

    def close(self):
        # a pending after() whose window is destroyed fails with "invalid command name" when it fires
        self.stop.set()
        if self.poll_job is not None:
            self.win.after_cancel(self.poll_job)
            self.poll_job = None

    def _poll(self):
        self.poll_job = None
        try:
            while True:
                (k, t, img) = self.done.get_nowait()
                self.times[k] = t
                self.photos[k] = ImageTk.PhotoImage(img)
                if self.canvas is not None:
                    self.canvas.itemconfigure(self.frames[k]['img'], image=self.photos[k])
                    self._color(k)
        except queue.Empty:
            pass
        except tk.TclError:
            return # the editor has gone
        if not self.stop.is_set():
            self.poll_job = self.win.after(200, self._poll)

    def _xy(self, k):
        return (self.LABEL_W + (k % SHEET_COLS) * self.CELL_W, (k // SHEET_COLS) * self.CELL_H)

    def show(self):
        if self.top is not None:
            self.top.deiconify()
            self.top.lift()
            return
        self._start()
        self.top = tk.Toplevel(self.win)
        self.top.title('pycommflag contact sheet')
        self.top.protocol('WM_DELETE_WINDOW', self.top.withdraw)
        rows = int(math.ceil(self.n / SHEET_COLS))
        width = self.LABEL_W + SHEET_COLS * self.CELL_W
        height = rows * self.CELL_H
        self.canvas = tk.Canvas(self.top, width=width, height=min(height, 900), scrollregion=(0, 0, width, height),
                                yscrollincrement=self.CELL_H)
        bar = tk.Scrollbar(self.top, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        bar.grid(row=0, column=1, sticky='ns')
        self.top.rowconfigure(0, weight=1)
        self.canvas.bind('<Button-1>', self._click)
        self.canvas.bind('<Button-4>', lambda e: self.canvas.yview_scroll(-1, 'units'))
        self.canvas.bind('<Button-5>', lambda e: self.canvas.yview_scroll(1, 'units'))
        self.canvas.bind('<MouseWheel>', lambda e: self.canvas.yview_scroll(-1 if e.delta > 0 else 1, 'units'))
        self.top.bind('n', lambda e: self.win.review(1))
        self.top.bind('p', lambda e: self.win.review(-1))

        (tw, th) = SHEET_THUMB
        for r in range(rows):
            m = r * SHEET_COLS * SHEET_STEP / 60
            self.canvas.create_text(self.LABEL_W - 6, r * self.CELL_H + 4, anchor='ne', text=f'{int(m)}:{int(m*60)%60:02}')
        for k in range(self.n):
            (x, y) = self._xy(k)
            ids = {}
            ids['frame'] = self.canvas.create_rectangle(x+1, y+1, x+tw+7, y+th+7, width=0)
            ids['img'] = self.canvas.create_image(x+4, y+4, anchor='nw')
            if self.photos[k] is not None:
                self.canvas.itemconfigure(ids['img'], image=self.photos[k])
            ids['prob'] = self.canvas.create_rectangle(x+4, y+th+9, x+4+max(1, round(self.prob[k]*tw)), y+th+12,
                                                        width=0, fill='darkred')
            ids['review'] = self.canvas.create_oval(x+tw-4, y+6, x+tw+2, y+12, width=0, fill='orange', state='hidden')
            self.frames[k] = ids
        self.cursor = self.canvas.create_rectangle(0, 0, 0, 0, outline='orange', width=3)
        self.recolor()
        self.mark_review()
        self.update_position()

    def _click(self, e):
        x = self.canvas.canvasx(e.x) - self.LABEL_W
        y = self.canvas.canvasy(e.y)
        if x < 0:
            return
        k = int(y // self.CELL_H) * SHEET_COLS + int(x // self.CELL_W)
        if 0 <= k < self.n and int(x // self.CELL_W) < SHEET_COLS:
            self.win.move(abs=self.times[k])

    def _color(self, k):
        c = SceneType.color_map().get(self.win.tag_at(self.times[k] + 0.5))
        self.canvas.itemconfigure(self.frames[k]['frame'], fill=c if c is not None else 'grey50')

    def recolor(self):
        if self.canvas is None:
            return
        for k in range(self.n):
            self._color(k)

    def mark_review(self):
        # a dot on each thumbnail holding a review stop not seen yet
        if self.canvas is None:
            return
        todo = set()
        for (t,_,_) in self.win._review_items():
            if not self.win._is_reviewed(t):
                todo.add(int(t // SHEET_STEP))
        for k in range(self.n):
            self.canvas.itemconfigure(self.frames[k]['review'], state='normal' if k in todo else 'hidden')

    def update_position(self):
        if self.canvas is None:
            return
        k = min(self.n - 1, max(0, int(self.win.position // SHEET_STEP)))
        (x, y) = self._xy(k)
        self.canvas.coords(self.cursor, x, y, x + self.CELL_W - 1, y + self.CELL_H - 1)
        # keep it in view
        height = int(math.ceil(self.n / SHEET_COLS)) * self.CELL_H
        (top, bottom) = self.canvas.yview()
        if y < top * height or y + self.CELL_H > bottom * height:
            self.canvas.yview_moveto(max(0, (y - (bottom - top) * height / 2) / height))

class _Cursor:
    """Decoded frames around a position in a Player, decoding forward rather than seeking when it can."""
    def __init__(self, player, keep=None, offset=0.0):
        self.player = player
        self.offset = offset # log time - the player's
        self.keep = keep if keep is not None else int(2*player.frame_rate) # 1080p frames are ~3MB each
        self.cache = [] # consecutive decoded frames, oldest first
        self.it = None # player.frames(), positioned just after cache[-1]

    def time(self, f):
        return round(f.time - self.player.vt_start + self.offset, 3)

    def window(self, seconds):
        """The decoded (prev, this, next) frames at `seconds`; any of them may be None."""
        want = round(seconds, 3)
        c = self.cache
        if not (self.it is not None and c and self.time(c[0]) < want <= self.time(c[-1]) + 1.5):
            # too far from what's decoded, so seek; starting a second early means
            # stepping backwards is served from the cache for a while
            if self.it is not None:
                self.it.close()
            c = self.cache = []
            f = self.player.seek_exact(max(0, seconds - self.offset - 1))
            if f is not None:
                c.append(f)
            self.it = self.player.frames()

        # decode forward until there's a frame after the wanted one
        while self.it is not None and (len(c) < 2 or self.time(c[-2]) < want):
            try:
                c.append(next(self.it))
            except Exception:
                self.it = None
            if len(c) > self.keep:
                del c[0]

        k = 0
        while k < len(c) and self.time(c[k]) < want:
            k += 1
        if k >= len(c):
            return (c[-2] if len(c) >= 2 else None, c[-1] if c else None, None)
        return (c[k-1] if k > 0 else None, c[k], c[k+1] if k+1 < len(c) else None)
