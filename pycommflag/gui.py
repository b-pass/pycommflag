import tkinter as tk
import tkinter.font as tkfont
import bisect
import math
import time
import numpy as np
from PIL import ImageTk, Image
from .player import Player
from . import logo_finder
from . import processor
from . import neural
from .feature_span import *

INFO_LINES = 16 # in the info box, which must not change height
LOGO_DEBOUNCE = 3 # log frames a new logo state must last before Logo |</>| count it as a change

class Window(tk.Tk):
    def __init__(self, opts, video, flog):
        tk.Tk.__init__(self)
        self.title("pycommflag editor")
        self.player = Player(video, no_deinterlace=True)
        # the main frames and the -5s/+5s thumbnails each get their own decoder, so they never
        # seek each other's position away (which is what makes small steps cheap)
        self.cursor = _Cursor(self.player)
        self.thumb_cursors = [_Cursor(Player(video, no_deinterlace=True), keep=int(1.5*self.player.frame_rate))
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
        
        b = tk.Button(skips, text="|< Break", command=lambda:self.prev('break'))
        b.grid(row=0, column=0, padx=5)
        self.misc.append(b)

        b = tk.Button(skips, text="|< Blank", command=lambda:self.prev('blank'))
        b.grid(row=0, column=1, padx=5)
        self.misc.append(b)

        b = tk.Button(skips, text="|< Logo", command=lambda:self.prev('logo'))
        b.grid(row=0, column=2, padx=5)
        self.misc.append(b)

        b = tk.Button(skips, text="|< Diff", command=lambda:self.prev('diff'))
        b.grid(row=0, column=3, padx=(5,10))
        self.misc.append(b)

        b = tk.Button(skips, text="Diff >|", command=lambda:self.next('diff'))
        b.grid(row=0, column=5, padx=(10,5))
        self.misc.append(b)

        b = tk.Button(skips, text="Logo >|", command=lambda:self.next('logo'))
        b.grid(row=0, column=6, padx=5)
        self.misc.append(b)
        
        b = tk.Button(skips, text="Blank >|", command=lambda:self.next('blank'))
        b.grid(row=0, column=7, padx=5)
        self.misc.append(b)
        
        b = tk.Button(skips, text="Break >|", command=lambda:self.next('break'))
        b.grid(row=0, column=8, padx=5)
        self.misc.append(b)
        
        tags = tk.Frame(self)
        tags.grid(row=6, column=0, columnspan=5)
        self.misc.append(tags)

        self.taggers = []
        
        self.tag_cancel = tk.Button(tags, text='Cancel This Flag')

        self.taggers.append((tk.Button(tags), SceneType.COMMERCIAL, 'Break'))
        self.taggers.append((tk.Button(tags), SceneType.SHOW, 'Show'))
        #self.taggers.append((tk.Button(tags), SceneType.TRANSITION, 'Transition'))
        self.taggers.append((tk.Button(tags), SceneType.INTRO, 'Intro'))
        self.taggers.append((tk.Button(tags), SceneType.CREDITS, 'Credits'))
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

        self.info = tk.Label(self, text=f'File: {video}; Length:{self.duration/60.0:0.1f} mins; {float(self.frame_rate)} fps')
        self.info.grid(row=10, column=0, sticky="se", columnspan=5)

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
        self.drawMap()

        self.move(abs=0)
    
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

    def run(self):
        tk.mainloop()
        return self.result

class _Cursor:
    """Decoded frames around a position in a Player, decoding forward rather than seeking when it can."""
    def __init__(self, player, keep=None):
        self.player = player
        self.keep = keep if keep is not None else int(2*player.frame_rate) # 1080p frames are ~3MB each
        self.cache = [] # consecutive decoded frames, oldest first
        self.it = None # player.frames(), positioned just after cache[-1]

    def time(self, f):
        return round(f.time - self.player.vt_start, 3)

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
            f = self.player.seek_exact(max(0, seconds - 1))
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
