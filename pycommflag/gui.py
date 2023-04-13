import tkinter as tk
import math
import time
import numpy as np
from PIL import ImageTk, Image
from .player import Player
from .scene import *
from .processor import process_scenes
from . import logo_finder
from . import processor

class Window(tk.Tk):
    def __init__(self, video, scenes:list[Scene]=[], logo:tuple=None):
        tk.Tk.__init__(self)
        self.title("pycommflag editor")
        self.player = Player(video, no_deinterlace=True)

        self.result = None
        self.scenes = scenes
        self.position = 0
        self.prev_frame_time = 0
        self.next_frame_time = 1/self.player.frame_rate

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
        
        b = tk.Button(skips, text="|< Break", command=lambda:self.prev(cbreak=True))
        b.grid(row=0, column=0, padx=5)
        self.misc.append(b)

        b = tk.Button(skips, text="|< Blank", command=lambda:self.prev(blank=True))
        b.grid(row=0, column=1, padx=5)
        self.misc.append(b)

        #b = tk.Button(skips, text="|< Silence", command=lambda:self.prev(silence=True))
        #b.grid(row=0, column=2, padx=5)
        #self.misc.append(b)

        b = tk.Button(skips, text="|< Scene", command=lambda:self.prev(scene=True))
        b.grid(row=0, column=3, padx=(5,10))
        self.misc.append(b)

        b = tk.Button(skips, text="Scene >|", command=lambda:self.next(scene=True))
        b.grid(row=0, column=5, padx=(10,5))
        self.misc.append(b)

        #b = tk.Button(skips, text="Silence >|", command=lambda:self.next(silence=True))
        #b.grid(row=0, column=6, padx=5)
        #self.misc.append(b)
        
        b = tk.Button(skips, text="Blank >|", command=lambda:self.next(blank=True))
        b.grid(row=0, column=7, padx=5)
        self.misc.append(b)
        
        b = tk.Button(skips, text="Break >|", command=lambda:self.next(cbreak=True))
        b.grid(row=0, column=8, padx=5)
        self.misc.append(b)
        
        self.vidMap = tk.Canvas(self, width=1280, height=50)
        self.vidMap.grid(row=6, column=0, columnspan=5)
        self.vidMap.bind("<Button-1>", lambda e:self.move(abs=float(e.x)/1280.0*self.player.duration))
        
        self.audMap = tk.Canvas(self, width=1280, height=50)
        self.audMap.grid(row=7, column=0, columnspan=5)
        
        self.scale_pos = tk.DoubleVar()
        self.scroller = tk.Scale(self, 
                                from_=0, to_=self.player.duration/60, resolution=1/60, tickinterval=5, showvalue=False,
                                length=1280+25, orient=tk.HORIZONTAL, sliderlength=25,
                                variable=self.scale_pos,
                                command=lambda n: self.move(abs=float(n)*60))
        self.scroller.grid(row=8, column=0, columnspan=5)

        tags = tk.Frame(self)
        tags.grid(row=9, column=0, columnspan=5)
        self.misc.append(tags)

        self.tag_break = tk.Button(tags, text="Start Break", command=lambda:self.tag(SceneType.COMMERCIAL))
        self.tag_break.grid(row=0, column=1, padx=5)
        
        self.tag_intro = tk.Button(tags, text="Start Intro", command=lambda:self.tag(SceneType.INTRO))
        self.tag_intro.grid(row=0, column=2, padx=5)
        
        self.tag_show = tk.Button(tags, text="Start Show", command=lambda:self.tag(SceneType.SHOW))
        self.tag_show.grid(row=0, column=3, padx=5)
        
        self.tag_postroll = tk.Button(tags, text="Start Credits", command=lambda:self.tag(SceneType.CREDITS))
        self.tag_postroll.grid(row=0, column=4, padx=5)
        
        self.truncate_here = tk.Button(tags, text="End Here", command=lambda:self.truncate_now())
        self.truncate_here.grid(row=0, column=5, padx=5)
        
        self.truncate_here = tk.Button(tags, text="Save & Exit", command=lambda:self.save_and_close())
        self.truncate_here.grid(row=0, column=6, padx=5)

        self.info = tk.Label(self, text=f'File: {video}; Length:{self.player.duration/60.0:0.1f} mins; {float(self.player.frame_rate)} fps')
        self.info.grid(row=10, column=0, sticky="se", columnspan=5)

        limg = logo_finder.toimage(logo)
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
        
        self.vinfo = tk.Label(self)
        self.vinfo.grid(row=1, column=4, sticky='nswe')

        self.images = [ImageTk.PhotoImage(Image.new("RGB", (320,180))), ImageTk.PhotoImage(Image.new("RGB", (640,360)))]
        for v in range(len(self.video_labels)):
            self.video_labels[v].configure(image=self.images[0 if v != 2 else 1])
        
        self.drawMap()

        self.move(abs=0)
    
    def prev(self,scene=False,blank=False,cbreak=False):
        if cbreak:
            want = True
            first = True
            for s in reversed(self.scenes):
                if first and s.start_time <= self.position:
                    first = False
                    want = not s.is_break
                if (s.stop_time - self.position) < -3/self.player.frame_rate and s.is_break == want:
                    self.move(abs=s.stop_time-1/self.player.frame_rate)
                    return
        
        for s in reversed(self.scenes):
            if blank and not s.is_blank:
                continue
            abs = s.middle_time if blank and s.is_blank else s.start_time
            if (abs - self.position) < -3/self.player.frame_rate:
                self.move(abs=abs)
                return

    def next(self,scene=False,blank=False,cbreak=False):
        if cbreak:
            want = True
            first = True
            for s in self.scenes:
                if first and s.start_time >= self.position:
                    first = False
                    want = not s.is_break
                if (s.start_time - self.position) > 3/self.player.frame_rate and s.is_break == want:
                    self.move(abs=s.start_time)
                    return
        
        for s in self.scenes:
            if blank and not s.is_blank:
                continue
            abs = s.middle_time if blank and s.is_blank else s.start_time
            if (abs - self.position) > 3/self.player.frame_rate:
                self.move(abs=abs)
                return

    def move_prev_frame(self):
        self.move(abs=self.prev_frame_time)
        
    def move_next_frame(self):
        self.move(abs=self.next_frame_time)

    def move(self, frames=0, seconds=0, abs=None):
        seconds += frames/self.player.frame_rate
        if abs is not None:
            seconds += abs
        else:
            seconds += self.position
        seconds = max(0, min(seconds, self.player.duration))

        self.prev_frame_time = seconds - 1/self.player.frame_rate
        self.position = seconds
        self.next_frame_time = seconds + 1/self.player.frame_rate
        self.images = [None]*5

        #print("SEEK", self.prev_frame_time, seconds, self.next_frame_time)

        if seconds >= 5:
            f = self.player.seek_exact(seconds - 5)
            if f is not None:
                self.images[0] = ImageTk.PhotoImage(f.to_image(height=180,width=320))
        
        f = self.player.seek_exact(max(0, min(seconds - 7/self.player.frame_rate, self.player.duration - 10/self.player.frame_rate)))
        
        frames = []
        if f is not None:
            frames.append(f)
        for f in self.player.frames():
            frames.append(f)
            if len(frames) >= 3 and round(f.time - self.player.vt_start, 3) > round(seconds,3) and round(frames[-2].time - self.player.vt_start, 3) >= round(seconds,3):
                break
        
        #print([f.time - self.player.vt_start for f in frames])
        if len(frames) >= 3:
            self.images[1] = ImageTk.PhotoImage(frames[-3].to_image(height=180,width=320))
        if len(frames) >= 2:
            self.position = frames[-2].time - self.player.vt_start
            self.images[2] = ImageTk.PhotoImage(frames[-2].to_image(height=360,width=640))
        if len(frames) >= 1:
            self.images[3] = ImageTk.PhotoImage(frames[-1].to_image(height=180,width=320))
        
        info = 'Diffs:'
        prev = None
        for frame in frames[-3:]:
            c = processor.columnize_frame(frame).astype('int16')
            if prev is None:
                prev = c
                continue
            diff = prev - c
            prev = c
            scm = np.mean(np.std(np.abs(diff), (0)))
            info += '\nDiff: %9.05f' % (scm,)
        self.vinfo.configure(text=info)

        f = self.player.seek_exact(seconds + 5)
        if f is not None:
            self.images[4] = ImageTk.PhotoImage(f.to_image(height=180,width=320))
        
        for n in range(len(self.images)):
            self.video_labels[n].configure(image=self.images[n])
            
        self.updatePosIndicators()
    
    def updatePosIndicators(self):
        self.pos_label.configure(text=f'{int(self.position/60):02}:{self.position%60:06.03f}')
        
        self.scale_pos.set(self.position/60) #self.scroller.set(self.position/60)
        x = self.position / (self.player.duration / 1280)
        self.vidMap.coords(self.vMapPos, x, 0, x, 50)
        self.audMap.coords(self.aMapPos, x, 0, x, 50)

    def drawMap(self):
        map_height = 50
        sec_per_pix = self.player.duration / 1280
        last = 0

        for x in self.vidMap.find_all():
            self.vidMap.delete(x)

        for s in self.scenes:
            if s.is_blank:
                continue
            startx = math.floor(s.start_time / sec_per_pix)
            stopx = math.ceil(s.stop_time / sec_per_pix)
            fill = s.type.color()
            if s.logo > .75: # indicate logo?
                fill = s.type.logo_color()
            else:
                fill = s.type.color()
            
            fill2 = None
            if hasattr(s, 'newtype'):
                fill2 = s.newtype.new_color()
            
            y = map_height
            if fill2:
                y = int(y/2)
                self.vidMap.create_rectangle(startx, y, stopx, map_height, width=0, fill=fill2)
            self.vidMap.create_rectangle(startx, 0, stopx, y, width=0, fill=fill)
        
        for s in self.scenes:
            if s.is_blank:
                startx = math.floor(s.start_time / sec_per_pix)
                stopx = math.ceil(s.stop_time / sec_per_pix)
                self.vidMap.create_rectangle(startx, 0, stopx, map_height, width=0, fill='black')
        
        # find a scale to exagerate the line graph features
        # FFmpeg channel layout for 5.1: FL+FR+FC+LFE+SL+SR
        maxmax = [0]*6
        for s in self.scenes:
            for c in range(6):
                maxmax[c] = max(maxmax[c], s.peak_peaks[c])
        scale = [1.0/x if x > 0 else 1.0 for x in maxmax]

        bott = 0
        height = map_height/3
        for cs in [(0,1),(2,),(4,5)]:
            bott += height
            line = []
            bline = []
            for s in self.scenes:
                startx = int(math.floor(s.start_time / sec_per_pix))
                stopx = int(math.ceil(s.stop_time / sec_per_pix))
                if startx == stopx:
                    stopx += 1

                y = 0
                for c in cs:
                    y += s.peak_peaks[c] * scale[c]
                y = int(bott - y/len(cs) * height)

                line.append(startx)
                line.append(y)
                line.append(stopx)
                line.append(y)

                y = 0
                for c in cs:
                    y += s.valley_peaks[c] * scale[c]
                y = int(bott - y/len(cs) * height)
                bline.append(startx)
                bline.append(y)
                bline.append(stopx)
                bline.append(y)
            if bline:
                self.audMap.create_line(*bline,fill='black')
            if line:
                self.audMap.create_line(*line,fill='blue')

        # lastly, add the positional indicator
        self.aMapPos = self.audMap.create_line(0,0,0,map_height,arrow=tk.BOTH,fill='orange')
        self.vMapPos = self.vidMap.create_line(0,0,0,map_height,arrow=tk.BOTH,fill='orange')
        self.updatePosIndicators()

    def tag(self, type):
        attr = 'type'
        old = None
        for s in self.scenes:
            if s.stop_time > self.position:
                if hasattr(s,'newtype'):
                    attr = 'newtype'
                    old = s.newtype
                else:
                    old = s.type
                break
        for s in self.scenes:
            if s.stop_time <= self.position:
                continue
            if s.start_time > self.position:
                if getattr(s, attr, None) != old:
                    break
            if s.type == type:
                if hasattr(s, 'newtype'):
                    delattr(s, 'newtype')
            else:
                s.newtype = type
        self.drawMap()
    
    def truncate_now(self):
        for s in self.scenes:
            if s.stop_time > self.position:
                setattr(s, 'newtype', SceneType.TRUNCATED)
        self.save_and_close()
    
    def save_and_close(self):
        self.result = self.scenes
        self.destroy()

    def run(self):
        tk.mainloop()
        return self.result
