import tkinter as tk
import time
import numpy as np
from PIL import ImageTk, Image
from .player import Player
from .scene import Scene
from .processor import process_scenes

class Window(tk.Tk):
    def __init__(self, video, scenes:list[Scene]=[]):
        tk.Tk.__init__(self)
        self.title("pycommflag editor")
        self.player = Player(video)

        self.scenes = scenes
        self.misc = []
        self.video_labels = []
        self.images = []
        for x in range(5):
            v = tk.Label(self)
            if x == 2:
                v.grid(row=1,column=0, columnspan=5)
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

        b = tk.Button(master=skipf, text="<<<< 30s", command=lambda:self.move(seconds=-30))
        b.grid(row=0, column=0, padx=5)
        self.misc.append(b)
        
        b = tk.Button(master=skipf, text="<<< 5s", command=lambda:self.move(seconds=-5))
        b.grid(row=0, column=1, padx=5)
        self.misc.append(b)

        b = tk.Button(master=skipf, text="<< 1s", command=lambda:self.move(seconds=-1))
        b.grid(row=0, column=3, padx=5)
        self.misc.append(b)
        
        b = tk.Button(master=skipf, text="< 1f", command=lambda:self.move(abs=self.prev_frame_time))
        b.grid(row=0, column=4, padx=(5,20))
        self.misc.append(b)

        b = tk.Button(master=skipf, text="1f >", command=lambda:self.move(abs=self.next_frame_time))
        b.grid(row=0, column=5, padx=(20,5))
        self.misc.append(b)
        
        b = tk.Button(master=skipf, text="1s >>", command=lambda:self.move(seconds=1))
        b.grid(row=0, column=6, padx=5)
        self.misc.append(b)
        
        b = tk.Button(master=skipf, text="5s >>>", command=lambda:self.move(seconds=5))
        b.grid(row=0, column=8, padx=5)
        self.misc.append(b)

        b = tk.Button(master=skipf, text="30s >>>>", command=lambda:self.move(seconds=30))
        b.grid(row=0, column=9, padx=5)
        self.misc.append(b)

        skips = tk.Frame(self)
        skips.grid(row=5, column=0, columnspan=5)
        self.misc.append(skips)
        
        b = tk.Button(master=skips, text="|< Break", command=lambda:self.prev(cbreak=True))
        b.grid(row=0, column=0, padx=5)
        self.misc.append(b)

        b = tk.Button(master=skips, text="|< Blank", command=lambda:self.prev(blank=True))
        b.grid(row=0, column=1, padx=5)
        self.misc.append(b)

        #b = tk.Button(master=skips, text="|< Silence", command=lambda:self.prev(silence=True))
        #b.grid(row=0, column=2, padx=5)
        #self.misc.append(b)

        b = tk.Button(master=skips, text="|< Scene", command=lambda:self.prev(scene=True))
        b.grid(row=0, column=3, padx=(5,10))
        self.misc.append(b)

        b = tk.Button(master=skips, text="Scene >|", command=lambda:self.next(scene=True))
        b.grid(row=0, column=5, padx=(10,5))
        self.misc.append(b)

        #b = tk.Button(master=skips, text="Silence >|", command=lambda:self.next(silence=True))
        #b.grid(row=0, column=6, padx=5)
        #self.misc.append(b)
        
        b = tk.Button(master=skips, text="Blank >|", command=lambda:self.next(blank=True))
        b.grid(row=0, column=7, padx=5)
        self.misc.append(b)
        
        b = tk.Button(master=skips, text="Break >|", command=lambda:self.next(cbreak=True))
        b.grid(row=0, column=8, padx=5)
        self.misc.append(b)

        self.info = tk.Label(self, text=f'File: {video}; Length:{self.player.duration/60.0:0.1f} mins; {float(self.player.frame_rate)} fps')
        self.info.grid(row=7, column=0, sticky="se", columnspan=5)

        # TODO: add a little display of the logo we found.

        self.images = [ImageTk.PhotoImage(Image.new("RGB", (320,180))), ImageTk.PhotoImage(Image.new("RGB", (640,360)))]
        for v in range(len(self.video_labels)):
            self.video_labels[v].configure(image=self.images[0 if v != 2 else 1])

        self.position = 0
        self.prev_frame_time = 0
        self.next_frame_time = 1/self.player.frame_rate
        self.move(abs=0)
    
    def prev(self,scene=False,blank=False,cbreak=False):
        for s in reversed(self.scenes):
            if blank and not s.start_blank:
                continue
            if cbreak and not s.start_break:
                continue
            if s.start_time <= self.posiiton and (self.position - s.start_time) > 2/self.player.frame_rate:
                self.move(abs=s.start_time)
                return

    def next(self,scene=False,blank=False,cbreak=False):
        for s in self.scenes:
            if blank and not s.start_blank:
                continue
            if cbreak and not s.start_break:
                continue
            if s.start_time > self.posiiton:
                self.move(abs=s.start_time)
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
        
        self.prev_frame_time = seconds - 1/self.player.frame_rate
        self.posiiton = seconds
        self.next_frame_time = seconds + 1/self.player.frame_rate
        self.images = [None]*5

        if seconds >= 5:
            (f,_) = self.player.seek_exact(seconds - 5)
            self.images[0] = ImageTk.PhotoImage(f.to_image(height=180,width=320))
        
        (f,_) = self.player.seek_exact(min(max(0, seconds - 10/self.player.frame_rate), self.player.duration - 5/self.player.frame_rate))
        
        frames = []
        if f is not None:
            frames.append(f)
        for (f,_) in self.player.frames():
            frames.append(f)
            if len(frames) > 1 and (f.time - self.player.vt_start) > seconds and (frames[-2].time - self.player.vt_start) >= seconds:
                break
        
        if len(frames) >= 3:
            self.prev_frame_time = frames[-3].time - self.player.vt_start
            self.images[1] = ImageTk.PhotoImage(frames[-3].to_image(height=180,width=320))
        if len(frames) >= 2:
            self.position = frames[-2].time - self.player.vt_start
            self.images[2] = ImageTk.PhotoImage(frames[-2].to_image(height=360,width=640))
        if len(frames) >= 1:
            self.next_frame_time = frames[-1].time - self.player.vt_start
            self.images[3] = ImageTk.PhotoImage(frames[-1].to_image(height=180,width=320))
        
        (f,_) = self.player.seek_exact(seconds + 5)
        if f is not None:
            self.images[4] = ImageTk.PhotoImage(f.to_image(height=180,width=320))
        
        self.pos_label.configure(text=f'{int(self.position/60):02}:{self.position%60:06.03f}')

        for n in range(len(self.images)):
            self.video_labels[n].configure(image=self.images[n])
    
    def run(self):
        tk.mainloop()
