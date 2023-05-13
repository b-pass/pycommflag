import tkinter as tk
import math
import time
import numpy as np
from PIL import ImageTk, Image
from .player import Player
from . import logo_finder
from . import processor
from .feature_span import *

class Window(tk.Tk):
    def __init__(self, video, spans:dict={}, logo:tuple=None, tags:FeatureSpan=None):
        tk.Tk.__init__(self)
        self.title("pycommflag editor")
        self.player = Player(video, no_deinterlace=True)

        self.result = None
        self.spans = spans
        self.position = 0
        self.prev_frame_time = 0
        self.next_frame_time = 1/self.player.frame_rate
        self.settype = None
        self.setpos = 0
        
        p = 0
        self.tags = []
        if type(tags) is FeatureSpan:
            tags = tags.to_list()
        if tags and len(tags[-1][1]) < 2:
            tags[-1] = (tags[-1][0], (tags[-1][1][0], self.player.duration))
        for (t,(b,e)) in tags:
            if b > p:
                self.tags.append((SceneType.SHOW,(p,b)))
            if type(t) is int:
                t = SceneType(t)
            self.tags.append((t,(b,e)))
            p = e
        tags = None
        if p < self.player.duration:
            self.tags.append((SceneType.SHOW,(p,self.player.duration)))

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

        b = tk.Button(skips, text="|< Silence", command=lambda:self.prev('audio'))
        b.grid(row=0, column=2, padx=5)
        self.misc.append(b)

        b = tk.Button(skips, text="|< Diff", command=lambda:self.prev('diff'))
        b.grid(row=0, column=3, padx=(5,10))
        self.misc.append(b)

        b = tk.Button(skips, text="Diff >|", command=lambda:self.next('diff'))
        b.grid(row=0, column=5, padx=(10,5))
        self.misc.append(b)

        b = tk.Button(skips, text="Audio >|", command=lambda:self.next('audio'))
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
        self.taggers.append((tk.Button(tags), SceneType.INTRO, 'Intro'))
        self.taggers.append((tk.Button(tags), SceneType.SHOW, 'Show'))
        self.taggers.append((tk.Button(tags), SceneType.CREDITS, 'Credits'))
        self.taggers.append((tk.Button(tags), SceneType.DO_NOT_USE, 'Bad'))

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
        
        self.map_height = 80
        self.map_width = 1280
        self.mapCanvas = tk.Canvas(self, width=self.map_width, height=self.map_height)
        self.mapCanvas.grid(row=7, column=0, columnspan=5)
        self.mapCanvas.bind("<Button-1>", lambda e:self.move(abs=float(e.x)/self.map_width*self.player.duration))
        
        self.scale_pos = tk.DoubleVar()
        self.scroller = tk.Scale(self, 
                                from_=0, to_=self.player.duration/60, resolution=1/60, tickinterval=5, showvalue=False,
                                length=self.map_width+25, orient=tk.HORIZONTAL, sliderlength=25,
                                variable=self.scale_pos,
                                command=lambda n: self.move(abs=float(n)*60))
        self.scroller.grid(row=9, column=0, columnspan=5)

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
    
    def prev(self,key='diff'):
        span = []
        if key == 'break':
            span = self.tags
        else:
            span = self.spans.get(key)
            
        for (t,(b,e)) in reversed(span):
            if type(t) is bool and not t:
                continue
            p = b+(e-b)/2 if key == 'blank' else b
            if (p - self.position) < -3/self.player.frame_rate:
                self.move(abs=p)
                return

    def next(self,key='diff'):
        span = []
        if key == 'break':
            span = self.tags
        else:
            span = self.spans.get(key)
            
        for (t,(b,e)) in span:
            if type(t) is bool and not t:
                continue
            p = b+(e-b)/2 if key == 'blank' else b
            if (p - self.position) > 3/self.player.frame_rate:
                self.move(abs=p)
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
        try:
            for f in self.player.frames():
                frames.append(f)
                if len(frames) >= 3 and round(f.time - self.player.vt_start, 3) > round(seconds,3) and round(frames[-2].time - self.player.vt_start, 3) >= round(seconds,3):
                    break
        except:
            pass

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
        x = math.ceil(self.position / (self.player.duration / self.map_width))
        self.mapCanvas.coords(self.vMapPos, x, 0, x, self.map_height)
        
        if self.vMaybe is not None:
            startx = math.floor(self.setpos / (self.player.duration / self.map_width))
            stopx = x
            if startx > stopx:
                (startx, stopx) = (stopx, startx)
            elif startx == stopx:
                stopx += 1
            self.mapCanvas.coords(self.vMaybe, startx, 0, stopx, self.map_height)
            #print(self.settype, self.vMaybe, self.setpos, self.position, startx, stopx)

    def drawSpan(self, span, colorMap, top, bottom, force_width=None):
        sec_per_pix = self.player.duration / self.map_width
        for (t,(b,e)) in span:
            startx = math.floor(b / sec_per_pix)
            if force_width is not None:
                stopx = startx + force_width
            else:
                stopx = max(startx+1,math.ceil(e / sec_per_pix))
            color = colorMap.get(t, None)
            if color is not None:
                self.mapCanvas.create_rectangle(startx, top, stopx, bottom, width=0, fill=color)
            #print(t,startx,stopx,color)
    
    def drawMap(self):
        for x in self.mapCanvas.find_all():
            self.mapCanvas.delete(x)
        self.vMaybe = None

        row = self.map_height/4
        pos = 0
        self.drawSpan(self.tags, top=pos, bottom=pos+row, colorMap=SceneType.color_map())
        pos += row
        self.drawSpan(self.spans.get('logo',[]), top=pos, bottom=pos+row, colorMap={True:'blue'})
        pos += row
        self.drawSpan(self.spans.get('diff',[]), top=pos, bottom=pos+row, colorMap={True:'purple'}, force_width=1)
        pos += row
        self.drawSpan(self.spans.get('audio',[]), top=pos, bottom=pos+row, colorMap=AudioSegmentLabel.color_map())
        pos += row

        self.drawSpan(self.spans.get('blank',[]), top=int(row*.3), bottom=pos-int(row*.3), colorMap={True:'black'})

        if self.settype is not None:
            color = SceneType.color_map()[self.settype]
            if color is not None:
                self.vMaybe = self.mapCanvas.create_rectangle(0, 0, 0, self.map_height, width=0, fill=color, stipple='gray50')

        # lastly, add the positional indicator
        self.vMapPos = self.mapCanvas.create_line(0,0,0,self.map_height,arrow=tk.BOTH,fill='orange')
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
        self.drawMap()
        self.next()
    
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
            if b > 0 and b-1 < len(self.tags) and self.tags[b-1][0] == settype and (startpos-self.tags[b-1][1][1]) < 0.02:
                b -= 1
                startpos = self.tags[b][1][0]
            # if the one after is the same type, merge with that
            if e > 0 and e < len(self.tags) and self.tags[e][0] == settype and (self.tags[e][1][0] - endpos) < 0.02:
                endpos = self.tags[e][1][1]
                e += 1
            
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

        self.drawMap()
    
    def truncate_now(self):
        self.setpos = self.player.duration
        self.settype = SceneType.DO_NOT_USE
        self.end_tag(None)
        self.save_and_close()
    
    def save_and_close(self):
        self.result = []
        for (t,(b,e)) in self.tags:
            if t != SceneType.SHOW:
                self.result.append((t.value,(b,e)))
        self.destroy()

    def run(self):
        tk.mainloop()
        return self.result
