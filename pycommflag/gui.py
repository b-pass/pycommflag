import tkinter as tk
from PIL import ImageTk, Image
from .player import Player

class Window(tk.Tk):
    def __init__(self, video, log):
        tk.Tk.__init__(self)
        self.title("pycommflag editor")
        self.player = Player(video)
        self.frame = tk.Frame(master=self, width=800, height=450)
        self.frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

        self.misc = []
        self.video_labels = []
        self.images = []
        for x in range(5):
            v = tk.Label(master=self.frame, relief=tk.SUNKEN)
            if x == 2:
                v.grid(row=1,column=0, columnspan=5)
            else:
                v.grid(row=2, column=x)
            self.video_labels.append(v)

        a = tk.Label(master=self.frame, text="-5s")
        a.grid(row=3, column=0, sticky="nswe")
        self.misc.append(a)
            
        a = tk.Label(master=self.frame, text="-1f")
        a.grid(row=3, column=1, sticky="nswe")
        self.misc.append(a)
            
        self.pos_label = tk.Label(master=self.frame, text="00:00.000")
        self.pos_label.grid(row=2, column=2, sticky="nswe")

        a = tk.Label(master=self.frame, text="+1f")
        a.grid(row=3, column=3, sticky="nswe")
        self.misc.append(a)
        
        a = tk.Label(master=self.frame, text="+5s")
        a.grid(row=3, column=4, sticky="nswe")
        self.misc.append(a)

        skipf = tk.Frame(self.frame)
        skipf.grid(row=4, column=0, columnspan=5)
        self.misc.append(skipf)

        b = tk.Button(master=skipf, text="<<<< 30s", command=lambda:self.move(seconds=-30))
        b.grid(row=0, column=0, padx=5)
        self.misc.append(b)
        
        b = tk.Button(master=skipf, text="<<< 5s", command=lambda:self.move(seconds=-5))
        b.grid(row=0, column=1, padx=5)
        self.misc.append(b)
        
        b = tk.Button(master=skipf, text="|< Scene", command=lambda:self.prev_scene())
        b.grid(row=0, column=2, padx=5)
        self.misc.append(b)

        b = tk.Button(master=skipf, text="< 1s", command=lambda:self.move(seconds=-1))
        b.grid(row=0, column=3, padx=5)
        self.misc.append(b)
        
        b = tk.Button(master=skipf, text="< 1f", command=lambda:self.move(frames=-1))
        b.grid(row=0, column=4, padx=(5,20))
        self.misc.append(b)

        b = tk.Button(master=skipf, text="1f >", command=lambda:self.move(frames=1))
        b.grid(row=0, column=5, padx=(20,5))
        self.misc.append(b)
        
        b = tk.Button(master=skipf, text="1s >", command=lambda:self.move(seconds=1))
        b.grid(row=0, column=6, padx=5)
        self.misc.append(b)
        
        b = tk.Button(master=skipf, text="Scene >|", command=lambda:self.next_scene())
        b.grid(row=0, column=7, padx=5)
        self.misc.append(b)
        
        b = tk.Button(master=skipf, text="5s >>>", command=lambda:self.move(seconds=5))
        b.grid(row=0, column=8, padx=5)
        self.misc.append(b)

        b = tk.Button(master=skipf, text="30s >>>>", command=lambda:self.move(seconds=30))
        b.grid(row=0, column=9, padx=5)
        self.misc.append(b)

        self.info = tk.Label(master=self.frame, text=f'File: {video}; Length:{self.player.duration/60.0:0.1f} mins; {self.player.frame_rate} fps')
        self.info.grid(row=7, column=0, sticky="se", columnspan=5)

        self.position = 0
        self.move(abs=0)
    
    def prev_scene(self):
        pass
    def next_scene(self):
        pass
    
    def move(self, frames=0, seconds=0, abs=None):
        seconds += frames/self.player.frame_rate
        if abs is not None:
            seconds += abs
        else:
            seconds += self.position

        self.position = seconds
        pos = f'{int(seconds/60):02}:{seconds%60:06.03f}'
        self.pos_label.configure(text=pos)
        
        self.images = [None]*5

        need = [seconds-5, seconds-1/self.player.frame_rate, seconds, seconds+1/self.player.frame_rate, seconds+5]
        if need[0] > 0:
            self.player.seek(need[0])
            n = 0
        else:
            self.player.seek(0)
            n = 1 if need[1] >= 0 else 2
        
        for (f,_) in self.player.frames():
            print(f,f.time, f.time - self.player.vt_start, need[n])
            if (f.time - self.player.vt_start) >= need[n]:
                if n == 2:
                    self.images[2] = ImageTk.PhotoImage(f.to_image(height=360,width=640))
                else:
                    self.images[n] = ImageTk.PhotoImage(f.to_image(height=180,width=320))
                n += 1
                if n >= len(need):
                    break
        
        for n in range(len(self.images)):
            self.video_labels[n].configure(image=self.images[n])
        
        #when = f.time - 1/self.player.frame_rate + 5
        #for (f,_) in self.player.frames():
        #    if f.time >= when:
        #        self.video_labels[4].configure(image=ImageTk.PhotoImage(f.to_image(height=360,width=640)))
        #        break
    
    def run(self):
        tk.mainloop()
