from __future__ import annotations
from typing import BinaryIO
import numpy as np
import struct
import pickle

class Scene:
    def __init__(self,ftime:float=None,column:np.ndarray=None,audio:np.ndarray=None,logo_present:bool=None,is_blank:bool=False,infile=None):
        if infile is not None:
            self.read_bin(infile)
        else:
            self.finished = False
            self.start_time = ftime
            self.stop_time = ftime
            self.frame_count = 1
            self.barcode = column.astype('int32')
            self.avg_peaks = audio
            self.peak_peaks = audio
            self.valley_peaks = audio
            self.logo_count = 1 if logo_present else 0
            self.is_blank = is_blank
            self.type = SceneType.UNKNOWN
            #s.newtype = None
        
    def __iadd__(self, tup:tuple[float,np.ndarray,np.ndarray,bool]):
        assert(not self.finished)
        self.stop_time = tup[0]
        self.frame_count += 1
        self.barcode += tup[1] #self.barcode = np.add(self.barcode, tup[1], casting='unsafe', dtype='uint32')
        self.avg_peaks += tup[2]
        self.peak_peaks = np.where(self.peak_peaks > tup[2], self.peak_peaks, tup[2])
        self.valley_peaks = np.where(self.valley_peaks < tup[2], self.valley_peaks, tup[2])
        if tup[3]: self.logo_count += 1
        return self
    
    def finish(self, next_frame_time:float=None):
        assert(not self.finished)
        if next_frame_time:
            self.stop_time = max(self.stop_time, next_frame_time)
        self.barcode = (self.barcode / self.frame_count).astype('uint8')
        self.avg_peaks = (self.avg_peaks / self.frame_count).astype('float32')
        self.logo = self.logo_count / self.frame_count 
        self.finished = True
    
    def difference(self, other:Scene):
        assert(self.finished and other.finished)
        s = np.std(self.barcode.astype('int16') - other.barcode, (0))
        return max(s)
    
    @property
    def middle_time(self):
        if self.frame_count > 2:
            return self.start_time + (self.stop_time - self.start_time)/2
        else:
            return self.start_time
    
    @property
    def duration(self):
        return self.stop_time - self.start_time

    def __len__(self):
        return self.frame_count
    
    @property
    def is_break(self):
        return self.type == SceneType.COMMERCIAL

    @is_break.setter
    def is_break(self, value:bool):
        if value:
            self.type = SceneType.COMMERCIAL
        elif self.type == SceneType.COMMERCIAL:
            self.type = SceneType.UNKNOWN
        else:
            pass
    
    def write_bin(self, fd:BinaryIO):
        assert(self.finished)
        fd.write(struct.pack('ffI', self.start_time, self.stop_time, self.frame_count))
        #self.barcode.astype('uint8').tofile(fd)
        self.avg_peaks.astype('float32').tofile(fd)
        self.peak_peaks.astype('float32').tofile(fd)
        self.valley_peaks.astype('float32').tofile(fd)
        fd.write(struct.pack('III', self.logo_count, int(self.is_blank), getattr(self, 'newtype', self.type).value))
    
    def read_bin(self, fd:BinaryIO):
        (self.start_time, self.stop_time, self.frame_count) = struct.unpack('ffI', fd.read(12))
        self.barcode = None #self.barcode = np.fromfile(fd, dtype='uint8', count=720*3).reshape((720,3))
        self.avg_peaks = np.fromfile(fd, dtype='float32', count=6).reshape((6,1))
        self.peak_peaks = np.fromfile(fd, dtype='float32', count=6).reshape((6,1))
        self.valley_peaks = np.fromfile(fd, dtype='float32', count=6).reshape((6,1))
        (self.logo_count, isblank, type) = struct.unpack('III', fd.read(12))
        self.is_blank = bool(isblank)
        self.type = SceneType(type)

        self.logo = self.logo_count / self.frame_count 
        self.finished = True
    

from enum import Enum
class SceneType(Enum):
    UNKNOWN = 0
    SHOW = 0
    INTRO = 1
    TRANSITION = 2
    COMMERCIAL = 3
    CREDITS = 4
    TRUNCATED = 5

    def count():
        return SceneType.TRUNCATED.value

    def color(self):
        colors = ['green','yellow','blue','red','yellow','gray']
        return colors[self.value]
    
    def logo_color(self):
        colors = ['light green','light yellow','light blue','pink','light yellow','gray']
        return colors[self.value]
    
    def new_color(self):
        colors = ['dark green','dark orange','dark blue','deep pink','dark orange','gray']
        return colors[self.value]
