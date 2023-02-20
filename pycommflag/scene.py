from __future__ import annotations
import av
import numpy as np

class Scene:
    def __init__(self,ftime:float,column:np.ndarray,audio:np.ndarray,logo_present:bool,start_blank:bool=False):
        self.finished = False
        self.start_time = ftime
        self.stop_time = ftime
        self.frame_count = 1
        self.barcode = column.astype('int32')
        self.avg_peaks = audio
        self.peak_peaks = audio
        self.valley_peaks = audio
        self.logo_count = 1 if logo_present else 0
        self.start_blank = start_blank
        self.end_blank = False
    
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
    
    def finish(self, next_frame_time:float=None, end_blank=False):
        assert(not self.finished)
        if next_frame_time:
            self.stop_time = max(self.stop_time, next_frame_time)
        if end_blank:
            self.end_blank = True
        self.barcode = (self.barcode / self.frame_count).astype('uint8')
        self.avg_peaks = (self.avg_peaks / self.frame_count).astype('float32')
        self.logo = self.logo_count / self.frame_count 
        self.finished = True
    
    def difference(self, other:Scene):
        assert(self.finished and other.finished)
        s = np.std(self.barcode.astype('int16') - other.barcode, (0))
        return max(s)
    
    def __len__(self):
        return self.frame_count

    def write_txt(self, tout):
        tout.write('{\n')
        tout.write(f'  "start": {self.start_time},\n')
        tout.write(f'  "stop":  {self.stop_time},\n')
        tout.write(f'  "frames":{self.frame_count},\n')
        tout.write(f'  "logo":  {self.logo},\n')
        tout.write(f'  "bar":   {self.barcode.tolist()},\n')
        tout.write(f'  "aavg":  {self.avg_peaks.tolist()},\n')
        tout.write(f'  "amax":  {self.peak_peaks.tolist()},\n')
        tout.write(f'  "amin":  {self.valley_peaks.tolist()},\n')
        tout.write('},\n')
