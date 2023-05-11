from .extern.ina_foss import AudioSegmentLabel

class FeatureSpan:
    def __init__(self):
        self._values = []
        self._spans = []
        self._lastval = None
    
    def _log(self, when, what):
        if self._spans:
            self._spans[-1] = (self._spans[-1][0], when)
        if what is not None:
            self._spans.append((when,))
            self._values.append(what)
    
    def start(self, when, what):
        self._log(when, what)
        self._lastval = what

    def add(self, when, what):
        if what != self._lastval:
            self._log(when, what)
        self._lastval = what
    
    def end(self, when):
        if self._spans:
            self._spans[-1] = (self._spans[-1][0], when)

        print(f"****** {len(self._spans)} spans ******")
        p = self._spans[0][0]
        for (v,(b,e)) in zip(self._values, self._spans):
            if b != p:
                print(f"*** NOT CONTIGUOUS! {b-p} gap!")
            p = e

            print(f"{b} - {e} ({e-b}) = {v}")

    def from_json(self, js:str|list, converter):
        if type(js) is str:
            import json
            js = json.loads(js)
        for (x,y) in js:
            self._values.append(converter(x))
            self._spans.append((y[0],y[1]))
        return True

    def to_json(self, converter=int):
        import json
        return json.dumps([(converter(x),y) for (x,y) in zip(self._values, self._spans)])

    def to_list(self):
        return [x for x in zip(self._values, self._spans)]

class SeparatorFeatureSpan(FeatureSpan):
    def __init__(self):
        super().__init__()
        self._prev = None
    
    def add(self, when, what):
        if self._lastval:
            if what:
                self._prev = when
            elif self._prev is not None:
                self._log(self._prev, self._lastval)
        elif what:
            self._log(when, what)
            self._prev = None
        else:
            pass
        self._lastval = what

class AudioFeatureSpan(FeatureSpan):
    def __init__(self):
        super().__init__()
    
    def add_all(self, audio:list[tuple[int,float,float]]):
        if not audio:
            return
        self._spans = [(0,0.000001)]
        self._values = [AudioSegmentLabel.SILENCE]
        for (t,b,e) in audio:
            if t == self._values[-1]:
                # not a change, just expand the existing entry
                self._spans[-1] = (min(b,self._spans[-1][0]), max(e, self._spans[-1][1]))
                continue
            if b > self._spans[-1][1]:
                # skipped a spot, add silence there
                if t == AudioSegmentLabel.SILENCE:
                    # will be silence, just start sooner
                    b = self._spans[-1][1]
                elif self._values[-1] == AudioSegmentLabel.SILENCE:
                    # was silence, end later
                    self._spans[-1] = (self._spans[-1][0], b)
                else:
                    # insert a silence between
                    self._spans.append((self._spans[-1][1],b))
                    self._values.append(AudioSegmentLabel.SILENCE)
            if b < self._spans[-1][1]:
                # going backwards, need to de-overlap the previous entry
                if e <= self._spans[-1][0]:
                    # would go entirely before the last entry, so just drop this
                    continue
                # replace the old segment 
                b = self._spans[-1][0] # now spanning both entries
                self._spans.pop() # drop old
                self._values.pop() # drop old
                if t == self._values[-1]:
                    # not a change, just expand the entry
                    self._spans[-1] = (min(b,self._spans[-1][0]), max(e, self._spans[-1][1]))
                    continue
            # add a new entry
            self._spans.append((b,e))
            self._values.append(t)
        
        p = self._spans[0][0]
        for (t,(b,e)) in zip(self._values, self._spans):
            assert(b == p)
            p = e
    
    def to_json(self):
        return super().to_json(converter=lambda x:x.value)

from enum import Enum
class SceneType(Enum):
    UNKNOWN = 0
    SHOW = 0
    INTRO = 1
    TRANSITION = 2
    COMMERCIAL = 3
    CREDITS = 4
    DO_NOT_USE = 5

    def count():
        return SceneType.DO_NOT_USE.value

    def color(self):
        colors = ['green','yellow','blue','red','yellow','gray']
        return colors[self.value]
    
    def logo_color(self):
        colors = ['light green','light yellow','light blue','pink','light yellow','gray']
        return colors[self.value]
    
    def new_color(self):
        colors = ['dark green','dark orange','dark blue','deep pink','dark orange','gray']
        return colors[self.value]
