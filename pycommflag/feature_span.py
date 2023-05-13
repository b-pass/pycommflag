from enum import Enum

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

        p = self._spans[0][0]
        for (v,(b,e)) in zip(self._values, self._spans):
            if b != p:
                import logging as log
                log.warn(f"FEATURE SPAN NOT CONTIGUOUS! at {p} throguh {b}, a {b-p} gap!")
            p = e

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

    def start(self):
        self._values = [AudioSegmentLabel.SILENCE]
        self._spans = [(0.0,0.0)]
    
    # de-overlap and fill in gaps with silence as we go
    def add(self, start, end, what):
        #if type(what) is int: what = AudioSegmentLabel[what]
        if self._values[-1] == what:
            self._spans[-1] = (self._spans[-1][0], max(end, self._spans[-1][1]))
            return
        if self._spans[-1][1] > start:
            # overlap with previous
            if self._spans[-1][0] >= start:
                # total overlap, just remove it
                del self._spans[-1]
                del self._values[-1]
                if self._spans:
                    start = max(start,self._spans[-1][1])
                    if start < end:
                        # retry
                        self.add(start, end, what)
                    return
                else:
                    pass
            else:
                # partial overlap, truncate previous
                self._spans[-1] = (self._spans[-1][0], start)
                pass
        elif self._spans[-1][1] < start:
            if what != AudioSegmentLabel.SILENCE:
                self.add(self._spans[-1][1], start, AudioSegmentLabel.SILENCE)
                # and retry
                self.add(start, end, what)
                return
            else:
                start = self._spans[-1][1]
        else:
            pass
        
        self._spans.append((start,end))
        self._values.append(what)
    
    def end(self, when):
        if self._spans and self._spans[-1][1] < when:
            self.add(self._spans[-1][1], when, AudioSegmentLabel.SILENCE)

        p = self._spans[0][0]
        for (v,(b,e)) in zip(self._values, self._spans):
            if b != p:
                import logging as log
                log.warn(f"FEATURE SPAN NOT CONTIGUOUS! at {p} throguh {b}, a {b-p} gap!")
            p = e

    def to_json(self):
        return super().to_json(converter=lambda x:x.value)

    def to_list(self,serializable=False):
        if serializable:
            return [(lab.value, tm) for (lab,tm) in zip(self._values, self._spans)]
        else:
            return [x for x in zip(self._values, self._spans)]
    
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

    def color_map():
        return {
            SceneType.SHOW : 'green',
            SceneType.INTRO : 'yellow',
            SceneType.TRANSITION : 'pink',
            SceneType.COMMERCIAL : 'red',
            SceneType.CREDITS : 'yellow',
            SceneType.DO_NOT_USE : None,# 'dark grey',
        }

from enum import Enum
class AudioSegmentLabel(Enum):
    noEnergy = 0
    silence = 0
    SILENCE = 0

    energy = 1
    speech = 1
    SPEECH = 1

    music = 2
    MUSIC = 2

    noise = 3
    NOISE = 3

    def count():
        return 4

    def color_map():
        return {
            AudioSegmentLabel.SILENCE : None,# 'dark grey',
            AudioSegmentLabel.SPEECH : 'dark blue',
            AudioSegmentLabel.MUSIC : 'light blue',
            AudioSegmentLabel.NOISE : 'pink',
        }
