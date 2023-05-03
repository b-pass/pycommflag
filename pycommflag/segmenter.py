from .extern.ina_foss import AudioSegmentLabel

class SceneSegmenter:
    def check(self, ftime:float, faudio:AudioSegmentLabel, logo_present:bool, is_blank:bool, is_diff:bool)->bool:
        return False

class NoneSegmenter(SceneSegmenter):
    def check(**kwargs):
        return False

class OrSegmenter(SceneSegmenter):
    def __init__(self, a, b):
        self._subs = []
        self.add(a)
        self.add(b)
    
    def add(self, sub):
        if sub is not None:
            self._subs.append(sub)

    def check(self, **kwargs):
        for s in self._subs:
            if s.check(**kwargs):
                return True
        return False

class AndSegmenter(SceneSegmenter):
    def __init__(self, a, b):
        self._subs = []
        self.add(a)
        self.add(b)
    
    def add(self, sub):
        if sub is not None:
            self._subs.append(sub)

    def check(self, **kwargs):
        for s in self._subs:
            if not s.check(**kwargs):
                return False
        return True

class TimeSegmenter(SceneSegmenter):
    def __init__(self, duration=1.0):
        self._dur = duration
        self._last = 0
    
    def check(self, ftime, **kwargs):
        diff = ftime - self._last
        if diff >= self._dur:
            self._last = ftime
            return True
        else:
            return False

class SilenceSegmenter(SceneSegmenter):
    def check(self, faudio, **kwargs):
        if faudio is int:
            faudio = AudioSegmentLabel(faudio)
        return faudio == AudioSegmentLabel.SILENCE

class AudioSegmenter(SceneSegmenter):
    def __init__(self):
        self._prev = AudioSegmentLabel.SILENCE
    
    def check(self, faudio, **kwargs):
        if faudio is int:
            faudio = AudioSegmentLabel(faudio)
        if faudio != self._prev:
            self._prev = faudio
            return True
        else:
            return False

class LogoSegmenter(SceneSegmenter):
    def __init__(self):
        self._prev = False
    
    def check(self, logo_present, **kwargs):
        if logo_present != self._prev:
            self._prev = logo_present
            return True
        else:
            return False

class BlankSegmenter(SceneSegmenter):
    def check(self, is_blank, **kwargs):
        return is_blank

class DiffSegmenter(SceneSegmenter):
    def check(self, is_diff, **kwargs):
        return is_diff

def _make_segmenter(name:str):
    name = name.lower().strip()
    types = {
        'none':NoneSegmenter,
        '1s':TimeSegmenter,
        'silence':SilenceSegmenter,
        'audio':AudioSegmenter,
        'logo':LogoSegmenter,
        'blank':BlankSegmenter,
        'black':BlankSegmenter,
        'diff':DiffSegmenter,
        'scene':DiffSegmenter,
        'change':DiffSegmenter,
    }
    if name not in types:
        raise Exception(f"{name} is not a valid scene segmenter name")
    return types[name]()

def sub_parse(eq:str, sep:str|None=None):
    p = 0
    start = p
    while p < len(eq) and eq[p] != sep:
        if eq[p].isalnum():
            p += 1
            continue
        if eq[p] == '(':
            if start + 1 < p:
                raise Exception(f"Bad text from {start} to {p}: '{eq[start:p-start]}'")
            (p,next) = sub_parse(eq[p+1:], ')')
            p += 1
            continue
        
        next = _make_segmenter(eq[start:p-start])
        start = p+1
        if eq[p] in ',|-/' and type(val) is not OrSegmenter:
            val = OrSegmenter(val, next)
        elif eq[p] in '^+&*' and type(val) is not AndSegmenter:
            val = AndSegmenter(val, next)
        elif type(val) in [OrSegmenter,AndSegmenter]:
            val.add(next)
        elif val is None:
            val = next
        else:
            raise Exception("Already have a segmenter... missing a logic operator?")
    
    return (p,val)

def parse(eq)->SceneSegmenter:
    if not eq:
        return NoneSegmenter()
    (p,val) = sub_parse(eq)
    if p+1 < len(eq):
        raise Exception(f"String parse failed... extra text after {p}")
    return val
