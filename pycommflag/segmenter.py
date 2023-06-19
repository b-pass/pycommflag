from .extern.ina_foss import AudioSegmentLabel

class SceneSegmenter:
    def check(self, ftime:float, faudio:AudioSegmentLabel, logo_present:bool, is_blank:bool, is_diff:bool)->bool:
        return False

class NoneSegmenter(SceneSegmenter):
    def check(**kwargs):
        return False
    
    def __str__(self): 
        return 'none'

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

    def __str__(self): 
        return '(' + ' | '.join([str(x) for x in self._subs]) + ')'
    
class AndSegmenter(SceneSegmenter):
    def __init__(self, a, b):
        self._subs = []
        self._prev = []
        self.add(a)
        self.add(b)
    
    def add(self, sub):
        if sub is not None:
            self._subs.append(sub)
            self._prev.append(False)

    def check(self, **kwargs):
        # fuzz the result with the previous frame
        current = list(self._prev)
        for i in range(len(self._subs)):
            res = self._subs[i].check(**kwargs)
            current[i] = current[i] or res
            self._prev[i] = res
        return False not in current

    def __str__(self) -> str:
        return '(' + ' & '.join([str(x) for x in self._subs]) + ')'

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
    
    def __str__(self) -> str:
        return str(int(self._dur)) + 's'

class SilenceSegmenter(SceneSegmenter):
    def check(self, faudio, **kwargs):
        if type(faudio) is int:
            faudio = AudioSegmentLabel(faudio)
        return faudio == AudioSegmentLabel.SILENCE
    
    def __str__(self) -> str:
        return 'silence'

class AudioSegmenter(SceneSegmenter):
    def __init__(self):
        self._prev = AudioSegmentLabel.SILENCE
    
    def check(self, faudio, **kwargs):
        if type(faudio) is int:
            faudio = AudioSegmentLabel(faudio)
        if faudio != self._prev:
            self._prev = faudio
            return True
        else:
            return False

    def __str__(self) -> str:
        return 'audio'
    
class LogoSegmenter(SceneSegmenter):
    def __init__(self):
        self._prev = [False]*30
    
    def check(self, logo_present, **kwargs):
        res = logo_present not in self._prev
        self._prev.append(logo_present)
        self._prev.pop(0)
        return res

    def __str__(self) -> str:
        return 'logo'
    
class BlankSegmenter(SceneSegmenter):
    def check(self, is_blank, **kwargs):
        return is_blank
    def __str__(self) -> str:
        return 'blank'

class DiffSegmenter(SceneSegmenter):
    def check(self, is_diff, **kwargs):
        return is_diff
    def __str__(self) -> str:
        return 'diff'

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
    #print('make segm',name)
    return types[name]()

def sub_parse(eq:str, sep:str|None=None):
    p = 0
    start = p
    val = None
    while True:
        #print(f'{start}...{p} ... [{eq[p]}] {eq[start:p]}')
        if eq[p].isalnum():
            p += 1
            if p < len(eq):
                continue
        if p < len(eq) and eq[p] == '(':
            if start + 1 < p:
                raise Exception(f"Bad text from {start} to {p}: '{eq[start:p]}'")
            (x,next) = sub_parse(eq[p+1:], ')')
            p += x
        elif p > start:
            next = _make_segmenter(eq[start:p])
        else:
            next = None
        if p < len(eq) and eq[p] in ',|-/' and type(val) is not OrSegmenter:
            val = OrSegmenter(val, next)
        elif p < len(eq) and eq[p] in '^+&*' and type(val) is not AndSegmenter:
            val = AndSegmenter(val, next)
        elif type(val) in [OrSegmenter,AndSegmenter]:
            val.add(next)
        elif val is None:
            val = next
        elif next is not None:
            raise Exception(f"Already have a segmenter... missing a logic operator at {p}?")
        if p+1 >= len(eq) or eq[p] == sep:
            break
        p += 1
        start = p
    
    if sep is not None:
        if p < len(eq) and eq[p] == sep:
            p += 1
        else:
            raise Exception(f"Expecting '{sep}' at {p} but not found!")

    return (p,val)

def parse(eq)->SceneSegmenter:
    if not eq:
        return NoneSegmenter()
    (p,val) = sub_parse(eq)
    if p+1 < len(eq):
        raise Exception(f"String parse failed... extra text after {p}")
    #print(repr(val),str(val))
    return val
