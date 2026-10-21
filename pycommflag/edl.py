import os
from contextlib import contextmanager
from .feature_span import SceneType

@contextmanager
def _replace(outfile):
    """Write to a temporary file and swap it in, so a player never reads half a file
    (a live run rewrites these while the recording plays)."""
    tmp = outfile + '.tmp'
    with open(tmp, 'wt') as of:
        yield of
    os.replace(tmp, outfile)

def output_edl(outfile, marks, flog=None):
    with _replace(outfile) as of:
        for (t,(s,e)) in marks:
            et = 3 if t in [SceneType.COMMERCIAL, SceneType.COMMERCIAL.value] else 2
            of.write(f'{round(s, 5)}\t{round(e, 5)}\t{et}\n')

def output_txt(outfile, marks, flog):
    rate = flog.get('frame_rate', 29.97)
    irate = round(rate * 100)
    frames = int(rate * flog.get('duration', 0.0))
    with _replace(outfile) as of:
        of.write(f'FILE PROCESSING COMPLETE {frames} FRAMES AT {irate}\n')
        of.write(f'-------------------\n')
        for (t,(s,e)) in marks:
            if t not in [SceneType.COMMERCIAL, SceneType.COMMERCIAL.value]:
                continue
            of.write(f'{int(s*rate)} {int(e*rate)}\n')
