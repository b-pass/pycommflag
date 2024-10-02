import os
from .feature_span import SceneType

def output_edl(outfile, marks, flog=None):
    with open(outfile, 'wt') as of:
        for (t,(s,e)) in marks:
            et = 3 if t in [SceneType.COMMERCIAL, SceneType.COMMERCIAL.value] else 2
            of.write(f'{round(s, 5)}\t{round(e, 5)}\t{et}\n')

def output_txt(outfile, marks, flog):
    rate = flog.get('frame_rate', 29.97)
    irate = round(rate * 100)
    frames = int(rate * flog.get('duration', 0.0))
    with open(outfile, 'wt') as of:
        of.write(f'FILE PROCESSING COMPLETE {frames} FRAMES AT {irate}\n')
        of.write(f'-------------------\n')
        for (t,(s,e)) in marks:
            if t not in [SceneType.COMMERCIAL, SceneType.COMMERCIAL.value]:
                continue
            of.write(f'{int(s*rate)} {int(e*rate)}\n')
