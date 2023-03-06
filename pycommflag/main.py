from typing import Any
import os
import sys
import tempfile
from . import processor
from . import gui
from . import mythtv

def run(opts:Any) -> None|int:
    if opts.rebuild:
        os.execvp("mythcommflag", ["mythcommflag", "--rebuild", "--chanid", opts.chanid, "--starttime", opts.starttime])
    if opts.queue:
        os.execvp("mythcommflag", ["mythcommflag", "--queue", "--chanid", opts.chanid, "--starttime", opts.starttime])
    
    if opts.reprocess:
        return processor.process_features(opts.reprocess, opts.feature_log, opts)
    
    if opts.chanid and opts.starttime:
        opts.filename = mythtv.get_filename(opts.chanid, opts.starttime)

    if not opts.filename:
        print('No file to work on (need one of: -f, -r, --chanid, etc)')
        return 1

    if opts.gui:
        logo = processor.read_logo(opts.feature_log) if opts.feature_log else None
        scenes = processor.process_scenes(opts.feature_log) if opts.feature_log else []
        w = gui.Window(video=opts.filename, scenes=scenes, logo=logo)
        return w.run()
    
    if opts.no_feature_log:
        feature_log = tempfile.TemporaryFile('w+b', prefix='cf_', suffix='.feat')
    elif not opts.feature_log:
        feature_log = tempfile.gettempdir() + os.path.sep + 'cf_'+os.path.basename(opts.filename)+'.feat'
    else:
        feature_log = opts.feature_log
    
    processor.process_video(opts.filename, feature_log, opts)
    return processor.process_features(feature_log, feature_log, opts)
