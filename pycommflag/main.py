from typing import Any
import os
import tempfile
from . import processor
from . import gui

def run(opts:Any) -> None|int:
    if opts.reprocess:
        return processor.process_features(opts.reprocess, opts.feature_log, opts)
    
    if not opts.filename:
        print('Either reprocess or video file are required')
        return 1

    if opts.gui:
        scenes = processor.process_scenes(opts.feature_log) if opts.feature_log else []
        w = gui.Window(video=opts.filename, scenes=scenes)
        return w.run()
    
    if opts.no_feature_log:
        feature_log = tempfile.TemporaryFile('w+b', prefix='cf_', suffix='.feat')
    elif not opts.feature_log:
        feature_log = tempfile.gettempdir() + os.path.sep + 'cf_'+os.path.basename(opts.filename)+'.feat'
    else:
        feature_log = opts.feature_log
    
    processor.process_video(opts.filename, feature_log, opts)
    return processor.process_features(feature_log, feature_log, opts)
