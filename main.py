from typing import Any
import tempfile
from . import processor

def run(opts:Any) -> None:
    if opts.reprocess:
        return processor.process_log(opts.reprocess, opts.frame_log, opts)

    frame_log = open(opts.frame_log, 'wb') if opts.frame_log else tempfile.TemporaryFile('wb', prefix='log_', suffix='.cfl')
    
    processor.process_video(opts.filename, frame_log, opts)
    processor.process_log(frame_log, frame_log, opts)
