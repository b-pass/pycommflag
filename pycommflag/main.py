from typing import Any
import os
import sys
import tempfile
from . import gui
from . import mythtv
from . import neural
from . import processor

def run(opts:Any) -> None|int:
    if opts.rebuild:
        os.execvp("mythcommflag", ["mythcommflag", "--rebuild", "--chanid", opts.chanid, "--starttime", opts.starttime])
    if opts.queue:
        os.execvp("mythcommflag", ["mythcommflag", "--queue", "--chanid", opts.chanid, "--starttime", opts.starttime])
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # shut up, tf

    if opts.reprocess:
        spans = processor.reprocess(opts.reprocess, opts=opts)
        # and run NN
        # and save tags
        # and write tags to DB
        return 0
    
    if opts.train:
        d = neural.load_data(opts)
        if not d:
            return 1
        neural.train(d, opts)
        return 0
    
    if not opts.filename:
        if not opts.chanid and not opts.starttime:
            import re
            if m:=re.match(r'(?:.*/)?cf_(\d{4,6})_(\d{12,})(?:\.[a-zA-Z0-9]{2,5}){1,4}', opts.feature_log):
                opts.chanid = m[1]
                opts.starttime = m[2]
        if opts.chanid and opts.starttime:
            opts.filename = mythtv.get_filename(opts.chanid, opts.starttime)
    
    if not opts.filename:
        print('No video file to work on (need one of: -f, -r, --chanid, etc)')
        return 1
    
    if opts.gui:
        flog = processor.read_feature_log(opts.feature_log)
        logo = processor.read_logo(flog)
        spans = processor.read_feature_spans(flog)
        tags = processor.read_tags(flog)
        if not tags:
            tags = processor.external_tags(opts=opts)
        w = gui.Window(video=opts.filename, spans=spans, tags=tags, logo=logo)
        res = w.run()
        if res is not None:
            flog['tags'] = res
            processor.write_feature_log(flog, opts.feature_log)
            return 0
        else:
            return 1
    
    if opts.no_feature_log:
        feature_log = tempfile.TemporaryFile('w+', prefix='cf_', suffix='.json')
    elif not opts.feature_log:
        feature_log = tempfile.gettempdir() + os.path.sep + 'cf_'+os.path.basename(opts.filename)+'.json'
    else:
        feature_log = opts.feature_log
    
    processor.process_video(opts.filename, feature_log, opts)
    flog = processor.read_feature_log(feature_log)
    spans = processor.read_feature_spans(flog)
    # and then run NN...
    # then, processor.write_tags_into(res, log_f=opts.feature_log)
    # and then save in DB or text or whatever
    return 0
