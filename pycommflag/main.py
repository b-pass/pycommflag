import os
import re
import sys
import tempfile
from typing import Any
from . import gui
from . import mythtv
from . import neural
from . import processor
from . import segmenter

def run(opts:Any) -> None|int:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # shut up, tf

    if opts.rebuild or opts.queue:
        os.execvp("mythcommflag", ["mythcommflag"] + sys.argv[1:])
    
    segmenter.parse(opts.segmeth) # parse early to detect raised parse exceptions

    if opts.train:
        d = neural.load_data(opts)
        if not d:
            return 1
        neural.train(d, opts)
        return 0
    
    if not opts.chanid and not opts.starttime:
        if opts.feature_log:
            if m:=re.match(r'(?:.*[\//])?cf_(\d{4,6})_(\d{12,})(?:\.[a-zA-Z0-9]{2,5}){1,4}', opts.feature_log):
                opts.chanid = m[1]
                opts.starttime = m[2]
        if opts.filename:
            if m:=re.match(r'(?:.*[\//])?(\d{4,6})_(\d{12,})(?:\.[a-zA-Z0-9]{2,5}){1,4}', os.path.realpath(opts.filename)):
                opts.chanid = m[1]
                opts.starttime = m[2]
    
    if not opts.filename and opts.chanid and opts.starttime:
        opts.filename = mythtv.get_filename(opts.chanid, opts.starttime)
    
    if opts.reprocess:
        processor.reprocess(opts.reprocess, opts=opts)
        result = neural.predict(opts.reprocess, opts=opts)
        if result and opts.chanid and opts.starttime:
            mythtv.set_breaks(opts.chanid, opts.starttime, result)
        return 0
    
    if opts.gui:
        flog = processor.reprocess(opts.feature_log, opts=opts)
        if not opts.filename:
            opts.filename = flog.get('filename','')
            
        if not opts.filename:
            print('No video file was found in the options')
            return 1
        elif not os.path.exists(opts.filename) or os.path.isdir(opts.filename):
            print(f'No such video file "{opts.filename}"')
            return 1
        
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
            if res and opts.chanid and opts.starttime:
                mythtv.set_breaks(opts.chanid, opts.starttime, res)
            return 0
        else:
            return 1
    
    if not opts.filename:
        print('No video file was found in the options')
        return 1
    elif not os.path.exists(opts.filename) or os.path.isdir(opts.filename):
        print(f'No such video file "{opts.filename}"')
        return 1
    
    if opts.no_feature_log:
        feature_log = tempfile.TemporaryFile('w+', prefix='cf_', suffix='.json')
    elif not opts.feature_log:
        feature_log = tempfile.gettempdir() + os.path.sep + 'cf_'+os.path.basename(os.path.realpath(opts.filename))+'.json'
    else:
        feature_log = opts.feature_log
    
    processor.process_video(opts.filename, feature_log, opts)
    result = neural.predict(feature_log, opts=opts)
    if result and opts.chanid and opts.starttime:
        mythtv.set_breaks(opts.chanid, opts.starttime, result)
    return 0
