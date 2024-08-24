import os
import sys

def run(opts) -> None|int:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # shut up, tf

    if opts.rebuild or opts.queue:
        os.execvp("mythcommflag", ["mythcommflag"] + sys.argv[1:])

    if opts.train:
        from .neural import train
        return train(opts=opts)
    
    if opts.gui and not opts.feature_log:
        opts.feature_log = opts.gui
        from .processor import read_feature_log
        if f := read_feature_log(opts.feature_log).get('filename', None):
            opts.filename = f

    if not opts.chanid and not opts.starttime:
        import re
        if opts.feature_log:
            if m:=re.match(r'(?:.*[\//])?cf_(\d{4,6})_(\d{12,})(?:\.[a-zA-Z0-9]{2,5}){1,4}', opts.feature_log):
                opts.chanid = m[1]
                opts.starttime = m[2]
        if opts.filename:
            if m:=re.match(r'(?:.*[\//])?(\d{4,6})_(\d{12,})(?:\.[a-zA-Z0-9]{2,5}){1,4}', os.path.realpath(opts.filename)):
                opts.chanid = m[1]
                opts.starttime = m[2]
    
    if not opts.filename and opts.chanid and opts.starttime:
        from .mythtv import get_filename
        opts.filename = get_filename(opts.chanid, opts.starttime)
    
    if opts.reprocess:
        from .processor import reprocess
        reprocess(opts.reprocess, opts=opts)
        
        from .neural import predict
        result = predict(opts.reprocess, opts=opts)
        if result and opts.chanid and opts.starttime:
            from .mythtv import set_breaks
            set_breaks(opts.chanid, opts.starttime, result)
        return 0
    
    if opts.gui:
        from . import gui, processor
        if not opts.feature_log:
            opts.feature_log = opts.gui
        
        flog = processor.reprocess(opts.gui, opts=opts)
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
                from .mythtv import set_breaks
                set_breaks(opts.chanid, opts.starttime, res)
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
        import tempfile
        feature_log = tempfile.TemporaryFile('w+', prefix='cf_', suffix='.json')
    elif not opts.feature_log:
        import tempfile
        feature_log = tempfile.gettempdir() + os.path.sep + 'cf_'+os.path.basename(os.path.realpath(opts.filename))+'.json'
    else:
        feature_log = opts.feature_log
    
    from .processor import process_video
    process_video(opts.filename, feature_log, opts)
    
    from .neural import predict
    result = predict(feature_log, opts=opts)

    print("Not saving result because this thing is probably broken")
    #if result and opts.chanid and opts.starttime:
    #    from .mythtv import set_breaks
    #    set_breaks(opts.chanid, opts.starttime, result)
    return 0
