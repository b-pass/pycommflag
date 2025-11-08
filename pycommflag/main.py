import os
import sys
import random

def run(opts) -> None|int:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # shut up, tf

    if opts.rebuild or opts.queue:
        os.execvp("mythcommflag", ["mythcommflag"] + sys.argv[1:])

    if opts.exitcode:
        # make sure to return >= 256 for unhandled exceptions
        # otherwise mythtv will stupidly interpret python's exit(1) as a number of commercials
        def myexcepthook(type, value, tb):
            sys.__excepthook__(type, value, tb)
            sys.exit(256)
        sys.excepthook = myexcepthook

    if opts.train:
        from .neural import train
        return train(opts=opts)
    
    if opts.gui and not opts.feature_log:
        opts.feature_log = opts.gui
        from .processor import read_feature_log
        if f := read_feature_log(opts.feature_log).get('filename', None):
            opts.filename = f

    if opts.mythjob:
        from .mythtv import get_filename
        f = get_filename(opts)
        if f:
            opts.filename = f
    
    if not opts.chanid or not opts.starttime:
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
        opts.filename = get_filename(opts)
    
    if opts.reprocess:
        import gc
        from .processor import reprocess
        i = 0
        for fl in opts.reprocess:
            i += 1
            if not os.path.exists(fl) or not os.path.isfile(fl):
                continue
            gc.collect()
            print(f'* Reprocessing {fl} [{i} of {len(opts.reprocess)}]')
            try:
                flog = reprocess(fl, opts=opts)
            except Exception as e:
                print('EXCEPTION')
                print(e)
                print()
                continue
            if not flog:
                print(f'Skipped (bad file)')
                continue

            vf = flog.get('filename', '')
            opts.chanid = flog.get('chanid', '')
            opts.starttime = flog.get('starttime', '')
            
            if vf and not os.path.exists(vf):
                if len(opts.reprocess) > 1 and os.path.exists(os.path.join(os.path.dirname(fl), 'old')):
                    print(f'Archived ({vf} does not exist)')
                    os.rename(fl, os.path.join(os.path.dirname(fl), 'old', os.path.basename(fl)))
                else:
                    print(f'Skipped ({vf} does not exist)')
                continue

            if opts.chanid:
                from .mythtv import check_method
                if not check_method(opts.chanid):
                    print("Channel has flagging disabled")
                    output(opts, [], flog)
                    continue
        
            from .neural import predict, diff_tags
            try:
                old = flog.get('tags', [])
                result = predict(flog, opts, fl)
                (missing,extra,all) = diff_tags(old, result)

                chng = ''
                for (t,b,e) in all:
                    if t != 0:
                        chng += f'{e-b}s {"missing" if t < 0 else "extra"} at {b} to {e}; '
                print(f'{vf}: {len(result)} breaks - changed {extra-missing} seconds : {chng}')

                output(opts, result, flog)
            except Exception as e:
                import traceback
                print('EXCEPTION')
                print(traceback.format_exc())
                print()
                continue

        return 0
    
    if opts.eval:
        from .neural import eval
        eval(opts)
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
        if opts.model_file or opts.reprocess:
            from .neural import predict
            tags = predict(flog, opts)
        w = gui.Window(video=opts.filename, spans=spans, tags=tags, logo=logo)
        res = w.run()
        if res is not None:
            flog['tags'] = res
            processor.write_feature_log(flog, opts.feature_log)
            output(opts, res, flog)
            return 0
        else:
            return 1
    
    if not opts.filename:
        print('No video file was found in the options')
        return 1
    elif not os.path.exists(opts.filename) or os.path.isdir(opts.filename):
        print(f'No such video file "{opts.filename}"')
        return 1
    
    if opts.chanid:
        from .mythtv import check_method
        if not check_method(opts.chanid):
            output(opts, [], None)
            return 0
    
    if opts.no_feature_log:
        import tempfile
        feature_log = tempfile.TemporaryFile('w+', prefix='cf_', suffix='.json')
    elif not opts.feature_log:
        import tempfile
        feature_log = tempfile.gettempdir() + os.path.sep + 'cf_'+os.path.basename(os.path.realpath(opts.filename))+'.json'
    else:
        feature_log = opts.feature_log
    
    from .processor import process_video, read_feature_log
    process_video(opts.filename, feature_log, opts)
    flog = read_feature_log(feature_log)
    
    from .neural import predict
    result = predict(flog, opts, feature_log)
    output(opts, result, flog)

    return 0

def output(opts, result, feature_log=None):
    if result is None:
        return
    
    if opts.output_type in ['mythtv', 'myth', 'auto', '', None]:
        from .mythtv import set_breaks
        if set_breaks(opts, result, feature_log):
            return
        if opts.output_type in ['mythtv', 'myth']:
            return # was not auto, so dont try edl or txt 
    
    vf:str = opts.filename
    if feature_log:
        vf = feature_log.get("filename", vf)
    if not vf:
        vf = opts.feature_log
    
    if not vf:
        return
    
    if vf.endswith('.gz'):
        vf = vf[:-3]
    if vf.endswith('.json'):
        vf = vf[:-5]
    
    if '.' in os.path.basename(vf):
        dot = vf.rfind('.')
    else:
        dot = -1
    ofn = vf[:dot] if dot > 0 else vf
    
    if opts.output_type in ['txt', 'text']:
        from .edl import output_txt
        output_txt(ofn + '.txt', result, feature_log)
    else:
        from .edl import output_edl
        output_edl(ofn + '.edl', result, feature_log)
