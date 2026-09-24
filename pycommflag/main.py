import os
import sys
from contextlib import contextmanager

from .options import Mode


def run(opts) -> int:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # shut up, tf

    if opts.mode is Mode.PASSTHROUGH:
        os.execvp("mythcommflag", ["mythcommflag"] + sys.argv[1:])

    if opts.exitcode:
        # make sure to return >= 256 for unhandled exceptions
        # otherwise mythtv will stupidly interpret python's exit(1) as a number of commercials
        def myexcepthook(type, value, tb):
            sys.__excepthook__(type, value, tb)
            sys.exit(256)
        sys.excepthook = myexcepthook

    # training has no single input file, so it runs before input resolution
    if opts.mode is Mode.TRAIN:
        from .neural import train
        return train(opts=opts)

    _resolve_inputs(opts)

    return {
        Mode.REPROCESS: _cmd_reprocess,
        Mode.EVAL: _cmd_eval,
        Mode.GUI: _cmd_gui,
        Mode.FLAG: _cmd_flag,
    }[opts.mode](opts)

# --- input resolution ---------------------------------------------------------

def _resolve_inputs(opts) -> None:
    """Fill in whichever of filename/feature_log/chanid/starttime the user left out.

    Mutates opts in place, which is also what mythtv.get_filename does when it
    resolves a job id into a recording.
    """
    if opts.gui:
        if not opts.feature_log:
            opts.feature_log = opts.gui
        if not opts.filename:
            from .processor import read_feature_log
            opts.filename = read_feature_log(opts.feature_log).get('filename', '')

    if opts.mythjob:
        from .mythtv import get_filename
        if f := get_filename(opts):
            opts.filename = f

    # must come after the two above: it reads opts.filename
    if not opts.chanid or not opts.starttime:
        _infer_chanid_starttime(opts)

    if not opts.filename and opts.chanid and opts.starttime:
        from .mythtv import get_filename
        opts.filename = get_filename(opts)


def _infer_chanid_starttime(opts) -> None:
    """Recover chanid/starttime from mythtv's <chanid>_<starttime>.ext naming."""
    import re
    if opts.feature_log:
        if m := re.match(r'(?:.*[\//])?cf_(\d{4,6})_(\d{12,})(?:\.[a-zA-Z0-9]{2,5}){1,4}', opts.feature_log):
            opts.chanid = m[1]
            opts.starttime = m[2]
    if opts.filename:
        if m := re.match(r'(?:.*[\//])?(\d{4,6})_(\d{12,})(?:\.[a-zA-Z0-9]{2,5}){1,4}', os.path.realpath(opts.filename)):
            opts.chanid = m[1]
            opts.starttime = m[2]


def _require_video(opts) -> None|str:
    """Return an error message, or None if opts.filename points at a usable video."""
    if not opts.filename:
        return 'No video file was found in the options'
    if not os.path.exists(opts.filename) or os.path.isdir(opts.filename):
        return f'No such video file "{opts.filename}"'
    return None


# --- commands -----------------------------------------------------------------

def _cmd_eval(opts) -> int:
    from .neural import eval
    eval(opts)
    return 0


def _cmd_gui(opts) -> int:
    from . import gui, processor

    flog = processor.reprocess(opts.gui, opts=opts)
    if not opts.filename:
        opts.filename = flog.get('filename','')

    if err := _require_video(opts):
        print(err)
        return 1

    w = gui.Window(opts=opts, video=opts.filename, flog=flog)
    res = w.run()
    if res is None:
        return 1

    flog['tags'] = res
    processor.write_feature_log(flog, opts.feature_log)
    output(opts, res, flog)
    return 0


def _cmd_reprocess(opts) -> int:
    import gc
    import traceback

    i = 0
    archived = 0
    for fl in opts.reprocess:
        i += 1
        if not os.path.isfile(fl):
            continue

        gc.collect()
        print(f'* Reprocessing {fl} [{i-archived} of {len(opts.reprocess)-archived}]')
        try:
            if _reprocess_one(opts, fl, allow_archive=len(opts.reprocess) > 1):
                archived += 1
        except Exception:
            print('EXCEPTION')
            print(traceback.format_exc())
            print()

    return 0


def _reprocess_one(opts, fl, allow_archive:bool) -> bool:
    """Re-predict one saved feature log. Returns True if the log got archived."""
    from .processor import reprocess

    flog = reprocess(fl, opts=opts)
    if not flog:
        print('Skipped (bad file)')
        return False

    vf = flog.get('filename', '')
    opts.chanid = flog.get('chanid', '')
    opts.starttime = flog.get('starttime', '')

    if vf and not os.path.exists(vf):
        old = os.path.join(os.path.dirname(fl), 'old')
        if allow_archive and os.path.exists(old):
            print(f'Archived ({vf} does not exist)')
            os.rename(fl, os.path.join(old, os.path.basename(fl)))
            return True
        print(f'Skipped ({vf} does not exist)')
        return False

    if opts.chanid:
        from .mythtv import check_method
        if not check_method(opts.chanid):
            print("Channel has flagging disabled")
            output(opts, [], flog)
            return False

    from . import processor
    from .neural import predict, diff_tags

    old = flog.get('tags', [])
    result = predict(flog, opts)
    (missing, extra, all) = diff_tags(old, result)

    chng = ''
    for (t,b,e) in all:
        if t != 0:
            chng += f'{e-b}s {"missing" if t < 0 else "extra"} at {b} to {e}; '
    print(f'{vf}: {len(result)} breaks - changed {extra-missing} seconds : {chng}')

    if chng or missing or extra or len(old) != len(all):
        processor.write_feature_log(flog, fl)
        output(opts, result, flog)
    
    return False


def _cmd_flag(opts) -> int:
    if err := _require_video(opts):
        print(err)
        return 1

    if opts.chanid:
        from .mythtv import check_method
        if not check_method(opts.chanid):
            print(f"Flagging disabled on channel {opts.chanid}")
            output(opts, [], None)
            return 0

    from .processor import process_video, read_feature_log
    from .neural import predict

    with _feature_log_path(opts) as feature_log:
        process_video(opts.filename, feature_log, opts)
        flog = read_feature_log(feature_log)
        result = predict(flog, opts, feature_log)

    output(opts, result, flog)
    return 0


@contextmanager
def _feature_log_path(opts):
    """Yield a path (never an open handle: process_video closes what it is given)."""
    import tempfile

    default_name = 'cf_' + os.path.basename(os.path.realpath(opts.filename)) + '.json'

    if opts.no_feature_log:
        import shutil
        d = tempfile.mkdtemp(prefix='pycf_')
        try:
            yield os.path.join(d, default_name)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    elif opts.feature_log:
        yield opts.feature_log
    else:
        yield os.path.join(tempfile.gettempdir(), default_name)

# --- output -------------------------------------------------------------------

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

    if vf:
        if vf.endswith('.gz'):
            vf = vf[:-3]
        if vf.endswith('.json'):
            vf = vf[:-5]

        if '.' in os.path.basename(vf):
            dot = vf.rfind('.')
        else:
            dot = -1
        ofn = vf[:dot] if dot > 0 else vf
        if not ofn:
            return
    else:
        return

    if opts.output_type in ['txt', 'text']:
        from .edl import output_txt
        output_txt(ofn + '.txt', result, feature_log or {})
    else:
        from .edl import output_edl
        output_edl(ofn + '.edl', result, feature_log)
