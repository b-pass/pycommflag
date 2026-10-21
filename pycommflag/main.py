import os
import sys
from contextlib import contextmanager

from .options import Mode


def run(opts) -> int:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # shut up, tf

    if opts.mode is Mode.PASSTHROUGH:
        os.execvp("mythcommflag", ["mythcommflag"] + sys.argv[1:])

    if opts.exitcode:
        # mythtv reads the exit status as the number of breaks, so errors must exit 255: python's
        # exit(1) would be "1 break", 256 wraps to 0, and myth_system(kMSLowExitVal) mangles 128-254
        def myexcepthook(type, value, tb):
            sys.__excepthook__(type, value, tb)
            sys.exit(255)
        sys.excepthook = myexcepthook

    if err := _check_mythtv_support(opts):
        print(err)
        return 255 if opts.exitcode else 1

    if opts.mythjob and opts.mode is Mode.FLAG:
        from .mythtv import is_rebuild_job
        if is_rebuild_job(opts.mythjob):
            # we're the JobQueueCommFlagCommand, but this job wants a seek-table rebuild, so the
            # real mythcommflag handles the whole job (including its status and exit code)
            os.execvp("mythcommflag", ["mythcommflag", "-j", str(opts.mythjob), "--noprogress"])

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

def _check_mythtv_support(opts) -> None|str:
    """MythTV support is optional, but asking for it explicitly without mysqlclient is an error."""
    explicit = opts.mythjob or opts.chanid or opts.starttime or opts.exitcode or opts.output_type in ('mythtv', 'myth')
    if not explicit:
        return None
    try:
        import MySQLdb
    except ImportError:
        return ('MythTV options need mysqlclient: run "pipx inject pycommflag mysqlclient" '
                'or "pip install \'pycommflag[mythtv]\'"')
    return None

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
            opts.filename = _find_video(read_feature_log(opts.feature_log).get('filename', ''))

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
    return 1 if output(opts, res, flog) is False else 0


def _cmd_reprocess(opts) -> int:
    import gc
    import traceback

    i = 0
    archived = 0
    failed = 0
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
            failed += 1
            print('EXCEPTION')
            print(traceback.format_exc())
            print()

    return 1 if failed else 0


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

    if vf and (found := _find_video(vf)) != vf:
        print(f'Using {found} ({vf} does not exist)')
        vf = found

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
            if output(opts, [], flog) is False:
                raise RuntimeError(f'Output failed for {vf}')
            return False

    from . import processor
    from .neural import predict, diff_tags

    old = flog.get('tags', [])
    result = predict(flog, opts)
    (missing, extra, all) = diff_tags(result, old)

    chng = ''
    for (t,b,e) in all:
        if t != 0:
            chng += f'{e-b}s {"removed" if t < 0 else "added"} at {b} to {e}; '
    print(f'{vf}: {len(result)} breaks - changed {extra-missing} seconds : {chng}')

    if chng or missing or extra or len(old) != len(all):
        processor.write_feature_log(flog, fl)
        if output(opts, result, flog) is False:
            raise RuntimeError(f'Output failed for {vf}')
    
    return False


def _find_video(vf:str) -> str:
    """vf, or the .mkv it was transcoded to if vf is a .ts that has since been deleted."""
    if vf and vf.endswith('.ts') and not os.path.exists(vf) and os.path.isfile(vf[:-3] + '.mkv'):
        return vf[:-3] + '.mkv'
    return vf


def _video_rate(vf:str) -> None|float:
    """The frame rate the way Player (and so every feature log) computes it."""
    import av
    try:
        with av.open(vf) as c:
            rate = c.streams.video[0].guessed_rate
            return round(float(rate), 3) if rate else None
    except Exception:
        return None


def _cmd_flag(opts) -> int:
    if err := _require_video(opts):
        print(err)
        return 1

    if opts.chanid:
        from .mythtv import check_method
        if not check_method(opts.chanid):
            print(f"Flagging disabled on channel {opts.chanid}")
            return 1 if output(opts, [], None) is False else 0

    from .processor import process_video, read_feature_log
    from .neural import predict
    from . import live

    rec = None
    if live.is_live(opts):
        rec = live.Recording(opts.filename, live.recording_check(opts))

    with _feature_log_path(opts) as feature_log:
        process_video(opts.filename, feature_log, opts, live=rec, on_snapshot=_interim_output(opts))
        if rec is not None and not opts.no_logo:
            _recheck_logo(opts, feature_log)
        flog = read_feature_log(feature_log)
        result = predict(flog, opts, feature_log)

    return 1 if output(opts, result, flog) is False else 0


def _interim_output(opts):
    """The on_snapshot callback for a live run: predict on what we have so far and output
    the breaks whenever they change."""
    import logging as log
    import time
    from .mythtv import set_job_status
    from .neural import predict, WINDOW_AFTER

    last = None
    def on_snapshot(flog):
        nonlocal last
        try:
            # the last WINDOW_AFTER seconds' windows run off the end into padding
            cutoff = flog['duration'] - WINDOW_AFTER
            result = [(t, (b, min(e, cutoff))) for (t, (b, e)) in predict(flog, opts) if b < cutoff]
            if result != last:
                last = result
                output(opts, result, flog, final=False)
            set_job_status(opts, f'Live: flagged to {time.strftime("%H:%M:%S", time.gmtime(cutoff))}, '
                                 f'{len(result)} break{"s" if len(result) != 1 else ""} so far')
        except Exception:
            # the final pass is what counts, so carry on extracting
            log.exception('Interim prediction failed')
    return on_snapshot


def _recheck_logo(opts, feature_log) -> bool:
    """A live run found its logo in the head start.  Search the whole recording now, and if
    that finds a different logo, extract again with it.  True if it did."""
    import logging as log
    from . import logo_finder
    from .mythtv import set_job_status
    from .player import Player
    from .processor import process_video, read_logo

    set_job_status(opts, 'Checking the logo against the whole recording')
    logo = logo_finder.search(Player(opts.filename, no_deinterlace=opts.no_deinterlace), opts=opts)
    old = read_logo(feature_log)
    if logo_finder.same_logo(logo, old):
        return False
    log.info(f'The whole recording has a different logo ({logo[:2] if logo else None}) than its start '
             f'({old[:2] if old else None}), extracting again')
    process_video(opts.filename, feature_log, opts, use_logo=logo)
    return True


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

def output(opts, result, feature_log=None, final=True) -> None|bool:
    """Returns False if the output could not be written (only -o channelsdvr reports that).

    final=False is an interim result from a live run, which leaves the MythTV job running
    and isn't sent to Channels.
    """
    if result is None:
        return

    if feature_log and (vf := _find_video(feature_log.get('filename', ''))):
        # the outputs that count frames (mythtv, txt) need the video's own rate, which isn't the
        # log's after a transcode changed it; the log itself keeps the rate its features were at
        feature_log = dict(feature_log, filename=vf)
        if rate := _video_rate(vf):
            feature_log['frame_rate'] = rate

    if opts.output_type in ['mythtv', 'myth', 'auto', '', None]:
        from .mythtv import set_breaks
        if set_breaks(opts, result, feature_log, final=final):
            return
        if opts.output_type in ['mythtv', 'myth']:
            return # was not auto, so dont try edl or txt

    if opts.output_type in ['channelsdvr', 'channels']:
        if not final:
            return
        from .channels import set_breaks
        return set_breaks(opts, result, (feature_log or {}).get('filename') or opts.filename)

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

    out_dir, out_name = getattr(opts, 'output_dir', None), getattr(opts, 'output_name', None)
    if out_dir or out_name:
        ofn = os.path.join(out_dir or os.path.dirname(ofn), out_name or os.path.basename(ofn))

    if opts.output_type in ['txt', 'text']:
        from .edl import output_txt
        output_txt(ofn + '.txt', result, feature_log or {})
    else:
        from .edl import output_edl
        output_edl(ofn + '.edl', result, feature_log)
