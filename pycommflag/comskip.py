"""pycommflag-comskip: a stand-in for the comskip binary.

For DVRs that run comskip themselves and offer no other hook, such as Channels DVR.
It takes comskip's command line, flags the video, and writes the .edl where comskip
would have. Like comskip on Linux, it exits 0 if it found breaks and 1 if it found none.

If flagging fails, it runs the real comskip with the same arguments instead, so the
recording still gets markers: $PYCOMMFLAG_COMSKIP_REAL if that is set, else a
`comskip.real` next to the path this was run as.

Set $PYCOMMFLAG_LOG_DIR to keep the feature logs there (for -g/-r later); otherwise
none are kept.

It never writes comskip's <name>.log: Channels DVR reads that in preference to the .edl.
"""
import argparse
import logging as log
import os
import sys

def parse_comskip_args(argv:list[str]) -> tuple[str, str, str]:
    """Returns (video, output dir, output base name) the way comskip works them out."""
    p = argparse.ArgumentParser(prog='comskip', add_help=False, allow_abbrev=False)
    # every comskip option that takes a value, so the value isn't mistaken for the input;
    # comskip's many flag-only options are left to parse_known_args to skip
    for short, opt in [('-v', '--verbose'), (None, '--ini'), (None, '--logo'), (None, '--cut'),
                       (None, '--output'), (None, '--output-filename'), (None, '--threads'),
                       (None, '--selftest')]:
        p.add_argument(*[a for a in (short, opt) if a], dest=opt[2:].replace('-', '_'))
    p.add_argument('input')
    p.add_argument('outdir', nargs='?')
    (args, _) = p.parse_known_args(argv)

    name = args.output_filename or os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.outdir or args.output or os.path.dirname(os.path.abspath(args.input))
    return (args.input, out_dir, name)

def _real_comskip() -> None|str:
    real = os.environ.get('PYCOMMFLAG_COMSKIP_REAL') or \
           os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'comskip.real')
    return real if os.path.isfile(real) and os.access(real, os.X_OK) else None

def _count_breaks(edl:str) -> int:
    with open(edl) as f:
        return sum(1 for line in f if line.split()[2:3] == ['3'])

def _flag(video:str, out_dir:str, name:str) -> int:
    from . import main, options

    os.makedirs(out_dir, exist_ok=True)
    args = ['-f', video, '-o', 'edl', '-q', '--output-dir', out_dir, '--output-name', name]
    if log_dir := os.environ.get('PYCOMMFLAG_LOG_DIR'):
        os.makedirs(log_dir, exist_ok=True)
        args += ['-l', os.path.join(log_dir, 'cf_' + os.path.basename(os.path.realpath(video)) + '.json')]
    else:
        args += ['--no-log']

    opts = options.get_options().parse_args(args)
    opts.mode = options.resolve_mode(opts)
    return main.run(opts)

def main():
    argv = sys.argv[1:]
    try:
        (video, out_dir, name) = parse_comskip_args(argv)
    except SystemExit:
        video = None # bad command line, which the real comskip can report

    if video:
        from .options import get_options, setup_logging
        setup_logging(get_options().parse_args([]))
        log.info(f'Run as comskip: {sys.argv}')

        edl = os.path.join(out_dir, name + '.edl')
        try:
            if _flag(video, out_dir, name) == 0 and os.path.isfile(edl):
                sys.exit(0 if _count_breaks(edl) else 1)
            log.error('Flagging failed')
        except Exception:
            log.exception('Flagging failed')
        if os.path.isfile(edl):
            os.unlink(edl)

    if real := _real_comskip():
        log.info(f'Running {real} instead')
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(real, [real] + argv)
    sys.exit(2)

if __name__ == '__main__':
    main()
