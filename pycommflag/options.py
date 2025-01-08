import argparse
import os

# todo: automatically gz/ungz the feaature logs

def get_options():
    parser = argparse.ArgumentParser(add_help=True,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--file', dest="filename", type=str,
                        help="Video input file name")
    parser.add_argument('-l', '--feature-log', dest="feature_log", type=str,
                        help="In/Out location of feature log. If a video file is processed then this file will be overwritten. cf_[filename].json")
    parser.add_argument('--no-log', dest="no_feature_log", action='store_true',
                        help="Use a temporary file for the feature log (which won't persist anywhere)")
    parser.add_argument('-o','--output-type', dest='output_type', type=str, default="auto",
                        help="Output type [auto, mythtv, edl, txt] (auto will try mythtv and fall back to edl). EDL/TXT files are saved alongside the input video file")
    parser.add_argument('-r', '--reprocess', dest="reprocess", type=str, nargs='+',
                        help="Input location of feature data log to be reprocessed.")
    parser.add_argument('-g', '--gui', dest='gui', type=str, 
                        help="Display this feature log for GUI editing.")
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true',
                        help="Do not print progress during processing")
    parser.add_argument('--loglevel', dest='loglevel', type=str,
                        help="Python logger's logging level")
    parser.add_argument('--yaml', dest='yaml', type=str,
                        help="Pull configuration from a yaml file and overwrite commandline configuration and default with that.")
    parser.add_argument('--deinterlace', dest='no_deinterlace', action='store_false',
                        help="Use a deinterlacing filter during video processing")
    parser.add_argument('--diff-threshold', dest='diff_threshold', type=float, default=15.0,
                        help="Threshold (column/bar) for marking an image difference/change [8-30 is sane].")
    
    logo = parser.add_argument_group('Logo Search')
    logo.add_argument('--no-logo', dest='no_logo', action='store_true', 
                    help="Disable logo searching")
    logo.add_argument('--logo-samples', dest="logo_samples", type=int, default=2,
                    help="How many video frames per-second to use across the full length of video to find the logo")
    #logo.add_argument('--check-blanks', dest="blanks_check_logo", action="store_true",
    #                help="Include logo area when checking for blank frames (tends towards not marking frames blank if they have logo)")
    parser.add_argument_group(logo)

    ml = parser.add_argument_group('Machine Learning')
    ml.add_argument('-t', '--train', dest="train", action='store_true', 
                  help="Train the ML model")
    ml.add_argument('--data', dest="ml_data", nargs='+',
                  help="Data to train/eval with, as a list of feature-log files")
    ml.add_argument('--batch-size', dest='tf_batch_size', type=int, default=1000,
                  help="Model training batch size")
    ml.add_argument('--models', dest='models_dir', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models'),
                  help="Path to use for models (output for training, input for infrencing)")
    ml.add_argument('--model', dest='model_file', default='',
                  help="Path to model to use for inference/prediction")
    ml.add_argument('--eval', dest="eval", nargs='+',
                  help="List of model files to evaluate against the supplied data")
    parser.add_argument_group(ml)
    
    mcf = parser.add_argument_group('MythTV Options', description="Commandline compatibility with mythcommflag")
    mcf.add_argument('--chanid', dest='chanid', type=int, 
                    help="Channel ID of recording, filename will be fetched from the mythtv database")
    mcf.add_argument('--starttime', dest="starttime",
                    help="Start timestamp of recording, filename will be fetched from the mythtv database")
    mcf.add_argument('-j','--jobid', dest="mythjob", type=int,
                    help="Update status information in the mythtv job queue")
    mcf.add_argument('-e', '--exitcode', dest='exitcode', action='store_true',
                    help='Use non-standard exit status compatible with running instead of mythcommflag in the jobqueue (exit number of commercial breaks)')
    mcf.add_argument('--noprogress', dest='quiet', action='store_true',
                    help="Quiet output (compatibility flag)")
    mcf.add_argument('--rebuild', dest="rebuild", action="store_true",
                    help="Rebuild recorded seek table (compatibility flag, this will just exec mythcommflag directly to do this)")
    mcf.add_argument('--queue', dest="queue", action="store_true",
                    help="Insert a job into the mythtv job queue (compatibility flag, this will just exec mythcommflag directly to do this)")
    # progress? logs?
    #mcf.add_argument('--mythtv-out', dest="mythtv_output", action="store_true",
    #                help="Write flagging output to mythtv database even though --chanid and --starttime were not specified")
    #mcf.add_argument('--no-mythtv-out', dest="no_mythtv_output", action="store_true",
    #                help="Do NOT write flagging output to mythtv database even though --chanid and --starttime were specified")
    parser.add_argument_group(mcf)

    tune = parser.add_argument_group('Fine Tuning')
    tune.add_argument('--break-max-len', dest="break_max_len", type=int, default=335,
                      help="Longest allowed commercial break (in seconds)")
    tune.add_argument('--break-min-len', dest="break_min_len", type=float, default=59,
                      help="Shortest allowed commercial break (in seconds)")
    tune.add_argument('--show-min-len', dest="show_min_len", type=int, default=59,
                      help="Shortest allowed show segment (in seconds)")
    parser.add_argument_group(tune)

    return parser

def parse_yaml(filename:str):
    class Options(dict):
        def __init__(self, *args, **kwargs):
            super(Options, self).__init__(*args, **kwargs)
            self.__dict__ = self
    
    import yaml
    return Options(yaml.safe_load(open(filename, 'r')))

def parse_argv():
    import sys
    import logging as log
    opts = get_options()
    if len(sys.argv) <= 1:
        opts.print_help()
        sys.exit(1)
    cfg = opts.parse_args()

    if cfg.yaml:
        for (k,v) in parse_yaml(cfg.yaml).items():
            setattr(cfg, k, v)

    log.basicConfig(encoding='utf-8', 
                    level=log.getLevelName((cfg.loglevel or 'debug').upper()),
                    format='[%(asctime)s] %(levelname)s: %(message)s')

    return cfg
