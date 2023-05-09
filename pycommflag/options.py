import argparse

# todo: automatically gz/ungz the frame logs

def get_options():
    parser = argparse.ArgumentParser(add_help=True,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--file', dest="filename", type=str,
                        help="Video input file name")
    parser.add_argument('-l', '--feature-log', dest="feature_log", type=str,
                        help="In/Out location of feature log. If a video file is processed then this file will be overwritten. cf_[filename].feat")
    parser.add_argument('--no-log', dest="no_feature_log", action='store_true',
                        help="Use a temporary file for the feature log (which won't persist anywhere)")
    parser.add_argument('-c','--break-text', dest='comm_file', type=str,
                        help="Output location for commercial break results")
    parser.add_argument('-r', '--reprocess', dest="reprocess", type=str, 
                        help="Input location of feature data log to be reprocessed.")
    parser.add_argument('-g', '--gui', dest='gui', action='store_true', 
                        help="(Re)Process and then display for GUI editing.  Must also supply --file and --feature-log for the thing being edited in the GUI.")
    parser.add_argument('--dump-text', dest='dumptext', action='store_true',
                        help="(Re)Process and Dump scene text and exit.  Must supply a --feature-log")
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true',
                        help="Do not print progress during processing")
    parser.add_argument('--loglevel', dest='loglevel', type=str,
                        help="Python logger's logging level")
    parser.add_argument('--yaml', dest='yaml', type=str,
                        help="Pull configuration from a yaml file and overwrite commandline configuration and default with that.")
    parser.add_argument('--deinterlace', dest='no_deinterlace', action='store_false',
                        help="Use a deinterlacing filter during video processing")
    parser.add_argument('--scene-threshold', dest='scene_threshold', type=float, default=15.0,
                        help="Difference threshold (column/bar) for marking a scene change [8-30 is sane].")
    
    logo = parser.add_argument_group('Logo Search')
    logo.add_argument('--no-logo', dest='no_logo', action='store_true', 
                    help="Disable logo searching")
    logo.add_argument('--skip', dest="logo_skip", type=int, default=4,
                    help="Only search every Nth frame during the logo search phase.  (Speeds up searching at a slight cost to accuracy.)")
    logo.add_argument('--logo-search-all', dest="logo_search_all", action='store_true',
                    help="Search the entire video for the logo (perfect detection)")
    #logo.add_argument('--check-blanks', dest="blanks_check_logo", action="store_true",
    #                help="Include logo area when checking for blank frames (tends towards not marking frames blank if they have logo)")
    parser.add_argument_group(logo)
    
    seg = parser.add_argument_group('Scene Segmentation')
    seg.add_argument('--segment', dest='segmeth', type=str, default='blank|audio',
                   help="Scene segmentation instruction; split video into scenes using the demuxers.\nPlus to AND them, comma or pipe to OR them.\nSegmenters: logo,silence,audio,blank,imagediff,1s")
    parser.add_argument_group(seg)

    ml = parser.add_argument_group('Machine Learning')
    ml.add_argument('--train', dest="train", action='store_true', 
                  help="Train the ML model")
    ml.add_argument('--data', dest="ml_data", nargs='+',
                  help="Data to train the model with, as a list of feature-log files")
    ml.add_argument('--batch-size', dest='tf_batch_size', type=int, default=1000,
                  help="Model training batch size")
    parser.add_argument_group(ml)
    
    mcf = parser.add_argument_group('MythTV Options', description="Commandline compatibility with mythcommflag")
    mcf.add_argument('--chanid', dest='chanid', type=int, 
                    help="Channel ID of recording, filename will be fetched from the mythtv database")
    mcf.add_argument('--starttime', dest="starttime",
                    help="Start timestamp of recording, filename will be fetched from the mythtv database")
    mcf.add_argument('--queue', dest="queue", action="store_true",
                    help="Insert a job into the mythtv job queue")
    mcf.add_argument('--rebuild', dest="rebuild", action="store_true",
                    help="Rebuild seek table (this will just exec mythcommflag directly to do this)")
    #mcf.add_argument('--mythtv-out', dest="mythtv_output", action="store_true",
    #                help="Write flagging output to mythtv database even though --chanid and --starttime were not specified")
    #mcf.add_argument('--no-mythtv-out', dest="no_mythtv_output", action="store_true",
    #                help="Do NOT write flagging output to mythtv database even though --chanid and --starttime were specified")
    parser.add_argument_group(mcf)

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
    cfg = opts.parse_args()

    if cfg.yaml:
        for (k,v) in parse_yaml(cfg.yaml).items():
            setattr(cfg, k, v)

    log.basicConfig(encoding='utf-8', 
                    level=log.getLevelName((cfg.loglevel or 'debug').upper()),
                    format='[%(asctime)s] %(levelname)s: %(message)s')

    return cfg
