import optparse

# todo: automatically gz/ungz the frame logs

def get_options():
    parser = optparse.OptionParser(add_help_option=True)

    parser.add_option('-f', '--file', dest="filename", type='string',
                        help="Video input file name")
    parser.add_option('-d', '--frame-data', dest="frame_log", type='string',
                        help="Output location of frame data log. If a video file is processed then this file will be overwritten.")
    parser.add_option('-r', '--reprocess', dest="reprocess_file", type='string', 
                        help="Input location of frame data log to be reprocessed.")
    parser.add_option('-q', '--quiet', dest='quiet', action='store_true',
                        help="Do not print status information during processing")
    parser.add_option('--loglevel', dest='loglevel', type='string',
                        help="Python logger's logging level")
    parser.add_option('--yaml', dest='yaml', type='string',
                        help="Pull configuration from a yaml file and overwrite commandline configuration and default with that.")
    
    logo = parser.add_option_group(title="Logo Search")
    logo.add_option('--no-logo', dest='no_logo', type='store_true', 
                    help="Disable logo searching")
    logo.add_option('--skip', dest="logo_skip", type="int", default=4,
                    help="Only search every Nth frame during the logo search phase.  (Speeds up searching at a slight cost to accuracy.)")
    
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
    (cfg,args) = opts.parse_args()

    if cfg.yaml:
        for (k,v) in parse_yaml(cfg.yaml).items():
            setattr(cfg, k, v)

    log.basicConfig(encoding='utf-8', 
                    level=log.getLevelName((cfg.loglevel or 'debug').upper()),
                    format='[%(asctime)s] %(levelname)s: %(message)s')

    if cfg.filename and cfg.reprocess:
        raise optparse.OptionConflictError("filename conflicts with reprocess")

    return cfg
