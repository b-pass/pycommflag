import optparse

# todo: automatically gz/ungz the frame logs

def get_options():
    parser = optparse.OptionParser(add_help_option=True)

    parser.add_option('-f', '--file', dest="filename", type='string',
                        help="Video input file name")
    parser.add_option('-l', '--feature-log', dest="feature_log", type='string',
                        help="In/Out location of feature log. If a video file is processed then this file will be overwritten. cf_[filename].feat")
    parser.add_option('--no-log', dest="no_feature_log", action='store_true',
                        help="Use a temporary file for the feature log (which won't persist anywhere)")
    parser.add_option('-c','--break-text', dest='comm_file', type='string',
                        help="Output location for commercial break results")
    parser.add_option('-r', '--reprocess', dest="reprocess", type='string', 
                        help="Input location of feature data log to be reprocessed.")
    parser.add_option('-g', '--gui', dest='gui', action='store_true', 
                        help="(Re)Process and then display for GUI editing.  Must also supply --file and --feature-log for the thing being edited in the GUI.")
    parser.add_option('-q', '--quiet', dest='quiet', action='store_true',
                        help="Do not print progress during processing")
    parser.add_option('--loglevel', dest='loglevel', type='string',
                        help="Python logger's logging level")
    parser.add_option('--yaml', dest='yaml', type='string',
                        help="Pull configuration from a yaml file and overwrite commandline configuration and default with that.")
    parser.add_option('--no-deinterlace', dest='no_deinterlace', action='store_true',
                        help="Never use a deinterlacing filter")
    parser.add_option('--scene-threshold', dest='scene_threshold', type='float',
                        help="Difference threshold (column/bar) for marking a scene change [8-30 is sane].")
    
    logo = optparse.OptionGroup(parser, 'Logo Search')
    logo.add_option('--no-logo', dest='no_logo', action='store_true', 
                    help="Disable logo searching")
    logo.add_option('--skip', dest="logo_skip", type="int", default=4,
                    help="Only search every Nth frame during the logo search phase.  (Speeds up searching at a slight cost to accuracy.)")
    logo.add_option('--logo-search-all', dest="logo_search_all", action='store_true',
                    help="Search the entire video for the logo (perfect detection)")
    #logo.add_option('--check-blanks', dest="blanks_check_logo", action="store_true",
    #                help="Include logo area when checking for blank frames (tends towards not marking frames blank if they have logo)")
    parser.add_option_group(logo)
    
    mcf = optparse.OptionGroup(parser, 'MythTV Options', description="Commandline compatibility with mythcommflag")
    mcf.add_option('--chanid', dest='chanid', type='int', 
                    help="Channel ID of recording, filename will be fetched from the mythtv database")
    mcf.add_option('--starttime', dest="starttime",
                    help="Start timestamp of recording, filename will be fetched from the mythtv database")
    mcf.add_option('--queue', dest="queue", action="store_true",
                    help="Insert a job into the mythtv job queue")
    mcf.add_option('--rebuild', dest="rebuild", action="store_true",
                    help="Rebuild seek table (execs mythcommflag directly to do this)")
    mcf.add_option('--mythtv-out', dest="mythtv_output", action="store_true",
                    help="Write flagging output to mythtv database even though --chanid and --starttime were not specified")
    mcf.add_option('--no-mythtv-out', dest="no_mythtv_output", action="store_true",
                    help="Do NOT write flagging output to mythtv database even though --chanid and --starttime were specified")
    parser.add_option_group(mcf)

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

    return cfg
