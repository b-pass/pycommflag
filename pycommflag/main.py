from typing import Any
import os
import sys
import tempfile
from . import gui
from . import mythtv
from . import neural
from . import processor
from .segmenter import parse as parse_segmenter
from .scene import *

def run(opts:Any) -> None|int:
    if opts.rebuild:
        os.execvp("mythcommflag", ["mythcommflag", "--rebuild", "--chanid", opts.chanid, "--starttime", opts.starttime])
    if opts.queue:
        os.execvp("mythcommflag", ["mythcommflag", "--queue", "--chanid", opts.chanid, "--starttime", opts.starttime])
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # shut up, tf

    parse_segmenter(opts.segmeth) # throw for validation errors

    if opts.reprocess:
        scenes = processor.segment_scenes(opts.reprocess, opts=opts)
        return 0
    
    if opts.train:
        d = neural.load_data(opts)
        if not d:
            return 1
        neural.train(d, opts)
        return 0
    
    if not opts.filename:
        if not opts.chanid and not opts.starttime:
            import re
            if m:=re.match(r'(?:.*/)?cf_(\d{4,6})_(\d{12,})\.[a-zA-Z0-9]{2,5}\.feat', opts.feature_log):
                opts.chanid = m[1]
                opts.starttime = m[2]
        if opts.chanid and opts.starttime:
            opts.filename = mythtv.get_filename(opts.chanid, opts.starttime)
    
    if opts.dumptext:
        scenes = processor.read_scenes(opts.feature_log) if opts.feature_log else []
        got_tags = False
        for s in scenes:
            if s.type != SceneType.UNKNOWN:
                got_tags = True
                break
        if not got_tags:
            processor.external_scene_tags(scenes,opts=opts)
        
        print("\nSCENES:")
        print(Scene.header())
        for s in scenes:
            print(s)
        print(f"{len(scenes)} total.")
        sys.exit(0)
    
    if not opts.filename:
        print('No video file to work on (need one of: -f, -r, --chanid, etc)')
        return 1
    
    if opts.gui:
        logo = processor.read_logo(opts.feature_log) if opts.feature_log else None
        scenes = processor.read_scenes(opts.feature_log) if opts.feature_log else []
        if not scenes:
            scenes = processor.segment_scenes(opts.feature_log, opts=opts) if opts.feature_log else []
        audio = processor.read_audio(opts.feature_log) if opts.feature_log else []
        got_tags = False
        for s in scenes:
            if s.type != SceneType.DO_NOT_USE and s.type != SceneType.UNKNOWN:
                got_tags = True
                break
        if not got_tags:
            processor.external_scene_tags(scenes,opts=opts)
        w = gui.Window(video=opts.filename, scenes=scenes, logo=logo, audio=audio)
        res = w.run()
        if res is None:
            return 1
        processor.rewrite_scenes(res, log_f=opts.feature_log)
        return 0
    
    if opts.no_feature_log:
        feature_log = tempfile.TemporaryFile('w+b', prefix='cf_', suffix='.feat')
    elif not opts.feature_log:
        feature_log = tempfile.gettempdir() + os.path.sep + 'cf_'+os.path.basename(opts.filename)+'.feat'
    else:
        feature_log = opts.feature_log
    
    processor.process_video(opts.filename, feature_log, opts)
    scenes = processor.segment_scenes(feature_log, opts=opts)
    
    return 0
