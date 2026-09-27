#!/usr/bin/env python3
"""
pycommflag : A commercial flagging utility written in Python.

This utility uses image, audio, video, and machine learning techniques to 
identify ("flag") segments of a video as being one of several categories: 
'content'(aka 'show'), 'commercial' (aka 'advertizing'), 'credits', etc.
"""

# submodules are imported on demand (see main.py) so that e.g. the tk GUI and
# tensorflow are only loaded by the modes that need them

import logging
logging.getLogger('h5py').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
