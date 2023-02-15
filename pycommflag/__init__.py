#!/usr/bin/env python3
"""
pycommflag : A commercial flagging utility written in Python.

This utility uses image, audio, video, and machine learning techniques to 
identify ("flag") segments of a video as being one of several categories: 
'content'(aka 'show'), 'commercial' (aka 'advertizing'), 'credits', etc.

This utility can be run directly, or can be integrated into other 
applications as a python module.
"""

from . import logo_finder, options, processor
from .player import Player
