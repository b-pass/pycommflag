# pycommflag
A commercial flagging utility written in Python.

This utility uses image, audio, video, and machine learning techniques to
identify ("flag") segments of a video as being one of several categories:
'content'(aka 'show'), 'commercial' (aka 'advertizing'), 'credits', etc.

This utility can be run directly, or can be integrated into other
applications as a python module.

# Terminology

"Diff" - A place in the video where the image changes significantly. Usually a change in camera, or screen overlay, or switch to a different scene, etc.

"Tag" - A "flag" on a section of video indicating if it is a commercial, part of the show, or part of an intro scene.

# How to run

`./run.sh -f videofile.ts`

This analyzes the video and creates a file named "/tmp/videofile.ts.json" which is "frame log" of the video.

`./run.sh -g -l framelog.json`

This will open a GUI which displays the video along with the analysis done on it.  The intent is that a human correct tags/flags on the video for use in model training.

`./run.sh -t --data log1.json log2.json etc.json`

Train the NN on a bunch of framelogs that have been corrected up by a human. (Can take several minutes and use a lot of memory.)

`ln -s best_run.h5 model.h5`

Once you have a decent data set and have trained the NN, take those results and name/link them "model.h5".  Future runs will use this model to automatically tag a framelog.

`./run.sh -r framelog.json`

Reprocess a frame log through the NN and re-write the "tags"/"flags".

## todo: document me
Output formats for getting your tags in text or a DB
