# pycommflag
A commercial flagging utility written in Python.

This utility uses image, audio, video, and machine learning techniques to
identify ("flag") segments of a video as being one of several categories:
'content' (aka 'show'), 'commercial' (aka 'advertizing'), 'credits', or 
'intro'.

pycommflag is also a drop-in replacement for mythcommflag.  mythcommflag was revolutionary in its day, but it is now old and has seen no significant algorithmic updates in more than 15 years.  pycommflag stands on the shoulders of giants, many ideas in pycommflag are inspired by mythcommflag, and pycommflag would absolutely not exist without it!

## Features

- Graphically edit flags and save them for training ML models
- Train ML models on hand-curated data, and use the trained models to flag or re-flag
- Run modes:
    - Command line
    - From the mythtv JobQueue as a user job
    - From the mythtv JobQueue as a replacement for mythcommflag
    - Integrate into a python application as a module
- Output options:
    - Json
    - MythTV `recordedmarkup` table (can then be used commercial flags in mythfrontend or with kodi-pvr-mythtv)

# Installation

sorry, I am lazy and didn't make a setup.py yet.  

1. Clone the repo (or download a release)
2. Put it somewhere
3. Install the required python dependencies listed in `requirements.txt`
4. Put the installation in your PATH, such as by running `sudo ln -s /absolute/path/to/pycommflag/run.sh /usr/bin/pycommflag` to create a `pycommflag` command

# Running

## Flagging commercials

This repository includes an ML model trained on over 60 hand-curated recordings from a variety of channels (broadcast and cable) available in the United States in 2024.  You can download this model (`model.h5`) from github and place it in `/path/to/pycommflag/models/model.h5`.

Then simply run `pycommflag -f /path/to/video` and pycommflag will run.  Typical runtimes take about 15 minutes to flag a 1 hour recording.  But this will vary greatly depending on the hardware running pycommflag and on the codec of the recording.

The output will be a json file in `/tmp` which is called a "Frame log" and includes the recordings in a key called "tags".  If you have a `~/.mythtv/config.xml` file and you run pycommflag on a recording file from mythtv, it will write its results directly into the mythtv database.

### Setting up mythtv

Because pycommflag is still experimental, we recommend you try pycommflag out on selected recordings.  First on the commandline, then via a user job on selected recording schedules, and finally as a replacement for mythcommflag.

### As a user-job

You can use `mythtv-setup` on your backend to add a user job for `pycommflag`.  The recommended commandline for pycommflag as a userjob is: `pycommflag --no-log -j %JOBID%`.

After you've added the job and restarted your mythtvbackend, you will see the job as an option on your recordings and schedules.

### As a mythcommflag replacement

You can change the "JobQueueCommFlagCommand" setting via mythweb or in the settings table on your backend.  (TBD, can you change this via mythtv-setup or the UI??)

You should include the `-e` flag on your command.  Our recommended value for this setting is: `pycommflag -e --no-log -j %JOBID%`.

Once you change this any schedule you have set to run commercial flagging will run pycommflag instead of mythcommflag.  pycommflag will be run as the mythtv user (so keep that in mind during your installation).

# Training your own pycommflag model

## Creating training data

If pycommflag gets a commercial wrong, the best way to fix it is to hand-curate a set of training data to train your own pycommflag model from your recordings.

When you run pycommflag *without* the `--no-logs` command, it will generate a "feature log" in `/tmp` which contains everything about the recording that pycommflag discovered, and also the result of the pycommflag flagging.  You hand-curate a training set by running `pycommflag -g /tmp/something-to-curate.json`.  This will open a tk GUI that will show you the video, features pycommflag found, and options for changing the flagging that pycommflag has figured out.

(TODO: screen shots? button-by-button walk through?)

When you are done, press "Save and Exit" and pycommflag will update the json file with your changes.  You should then save off the json file to use in training.

## Training

To train your own model, simply run `pycommflag -t --data /path/to/datafiles/*json`.  pycommflag will train a new model on your data.  Keep an eye on "val_categorical_accuracy", this is the most important measure of the success of the training process.  Anything less than 0.98 (98%) is a bad result. Also "val_categorical_accuracy" should be higher than "categorical_accuracy", if it is not then you do not have enough training data.

After training is complete, pycommflag will save the results in `/path/to/pycommflag/models/` as `pycf.*stuff*.h5`.  The "stuff" includes the val_categorical_accuracy so you can see which training runs have produced useful output models.  When you have a model that you want to use for `pycommflag` flagging, you can either specify it on the commandline or move/link it to `model.h5` in the directory.

## Reprocessing

The models the pycommflag uses are run based on "features" extracted from the video file and not the file itself.  The features are saved in the "feature log" files.  If you keep these files, then you can quickly (less than 60 seconds) re-run a new model against something that pycommflag has previously run against, without the lengthy (15+ minute) process of re-processing the video.

Simply run `pycommflag -r /path/to/feature_log.json`.  If you don't keep your feature logs around, then you will have to reprocess the full video (with `-f`).

### A note about advanced options

Note that the data available for flagging and training has to be present both in the training data and in the recordings you want to use.  So if you change options like logo or audio detection, it might make the model not work well with your data.  You would need to retrain it, probably you would need to curate new data processed with the same settings.  The models available on github have been trained using the `pycommflag` defaults, which are suitable for cable and HD broadcast TV in the US, but maybe not in other locations.

# How does it work

It extracts "features" from the video.

Features:
- Rate of "difference"/change between frames of the video
- Presence of a station-identification logo in any of the corners of the screen (detected using the Sobel edge detection algorithm)
- Two different measures of time/position in the video
- Using the INAFOSS "speech/music/noise" audio detection algorithm and library
- RMS audio peek (rolling 0.5s window) of the "main" and "surround" channels
- And also the presence of all-black frames

Each of these features is available for every frame of the video, and each frame and those around it are fed to a small recurrent neural network (LSTM).

Frame logs save these in a convienient json format, which is (usually) much less than 1% the size of the original video file.  That makes these files easy to keep around for training and experimentation.

# To do / ideas

- Setup.  It would be good to get proper installation working, and then this could be put on pypi.

- Per-channel configuration.  Some channels might be more useful with different thresholds, or different settings.  So far the models seem to generalize pretty well with enough data.

- I suppose it might be possible to do per-channel models?  I don't know how that would work.

- New video/audio features.  Would be cool, might really help flagging?  I am running out of ideas.  Also usually these mean starting over on making new training data, which is really time consuming, so I am not super motivated.
