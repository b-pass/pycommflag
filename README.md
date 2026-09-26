# pycommflag

pycommflag finds the commercials in TV recordings. It extracts video and audio
features from a recording, runs them through a small neural network, and
outputs the commercial breaks. You can send those breaks to MythTV, or write
them as an EDL or comskip-style `.txt` file.

It is a drop-in replacement for MythTV's `mythcommflag`: it accepts the same
job-queue arguments and writes breaks into the same database table, so
mythfrontend, MythWeb, and kodi-pvr-mythtv can use them as they are. It can also
stand in for [comskip](https://github.com/erikkaashoek/Comskip) wherever you
currently use comskip's `.edl` or `.txt` output.

pycommflag builds on many ideas from mythcommflag and would not exist without it.
mythcommflag was revolutionary in its day, but its algorithms have not changed
significantly in more than 15 years.

## How it differs from mythcommflag and comskip

| | mythcommflag / comskip | pycommflag |
|---|---|---|
| Detection | Hand-tuned heuristics (blank frames, logo, scene changes, aspect ratio, …) | A neural network that learns from those same kinds of signals |
| Tuning | Many knobs (`comskip.ini`, detection method bitmasks) | A few limits on break length. To improve accuracy, you correct its mistakes and retrain |
| Re-running | Decodes the whole video again | Saves a small "feature log" and can re-flag from it in seconds |
| Speed | Fast | Slower (see [Speed](#speed)) |

pycommflag does not read `comskip.ini` or MythTV's per-channel detection method
bitmask. The only per-channel MythTV setting it honors is "commercial flagging
disabled".

## Installation

There is no `setup.py` or PyPI package yet.

1. Clone the repository (or download a release):
   ```sh
   git clone https://github.com/b-pass/pycommflag.git /opt/pycommflag
   ```
2. Install Python 3.10 or newer. You also need the build headers for
   `mysqlclient` (e.g. `libmysqlclient-dev` or `default-libmysqlclient-dev`, plus
   `pkg-config`). The editing GUI needs Tk (e.g. `python3-tk`). `libjemalloc2`
   is optional, and `run.sh` preloads it if present.
3. Install the Python dependencies into a virtualenv named `venv` inside the
   checkout. `run.sh` activates it automatically:
   ```sh
   cd /opt/pycommflag
   python3 -m venv venv
   ./venv/bin/pip install -r requirements.txt
   ```
4. Put a model at `/opt/pycommflag/models/model.keras` (see [Models](#models)).
5. Optionally, add a `pycommflag` command to your `PATH`:
   ```sh
   sudo ln -s /opt/pycommflag/run.sh /usr/local/bin/pycommflag
   ```

The first run downloads the inaSpeechSegmenter audio model into
`~/.keras/datasets/inaSpeechSegmenter`, so it needs network access once. Note
that this happens separately for each user: if MythTV runs pycommflag as the
`mythtv` user, that user needs network access the first time too.

### Models

pycommflag cannot flag anything without a trained model. Models are not stored
in git. It looks for one in this order:

1. `--model /path/to/file.keras`
2. `models/model.keras` in the checkout (use `--models DIR` to change the directory)
3. `models/model.h5`

A pre-trained model is published on the
[releases page](https://github.com/b-pass/pycommflag/releases). A model only
works with the version of pycommflag whose features it was trained on. If you
update pycommflag and the model's input shape no longer matches, get a newer
model or [train your own](#training-your-own-model).

## Flagging a recording

```sh
pycommflag -f /path/to/recording.ts
```

What happens to the results depends on `-o` / `--output-type`:

| `-o` | Result |
|---|---|
| `auto` (default) | If the recording is in MythTV, the breaks are written to the database. Otherwise, an `.edl` file is written. |
| `mythtv` | Write the breaks only to the MythTV database. |
| `edl` | Write `recording.edl` next to the video: one `start<TAB>end<TAB>type` line per segment, in seconds, with `3` (commercial break) for commercials. Kodi reads this format. |
| `txt` | Write `recording.txt` next to the video in comskip's format (`FILE PROCESSING COMPLETE N FRAMES AT R`, then `start end` frame numbers for each break). |

pycommflag also keeps a **feature log** at `$TMPDIR/cf_<video filename>.json`
(normally in `/tmp`). This file holds everything pycommflag extracted from the
video, plus the breaks it found (under the `"tags"` key). It is usually well
under 1% of the video's size. Keep it if you might want to re-flag the
recording later or use it for training. Use `-l FILE` to choose its location,
or `--no-log` to skip keeping it. If you flag the same video again with an
existing log, pycommflag reuses the station logo saved in that log.

Other useful options (`pycommflag --help` lists them all):

- `--break-min-len` / `--break-max-len` / `--show-min-len`: the shortest
  allowed break (default 59 s), the longest allowed break (default 335 s), and
  the shortest allowed show segment between breaks (default 59 s).
- `--no-logo`: skip the station logo search.
- `--deinterlace`: turn on deinterlacing (off by default).
- `-q` / `--noprogress`: don't print progress.
- `--yaml FILE`: load options from a YAML file. Keys are the option names as
  pycommflag stores them internally (e.g. `break_max_len: 300`), and they
  override the command line.

### Speed

A 1-hour recording typically takes about 10 minutes, but this varies a lot with
your hardware and the recording's codec. Extraction lowers its own CPU priority
(`nice 10`) so it can run in the background.

Re-flagging from a saved feature log skips the video decode and takes seconds:

```sh
pycommflag -r /tmp/cf_*.json
```

With several logs, `-r` rewrites a log and its output only when the breaks
changed. `-r` is how you apply a new model to old recordings.

## Using it with MythTV

pycommflag reads the database credentials from `~/.mythtv/config.xml` of the
user that runs it. If that file doesn't exist, all MythTV features are quietly
skipped. For MythTV recordings named in the usual `<chanid>_<starttime>.ts`
format, pycommflag works out the channel and start time from the filename.
You can also pass `--chanid`/`--starttime` or `-j JOBID`, just like with
mythcommflag.

We recommend trying pycommflag in three stages: from the command line first,
then as a user job on a few schedules, and finally as a full replacement for
mythcommflag.

Whenever MythTV runs pycommflag, it runs as the `mythtv` user. That user needs
read access to the checkout and its `venv` and `models`, and it needs its own
`~/.mythtv/config.xml`.

### Command line

```sh
pycommflag -f /var/lib/mythtv/recordings/1051_20240101200000.ts
```

This writes the breaks straight into `recordedmarkup`, and the next time you
play the recording, mythfrontend uses them.

### Where the settings are

Both setups below use the backend's job queue settings. The same settings
appear in two places:

- **Web app (recommended):** open `http://<your-backend>:6544`, choose
  **Backend Setup** in the side menu, go to the **General** step, and expand
  the section named below.
- **`mythtv-setup` (deprecated, but still works):** go to **General** and page
  through to the section of the same name.

Restart `mythbackend` after changing them.

### As a user job

1. In **Job Queue (Job Commands)**, fill in an unused User Job's description
   (e.g. `pycommflag`) and set its command to:
   ```
   pycommflag --no-log -j %JOBID%
   ```
2. In **Job Queue (Backend-Specific)**, tick the checkbox that allows that user
   job to run on this backend. In the web app, the checkbox is labeled with the
   job's description; in `mythtv-setup`, it reads "Allow User Job #N jobs".

The job now appears in the recording menu and in the schedule options.

### As a mythcommflag replacement

In **Job Queue (Global)**, set **Commercial Detection Command** (the
`JobQueueCommFlagCommand` setting) to:

```
pycommflag -e --no-log -j %JOBID%
```

From then on, every schedule that has commercial flagging turned on runs
pycommflag.

This setting holds the whole command line, not just the program name. Unless
it is exactly `mythcommflag`, MythTV runs it as written: it replaces `%JOBID%`
(and the other `%…%` variables that user jobs support) but adds no arguments
of its own. So you must include `-j %JOBID%` yourself, and use a full path if
`pycommflag` isn't on the backend's `PATH`.

`-e` matters here. MythTV reads the command's exit status as the number of
breaks found and treats 256 or higher as a failure. `-e` makes pycommflag exit
with its break count, and makes crashes exit with 256, so MythTV doesn't read a
crash as "1 break".

For command-line compatibility, pycommflag also accepts mythcommflag's
`--rebuild` (rebuild the seek table) and `--queue` options. It doesn't
implement them: it just runs the real `mythcommflag` from the `PATH` with the
same arguments. For that reason, don't replace the `mythcommflag` binary itself
with pycommflag. Use the setting above instead.

## Coming from comskip

- Use `-o txt` to get comskip's `.txt` format, or `-o edl` to get an `.edl` for
  Kodi and other players that understand EDL action `3` (commercial break).
  Both files are written next to the video with the extension replaced, like
  comskip's output.
- There is no `.ini` file. Instead of tuning detection, you correct the
  recordings pycommflag gets wrong in the GUI and retrain (see below). Options
  can go in a YAML file (`--yaml`) if you'd rather not put them on the command
  line.
- Post-processing scripts that run comskip can call
  `pycommflag --no-log -o edl -f "$FILE"` instead.

## Training your own model

If pycommflag keeps getting a channel or show wrong, the fix is more training
data, not more knobs.

### Curating recordings

Flag a recording without `--no-log` so its feature log is kept, then open the
log in the editor:

```sh
pycommflag -g /tmp/cf_recording.ts.json
```

The GUI needs the original video (it uses the path saved in the log) and a
model. It shows the video, a timeline of the extracted features, and the
current flags. You can step through the recording by frame, second, or
blank/diff/audio change, and jump between breaks. Use the `Flag Break` /
`Flag Show` / `Flag Intro` / `Flag Credits` / `Flag Ignore` buttons to mark
segments. `Ignore` excludes a segment from training. When you're done, click
`Save & Exit`. This saves your flags into the feature log and also writes them
to MythTV or an EDL, just like a normal run. Then copy the log somewhere
permanent: `/tmp` is not a good place to keep training data.

### Training

```sh
pycommflag -t --data /path/to/training/*.json TEST /path/to/validation/*.json
```

Logs listed after the literal word `TEST` are held out for validation. You need
some, because validation accuracy decides when training stops and which epoch
is kept. Watch `val_weighted_accuracy`: if it is much lower than the training
accuracy, you need more (or more varied) training data.

If the best epoch reaches at least 0.95 validation accuracy, it is saved to
`models/pycf-<val_acc>-….keras`. The filename starts with the accuracy, so you
can easily compare runs. To use a model, pass it with `--model` or link it as
`models/model.keras`. Then re-flag your saved logs with `-r`.

To compare models on the same data (this needs `scikit-learn`):

```sh
pycommflag --eval models/model.keras models/pycf-0.97-….keras --data /path/to/curated/*.json
```

The first time training uses a feature log, it caches preprocessed data next to
the log as `*.data.npy` and `*.nologo.data.npy`. **Delete these after you
re-curate a log in the GUI**, or training will keep using the old flags.

Training runs at the lowest CPU priority (`nice 19`).

### A note about feature extraction options

A model can only use the features it was trained with. If you flag with
non-default extraction options (e.g. `--no-logo`, `--diff-threshold`,
`--deinterlace`), the features no longer match what the published model
expects, and accuracy may drop. In that case, train on logs extracted with the
same options. The published model uses the defaults, which suit US cable and HD
broadcast TV. It may do less well elsewhere.

## How it works

For every video frame, pycommflag records:

- whether a station logo is present (found by sampling the whole recording for
  static Sobel edges in the screen corners)
- whether the frame is blank
- how much the picture changed from the previous frame
- the RMS volume of the front and rear/surround audio channels, and silence
- speech/music/noise classification from
  [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter)
- the frame's time and position in the recording

It then derives more features from those, such as how long the logo has been
present and how far the frame is from the nearest blank. Everything is
condensed into one row per second. For each second, a window from 60 seconds
before to 60 seconds after is fed to a small temporal convolutional network
(dilated 1D convolutions), which outputs the probability that the second is a
commercial. Finally, those probabilities are turned into breaks: each break is
snapped to nearby blank frames or scene changes, and the length limits above
are applied.

## Ideas / to do

- Proper packaging (`setup.py` / PyPI).
- Per-channel settings or models. So far, one model trained on enough varied
  data seems to generalize well.
- New video/audio features. These mean retraining from scratch, and possibly
  re-curating data.
