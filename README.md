# 📺 pycommflag

pycommflag finds the commercials in TV recordings using a small neural network.

- It is a drop-in replacement for MythTV's `mythcommflag`, and it can also write
  comskip-style `.edl` and `.txt` files for Kodi, Tvheadend, Jellyfin, Emby,
  NextPVR, and others.
- It was built and tuned in the 2020s for US cable and broadcast TV.
- It works on shows that have no station logo.
- It uses the audio as well as the picture.
- The trade-off: it is slower than the older tools (about 10 minutes per hour of
  video), and there's no `.ini` to tune. If it gets something wrong, you correct
  it and retrain.

pycommflag builds on many ideas from mythcommflag and would not exist without it.
mythcommflag was revolutionary in its day, but its algorithms have not changed
significantly in more than 15 years.

**Contents:**
[Quick start](#-quick-start) ·
[How it compares](#-how-it-compares) ·
[Installation](#-installation) ·
[Flagging a recording](#-flagging-a-recording) ·
[MythTV](#-using-it-with-mythtv) ·
[Coming from comskip](#-coming-from-comskip) ·
[Other DVRs and players](#-using-it-with-other-dvrs-and-players) ·
[Training](#-training-your-own-model) ·
[How it works](#-how-it-works) ·
[Ideas / to do](#-ideas--to-do)

## 🚀 Quick start

```sh
pipx install git+https://github.com/b-pass/pycommflag
pycommflag -f /path/to/recording.ts
```

If the recording is in MythTV, the breaks go straight into its database.
Otherwise, you get a `recording.edl` next to the video. The first run downloads
the models. See [Installation](#-installation) for MythTV backends and other
setups, then [Using it with MythTV](#-using-it-with-mythtv) or
[other DVRs and players](#-using-it-with-other-dvrs-and-players).

## 🆚 How it compares

| | pycommflag | mythcommflag | comskip |
|---|---|---|---|
| Approach | Neural network trained on hand-labeled recordings | Hand-written heuristics | Hand-written heuristics, with about 100 `.ini` settings |
| Video cues | Station logo, blank frames, scene changes | Blank frames, scene changes, station logo | Blank frames, station logo, scene changes, aspect ratio and resolution changes |
| Audio cues | Speech/music/noise classification, front and surround volume, silence | None | Volume and silence |
| Shows with no station logo | Handled; it is also trained on recordings with the logo removed | Blank frames and scene changes only | Falls back on its other methods |
| Tuned for | 2020s US cable and broadcast TV | 2000s-era TV (largely unchanged since) | Whatever your `.ini` says; many community `.ini` files exist |
| Improving accuracy | Correct mistakes in the GUI and retrain | Pick a detection method per channel | Edit `comskip.ini` |
| Re-flagging with a new model or settings | Seconds, from a saved "feature log" | Decodes the whole video again | Decodes the whole video again |
| Speed | About 10 min per hour of video (see [Speed](#speed)) | Fast | Very fast |
| Output | MythTV database, EDL, comskip `.txt` | MythTV database | EDL, `.txt`, and many player-specific formats |
| Runs on | Linux, with Python 3.10+ and TensorFlow | Part of MythTV | Windows, Linux, macOS |
| License | MIT | GPL-2 | GPL-2 |

pycommflag does not read `comskip.ini` or MythTV's per-channel detection method
bitmask. The only per-channel MythTV setting it honors is "commercial flagging
disabled".

What about other flaggers? Plex, Channels DVR, NextPVR, and the Emby and
Jellyfin commercial-skip plugins all run comskip under the hood. MCEBuddy is a
Windows wrapper around comskip or ShowAnalyzer, and ShowAnalyzer is
closed-source and no longer developed.
[tv-detect](https://github.com/simonchrz/tv-detect) is another machine-learning
detector, aimed at German TV. Jellyfin's Intro Skipper finds repeated intros,
not commercials.

## 📦 Installation

You need Python 3.10 or newer. A few features also need system packages:

- **MythTV integration** needs the `mysqlclient` build headers (e.g.
  `libmysqlclient-dev` or `default-libmysqlclient-dev`, plus `pkg-config`).
- **The editing GUI** (`-g`) needs Tk (e.g. `python3-tk`).

### For your own user (comskip-style use)

Install with [pipx](https://pipx.pypa.io/), which puts pycommflag in its own
virtualenv and adds a `pycommflag` command:

```sh
pipx install git+https://github.com/b-pass/pycommflag
```

To use it with MythTV, install `pycommflag[mythtv]` instead. You can also add
MythTV support to an existing install later with
`pipx inject pycommflag mysqlclient`.

### On a MythTV backend

MythTV runs flagging jobs as the `mythtv` user, so install pycommflag
somewhere every user can reach it, such as a system-wide virtualenv:

```sh
sudo python3 -m venv /opt/pycommflag
sudo /opt/pycommflag/bin/pip install 'pycommflag[mythtv] @ git+https://github.com/b-pass/pycommflag'
sudo ln -s /opt/pycommflag/bin/pycommflag /usr/local/bin/pycommflag
```

With pipx 1.5 or newer, `sudo pipx install --global 'pycommflag[mythtv] @ git+https://github.com/b-pass/pycommflag'`
does the same thing. To upgrade, run the same `pip install` again with
`--upgrade`.

### From a source checkout (for development)

```sh
git clone https://github.com/b-pass/pycommflag.git
cd pycommflag
python3 -m venv venv
./venv/bin/pip install -e '.[all]'
```

The `pycommflag` command is then in `venv/bin`. You can also use `./run.sh`
from the checkout, which activates `./venv` and, if `libjemalloc2` is
installed, preloads it to reduce memory use.

Optional extras: `mythtv` (MythTV database support), `train` (scikit-learn,
for `--eval`), `yaml` (for `--yaml`), and `all`.

### Models

pycommflag needs a trained model to flag anything. It looks for one in this
order:

1. `--model /path/to/file.keras`
2. `model.keras`, then `model.h5`, then the published model's file (e.g.
   `pycommflag-f23-w60x60.keras`) in the models directory. This is `models/`
   in a source checkout, or `~/.keras/pycommflag/` otherwise. Use `--models DIR`
   to change it.
3. Otherwise, it downloads the published model that matches this version of
   pycommflag into the models directory. If that directory isn't writable, it
   downloads to `~/.keras/pycommflag/` instead. Pass `--no-download` to get an
   error instead.

Put your own model in the models directory as `model.keras` (a symlink works)
to use it in place of the published one.

The first run also downloads the inaSpeechSegmenter audio model into
`~/.keras/inaSpeechSegmenter/`. Both downloads happen once per user, so if
MythTV runs pycommflag as the `mythtv` user, that user needs network access
the first time too (or copy `~/.keras` over from another user).

A model only works with the version of pycommflag whose features it was trained
on. If they don't match, pycommflag stops with an error saying so, rather than
flagging badly. You can [train your own](#-training-your-own-model).

## 🎬 Flagging a recording

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

## 📼 Using it with MythTV

pycommflag reads the database credentials from `~/.mythtv/config.xml` of the
user that runs it. If that file doesn't exist, all MythTV features are quietly
skipped, and the same happens if `mysqlclient` isn't installed. If you
explicitly ask for MythTV (`-j`, `--chanid`, `--starttime`, `-e` or
`-o mythtv`) without `mysqlclient`, pycommflag stops with an error. For MythTV recordings named in the usual `<chanid>_<starttime>.ts`
format, pycommflag works out the channel and start time from the filename.
You can also pass `--chanid`/`--starttime` or `-j JOBID`, just like with
mythcommflag.

We recommend trying pycommflag in three stages: from the command line first,
then as a user job on a few schedules, and finally as a full replacement for
mythcommflag.

Whenever MythTV runs pycommflag, it runs as the `mythtv` user. That user needs
pycommflag installed somewhere it can reach (see
[On a MythTV backend](#on-a-mythtv-backend)), `mysqlclient` installed, and its
own `~/.mythtv/config.xml`.

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
breaks found, and treats values of 128 or more as a failure. `-e` makes
pycommflag exit with its break count, and makes errors exit with 255, so
MythTV doesn't read an error as "1 break". (255 is the only error value that
survives MythTV's exit status handling.)

For command-line compatibility, pycommflag also accepts mythcommflag's
`--rebuild` (rebuild the seek table) and `--queue` options. It doesn't
implement them: it just runs the real `mythcommflag` from the `PATH` with the
same arguments. Likewise, if the job queue hands pycommflag a job that
`mythcommflag --queue --rebuild` created, pycommflag passes the whole job to the
real `mythcommflag`. For these reasons, don't replace the `mythcommflag` binary
itself with pycommflag. Use the setting above instead.

## 🔄 Coming from comskip

- Use `-o txt` to get comskip's `.txt` format, or `-o edl` to get an `.edl` for
  Kodi and other players that understand EDL action `3` (commercial break).
  Both files are written next to the video with the extension replaced, like
  comskip's output.
- There is no `.ini` file. Instead of tuning detection, you correct the
  recordings pycommflag gets wrong in the GUI and retrain (see below). Options
  can go in a YAML file (`--yaml`) if you'd rather not put them on the command
  line.
- Post-processing scripts that run comskip can call
  `pycommflag --no-log -o edl -f "$FILE"` instead. See the next section for
  specific DVRs.

## 🔌 Using it with other DVRs and players

Most DVRs can run a command when a recording finishes, and many players and
plugins read `.edl` files that sit next to the video. pycommflag fits anywhere
comskip does in that kind of setup.

A few things apply to all of them:

- The DVR runs pycommflag as its own service user (e.g. `hts` for Tvheadend),
  so install it system-wide as in
  [On a MythTV backend](#on-a-mythtv-backend), without the `[mythtv]` extra.
  That user also needs network access the first time, to download the models
  (see [Models](#models)).
- Flagging takes a while (see [Speed](#speed)), so the skip markers show up a
  few minutes after the recording ends.
- The `.edl` from a normal run contains only commercial breaks (type `3`). If
  you re-save a recording from the GUI, any Intro/Credits/Ignore segments you
  marked are also written, as type `2`. Kodi treats `2` as a scene marker, but
  some plugins read it differently (see Jellyfin below).

### Tvheadend (with Kodi)

Tvheadend reads `.edl` and comskip `.txt` files next to a recording and sends
them to Kodi's Tvheadend client as commercial skip points. In the web UI, go to
Configuration → Recording → Digital Video Recorder Profiles, and set
**Post-processor command** to:

```
/usr/local/bin/pycommflag --no-log -o edl -f "%f"
```

`%f` is the full path of the recording. Tvheadend runs the command without a
shell, so use the full path to `pycommflag`.

### Kodi on its own

Kodi reads `.edl` and comskip `.txt` files next to a video, and skips entries
of type `3` (commercial break). If Kodi plays your recordings as files, write
either format with `-o edl` or `-o txt`, from whatever runs after each
recording.

### Jellyfin

Jellyfin doesn't read EDL files itself. The
[EdlToMediaSegments](https://github.com/rrhett/EdlToMediaSegments) plugin
turns them into Jellyfin "media segments", which clients can skip. Type `3`
becomes a Commercial segment. That plugin treats type `2` as "Recap", so only
use EDL files from normal flagging runs with it, not ones re-saved from the
GUI.

To run pycommflag after each recording, create a small wrapper script, e.g.
`/usr/local/bin/pycommflag-jellyfin`:

```sh
#!/bin/sh
exec /usr/local/bin/pycommflag --no-log -o edl -f "$1"
```

Make it executable. Then, in Dashboard → DVR → Recording Post Processing, set
**Post-processing application** to the script, and **Post-processor command
line arguments** to `"{path}"`. Jellyfin's docs recommend a wrapper script
because of how it quotes arguments. New segments appear after the plugin's next
scan. You can also start one from Scheduled Tasks → "Media Segment Scan".

### Emby

The [Emby.ComSkipper](https://github.com/BillOatmanWork/Emby.ComSkipper) plugin
skips commercials using the `.edl` files next to recordings. Use Emby's DVR
post-processing setting to run pycommflag with `-o edl`, as for Jellyfin.

### NextPVR

NextPVR reads a `.edl` next to each recording and shows the breaks on its
timeline. Put a post-processing script in NextPVR's `Scripts` directory
(`PostProcessing.bat` on Windows). NextPVR passes the recording's filename as
the first argument, and the script should run:

```
pycommflag --no-log -o edl -f <filename>
```

### Plex

Plex has its own comskip-based commercial detection, and it doesn't read
`.edl` files. Its **Postprocessing Script** setting (Settings → Live TV & DVR →
DVR Settings) can run pycommflag, but the breaks then have to be cut out of the
file itself. For example, a script could run `pycommflag -o edl` and then use
the `.edl` to drive ffmpeg. pycommflag doesn't include such a script.

### Channels DVR and HDHomeRun DVR

Channels DVR has built-in, comskip-based commercial detection, and there's no
supported way to plug in a different detector. HDHomeRun's DVR has no
commercial skipping of its own. In both cases, pycommflag is only useful if
your player can read `.edl` files.

## 🧠 Training your own model

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

## 🔬 How it works

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

## 💡 Ideas / to do

- Publish to PyPI.
- Per-channel settings or models. So far, one model trained on enough varied
  data seems to generalize well.
- New video/audio features. These mean retraining from scratch, and possibly
  re-curating data.
