# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

pycommflag flags segments of a TV recording as show/commercial (plus intro/credits/etc.) using per-frame video+audio features and a small Keras model. It is a drop-in replacement for MythTV's `mythcommflag`.

## Running

No setup.py, no test suite, no linter config. Deps are in `requirements.txt` (TensorFlow/Keras, PyAV, scipy, scikit-image, mysqlclient, Pillow; scikit-learn only for `--eval`, pyyaml only for `--yaml`). Entry point is `python3 -m pycommflag` (from repo root) or `./run.sh`, which activates `./venv` if present and preloads jemalloc.

```sh
./run.sh -f video.ts                 # full pipeline: extract features -> feature log -> predict -> output
./run.sh -r /tmp/cf_*.json           # re-predict from saved feature logs (fast, no video decode)
./run.sh -g cf_foo.json              # tk GUI to hand-edit tags; saves back into the feature log
./run.sh -t --data a.json b.json TEST c.json d.json   # train; files after the literal TEST token are the validation set
./run.sh --eval m1.keras m2.keras --data ...          # compare models
./run.sh --no-log -o edl -f video.ts # don't keep feature log; write .edl next to video
```

Models live in `models/` (gitignored). Prediction uses `--model`, else `models/model.keras`, falling back to `models/model.h5`. Training writes `models/pycf-<val_acc>-...keras` only if val accuracy >= 0.95.

The ina_foss audio segmenter downloads its model on first use via `keras.utils.get_file` (needs network, cached in `~/.keras/datasets/inaSpeechSegmenter`).

## Architecture

Pipeline (orchestrated by `main.run`):

1. **Feature extraction** — `processor.process_video`: `Player` (PyAV wrapper) decodes; `logo_finder.search` first samples the whole video to find a static station logo (Sobel edges in corners). Then frames are fed to `VideoProc` thread (logo present, blank frame, column-mean diff) and audio to `AudioProc` thread (RMS of front/rear channels + `extern/ina_foss` speech/music/noise segmentation). Merged per-frame rows are streamed to the feature log.
2. **Feature log** (`cf_<basename>.json`, optionally `.gz`) — the central data artifact. Keys: `file_version`, `chanid`, `starttime`, `filename`, `duration`, `frame_rate`, `logo`, `frames_header`, `frames` (one row per video frame: `time, logo_present, is_blank, diff, fvol, rvol, silence, speech, music, noise`), and `tags` (list of `(SceneType value, (start_sec, end_sec))`). Tags are both the prediction output and the training labels (hand-curated via GUI). `read_feature_log`/`write_feature_log` in `processor.py` handle it.
3. **Model input** — `neural.load_nonpersistent` turns frame rows into a float matrix, appends derived columns (logo run, have_logo, have_rvol, percent time), builds answers/weights from tags, then `condense` downsamples to `SUMMARY_RATE` (1/sec) and adds summary columns. `load_data_sliding_window` produces windows of `WINDOW_BEFORE + 1 + WINDOW_AFTER` seconds. Column indices are the module constants at the top of `neural.py` (`NORMTIME`…`WEIGHTS`, `FEATURE_WIDTH`); the `assert`s in `load_nonpersistent` enforce column order, so any feature change must update those constants and will invalidate existing models. For training, `load_persistent` caches condensed arrays next to each log as `*.data.npy` / `*.nologo.data.npy` — delete these after changing feature code.
4. **Model** — `neural.build_model`: dilated 1D-conv TCN with a single sigmoid output (commercial probability per second). Hyperparameters are module constants (`F`, `K`, `DILATIONS`, `DROPOUT`, `EPOCHS`, …). Training monitors `val_weighted_accuracy`.
5. **Post-processing** — `neural.post_predict` thresholds predictions into tags and enforces `--break-min-len`, `--break-max-len`, `--show-min-len`.
6. **Output** — `main.output`: `-o auto` tries MythTV DB (`mythtv.set_breaks` into `recordedmarkup`) and falls back to EDL; `txt` is comskip format (`edl.py`).

`SceneType` / `AudioSegmentLabel` enums in `feature_span.py` define label integer values stored in logs; `SceneType.DO_NOT_USE` regions get zero training weight. Note `UNKNOWN` and `SHOW` share value 0.

### MythTV integration

`mythtv.py` reads DB credentials from `~/.mythtv/config.xml`; if missing, all MythTV functions silently no-op. `chanid`/`starttime` are parsed from `<chanid>_<starttime>.ext` filenames when not given. `-j JOBID` updates job-queue status; `-e` makes the exit code the number of breaks (and forces uncaught exceptions to exit 256 so MythTV doesn't misread them). `--rebuild`/`--queue` just exec the real `mythcommflag`. `check_method(chanid)` respects per-channel commflag-disabled settings.

`extern/` contains vendored code (inaSpeechSegmenter, sidekit MFCC, pyannote viterbi) — avoid restyling it.
