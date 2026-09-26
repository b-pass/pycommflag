# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Keeping this file accurate

This file is only useful if it matches the code. Treat updating it as part of any change, not a separate task:

- When a change affects something described here (CLI options or mode precedence, feature log keys or `file_version`, `neural.py` column constants/`FEATURE_WIDTH`/window sizes, model lookup or naming, label/weight rules, `SceneType` values, output formats, MythTV behavior, file locations), update the matching text in the same edit session.
- If you notice a statement here that is already wrong, fix it (or remove it) rather than working around it. Verify against the code first; don't "correct" it from memory.
- Keep it to non-obvious facts and gotchas that take reading several files to learn. Don't add per-function docs, change history, or anything easily found by grepping. Prefer editing an existing line over appending a new one.
- Mention CLAUDE.md edits in your final summary so the user can review them.

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

All options live in `options.get_options()` as one flat argparse parser (no subparsers, for mythcommflag CLI compatibility). The mode is derived afterwards by `options.resolve_mode` with fixed precedence: `--rebuild`/`--queue` > `-t` > `-r` > `--eval` > `-g` > flag. `--yaml FILE` overwrites any parsed option with the YAML's keys. Note `--deinterlace` stores `False` into `no_deinterlace` (deinterlacing is off by default). `main.run`'s return value is not used as the process exit code.

Without `-l` or `--no-log`, the feature log is written to `$TMPDIR/cf_<video basename>.json` and kept. When re-running `-f` against an existing log, its saved logo is reused and the logo search is skipped.

Models live in `models/` (gitignored; `--models` overrides the dir). Prediction uses `--model`, else `models/model.keras`, falling back to `models/model.h5` — this lookup is duplicated in `neural.predict` and `neural.raw_predict`. The GUI calls `raw_predict`, so it needs a model too. Training writes `models/pycf-<val_acc>-...keras` only if val accuracy >= 0.95.

The ina_foss audio segmenter downloads its model on first use via `keras.utils.get_file` (needs network, cached in `~/.keras/datasets/inaSpeechSegmenter`).

Extraction and training both `os.nice()` themselves (10 and 19) because they are meant to run as background jobs.

## Architecture

Pipeline (orchestrated by `main.run`, which dispatches to the `_cmd_*` functions):

1. **Feature extraction** — `processor.process_video`: `Player` (PyAV wrapper, with error recovery in `_Recovery`) decodes; `logo_finder.search` first samples the whole video (`--logo-samples` fps) to find a static station logo (Sobel edges in the corners, ignoring overscan margins). Then frames are fed to the `VideoProc` thread (logo present, blank frame, column-mean diff) and audio to the `AudioProc` thread (RMS of front/rear channels + `extern/ina_foss` speech/music/noise segmentation, run in chunks). Merged per-frame rows are streamed to the feature log.
2. **Feature log** (`cf_<basename>.json`, optionally `.gz`) — the central data artifact. Keys: `file_version` (currently 10), `chanid`, `starttime`, `filename` (realpath of the video), `duration`, `frame_rate`, `logo`, `frames_header`, `frames` (one row per video frame: `time, logo_present, is_blank, diff, fvol, rvol, silence, speech, music, noise`), and `tags` (list of `(SceneType value, (start_sec, end_sec))`). Tags are both the prediction output and the training labels (hand-curated via GUI). `read_feature_log`/`write_feature_log` in `processor.py` handle it; `read_feature_spans` turns frame columns into spans (used by the GUI and tag adjustment).
3. **Model input** — `neural.load_nonpersistent` turns frame rows into a float matrix. Columns are the module constants at the top of `neural.py`:
   - 0–9 (`NORMTIME`…`NOISE`): the raw frame columns;
   - 10–15 (`LOGO_RUN`, `HAVE_LOGO`, `HAVE_RVOL`, `PERCTIME`, `BLANK_SINCE`, `BLANK_UNTIL`): derived per-frame;
   - 16–22 (`GENERATED_FEATURES_START`…`DIFF_MAX`): summary stats added by `condense` when it downsamples to `SUMMARY_RATE` (1/sec);
   - 23–25 (`TIMESTAMPS`, `ANSWERS`, `WEIGHTS`): bookkeeping, not model input. `FEATURE_WIDTH` = 23 is the model's input width.

   The condensed array is padded by repeating the first/last row, and `load_data_sliding_window` produces windows of `WINDOW_BEFORE + 1 + WINDOW_AFTER` (121) seconds, one per output second. Logs shorter than two windows yield `None` and are skipped. The `assert`s in `load_nonpersistent` enforce column order, so any feature change must update those constants and will invalidate existing models.

   Training labels/weights: tags are first snapped to nearby blanks/scene changes (`_adjust_tags`) and small gaps merged; `COMMERCIAL` → answer 1; `DO_NOT_USE` → weight 0; other non-show types → weight 0.75; frames near show/commercial boundaries are upweighted (up to 2×). For training, `load_persistent` caches condensed arrays next to each log as `*.data.npy` and `*.nologo.data.npy` (a synthetic variant with the logo features removed; training only keeps its windows with weight > 1, and it is never used for validation). Caches with the wrong column count are refused, but after changing feature *values* you must delete the `.npy` files yourself.
4. **Model** — `neural.build_model`: dilated 1D-conv TCN with a single sigmoid output (commercial probability per second). Hyperparameters are module constants (`F`, `K`, `DILATIONS`, `DROPOUT`, `EPOCHS`, …), and the saved model filename encodes them. Training monitors `val_weighted_accuracy` (early stopping, LR reduction, and best-epoch checkpoint).
5. **Post-processing** — `neural.post_predict` thresholds predictions at 0.5 into tags and enforces `--break-min-len`, `--break-max-len`, `--show-min-len`. `neural.diff_tags` compares two tag lists (used by `-r` and `--eval`).
6. **Output** — `main.output`: `-o auto` tries MythTV DB (`mythtv.set_breaks` into `recordedmarkup`) and falls back to EDL; `txt` is comskip format (`edl.py`). EDL/TXT go next to the video (the log's `filename`), with the extension replaced.

`-r` with several logs only rewrites a log and its output when the tags changed; if a log's video no longer exists and an `old/` directory sits beside the log, the log is moved there.

`SceneType` / `AudioSegmentLabel` enums in `feature_span.py` define label integer values stored in logs (`SHOW`=0, `INTRO`=1, `TRANSITION`=2, `COMMERCIAL`=3, `CREDITS`=4, `DO_NOT_USE`=5). Note `UNKNOWN` and `SHOW` share value 0, as do several `AudioSegmentLabel` aliases.

### MythTV integration

`mythtv.py` reads DB credentials from `~/.mythtv/config.xml`; if missing, all MythTV functions silently no-op. `main._resolve_inputs` fills in whatever is missing: filename from `-j` or from `--chanid`/`--starttime`, and `chanid`/`starttime` from `<chanid>_<starttime>.ext` (or `cf_<chanid>_<starttime>...`) filenames. `-j JOBID` updates job-queue status; `-e` makes `set_breaks` `sys.exit()` with the number of breaks (and forces uncaught exceptions to exit 256 so MythTV doesn't misread them). `--rebuild`/`--queue` just exec the real `mythcommflag`. `check_method(chanid)` respects per-channel commflag-disabled settings; when disabled, an empty break list is still output. `mythtv.get_breaks`/`processor.guess_external_breaks` can read existing breaks back out of MythTV.

`extern/` contains vendored code (inaSpeechSegmenter, sidekit MFCC, pyannote viterbi) — avoid restyling it.
