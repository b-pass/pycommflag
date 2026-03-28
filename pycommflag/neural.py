import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # shut up, tf

import logging as log
import gc
from math import ceil, floor, isqrt
from queue import Empty as QueueEmpty
from multiprocessing import Process, Queue

import resource
import sys
import tempfile
import time
from typing import Any, Iterator,TextIO,BinaryIO,List,Tuple

import numpy as np
import signal
import random

from .feature_span import *
from . import processor
from . import neural

SEED = 1711
random.seed(SEED)

# data params, both for train and for inference
WINDOW_BEFORE = 60
WINDOW_AFTER = 60

# --- Tunable Hyperparameters ---
MTYPE     = 'tcn'
EPOCHS    = 50
BATCH_SIZE= 128
TEST_PERC = 0.25
PATIENCE  = 10
F         = 32 # TCN filter count
K         = 5  # TCN kernel size
DILATIONS = [1, 2, 4, 8] # TCN dilation schedule
GRU_UNITS = 24
F_UNITS   = 32
GRU_DROPOUT = 0.15
DROPOUT   = 0.4
START_DROP= 0.075
L2        = 0.0002

def build_model(input_shape=(121, 21)):
    from keras import layers, regularizers, utils, Input, Model
    random.seed(SEED)
    utils.set_random_seed(SEED)

    inputs = Input(shape=input_shape[-2:], dtype='float32', name="input")

    # some features are unreliable ...
    x = layers.SpatialDropout1D(START_DROP)(inputs)

    x = layers.Dense(32, name="projection")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    # --- TCN Blocks ---
    for i, dilation_rate in enumerate(DILATIONS, start=1):
        name_prefix = f"tcn{i}"

        residual = x
        x = layers.Conv1D(F, K,
                          padding="same",
                          dilation_rate=dilation_rate,
                          kernel_regularizer=regularizers.l2(L2),
                          name=f"{name_prefix}_conv1")(x)
        x = layers.LayerNormalization(name=f"{name_prefix}_ln1")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)

        x = layers.Conv1D(F, K,
                          padding="same",
                          dilation_rate=dilation_rate,
                          kernel_regularizer=regularizers.l2(L2),
                          name=f"{name_prefix}_conv2")(x)
        x = layers.LayerNormalization(name=f"{name_prefix}_ln2")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)

        if residual.shape[-1] != F:
            residual = layers.Conv1D(F, 1, padding="same",
                                     name=f"{name_prefix}_res_proj")(residual)

        x = layers.Add(name=f"{name_prefix}_res")([x, residual])

    x = layers.Bidirectional(layers.GRU(GRU_UNITS, dropout=GRU_DROPOUT, recurrent_dropout=0.0),name="bigru")(x)

    #attn = layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=GRU_DROPOUT)(x, x)  # self-attention
    #x = layers.Add(name="mha_residual")([x, attn])
    #x = layers.LayerNormalization()(x)
    #x = x[:, 60, :]

    x = layers.Dense(F_UNITS, kernel_regularizer=regularizers.l2(L2), name="classifier")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(DROPOUT)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs, outputs)

def _adjust_tags(tags: List[Tuple[int, Tuple[float, float]]], 
                 blanks: List[Tuple[bool, Tuple[float, float]]], 
                 diffs: List[Tuple[float, Tuple[float, float]]]) \
        -> List[Tuple[int, Tuple[float, float]]]:
    """
    Adjust tag boundaries to align with scene transitions using blank frames and diff values.
    This is because the training data is supplied by humans, and might be off a bit.

    Args:
        tags: List of (tag_type, (start_time, end_time))
        blanks: List of (is_blank, (start_time, end_time))
        diffs: List of (diff_value, (start_time, end_time))
    
    Returns:
        Adjusted tags list with updated boundaries
    """

    def find_highest_diff_boundary(target_time: float, 
                                 search_window: float, 
                                 diffs: List[Tuple[float, Tuple[float, float]]]) -> float:
        # Find diffs within window
        window_start = target_time - search_window
        window_end = target_time + search_window
        best = 0
        when = None

        for (val, (s,e)) in diffs:
            time = s + (e-s)/2
            if time < window_start or not val:
                continue
            if time > window_end:
                break
            if val > best:
                best = val
                when = time
        
        return when if when is not None else target_time

    def find_nearest_blank(target_time: float, 
                         blanks: List[Tuple[bool, Tuple[float, float]]], 
                         max_distance: float) -> float:

        best = None
        dist = max_distance
        
        for is_blank, (start, end) in blanks:
            if not is_blank:
                continue

            when = start + (end - start)/2
            
            if when > target_time + max_distance:
                break

            if abs(when - target_time) < dist:
                dist = abs(when - target_time)
                best = when
        
        return best if best is not None else target_time

    # Process each tag
    filtered_tags = []
    for tag_type, (start_time, end_time) in tags:
        if tag_type in (SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value):
            filtered_tags.append((tag_type, (start_time, end_time)))
            continue

        # Different max distances based on tag type
        max_distance = 10 if tag_type in {SceneType.SHOW, SceneType.SHOW.value, 
                                        SceneType.COMMERCIAL, SceneType.COMMERCIAL.value} else 2
        
        # First try to align with blank frames
        new_start = find_nearest_blank(start_time, blanks, max_distance)
        new_end = find_nearest_blank(end_time, blanks, max_distance)
        
        # If still at original positions, try aligning with diff boundaries
        if new_start == start_time:
            new_start = find_highest_diff_boundary(start_time, 2, diffs)
        #    if new_start != start_time:
        #        print(f"MOVED tag start {tag_type} from {start_time} {new_start}")
        #else:
        #    print(f"ALIGNED tag start {tag_type} from {start_time} {new_start}")
        if new_end == end_time:
            new_end = find_highest_diff_boundary(end_time, 2, diffs)
        #    if new_end != end_time:
        #        print(f"MOVED tag end {tag_type} from {end_time} {new_end}")
        #else:
        #    print(f"ALIGNED tag end {tag_type} from {end_time} {new_end}")
        
        # Only keep valid tags
        if new_start < new_end:
            filtered_tags.append((tag_type, (new_start, new_end)))
    
    return filtered_tags

def aggregate_to_1hz(frames: np.ndarray, total_duration: float) -> np.ndarray:
    """
    Input:  (n_frames, 12) — time, logo, blank, diff, fvol, rvol, silence, speech, music, noise, answer, weight
    Output: (n_seconds, 41) — all features in [0,1] except col 38=time, col 39=answer, col 40=weight
    """
    
    from numpy.lib.stride_tricks import sliding_window_view

    # Input column indices
    _TIME    = 0
    _LOGO    = 1
    _BLANK   = 2
    _DIFF    = 3
    _FVOL    = 4
    _RVOL    = 5
    _SILENCE = 6
    _SPEECH  = 7
    _MUSIC   = 8
    _NOISE   = 9
    _ANSWER  = 10
    _WEIGHT  = 11
    _TOTAL_INPUT = 12
    assert(frames.shape[1] == _TOTAL_INPUT)

    # Normalization constants
    DIFF_CAP         = 30.0
    DIFF_CUT_THRESH  = 15.0
    CONSEC_LOGO_CAP  = 600.0
    CONSEC_BLANK_CAP = 15.0
    VOL_ROLL_SECS    = 15
    MAX_AUDIO_TRANS  = 10.0
    AMBIGUITY_LOW    = 0.3
    AMBIGUITY_HIGH   = 0.7

    def _consecutive_runs(binary_arr):
        """Vectorized: for each position, how many consecutive 1s up to and including here."""
        result = np.zeros(len(binary_arr), dtype=np.float32)
        padded = np.concatenate([[0], binary_arr.astype(int), [0]])
        starts = np.where(np.diff(padded) == 1)[0]
        ends   = np.where(np.diff(padded) == -1)[0]
        for s, e in zip(starts, ends):
            result[s:e] = np.arange(1, e - s + 1, dtype=np.float32)
        return result

    def _rolling_window(arr, W, fn):
        padded = np.pad(arr, (W - 1, 0), mode='edge')
        windows = sliding_window_view(padded, W)
        return fn(windows, axis=1).astype(np.float32)

    n_seconds = max(1, int(np.ceil(total_duration)))

    # ── Sort frames by second ──────────────────────────────────────────────
    frame_sec = np.clip(np.floor(frames[:, _TIME]).astype(int), 0, n_seconds - 1)
    fs        = frames
    fss       = frame_sec

    sec_counts  = np.bincount(fss, minlength=n_seconds)                     # (n_seconds,)
    sec_offsets = np.concatenate([[0], np.cumsum(sec_counts)])               # (n_seconds+1,)
    nonempty    = np.where(sec_counts > 0)[0]
    ne_starts   = sec_offsets[nonempty]                                      # frame index of first frame in each non-empty second
    assert(len(ne_starts) > 0)

    # ── Position within second (for half-window deltas) ───────────────────
    pos_in_sec   = np.arange(len(fs)) - np.repeat(sec_offsets[:-1], sec_counts)
    half_per_sec = np.maximum(1, sec_counts) // 2
    is_first     = pos_in_sec < np.repeat(half_per_sec, sec_counts)
    is_second    = ~is_first

    # ── Vectorized helpers ─────────────────────────────────────────────────
    def ps_sum(arr):
        out = np.zeros(n_seconds, dtype=np.float64)
        out[nonempty] = np.add.reduceat(arr.astype(np.float64), ne_starts)
        return out

    def ps_mean(arr):
        return (ps_sum(arr) / np.maximum(1, sec_counts)).astype(np.float32)

    def ps_sum_masked(arr, mask):
        out = np.zeros(n_seconds, dtype=np.float64)
        np.add.at(out, fss[mask], arr[mask].astype(np.float64))
        return out

    def ps_half_delta(arr):
        """(mean_2nd_half - mean_1st_half) remapped from [-1,1] → [0,1]"""
        cnt2 = np.maximum(1, sec_counts - half_per_sec)
        m1 = (ps_sum_masked(arr, is_first)  / np.maximum(1, half_per_sec)).astype(np.float32)
        m2 = (ps_sum_masked(arr, is_second) / cnt2).astype(np.float32)
        return np.clip((m2 - m1 + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

    def ps_max(arr):
        out = np.zeros(n_seconds, dtype=np.float32)
        out[nonempty] = np.maximum.reduceat(arr.astype(np.float32), ne_starts)
        return out

    def ps_min(arr):
        out = np.zeros(n_seconds, dtype=np.float32)
        out[nonempty] = np.minimum.reduceat(arr.astype(np.float32), ne_starts)
        return out

    def ps_std(arr):
        m  = ps_mean(arr)
        m2 = ps_mean(arr ** 2)
        return np.sqrt(np.maximum(0.0, m2 - m ** 2)).astype(np.float32)

    # ── Time ──────────────────────────────────────────────────────────────
    t_center      = np.arange(n_seconds, dtype=np.float32) + 0.5
    time_in_block = (t_center % 1800.0) / 1800.0
    pct_through   = np.clip(t_center / total_duration, 0.0, 1.0).astype(np.float32)

    # ── Logo ──────────────────────────────────────────────────────────────
    logo                = fs[:, _LOGO]
    logo_rate           = ps_mean(logo)
    logo_majority       = (logo_rate >= 0.5).astype(np.float32)
    logo_transition     = np.abs(np.diff(logo_majority, prepend=logo_majority[0])).astype(np.float32)
    logo_window_present = (logo_rate > 0).astype(np.float32)
    consec_logo_norm    = np.clip(_consecutive_runs(logo_majority) / CONSEC_LOGO_CAP, 0.0, 1.0)

    # ── Blank ─────────────────────────────────────────────────────────────
    blank               = fs[:, _BLANK]
    blank_rate          = ps_mean(blank)
    blank_any           = (blank_rate > 0).astype(np.float32)
    blank_window_present = blank_any.copy()

    b              = (blank >= 0.5).astype(np.int8)
    b_prev         = np.roll(b, 1)
    b_prev[sec_offsets[nonempty]] = 0          # treat start of each second as coming from 0
    blank_events   = ps_sum(np.maximum(0, b - b_prev))
    blank_event_rate = np.clip(blank_events / np.maximum(1, sec_counts), 0.0, 1.0).astype(np.float32)
    consec_blank_norm = np.clip(_consecutive_runs(blank_any) / CONSEC_BLANK_CAP, 0.0, 1.0)

    # ── Frame diff ────────────────────────────────────────────────────────
    diff_raw  = fs[:, _DIFF]
    diff      = np.clip(diff_raw / DIFF_CAP, 0.0, 1.0).astype(np.float32)
    diff_mean = ps_mean(diff)
    diff_std  = ps_std(diff)
    diff_max  = ps_max(diff)
    cut_rate  = ps_mean((diff_raw >= DIFF_CUT_THRESH).astype(np.float32))
    diff_delta = ps_half_delta(diff)

    # median and p90 — small per-second loop, only ~30 frames each
    diff_median = np.zeros(n_seconds, dtype=np.float32)
    diff_p90    = np.zeros(n_seconds, dtype=np.float32)
    for i in nonempty:
        d = diff[sec_offsets[i]:sec_offsets[i + 1]]
        diff_median[i] = np.median(d)
        diff_p90[i]    = np.percentile(d, 90)

    # ── Volume ────────────────────────────────────────────────────────────
    fvol = fs[:, _FVOL]
    rvol = fs[:, _RVOL]

    fvol_mean = ps_mean(fvol);  fvol_std = ps_std(fvol)
    fvol_max  = ps_max(fvol);   fvol_min = ps_min(fvol)
    rvol_mean = ps_mean(rvol);  rvol_std = ps_std(rvol)
    rvol_max  = ps_max(rvol);   rvol_min = ps_min(rvol)

    eps              = 1e-6
    ratio            = fvol / (fvol + rvol + eps)
    frvol_ratio_mean = ps_mean(ratio)
    frvol_ratio_std  = ps_std(ratio)
    fvol_delta       = ps_half_delta(fvol)
    rvol_delta       = ps_half_delta(rvol)

    # ── Audio type ────────────────────────────────────────────────────────
    silence_mean = ps_mean(fs[:, _SILENCE])
    speech_mean  = ps_mean(fs[:, _SPEECH])
    music_mean   = ps_mean(fs[:, _MUSIC])
    noise_mean   = ps_mean(fs[:, _NOISE])

    audio_stack = fs[:, _SILENCE:_NOISE+1]
    dominant    = np.argmax(audio_stack, axis=1)
    dom_diff    = (np.diff(dominant, prepend=dominant[0]) != 0).astype(np.float32)
    dom_diff[sec_offsets[nonempty]] = 0.0      # don't count boundary as transition
    audio_trans_norm = np.clip(ps_sum(dom_diff) / MAX_AUDIO_TRANS, 0.0, 1.0).astype(np.float32)

    # ── Rolling volume ────────────────────────────────────────────────────
    W = VOL_ROLL_SECS
    roll_fvol_max = _rolling_window(fvol_max, W, np.max)
    roll_rvol_max = _rolling_window(rvol_max, W, np.max)
    roll_fvol_std = _rolling_window(fvol_std, W, np.mean)
    roll_rvol_std = _rolling_window(rvol_std, W, np.mean)

    # ── Answer + weight ───────────────────────────────────────────────────
    commercial_frac = ps_mean(fs[:, _ANSWER])
    answer          = (commercial_frac >= 0.5).astype(np.float32)
    base_weight     = ps_min(fs[:, _WEIGHT])
    #ambiguous       = (commercial_frac > AMBIGUITY_LOW) & (commercial_frac < AMBIGUITY_HIGH)
    weight          = base_weight #np.where(ambiguous, 0.0, base_weight).astype(np.float32)

    # ── Assemble ──────────────────────────────────────────────────────────
    return np.column_stack([
        pct_through,          #  0
        time_in_block,        #  1
        logo_rate,            #  2
        logo_transition,      #  3
        logo_window_present,  #  4
        consec_logo_norm,     #  5
        blank_rate,           #  6
        blank_event_rate,     #  7
        blank_window_present, #  8
        consec_blank_norm,    #  9
        diff_mean,            # 10
        diff_median,          # 11
        diff_std,             # 12
        diff_max,             # 13
        diff_p90,             # 14
        cut_rate,             # 15
        diff_delta,           # 16
        fvol_mean,            # 17
        fvol_min,             # 18
        fvol_max,             # 19
        fvol_std,             # 20
        rvol_mean,            # 21
        rvol_min,             # 22
        rvol_max,             # 23
        rvol_std,             # 24
        frvol_ratio_mean,     # 25
        frvol_ratio_std,      # 26
        fvol_delta,           # 27
        rvol_delta,           # 28
        silence_mean,         # 29
        speech_mean,          # 30
        music_mean,           # 31
        noise_mean,           # 32
        audio_trans_norm,     # 33
        roll_fvol_max,        # 34
        roll_rvol_max,        # 35
        roll_fvol_std,        # 36
        roll_rvol_std,        # 37
        t_center,             # 38
        answer,               # 39
        weight,               # 40
    ]).astype(np.float32)

def load_nonpersistent(flog:dict, for_training=False)->np.ndarray:
    version = flog.get('file_version', 10)
    frame_rate = flog.get('frame_rate', 29.97)
    endtime = flog.get('duration', 0)

    if endtime < (WINDOW_BEFORE + 1 + WINDOW_AFTER) * 2:
        return None

    have_logo = not not flog.get('logo', None)

    tags = flog.get('tags', [])
    
    frames_header = flog['frames_header']
    assert('time' in frames_header[0])
    assert('diff' in frames_header[3])

    if tags and for_training:
        spans = processor.read_feature_spans(flog, 'blank')
        tags = _adjust_tags(tags, spans.get('blank', []), spans.get('diff', []))
        
        # clean up tiny gaps between identified breaks (including true 0-length gaps)
        i = 1
        while i < len(tags):
            if tags[i][0] == tags[i-1][0] and (tags[i][1][0] - tags[i-1][1][1]) < 15:
                tags[i-1] = (tags[i][0], (tags[i-1][1][0], tags[i][1][1]))
                del tags[i]
            else:
                i += 1
        
        i = 0
        while i < len(tags):
            clen = tags[i][1][1] - tags[i][1][0]
            if clen < 10 and not (tags[i][1][1]+clen+10 >= endtime or tags[i][0] in [SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value]):
                # delete the tiny segment
                del tags[i]
            else:
                i += 1
    
    frames = flog['frames']

    if frames and frames[0] is None: 
        frames = frames[1:]
    
    if len(frames) < frame_rate:
        return None
    
    if tags and tags[-1][0] in (SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value) and tags[-1][1][1]+10 >= endtime:
        flog['duration'] = endtime = min(endtime, tags[-1][1][0])
        e = len(frames) - 1
        while frames[e][0] >= endtime and e > 0:
            e -= 1
        if e > 0:
            del frames[e+1:]
        del tags[-1]
    
    if tags and tags[0][0] in (SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value) and tags[0][1][0] <= 5:
        b = 0
        while frames[b][0] < tags[0][1][1]:
            b += 1
        if b:
            del frames[:b]
        del tags[0]

    if len(frames) < (WINDOW_BEFORE + WINDOW_AFTER) * 2:
        return None
    
    # ok now we can numpy....
    frames = np.array(frames, dtype='float32')

    if not have_logo:
        frames[..., 1] = 0

    # add a column for answers [-2]
    frames = np.append(frames, np.zeros((frames.shape[0],1), dtype=np.float32), axis=1)

    # add a column for weights [-1]
    frames = np.append(frames, np.ones((frames.shape[0],1), dtype=np.float32), axis=1)

    timestamps = frames[:, 0]
    answers = frames[:, -2]
    weights = frames[:, -1]

    for (tt,(st,et)) in tags:
        if type(tt) is not int: tt = tt.value

        si = np.searchsorted(timestamps, st, 'left')
        ei = np.searchsorted(timestamps, et, 'right')

        if tt == SceneType.DO_NOT_USE.value:
            weights[si:ei] = 0 # ignore this entire section
        elif tt == SceneType.COMMERCIAL.value:
            answers[si:ei] = 1.0
        elif tt != SceneType.SHOW.value:
            weights[si:ei] = 0.5 # de-value the information here, it might be wonky
    
    condensed = aggregate_to_1hz(frames, endtime)
    
    #for x in [0,1]:
    #    print(f'{x}) {np.count_nonzero(answers == x)}')

    condensed = np.concatenate((
        np.tile(condensed[0], (WINDOW_BEFORE,1)),
        condensed,
        np.tile(condensed[-1], (WINDOW_AFTER,1)),
    ))

    return condensed

def load_persistent(flogname:str,for_training=True):
    fname = flogname
    if fname.endswith('.npy'):
        fname = fname[:-4]
    if fname.endswith('.gz'):
        fname = fname[:-3]
    if fname.endswith('.json'):
        fname = fname[:-5]
    
    if not os.path.exists(fname + '.data.npy'):
        condensed = load_nonpersistent(processor.read_feature_log(flogname), for_training)
        np.save(fname+'.data.npy', condensed)
    
    condensed = np.load(fname+'.data.npy', mmap_mode='r')
    return condensed

def make_data_generator(*args, **kwargs):
    from keras.utils import Sequence
    class DataGenerator(Sequence):
        def __init__(self, data, answers=None, weights=None):
            super().__init__()
            self.data = np.array(data, dtype='float32')
            self.answers = np.array(answers, dtype='float32') if answers else None
            self.weights = np.array(weights, dtype='float32') if weights else None
            self.len = ceil(len(self.data) / BATCH_SIZE)
            self.shape = (self.len, BATCH_SIZE, len(data[0]), len(data[0][0]))
            self.shuf = np.arange(len(self.data), dtype='int')
            self.do_shuf = False
        
        def __len__(self):
            return self.len
        
        def __getitem__(self, index):
            indexes = self.shuf[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            d = self.data[indexes]
            if self.answers is not None:
                a = self.answers[indexes]
                if self.weights is not None:
                    w = self.weights[indexes]
                    return d,a,w
                else:
                    return d,a
            else:
                return d
        
        def on_epoch_end(self):
            if self.do_shuf:
                self.shuffle()
            return super().on_epoch_end()
        
        def shuffle(self):
            #print("Doing the data generator shufflehussle")
            self.do_shuf = True
            self.shuf = np.random.randint(len(self.data), size=len(self.data))

    return DataGenerator(*args, **kwargs)

def load_data_sliding_window(condensed:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if condensed is None:
        return ([],[],[],[])
    
    timestamps = condensed[WINDOW_BEFORE:-WINDOW_AFTER,-3]
    answers = condensed[WINDOW_BEFORE:-WINDOW_AFTER,-2]
    weights = condensed[WINDOW_BEFORE:-WINDOW_AFTER,-1]
    condensed = condensed[:, :-3]
    
    from numpy.lib.stride_tricks import sliding_window_view
    frames = sliding_window_view(condensed, (WINDOW_BEFORE+1+WINDOW_AFTER, condensed.shape[1],)).squeeze()

    #print(len(self.frames), len(self.timestamps))

    assert(np.shares_memory(condensed, frames))
    assert(len(frames) == len(timestamps))
    assert(len(frames) == len(answers))
    assert(len(frames) == len(weights))

    return frames, answers, weights, timestamps

def load_data(opts, do_not_test=False) -> tuple:
    datafiles = opts.ml_data
    if not datafiles:
        return None
    
    testfiles = []

    if 'TEST' in datafiles:
        i = datafiles.index('TEST')
        testfiles = datafiles[i+1:]
        datafiles = datafiles[:i]

        i = 0
        while i < len(datafiles):
            if not os.path.exists(datafiles[i]):
                print(datafiles[i], "does not exist!!")
                del datafiles[i]
            elif datafiles[i] in testfiles or not os.path.isfile(datafiles[i]):
                del datafiles[i]
            else:
                i += 1

    dlen = 0
    data = ([],[],[])
    tlen = 0
    test = ([],[],[])

    for f in datafiles:
        if os.path.isdir(f) or f.endswith('.npy'):
            continue
        print("Loading",f)
        stuff = load_data_sliding_window(load_persistent(f))
        if stuff is not None:
            dlen += len(stuff[0])
            for x in range(3):
                for i in range(len(stuff[x])):
                    data[x].append(stuff[x][i])
    
    for f in testfiles:
        if os.path.isdir(f) or f.endswith('.npy'):
            continue
        print("Loading test",f)
        stuff = load_data_sliding_window(load_persistent(f))
        if stuff is not None:
            tlen += len(stuff[0])
            for x in range(3):
                for i in range(len(stuff[x])):
                    test[x].append(stuff[x][i])
    stuff = None

    if not do_not_test:
        need = int(dlen*TEST_PERC+1) - tlen
        if need > dlen/100 and tlen/(tlen+dlen) < 0.1:
            print(f'Need to move {need} of {dlen} elements to the test/eval set (have {tlen} will have ~{need+tlen})')
            data = list(zip(*data))
            random.shuffle(data)
            for i in range(need):
                e = data[i]
                test[0].append(e[0])
                test[1].append(e[1])
                test[2].append(e[2])
            data = data[need:]
            data = zip(*data)
    
    data = make_data_generator(*data)
    data.shuffle()

    test = make_data_generator(*test) if test else None
    
    return data,test

def train(opts:Any=None):
    # yield CPU time to useful tasks, this is a background thing...
    try: os.nice(19)
    except: pass

    (data,test) = load_data(opts)
    
    #print('Calculating loss weights')
    #sums = np.sum((np.sum(answers, axis=0), np.sum(test_answers, axis=0)), axis=0)
    ##weights = [1] * len(sums)
    ##weights[SceneType.SHOW.value] = sums[SceneType.COMMERCIAL.value] / sums[SceneType.SHOW.value]
    ##weights[SceneType.COMMERCIAL.value] = sums[SceneType.SHOW.value] / sums[SceneType.COMMERCIAL.value]
    #sums += 1
    #weights = (np.sum(sums)-sums)/np.sum(sums)
    #print("Loss Weights",weights)
    
    print(f"Data shape (x):{data.shape} - Test shape (y):{test.shape if test is not None else 'None'}")
    
    tfile = tempfile.NamedTemporaryFile(prefix='train-', suffix='.pycf.model.keras', )
    model_path = tfile.name
    
    stop = False
    epoch = 0

    #model_path = '/tmp/x.keras'
    #epoch = 10

    _train_some(model_path, data, test, epoch)
        
    print()
    print("Done")
    print()
    print('Final Evaluation...')

    import keras
    #dmetrics = model.evaluate(data, verbose=0)
    tmetrics = keras.models.load_model(model_path).evaluate(test, verbose=1)
    print()
    #print(dmetrics)
    print(tmetrics)

    if tmetrics[1] >= 0.80:
        name = f'{opts.models_dir if opts and opts.models_dir else "."}{os.sep}pycf-{tmetrics[1]:.04f}-{MTYPE}-{F}x{K}-x{len(DILATIONS)}-g{GRU_UNITS}-f{F_UNITS}-w{WINDOW_BEFORE}x{WINDOW_AFTER}-{int(time.time())}.keras'
        print()
        print('Saving as ' + name)

        import shutil
        shutil.copy(model_path, name)
        try: os.chmod(name, 0o644)
        except: pass
    
    print()

    return 0

def _train_some(model_path, train_dataset, test_dataset, epoch=0) -> tuple[int,bool]:
    import keras

    model:keras.models.Model = None
    if epoch > 0:
        model = keras.models.load_model(model_path)
    else:
        from keras.metrics import Recall, Precision
        from keras.losses import BinaryFocalCrossentropy, BinaryCrossentropy
        model = build_model(train_dataset.shape)
        model.summary()
        model.compile(optimizer="adam", loss=BinaryFocalCrossentropy(alpha=0.667, gamma=1.5), metrics=['accuracy', Recall(), Precision()])
        model.save(model_path)
    
    gc.collect()

    cb = []

    from keras import callbacks

    cb.append(callbacks.EarlyStopping(monitor='loss', patience=PATIENCE))
    cb.append(callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE))

    def cosine_annealing_with_warmup(epoch, lr):
        WARMUP = 4
        TOTAL_EPOCHS = 35
        MAX_LR = 0.001
        MIN_LR = 0.00001 # Set your true floor here
        
        if epoch < WARMUP:
            return MAX_LR * (epoch + 1) / WARMUP
            
        # Option A: Smooth decay all the way to the ending epoch
        progress = (epoch - WARMUP) / (TOTAL_EPOCHS - WARMUP)
        return MIN_LR + (MAX_LR - MIN_LR) * 0.5 * (1 + np.cos(np.pi * progress))
    
    cb.append(callbacks.LearningRateScheduler(cosine_annealing_with_warmup))
    #cb.append(callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=PATIENCE-1))
    
    class EpochModelCheckpoint(callbacks.ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            self.last_epoch = epoch
            return super().on_epoch_end(epoch, logs)

    ecp = EpochModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)
    cb.append(ecp)

    def handler(signum, frame):
        print("\nStopping (gracefully)...\n")
        model.stop_training = True
        signal.signal(signal.SIGINT, oldsint)
        signal.signal(signal.SIGTERM, oldterm)
        return
    oldsint = signal.signal(signal.SIGINT, handler)
    oldterm = signal.signal(signal.SIGTERM, handler)

    # no class weights with Focal loss: , class_weight={0:0.65, 1:1/0.65}
    model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, initial_epoch=epoch, callbacks=cb)

    #model.save(model_path) the checkpoint already saved the vest version
    return (ecp.last_epoch+1, model.stop_training or ecp.last_epoch+1 >= EPOCHS)

def predict(feature_log:str|TextIO|dict, opts:Any, write_log=None)->list:
    from .mythtv import set_job_status
    set_job_status(opts, "Inferencing...")

    import tensorflow as tf
    import keras

    flog = processor.read_feature_log(feature_log)
    frame_rate = flog.get('frame_rate', 29.97)
    
    assert(flog['frames'][-1][0] > frame_rate)

    mf = opts.model_file
    if not mf and opts:
        mf = f'{opts.models_dir or "."}{os.sep}model.keras'
    if not os.path.exists(mf):
        blah = mf
        mf = f'{opts.models_dir or "."}{os.sep}model.h5'
        if not os.path.exists(mf):
            raise Exception(f"Model files '{blah}' or '{mf}' do not exist")
    
    model:keras.models.Model = keras.models.load_model(mf)
    assert(model.output_shape[-1] == 1)

    data,_,_,times = load_data_sliding_window(load_nonpersistent(flog, False))
    prediction = model.predict(make_data_generator(data), verbose=True)

    results = post_predict(flog, prediction, times, opts)
    if not results:
        results = []

    #if orig_tags := flog.get('tags', []):
    #    log.debug(f'OLD tags n={len(orig_tags)} -> {str(orig_tags)}')
    #    log.debug(f'NEW tags n={len(results)} -> {str(results)}')
    #else:

    log.debug(f'Final tags n={len(results)}: {str(results)}')

    flog['tags'] = results

    if write_log is not None:
        processor.write_feature_log(flog, write_log)
    
    return results

def post_predict(flog:dict, prediction, times, opts:Any, threshold=0.5):
    duration = flog.get('duration', 0)

    results = [(0,(0,0))]
    for i in range(len(prediction)):
        when = float(times[i])
        ans = SceneType.COMMERCIAL.value if prediction[i] >= threshold else SceneType.SHOW.value
        
        results[-1] = (results[-1][0], (results[-1][1][0], when))
        if ans != results[-1][0]:
            results.append((ans, (when, when)))
    results[-1] = (results[-1][0], (results[-1][1][0], duration))

    i = 0
    while i < len(results):
        if results[i][0] == SceneType.SHOW.value:
            del results[i]
        else:
            i += 1

    #log.debug(f'Raw result n={len(results)}')

    # clean up tiny gaps between identified breaks (including true 0-length gaps)
    # show must be at least 30 seconds long (opts.show_min_len), or we just combine it into the commercial break its in the middle of
    i = 1
    while i < len(results):
        if results[i][0] == results[i-1][0] and (results[i][1][0] - results[i-1][1][1]) < opts.show_min_len:
            results[i-1] = (results[i][0], (results[i-1][1][0], results[i][1][1]))
            del results[i]
        else:
            i += 1

    spans = processor.read_feature_spans(flog, 'diff', 'blank')
    
    results = _adjust_tags(results, spans.get('blank', []), spans.get('diff', []))
    i = 1
    while i < len(results):
        if results[i][0] == results[i-1][0] and (results[i][1][0] - results[i-1][1][1]) < opts.show_min_len:
            results[i-1] = (results[i][0], (results[i-1][1][0], results[i][1][1]))
            del results[i]
        else:
            i += 1

    #log.debug(f'Merge/Adjust n={len(results)}: {str(results)}')

    # commercials must be at least 60 (opts.comm_min_len) seconds long, if it's less, it is deleted
    # commercials must be less than 360 seconds long (opts.comm_max_len), if it's more then it is just show after that
    i = 0
    while i < len(results):
        clen = results[i][1][1] - results[i][1][0]
        if clen < 10 or (clen < opts.break_min_len and results[i][0] == SceneType.COMMERCIAL.value):
            if i+1 >= len(results) and clen >= 5 and results[i][1][1]+clen+10 >= duration:
                # dont require full length if it is near the end of the recording
                break
            elif i == 0 and clen >= 5 and results[i][1][0] <= 5:
                # don't require full length at the beginning of the recording
                i += 1
                pass
            else:
                # tiny commercial, delete it
                del results[i]
            continue

        if clen >= opts.break_max_len:
            # huge commercial, truncate it
            results[i] = (results[i][0], (results[i][1][0], results[i][1][0] + opts.break_max_len))
            nextstart = results[i][1][1] + opts.show_min_len
            # check to make sure we didn't somehow create a small gap, if we did then WIDEN it to be show_min_len
            while i+1 < len(results) and results[i+1][1][0] < nextstart:
                if results[i+1][1][1] <= nextstart:
                    del results[i+1]
                else:
                    results[i+1] = (results[i+1][0], (nextstart, results[i+1][1][1]))
                    break

        # its ok now, move on
        i += 1
    
    #log.debug(f'Post n={len(results)}: {str(results)}')

    return results

def diff_tags(realtags, result) -> tuple[float,float,list]:
    # we create a list of tag pairs where they always exactly line up with boundaries in another list
    # this means we don't have to handle overlaps or tags spanning multiple other tags
    def split_upon(inlist, splitlist):
        sres = []
        for it,(ib,ie) in inlist:
            if it not in [SceneType.COMMERCIAL, SceneType.COMMERCIAL.value]:
                continue
            for st,(sb,se) in splitlist:
                if st not in [SceneType.COMMERCIAL, SceneType.COMMERCIAL.value,SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value]:
                    continue
                if ib < sb and sb < ie:
                    sres.append( (ib,sb) )
                    ib = sb
                if se > ib and se <= ie:
                    sres.append( (ib, se) )
                    ib = se
                if ib+1/30 > ie:
                    break
            if ib + 1/30 <= ie:
                sres.append( (ib, ie) )
        
        for st,(sb,se) in splitlist:
            if st in [SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value]:
                for i in range(len(sres)):
                    rb,re = sres[i]
                    if rb < se and re > sb:
                        del sres[i]
                        break

        #print("SPLIT", inlist, "into", sres)
        return sres

    orig = split_upon(realtags, result)
    result = split_upon(result, realtags)

    #print(orig)
    #print(result)

    missing = 0
    extra = 0
    rlist = []

    # now we have no overlaps, so all entries are one of: missing, extra, same
    ri = 0
    for ob,oe in orig:
        while ri < len(result):
            rb,re = result[ri]
            if ob < re:
                break
            else:
                extra += re - rb
                rlist.append( (1,rb,re) )
            ri += 1
        if ri >= len(result) or oe <= rb:
            missing += oe - ob
            rlist.append( (-1,ob,oe) )
        else:
            rlist.append( (0,ob,re) )
            assert(rb == ob and re == oe)
            ri += 1
    while ri < len(result):
        rb,re = result[ri]
        extra += re - rb
        rlist.append( (1,rb,re) )
        ri += 1
    
    #print("rlist:",rlist)
    
    return missing, extra, rlist


def eval(opts:Any):
    # yield CPU time to useful tasks, this is a background thing...
    try: os.nice(19)
    except: pass

    datafiles = []
    if opts.ml_data is None: opts.ml_data = []
    
    for f in opts.ml_data:
        if os.path.isdir(f) or f.endswith('.npy'):
            continue
        else:
            datafiles.append(f)
    
    print("EVALUATE", len(opts.eval), "on", len(datafiles))

    import keras

    models = {}
    total_time = 0
    all_missing = {}
    all_extra = {}
    for mf in opts.eval:
        try:
            models[mf] = keras.models.load_model(mf)
            print(mf)
            models[mf].summary()
            all_missing[mf] = 0
            all_extra[mf] = 0
        except Exception as e:
            log.exception(f"Unable to load MODEL {mf}")

    for f in datafiles:
        try:
            flog = processor.read_feature_log(f)
            if not flog:
                print('Load failed')
                continue

            spans = processor.read_feature_spans(flog, 'diff', 'blank')

            duration = flog.get('duration',0.00001)
            real_dur = duration
            realtags = []

            for (t,(b,e)) in flog.get('tags', []):
                if t in [SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value]:
                    if e+10 > real_dur and e < real_dur and b < real_dur:
                        e = real_dur
                    if b < 10 and e > 0:
                        b = 0
                    duration -= e-b
                elif t in [SceneType.COMMERCIAL, SceneType.COMMERCIAL.value]:
                    realtags.append( (t,(b,e)) )

            realtags = _adjust_tags(realtags, spans.get('blank', []), spans.get('diff', []))
            
            total_time += duration
        except Exception as e:
            log.exception(f"Unable to load {f}")
            continue
        
        #etotal = np.zeros(100)
        etotal = None
        for (mf, model) in models.items():
            try:
                data,_,_,times = load_data_sliding_window(load_nonpersistent(flog, False))

                prediction = model.predict(make_data_generator(data), verbose=True)
                
                if etotal is not None:
                    best = []
                    best_count = duration
                    for it in range(0, 100):
                        thresh = it / 100.0
                        result = post_predict(flog, prediction, times, opts, threshold=thresh)
                        (missing,extra,_) = diff_tags(realtags, result)
                        etotal[it] += missing+extra
                        if missing+extra == best_count:
                            best.append(thresh)
                        elif missing+extra < best_count:
                            best_count = missing+extra
                            best = [thresh]
                    #print(f"BEST THRESHOLD = {best[0]} to {best[-1]}")

                result = post_predict(flog, prediction, times, opts) #, threshold=best[0])
                (missing,extra,_) = diff_tags(realtags, result)
                acc = 100 - 100*(missing+extra)/duration
                print(f'{f} @ {mf} -> Acc {round(acc,4)}% <- FN:{round(missing,3)} + FP:{round(extra,3)} = {round(missing+extra,3)} seconds WRONG')
                all_missing[mf] += missing
                all_extra[mf] += extra
                #print(f,mf,model.evaluate(dataset_cat if model.output_shape[-1] > 1 else dataset_bin))
            except Exception as e:
                log.exception(f"Unable to load MODEL {mf}")
        print()
    
    print()
    print()
    print("Results:\nName\t\tMissing\t\tExtra\t\tAbsDiffSum\t\tAcc")

    for mf in models.keys():
        total_wrong = all_extra[mf] + all_missing[mf]
        acc = '%.5f %%' % ((total_time - total_wrong) * 100.0 / total_time,)
        print(f'{mf}: \t-{all_missing[mf]} \t+{all_extra[mf]} \t{total_wrong} \t{acc}')
    print()

    if etotal is not None:
        best = 0
        for i in range(100):
            print(f"Threshold {i/100} error = {etotal[i]}")
            if etotal[i] < etotal[best]:
                best = i
        print("BEST overall error rate at threshold", best/100)

    