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

SEED = 121711

# array indexes
NORMTIME = 0
LOGO = 1
BLANK = 2
DIFF = 3
FVOL = 4
RVOL = 5
SILENCE = 6
SPEECH = 7
MUSIC = 8
NOISE = 9
LOGO_RUN = 10
HAVE_LOGO = 11
HAVE_RVOL = 12
PERCTIME = 13

GENERATED_FEATURES_START = 14
FVOL_MAX = 14
FVOL_STDDEV = 15
RVOL_MAX = 16
RVOL_STDDEV = 17
LOGO_RUN_MIN = 18
DIFF_90TH = 19
DIFF_MAX = 20

FEATURE_WIDTH = 21
TIMESTAMPS = 21
ANSWERS = 22
WEIGHTS = 23

# data params, both for train and for inference
WINDOW_BEFORE = 60
WINDOW_AFTER = 60
SUMMARY_RATE = 1
RATE = 29.97

# training params
MTYPE = ''
EPOCHS = 50
BATCH_SIZE = 64
TEST_PERC = 0.25
PATIENCE = floor(EPOCHS * .2)

F         = 32 # TCN filter count
K         = 7  # TCN kernel size
NUM_LAYERS = 5
DROPOUT   = 0.4
START_DROP= 0.1
TCN_DROP  = 0.2
POOL_HEADS= 4

def build_model(input_shape=(None, 121, 21)):
    return build_model_BEST(input_shape)

    # training params
    global MTYPE
    MTYPE = 'sides'

    from keras import layers, utils, Input, Model
    random.seed(SEED)
    utils.set_random_seed(SEED)

    inputs = Input(shape=input_shape[-2:], dtype='float32', name="input")

    # some features are unreliable ...
    x = layers.SpatialDropout1D(START_DROP)(inputs)

    x = layers.Dense(F, "relu", name="projection")(x)
    #x = layers.SpatialDropout1D(TCN_DROP//2)(x)

    # middle step [60] included in both sides intentionally since the center is important
    # since we are attempting to find a boundary between two things at point[60], we split
    # the pooling into two halves with their own weights and attentions 
    left = x[:, :60, : ]
    right = x[:, 60:120, : ]
    for i in range(NUM_LAYERS):
        blocks = [
            layers.Conv1D(F, K, padding="same", name=f"b{i}_conv1"),
            layers.LayerNormalization(name=f"b{i}_norm1"),
            layers.Activation("swish", name=f"b{i}_act1"),
            layers.SpatialDropout1D(TCN_DROP, name=f"b{i}_sd1"), # Add this to both Conv units in the block

            layers.Conv1D(F, K, padding="same", name=f"b{i}_conv2"),
            layers.LayerNormalization(name=f"b{i}_norm2"),
            layers.Activation("swish", name=f"b{i}_act2"),
            layers.SpatialDropout1D(TCN_DROP, name=f"b{i}_sd2"), # Add this to both Conv units in the block
        ]
        
        x = left
        for b in blocks:
            x = b(x)
        x = layers.Add(name=f"left_residual_b{i}")([x, left])
        if i+1 < NUM_LAYERS and x.shape[-2] > 1:
            x = layers.AveragePooling1D(2, padding="same", name=f"left_pool{i}")(x)
        left = x

        x = right
        for b in blocks:
            x = b(x)
        x = layers.Add(name=f"right_residual_b{i}")([x, right])
        if i+1 < NUM_LAYERS and x.shape[-2] > 1:
            x = layers.AveragePooling1D(2, padding="same", name=f"right_pool{i}")(x)
        right = x

        if x.shape[-2] <= 1:
            break

    def pool(side):
        if side.shape[-2] <= 1:
            return layers.Flatten()(side)
        return layers.Concatenate()([
            layers.GlobalAveragePooling1D()(side),
            layers.GlobalMaxPooling1D()(side),
        ])
        # Compute attention logits for all pool heads simultaneously
        x = layers.Conv1D(POOL_HEADS, 1, use_bias=False)(side)
        x = layers.Softmax(axis=1)(x) 
        # We transpose so we can multiply (batch, POOL_HEADS, 61) x (batch, 61, F)
        # Resulting shape: (batch, POOL_HEADS, F)
        x = layers.Permute((2, 1))(x)
        x = layers.Dot(axes=(2, 1))([x, side])
        # now make it (batch, POOL_HEADS*F)
        x = layers.Flatten()(x)
        x = layers.Dense(32, 'relu')(x)
        x = layers.Dropout(DROPOUT)(x)
        return x

    left = pool(left)
    right = pool(right)

    diff = layers.Subtract()([left, right])
    mult = layers.Multiply()([left, right])

    x = layers.Concatenate()([left, diff , mult, right])

    x = layers.Dense(32, 'relu', name="reclassifier")(x) 
    x = layers.Dropout(DROPOUT)(x)
    
    outputs = layers.Dense(1, 'sigmoid', name="output")(x)

    return Model(inputs, outputs)

def build_model_BLAH(input_shape=(121, 21)):
    # training params
    global MTYPE
    MTYPE = 'tcn'

    from keras import layers, utils, Input, Model
    random.seed(SEED)
    utils.set_random_seed(SEED)

    inputs = Input(shape=input_shape[-2:], dtype='float32', name="input")

    # some features are unreliable ...
    x = layers.SpatialDropout1D(START_DROP)(inputs)

    x = layers.Dense(F, "relu", name="projection")(x)
    x = layers.SpatialDropout1D(TCN_DROP//2)(x)

    # --- TCN Blocks ---
    for i, dilation_rate in enumerate(DILATIONS, start=1):
        name_prefix = f"tcn{i}"

        residual = x
        x = layers.Conv1D(F, K,
                          padding="same",
                          dilation_rate=dilation_rate,
                          name=f"{name_prefix}_conv1")(x)
        x = layers.LayerNormalization(name=f"{name_prefix}_ln1")(x)
        x = layers.Activation("swish", name=f"{name_prefix}_act1")(x)
        x = layers.SpatialDropout1D(TCN_DROP)(x) # Add this to both Conv units in the block

        x = layers.Conv1D(F, K,
                          padding="same",
                          dilation_rate=dilation_rate,
                          name=f"{name_prefix}_conv2")(x)
        x = layers.LayerNormalization(name=f"{name_prefix}_ln2")(x)
        x = layers.Activation("swish", name=f"{name_prefix}_act2")(x)
        x = layers.SpatialDropout1D(TCN_DROP)(x) # Add this to both Conv units in the block
        
        # Squeeze
        se = layers.GlobalAveragePooling1D()(x)
        # Excite
        se = layers.Dense(x.shape[-1]//4, activation='relu', use_bias=False)(se)
        se = layers.Dense(x.shape[-1], activation='sigmoid', use_bias=False)(se)
        # Apply
        se = layers.Reshape((1, x.shape[-1]))(se)
        x = layers.Multiply()([x, se])

        #if residual.shape[-1] != F:
        #    residual = layers.Conv1D(F, 1, padding="same",
        #                             name=f"{name_prefix}_res_proj")(residual)

        x = layers.Add(name=f"{name_prefix}_res")([x, residual])
        x = layers.LayerNormalization(name=f"{name_prefix}_res_norm")(x)

    #attn = layers.MultiHeadAttention(num_heads=2, key_dim=16, dropout=.1)(x, x) 
    #x = layers.Add(name="mha_residual")([x, attn])

    # use light attention to focus on a few slices instead of forcing just [60]
    # middle step [60] included in both sides intentionally since the center is important
    # since we are attempting to find a boundary between two things at point[60], we split
    # the pooling into two halves with their own weights and attentions in order to find
    # signal on each independently, and then we let a final dense sort it out.
    left = x[:, :61, : ]
    right = x[:, 60:, : ]
    x = None
    poolt = []

    for side in (left, right):
        # Compute attention logits for all pool heads simultaneously
        x = layers.Conv1D(POOL_HEADS, 1, use_bias=False)(side)
        x = layers.Softmax(axis=1)(x) 
        # We transpose so we can multiply (batch, POOL_HEADS, 61) x (batch, 61, F)
        # Resulting shape: (batch, POOL_HEADS, F)
        x = layers.Permute((2, 1))(x)
        x = layers.Dot(axes=(2, 1))([x, side])
        # now make it (batch, POOL_HEADS*F)
        x = layers.Flatten()(x)
        x = layers.Dense(16, 'relu')(x)
        x = layers.Dropout(DROPOUT)(x)
        x = layers.Dense(8, 'relu')(x)
        x = layers.Dropout(DROPOUT)(x)
        #x = layers.Dense(1, 'relu')(x)
        poolt.append(x)
    
    diff = layers.Subtract()(poolt)
    mult = layers.Multiply()(poolt)

    x = layers.Concatenate()(poolt + [diff , mult])

    x = layers.Dense(32, 'relu', name="reclassifier")(x) 
    x = layers.Dropout(DROPOUT)(x)
    
    outputs = layers.Dense(1, 'sigmoid', name="output")(x)

    return Model(inputs, outputs)

def build_model_BEST(input_shape=(121, 21)):
    global MTYPE, F, K, DILATIONS, DROPOUT, START_DROP, NUM_LAYERS, TCN_DROP

    F        = 32 # TCN filter count
    K        = 5  # TCN kernel size
    DILATIONS = [1, 2, 4, 8] # TCN dilation schedule
    NUM_LAYERS = len(DILATIONS)
    DROPOUT   = 0.4
    START_DROP= 0.2
    TCN_DROP = 0.1

    from keras import layers, regularizers, utils, Input, Model
    random.seed(SEED)
    utils.set_random_seed(SEED)

    inputs = Input(shape=input_shape[-2:], dtype='float32', name="input")

    # some features are unreliable ...
    x = layers.SpatialDropout1D(START_DROP)(inputs)

    x = layers.Dense(F, name="projection")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # --- TCN Blocks ---
    for i, dilation_rate in enumerate(DILATIONS, start=1):
        name_prefix = f"tcn{i}"

        residual = x
        x = layers.Conv1D(F, K,
                          padding="same",
                          dilation_rate=dilation_rate,
                          name=f"{name_prefix}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_ln1")(x)
        x = layers.Activation("swish", name=f"{name_prefix}_act1")(x)
        x = layers.SpatialDropout1D(TCN_DROP)(x) # Add this to both Conv units in the block

        x = layers.Conv1D(F, K,
                          padding="same",
                          dilation_rate=dilation_rate,
                          name=f"{name_prefix}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_ln2")(x)
        x = layers.Activation("swish", name=f"{name_prefix}_act2")(x)
        x = layers.SpatialDropout1D(TCN_DROP)(x) # Add this to both Conv units in the block
        
        # Squeeze
        se = layers.GlobalAveragePooling1D()(x)
        # Excite
        se = layers.Dense(x.shape[-1]//4, activation='relu', use_bias=False)(se)
        se = layers.Dense(x.shape[-1], activation='sigmoid', use_bias=False)(se)
        # Apply
        se = layers.Reshape((1, x.shape[-1]))(se)
        x = layers.Multiply()([x, se])

        #if residual.shape[-1] != F:
        #    residual = layers.Conv1D(F, 1, padding="same",
        #                             name=f"{name_prefix}_res_proj")(residual)

        x = layers.Add(name=f"{name_prefix}_res")([x, residual])

    #attn = layers.MultiHeadAttention(num_heads=2, key_dim=16, dropout=.1)(x, x) 
    #x = layers.Add(name="mha_residual")([x, attn])

    # use light attention to focus on a few slices instead of forcing just [60]
    attn = layers.Dense(1, use_bias=False)(x) 
    attn = layers.Softmax(axis=1, name="attn")(attn)
    x = layers.Dot(axes=1)([x,attn])
    x = layers.Flatten()(x)

    x = layers.Dense(64, name="classifier")(x)
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
    This is because the training data is supplied by humans, and might be off a couple frames.
    Using this allows us to repeatably programmatically fine-tune tag locations whether 
    supplied by humans or AI.

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
    prev_end = 0
    for tag_type, (start_time, end_time) in tags:
        if start_time < prev_end:
            start_time = prev_end
            if end_time < start_time:
                continue
        
        if tag_type in (SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value):
            filtered_tags.append((tag_type, (max(start_time,prev_end), end_time)))
            prev_end = end_time
            continue

        # Different max distances based on tag type
        common_tag_types = (SceneType.SHOW, SceneType.SHOW.value, SceneType.COMMERCIAL, SceneType.COMMERCIAL.value)
        max_distance = 5 if tag_type in common_tag_types else 2
        
        # First try to align with blank frames
        new_start = find_nearest_blank(start_time, blanks, max_distance)
        new_end = find_nearest_blank(end_time, blanks, max_distance)
        
        # If still at original positions, try aligning with diff boundaries
        if new_start == start_time:
            new_start = find_highest_diff_boundary(start_time, max_distance/2, diffs)
            if new_start < prev_end:
                new_start = prev_end
        #    if new_start != start_time:
        #        print(f"MOVED tag start {tag_type} from {start_time} {new_start}")
        #else:
        #    print(f"ALIGNED tag start {tag_type} from {start_time} {new_start}")
        if new_end == end_time:
            new_end = find_highest_diff_boundary(end_time, max_distance/2, diffs)
            if new_end < prev_end:
                new_end = prev_end
        #    if new_end != end_time:
        #        print(f"MOVED tag end {tag_type} from {end_time} {new_end}")
        #else:
        #    print(f"ALIGNED tag end {tag_type} from {end_time} {new_end}")
        
        # invalid after fine tuning, revert to original
        #if new_start > new_end:
        #    new_start = max(start_time, prev_end)
        #    new_end = max(end_time, new_start)
        # Only keep valid tags
        if new_start < new_end:
            filtered_tags.append((tag_type, (new_start, new_end)))
            prev_end = new_end

    #print(tags)
    #print(filtered_tags)
    return filtered_tags

def condense(frames: np.ndarray, timestamps: np.ndarray, answers: np.ndarray, weights: np.ndarray, step: int) -> np.ndarray:
    """
    Summarize video features by aggregating the specified step size.
    """
    def doit(a, atimestamps, aanswers, aweights):
        res = []

        res.append(a[:, a.shape[1]//2, NORMTIME]) # middle relative timestamp
        res.append(np.average(a[:, :, LOGO], axis=1))
        res.append(np.average(a[:, :, BLANK], axis=1))
        res.append(np.count_nonzero(a[:, :, DIFF] >= 0.5, axis=1) / a.shape[1])  # Diff count above 0.5
        for x in (FVOL,RVOL,SILENCE,SPEECH,MUSIC,NOISE):
            res.append(np.average(a[:, :, x], axis=1))
        for x in (LOGO_RUN, HAVE_LOGO, HAVE_RVOL):
            res.append(a[:, a.shape[1]-1, x]) # end of the logo run feature
        res.append(a[:, a.shape[1]//2, PERCTIME]) # middle percentage timestamp

        assert(len(res) == GENERATED_FEATURES_START)
        
        for f in (FVOL,RVOL) :
            res.append(np.max(a[:, :, f], axis=1)) # vol max
            res.append(np.std(a[:, :, f], axis=1)) # vol std dev
        res.append(np.min(a[:, :, LOGO_RUN], axis=1)) # min of the logo run feature
        res.append(np.percentile(a[:, :, DIFF], 90, axis=1)) # diff 90th
        res.append(np.max(a[:, :, DIFF], axis=1)) # diff max

        assert(len(res) == FEATURE_WIDTH)

        res.append(atimestamps[:, a.shape[1]//2]) # center timestamp
        res.append(np.max(aanswers, axis=1)) # answer
        res.append(np.min(aweights, axis=1)) # weight
        
        #res = [np.average(a[:, :, 0:6], axis=1)] + [x.reshape(x.shape[0], 1) for x in res] + [np.average(a[:, :, 6:], axis=1)]
        #res[-1][:, -2] = (np.count_nonzero(a[:, :, -2] >= 0.5, axis=1) >= a.shape[1]/2).astype('float32')
        
        return np.concatenate([x.reshape((x.shape[0], 1)) if len(x.shape) == 1 else x for x in res], axis=1)
    
    n_frames = len(frames)
    remaining = n_frames % step
    if n_frames >= step:
        # Reshape the array to group frames by step size
        condensed = doit( 
            frames[:(n_frames//step)*step].reshape(-1, step, frames.shape[1]),
            timestamps[:(n_frames//step)*step].reshape(-1, step, 1),
            answers[:(n_frames//step)*step].reshape(-1, step, 1),
            weights[:(n_frames//step)*step].reshape(-1, step, 1),
        )
    else:
        condensed = None
    
    if remaining > 0:
        # Do the final, partial condensing
        partial = doit( 
            frames[-remaining:].reshape(-1, remaining, frames.shape[1]),
            timestamps[-remaining:].reshape(-1, remaining, 1),
            answers[-remaining:].reshape(-1, remaining, 1),
            weights[-remaining:].reshape(-1, remaining, 1),
        )

        if condensed is None:
            return partial
        return np.vstack((condensed, partial))
    return condensed

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
        spans = processor.read_feature_spans(flog, 'blank', 'diff')
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

    if len(frames) < (WINDOW_BEFORE + WINDOW_AFTER) * 2 * round(RATE):
        return None
    
    # ok now we can numpy....
    frames = np.array(frames, dtype='float32')

    if not have_logo:
        frames[:, LOGO] = 0

    # change the diff column to be normalized [0,30] -> [0,1]
    frames[:,DIFF] = np.clip(frames[:,DIFF] / 30, 0, 1.0)

    # add a column for time since logo
    run = np.zeros((len(frames),1), dtype='float32')
    nlogo_dist = 0
    for n in range(len(frames)):
        if frames[n][LOGO] > 0.5:
            nlogo_dist = 0
        else:
            nlogo_dist += 1
            run[n][0] = min(nlogo_dist/(frame_rate * 300), 1.0) 
    assert(frames.shape[-1] == LOGO_RUN)
    frames = np.append(frames, run, axis=1)
    assert(frames.shape[-1] == HAVE_LOGO)
    pres = (np.ones if have_logo else np.zeros)((len(frames),1), dtype='float32')
    frames = np.append(frames, pres, axis=1)

    assert(frames.shape[-1] == HAVE_RVOL)
    have_rvol = np.count_nonzero(frames[:, RVOL] > 0.001) >= frame_rate
    pres = (np.ones if have_rvol else np.zeros)((len(frames),1), dtype='float32')
    frames = np.append(frames, pres, axis=1)

    # save off the times
    timestamps = frames[:,NORMTIME].copy()

    # add a column for time percentage
    assert(frames.shape[-1] == PERCTIME)
    frames = np.append(frames, (frames[:,NORMTIME]/endtime)[:,np.newaxis], axis=1)

    # change the first column to be normalized timestamps (30 minute segments)
    frames[:,NORMTIME] = (frames[:,NORMTIME] % 1800.0) / 1800.0

    answers = np.zeros((frames.shape[0],1), dtype=np.float32)
    weights = np.ones((frames.shape[0],1), dtype=np.float32)

    for (tt,(st,et)) in tags:
        if type(tt) is not int: tt = tt.value

        si = np.searchsorted(timestamps, st)
        ei = np.searchsorted(timestamps, et)

        if tt == SceneType.DO_NOT_USE.value:
            weights[si:ei] = 0 # ignore this entire section
        elif tt == SceneType.COMMERCIAL.value:
            answers[si:ei] = 1.0
        elif tt != SceneType.SHOW.value:
            weights[si:ei] = 0.75 # weight these areas as less important because they might be confusing
    
    condensed = condense(frames, timestamps, answers, weights, round(frame_rate/SUMMARY_RATE))

    # massively up weight near boundaries. however, _adjust_tags makes this only really matter outside of 5
    # within 5, it can be wrong ... so we only upweight the closest ones to the boundary
    # the others that are a little farther are not upweighted at all because they are in "dont care" territory
    prev_t = condensed[0][ANSWERS]
    valid_tags = (SceneType.SHOW.value, SceneType.SHOW, SceneType.COMMERCIAL.value, SceneType.COMMERCIAL)
    def upweight(t):
        return 1 + 4 * ((((60 - t) / 60) ** 2))
    for i in range(1,len(condensed)):
        next_t = condensed[i][ANSWERS]
        if prev_t != next_t:
            if next_t in valid_tags and prev_t in valid_tags:
                for t in range(5,WINDOW_BEFORE):
                    if i >= t and condensed[i-t][WEIGHTS] >= 1.0:
                        condensed[i-t][WEIGHTS] = max(condensed[i-t][WEIGHTS], upweight(t))
                for t in range(5,WINDOW_AFTER):
                    if i+t < len(condensed) and condensed[i+t][WEIGHTS] >= 1.0:
                        condensed[i+t][WEIGHTS] = max(condensed[i+t][WEIGHTS], upweight(t))

                condensed[i-1][WEIGHTS] = max(condensed[i-1][WEIGHTS], upweight(0))
                condensed[i][WEIGHTS] = max(condensed[i][WEIGHTS], upweight(0))
                
            prev_t = next_t

    #for x in [0,1]:
    #    print(f'{x}) {np.count_nonzero(answers == x)}')
    condensed = np.concatenate((
        np.tile(condensed[0], (round(WINDOW_BEFORE * SUMMARY_RATE),1)),
        condensed,
        np.tile(condensed[-1], (round(WINDOW_AFTER * SUMMARY_RATE),1)),
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
    if fname.endswith('.data'):
        fname = fname[:-5]
    
    if not os.path.exists(fname + '.data.npy'):
        condensed = load_nonpersistent(processor.read_feature_log(flogname), for_training)
        np.save(fname+'.data.npy', condensed)
        condensed = None
        gc.collect()
    
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
            self.shuf = np.random.permutation(len(self.data))

    return DataGenerator(*args, **kwargs)

def load_data_sliding_window(condensed:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if condensed is None:
        return ([],[],[],[])
    
    wbefore = round(WINDOW_BEFORE * SUMMARY_RATE)
    wafter = round(WINDOW_AFTER * SUMMARY_RATE)

    timestamps = condensed[wbefore:-wafter, TIMESTAMPS]
    answers = condensed[wbefore:-wafter, ANSWERS]
    weights = condensed[wbefore:-wafter, WEIGHTS]
    condensed = condensed[:, :FEATURE_WIDTH]
    
    from numpy.lib.stride_tricks import sliding_window_view
    frames = sliding_window_view(condensed, (wbefore+1+wafter, condensed.shape[1],)).squeeze()

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

    if False: #if not do_not_test:
        need = int(dlen*TEST_PERC+1) - tlen
        if need > dlen/100 and tlen/(tlen+dlen) < 0.1:
            print(f'WARNING: Need to move {need} of {dlen} elements to the test/eval set (have {tlen} will have ~{need+tlen})')
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

    import keras
    from keras import utils, callbacks
    from keras.metrics import Recall, Precision, TrueNegatives, TruePositives, FalseNegatives, FalsePositives
    from keras.losses import BinaryFocalCrossentropy, BinaryCrossentropy

    utils.set_random_seed(SEED)

    (data,test) = load_data(opts)
    
    print(f"Data shape (x):{data.shape} - Test shape (y):{test.shape if test is not None else 'None'}")
    
    tfile = tempfile.NamedTemporaryFile(prefix='train-', suffix='.pycf.model.keras', )
    model_path = tfile.name
    
    epoch = 0

    #model_path = '/tmp/x.keras'
    #epoch = 10

    model:keras.models.Model = None
    if epoch > 0:
        model = keras.models.load_model(model_path)
    else:
        model = build_model(data.shape)
        model.summary()
        model.compile(optimizer=keras.optimizers.AdamW(weight_decay=0.004), 
                    loss=BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.67, gamma=2, label_smoothing=0.01), 
                    metrics=['accuracy'],
                    weighted_metrics=['accuracy', 'recall', 'precision'])
        model.save(model_path)
    
    gc.collect()

    cb = []

    cb.append(callbacks.EarlyStopping(monitor='val_weighted_accuracy', mode="max", patience=PATIENCE, restore_best_weights=True))

    def cosine_annealing_with_warmup(epoch, lr):
        WARMUP = 4
        TOTAL = ceil(EPOCHS * .75)
        MAX_LR = 0.001
        MIN_LR = 0.00001 # Set your true floor here
        
        if epoch < WARMUP:
            return MAX_LR * (epoch + 1) / WARMUP
        
        progress = min(1.0, (epoch - WARMUP) / (TOTAL - WARMUP))
        return min(lr, MIN_LR + (MAX_LR - MIN_LR) * 0.5 * (1 + np.cos(np.pi * progress)))
    
    cb.append(callbacks.LearningRateScheduler(cosine_annealing_with_warmup))
    cb.append(callbacks.ReduceLROnPlateau(monitor='val_weighted_accuracy', mode="max", patience=PATIENCE-3, factor=0.5))
    
    class EpochModelCheckpoint(callbacks.ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            self.last_epoch = epoch
            return super().on_epoch_end(epoch, logs)

    ecp = EpochModelCheckpoint(model_path, monitor='val_weighted_accuracy', mode="max", verbose=1, save_best_only=True)
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
    model.fit(data, validation_data=test, epochs=EPOCHS, initial_epoch=epoch, callbacks=cb)

    print()
    print("Done")
    print()
    print('Final Evaluation...')

    # reload the best epoch
    model = keras.models.load_model(model_path)

    model.compile(optimizer="adam", loss='binary_crossentropy',
                  metrics=['accuracy', Precision(), Recall(), TrueNegatives(), TruePositives(), FalseNegatives(), FalsePositives()])

    dmetrics = model.evaluate(data, verbose=0, return_dict=True)
    print()
    for name, value in dmetrics.items():
        print(f"train {name}: {value:.4f}")
    
    tmetrics = model.evaluate(test, verbose=0, return_dict=True)
    print()
    for name, value in tmetrics.items():
        print(f"val {name}: {value:.4f}")

    tacc = tmetrics["accuracy"]
    if tacc >= 0.95:
        name = f'{opts.models_dir if opts and opts.models_dir else "."}{os.sep}pycf-{tacc:.04f}-{MTYPE}-{F}x{K}x{NUM_LAYERS}+{POOL_HEADS}-{DROPOUT}-w{WINDOW_BEFORE}x{WINDOW_AFTER}-{int(time.time())}.keras'
        print()
        print('Saving as ' + name)

        import shutil
        shutil.copy(model_path, name)
        try: os.chmod(name, 0o644)
        except: pass
    
    print()

    return 0

def raw_predict(feature_log:str|TextIO|dict, opts:Any=None)->list:
    import keras

    flog = processor.read_feature_log(feature_log)
    frame_rate = flog.get('frame_rate', 29.97)
    
    assert(flog['frames'][-1][0] > frame_rate)

    mf = opts.model_file if opts is not None else './model.keras'
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

    return list(zip(times, prediction.flatten()))

def predict(feature_log:str|TextIO|dict, opts:Any, write_log=None)->list:
    from .mythtv import set_job_status
    set_job_status(opts, "Inferencing...")

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

    results = post_predict(flog, prediction.flatten(), times, opts)
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

    #print(realtags)
    #print(orig)
    #print(result)
    #print()

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
    y_pred = []
    y_true = []
    y_dist = []
    
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
        
        for (mf, model) in models.items():
            try:
                data,answers,_,times = load_data_sliding_window(load_nonpersistent(flog, False))

                def distances_to_nearest(times, tags):
                    transitions = []
                    for (t,(b,e)) in tags:
                        if t != SceneType.DO_NOT_USE.value and t != SceneType.DO_NOT_USE:
                            transitions += [b,e]
                    transitions = np.array(transitions)
                    idx = np.searchsorted(transitions, times)
                    idx_left = np.clip(idx - 1, 0, len(transitions) - 1)
                    idx_right = np.clip(idx, 0, len(transitions) - 1)
                    dist_left = np.abs(times - transitions[idx_left])
                    dist_right = np.abs(times - transitions[idx_right])
                    return np.minimum(dist_left, dist_right)

                distances = distances_to_nearest(times, realtags)
                y_dist += distances.tolist()
                y_true += answers.tolist()
                
                prediction = model.predict(make_data_generator(data), verbose=True)
                
                result = post_predict(flog, prediction, times, opts) #, threshold=best[0])

                y_pred += prediction.flatten().tolist()

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
    print()

    if len(models) < 2:
        from tensorflow.math import confusion_matrix

        y_true = np.array(y_true, dtype='float32')
        y_pred = np.array(y_pred, dtype='float32')
        cm = confusion_matrix(y_true >= 0.5, 
                              y_pred >= 0.5, 
                              num_classes=2).numpy()
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  TP={tp} FP={fp} FN={fn} TN={tn} "
            f"| precision={precision:.4f} recall={recall:.4f}")
        
        bin_edges = [0,2,3,4, 5, 10, 15, 30, 45, 60, np.inf]
        bin_labels = ["0-1s", "3s", "4s", "5s", "5-10s", "10-15s", "15-30s", "30-45s", "45-60s", "60+s"]
        bucket_idx = np.digitize(y_dist, bin_edges) - 1  # 0-indexed bucket per example
        def auc_score(y_true, y_prob):
            pos = y_prob[y_true == 1]
            neg = y_prob[y_true == 0]
            if len(pos) == 0 or len(neg) == 0:
                return np.nan
            # count pairs where positive score > negative score (ties count as 0.5)
            diff = pos[:, None] - neg[None, :]
            return (np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg))

        print(f"{'bucket':<10} {'n':>7} {'accuracy':>10} {'auc':>8}")
        for i, label in enumerate(bin_labels):
            import gc
            gc.collect()
            mask = bucket_idx == i
            n = mask.sum()
            if n == 0:
                print(f"{label:<10} {0:>7} {'--':>10} {'--':>8}")
                continue
            acc = (y_true[mask] == (y_pred[mask] >= 0.5).astype('float32')).mean()
            auc = auc_score(y_true[mask], y_pred[mask])
            auc_str = f"{auc:.3f}" if not np.isnan(auc) else "n/a"
            print(f"{label:<10} {n:>7} {acc:>10.3f} {auc_str:>8}")

