from bisect import bisect_left
import logging as log
import gc
from math import ceil, floor
from queue import Empty as QueueEmpty
from multiprocessing import Process, Queue
import os
import pickle
import re
import resource
import sys
import tempfile
import time
from typing import Any,TextIO,BinaryIO,List,Tuple

import numpy as np

from .feature_span import *
from . import processor

# data params, both for train and for inference
WINDOW_BEFORE = 30
WINDOW_AFTER = 30
SUMMARY_RATE = 2
RATE = 29.97

# training params
RNN = 'lstm'
UNITS = 32
DROPOUT = 0.4
EPOCHS = 40
BATCH_SIZE = 256
TEST_PERC = 0.25

SceneType_one_hot = {SceneType.DO_NOT_USE: [0] * SceneType.count(),
                     SceneType.DO_NOT_USE.value: [0] * SceneType.count()}
for i in range(0, SceneType.count()):
    import numpy as np
    x = [0] * SceneType.count()
    x[i] = 1
    SceneType_one_hot[i] = np.array(x, dtype='float32')
    SceneType_one_hot[SceneType(i)] = SceneType_one_hot[i]


def _adjust_tags(tags: List[Tuple[int, Tuple[float, float]]], 
                blanks: List[Tuple[bool, Tuple[float, float]]], 
                diffs: List[Tuple[float, Tuple[float, float]]], 
                duration: float) -> List[Tuple[int, Tuple[float, float]]]:
    """
    Adjust tag boundaries to align with scene transitions using blank frames and diff values.
    This is because the training data is supplied by humans, and might be off a bit.

    Args:
        tags: List of (tag_type, (start_time, end_time))
        blanks: List of (is_blank, (start_time, end_time))
        diffs: List of (diff_value, (start_time, end_time))
        duration: Total duration of the video
    
    Returns:
        Adjusted tags list with updated boundaries
    """
    def find_highest_diff_boundary(target_time: float, 
                                 search_window: float, 
                                 diffs: List[Tuple[float, Tuple[float, float]]]) -> float:
        # Find diffs within window
        window_start = target_time - search_window
        window_end = target_time + search_window
        
        # Filter diffs within window
        candidates = [
            (diff_val, (start, end)) for diff_val, (start, end) in diffs 
            if window_start <= start <= window_end or 
               window_start <= end <= window_end
        ]
        
        if not candidates:
            return target_time
            
        # Find highest diff value and its corresponding boundary
        max_diff = max(candidates, key=lambda x: x[0])
        start, end = max_diff[1]
        
        # Return the closest boundary to target_time
        return start if abs(start - target_time) < abs(end - target_time) else end

    def find_nearest_blank(target_time: float, 
                         blanks: List[Tuple[bool, Tuple[float, float]]], 
                         max_distance: float) -> float:
        nearest_time = target_time
        min_distance = max_distance
        
        for is_blank, (start, end) in blanks:
            if not is_blank:
                continue
            
            if start > target_time + max_distance:
                break

            if abs(start - target_time) < min_distance:
                min_distance = abs(start - target_time)
                nearest_time = start
            
            if abs(end - target_time) < min_distance:
                min_distance = abs(end - target_time)
                nearest_time = end
        
        return nearest_time if min_distance < max_distance else target_time

    # Process each tag
    filtered_tags = []
    for tag_type, (start_time, end_time) in tags:
        # Different max distances based on tag type
        max_distance = 10 if tag_type in {SceneType.SHOW, SceneType.SHOW.value, 
                                        SceneType.COMMERCIAL, SceneType.COMMERCIAL.value} else 2
        
        # First try to align with blank frames
        new_start = find_nearest_blank(start_time, blanks, max_distance)
        new_end = find_nearest_blank(end_time, blanks, max_distance)
        
        # If still at original positions, try aligning with diff boundaries
        #if new_start == start_time:
        #    new_start = find_highest_diff_boundary(start_time, 1.0, diffs)
        #if new_end == end_time:
        #    new_end = find_highest_diff_boundary(end_time, 1.0, diffs)
           
        # Only keep valid tags
        if new_start < new_end:
            filtered_tags.append((tag_type, (new_start, new_end)))
    
    return filtered_tags

def condense(frames: np.ndarray, step: int) -> np.ndarray:
    """
    Condense video frames by averaging specific features over specified step sizes.
    
    Args:
        frames: numpy array of shape (n_frames, n_features) where features at indices:
               1: logo
               2: blank
               3: diff
               Are summarized.
               All other features are preserved from the first frame of each group.
        step: number of frames to combine into one
    
    Returns:
        Condensed numpy array with averaged features for logo, blank, diff
        and preserved values for other features
    """
    if step <= 1:
        return frames

    # audio features are spread across time anyway so we dont need to sumarize them unless the condense step is more than 0.5s
    
    n_frames = len(frames)
    remaining = n_frames % step
    if n_frames >= step:
        # Reshape the array to group frames by step size
        valid_frames = frames[:(n_frames // step) * step].reshape(-1, step, frames.shape[1])
        
        # Initialize condensed array with first frame of each group
        # This preserves all features that don't need averaging
        condensed = valid_frames[:, 0].copy()
        
        # Update only the features that need condensing
        condensed[:, 1] = np.mean(valid_frames[:, :, 1], axis=1, dtype='float32')  # Average logo
        condensed[:, 2] = np.mean(valid_frames[:, :, 2], axis=1, dtype='float32')  # Average blank
        condensed[:, 3] = np.count_nonzero(valid_frames[:, :, 3] > 15, axis=1) / step  # Diff ratio
    else:
        condensed = None
    
    if remaining:
        # Do the final, partial condensing
        last_frames = frames[-remaining:]
        last_condensed = last_frames[0].copy()  # Keep all features from first remaining frame
        last_condensed[1] = np.mean(last_frames[:, 1], dtype='float32')  # Average logo
        last_condensed[2] = np.mean(last_frames[:, 2], dtype='float32')  # Average blank
        last_condensed[3] = np.count_nonzero(last_frames[:, 3] > 15) / remaining  # Diff ratio
        if condensed is not None:
            return np.vstack((condensed, last_condensed))
        else:
            return last_condensed
    else:
        return condensed

def flog_to_vecs(flog:dict, fitlerForTraining=False, clean=True)->tuple[list[float], list[list[float]], list[float], list[float]]:
    version = flog.get('file_version', 10)
    frame_rate = flog.get('frame_rate', 29.97)
    endtime = flog.get('duration', 0)
    have_logo = not not flog.get('logo', None)

    tags = flog.get('tags', []) if clean else []
    
    frames_header = flog['frames_header']
    assert('time' in frames_header[0])
    assert('diff' in frames_header[3])

    if tags and fitlerForTraining:
        spans = processor.read_feature_spans(flog, 'blank')
        tags = _adjust_tags(tags, spans.get('blank', []), spans.get('diff', []), endtime)
        
        # clean up tiny gaps between identified breaks (including true 0-length gaps)
        i = 1
        while i < len(tags):
            if tags[i][0] == tags[i-1][0] and (tags[i][1][0] - tags[i-1][1][1]) < 20:
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
    
    if len(tags) > 1 and tags[-1][0] in (SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value) and tags[-1][1][1]+10 >= endtime:
        flog['duration'] = endtime = min(endtime, tags[-1][1][0])
        e = len(frames) - 1
        while frames[e][0] >= endtime and e > 0:
            e -= 1
        if e > 0:
            del frames[e+1:]
    
    if len(tags) > 1 and tags[0][0] in (SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value) and tags[0][1][0] <= 5:
        b = 0
        while frames[b][0] < tags[0][1][1]:
            b += 1
        if b:
            del frames[:b]

    if len(frames) < (WINDOW_BEFORE + WINDOW_AFTER) * 2 * round(RATE):
        return None
    
    # ok now we can numpy....
    frames = np.array(frames, dtype='float32')

    if not have_logo:
        frames[..., 1] = 0
    
    # normalize frame rate
    frames = condense(frames, round(frame_rate/RATE))

    # add a column for time percentage
    time_perc = frames[...,0]/endtime
    frames = np.append(frames, time_perc[:,np.newaxis], axis=1)
    time_perc = None

    # copy the real timestamps
    timestamps = frames[...,0].tolist()
    
    # normalize the timestamps upto 30 minutes
    frames[...,0] = (frames[...,0] % 1800.0) / 1800.0

    # normalize frame diffs to ~30
    frames[...,3] = np.clip(frames[...,3] / 30, 0, 1.0)

    # +/- 1s is all frames, plus the WINDOW before/after which is condensed to SUMMARY_RATE 
    rate = round(RATE)
    summary = round(RATE/SUMMARY_RATE)
    wbefore = round(WINDOW_BEFORE * SUMMARY_RATE)
    wafter = round(WINDOW_AFTER * SUMMARY_RATE)
    data = []

    # we do not use numpy for these because there is MUCH duplication, so we save memory by using pylists.
    # alternate is np.repeat, which would make new arrays for each iteration below.
    condensed = condense(frames, summary).tolist()
    frames = frames.tolist()

    i = 0
    while True:
        ci = i//summary
        if ci > wbefore: 
            break
        data.append(
            condensed[0:1] * (wbefore - ci) + 
            condensed[0 : ci] +
            frames[0:1] * max(0, rate-i) +
            frames[max(0, i-rate) : i+rate+1] +
            condensed[ci + 1 : ci + wafter + 1]
        )
        #if len(data[-1]) != len(data[0]): raise Exception(f"{i}, {np.array(data[0]).shape} {np.array(data[-1]).shape}")
        i += 1
    
    while i < len(frames) - (WINDOW_AFTER+1)*rate:
        ci = i//summary
        data.append(
            condensed[ci - (wbefore + 1) + 1 : ci] +
            frames[i-rate : i+rate+1] +
            condensed[ci+1 : ci + wafter + 1]
        )
        #if len(data[-1]) != len(data[0]): raise Exception(f"{i}, {np.array(data[0]).shape} {np.array(data[-1]).shape}")
        i += 1
    
    while i < len(frames):
        ci = i//summary
        data.append(
            condensed[ci - (wbefore + 1) + 1 : ci] +
            frames[i-rate : i+rate+1] +
            frames[-1:] * max(0, i + rate + 1 - len(frames)) +
            condensed[ci + 1 : ci + wafter + 1] +
            condensed[-1:] * max(0,ci + wafter + 1 - len(condensed))
        )
        #if len(data[-1]) != len(data[0]): raise Exception(f"{i}, {np.array(data[0]).shape} {np.array(data[-1]).shape}")
        i += 1

    # convert tags to a one-hot per-frame
    answers = []
    prev = SceneType.DO_NOT_USE.value
    weights = [1.0] * len(timestamps)
    tit = iter(tags)
    tag = next(tit, (SceneType.UNKNOWN, (0, endtime+9999)))
    for ts in timestamps:
        i = len(answers)
        if ts >= endtime:
            del timestamps[i:]
            del data[i:]
            del weights[i:]
            break

        while ts >= tag[1][1]:
            tag = next(tit, (SceneType.UNKNOWN, (0, endtime+9999)))
        tt = tag[0] if ts >= tag[1][0] else SceneType.UNKNOWN
        if type(tt) is not int: tt = tt.value
        
        if tt == SceneType.DO_NOT_USE.value:
            weights[i] = 0
        elif prev != tt and prev != SceneType.DO_NOT_USE.value:
            #for x in range(max(0,i-round(WINDOW_BEFORE*RATE)), min(i+1+round(WINDOW_AFTER*RATE),len(weights))):
            #    if weights[x] > 0:
            #        weights[x] = 1.5 if tt in [SceneType.COMMERCIAL.value, SceneType.SHOW.value] else 1.2
            for x in range(max(0,i-round(RATE)), min(i+1+round(RATE),len(weights))):
                if weights[x] > 0:
                    weights[x] = 2.0
        
        prev = tt
        answers.append(SceneType_one_hot[tt])
    
    assert(len(timestamps) == len(data))
    assert(len(data) == len(answers))
    assert(len(answers) == len(weights))

    return (timestamps,data,answers,weights)

def load_data(opts, do_not_test=False)->tuple[list,list,list,list,list]:
    data = opts.ml_data
    if not data:
        return None
    
    test = []

    if 'TEST' in data:
        i = data.index('TEST')
        test = data[i+1:]
        data = data[:i]

        i = 0
        while i < len(data):
            if not os.path.exists(data[i]):
                print(data[i], "does not exist!!")
                del data[i]
            elif data[i] in test or not os.path.isfile(data[i]):
                del data[i]
            else:
                i += 1

    x = []
    y = []
    w = []
    for f in data:
        if os.path.isdir(f):
            continue
        print("Loading",f)
        if flog := processor.read_feature_log(f):
            (_,a,b,c) = flog_to_vecs(flog,True)
            x += a
            y += b
            w += c
    
    xt = []
    yt = []
    for f in test:
        if os.path.isdir(f):
            continue
        print("Loading test",f)
        if flog := processor.read_feature_log(f):
            (_,a,b,_) = flog_to_vecs(flog,True)
            xt += a
            yt += b
    
    a = None
    b = None
    c = None
    flog = None
    
    z = list(zip(x,y,w))
    from random import seed, shuffle
    seed(time.time())
    shuffle(z)

    need = int(len(x)*TEST_PERC+1) - len(xt)
    if need > len(x)/100 and not do_not_test:
        print('Need to move',need,'datums to the test/eval set')
        (x,y,w) = zip(*z[need:])
        if len(xt) == 0:
            (xt,yt,_) = zip(*z[:need])
        else:
            (a,b,_) = zip(*z[:need])
            xt += a
            yt += b
    else:
        (x,y,w) = zip(*z)
    
    #x = np.array(x, dtype='float32')
    #y = np.array(y, dtype='float32')
    #w = np.array(w, dtype='float32')
    #xt = np.array(xt, dtype='float32')
    #yt = np.array(yt, dtype='float32')
    
    a = None
    b = None
    c = None
    flog = None
    gc.collect()

    return (x,y,w,xt,yt)

from keras.utils import Sequence
class DataGenerator(Sequence):
    def __init__(self, data, answers=None, weights=None):
        self.data = data
        self.answers = answers
        self.weights = weights
        self.len = ceil(len(self.data) / BATCH_SIZE)
        self.shape = (len(data[0]), len(data[0][0]))
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        #gc.collect()
        index *= BATCH_SIZE
        d = np.array(self.data[index:index+BATCH_SIZE], dtype='float32')
        if self.answers:
            a = np.array(self.answers[index:index+BATCH_SIZE], dtype='float32')
            if self.weights:
                w = np.array(self.weights[index:index+BATCH_SIZE], dtype='float32')
                return d,a,w
            else:
                return d,a
        else:
            return d

def train(opts:Any=None):
    # yield CPU time to useful tasks, this is a background thing...
    try: os.nice(19)
    except: pass

    (data,answers,sample_weights,test_data,test_answers) = load_data(opts)
    gc.collect()
    
    #print('Calculating loss weights')
    #sums = np.sum((np.sum(answers, axis=0), np.sum(test_answers, axis=0)), axis=0)
    ##weights = [1] * len(sums)
    ##weights[SceneType.SHOW.value] = sums[SceneType.COMMERCIAL.value] / sums[SceneType.SHOW.value]
    ##weights[SceneType.COMMERCIAL.value] = sums[SceneType.SHOW.value] / sums[SceneType.COMMERCIAL.value]
    #sums += 1
    #weights = (np.sum(sums)-sums)/np.sum(sums)
    #print("Loss Weights",weights)
    
    nsteps = len(data[0])
    nfeat = len(data[0][0])
    print("Data (x):",len(data)," Test (y):", len(test_data), "; Samples=",nsteps,"; Features=",nfeat)

    #train_dataset = tf.data.Dataset.from_tensor_slices((data, answers)).batch(BATCH_SIZE)
    train_dataset = DataGenerator(data, answers, sample_weights)
    data = None
    answers = None
    gc.collect()

    #test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_answers)).batch(BATCH_SIZE)
    test_dataset = DataGenerator(test_data, test_answers)
    have_test = test_answers is not None and len(test_answers) > 0
    test_data = None
    test_answers = None
    gc.collect()

    tfile = tempfile.NamedTemporaryFile(prefix='train-', suffix='.pycf.model.h5', )
    model_path = tfile.name
    
    stop = False
    epoch = 0

    #model_path = '/tmp/blah.h5'
    #epoch = 19

    if True:
        _train_some(model_path, train_dataset, test_dataset, epoch)
    else:
      while not stop:
        queue = Queue(1)
        sub = Process(target=_train_proc, args=(model_path, train_dataset, test_dataset, epoch, queue))
        sub.start()

        while sub.is_alive():
            try:
                epoch,stop = queue.get(timeout=0.1)
            except QueueEmpty:
                pass
            except KeyboardInterrupt:
                stop = True
                os.kill(sub.pid, 2)
        
        sub.join()
    
    print()
    print("Done")
    print()
    print('Final Evaluation...')

    import keras
    #dmetrics = model.evaluate(train_dataset, verbose=0)
    tmetrics = keras.models.load_model(model_path).evaluate(test_dataset, verbose=1)
    print()
    #print(dmetrics)
    print(tmetrics)

    if tmetrics[1] >= 0.80:
        name = f'{opts.models_dir if opts and opts.models_dir else "."}{os.sep}pycf-{tmetrics[1]:.04f}-{RNN}{UNITS}-d{DROPOUT}-w{WINDOW_BEFORE}x{WINDOW_AFTER}x{SUMMARY_RATE}-{int(time.time())}.h5'
        print()
        print('Saving as ' + name)

        import shutil
        shutil.copy(model_path, name)
        try: os.chmod(name, 0o644)
        except: pass
    
    print()

    return 0

def _train_proc(model_path, train_dataset, test_dataset, epoch, queue):
    queue.put( _train_some(model_path, train_dataset, test_dataset, epoch) )

def _train_some(model_path, train_dataset, test_dataset, epoch=0) -> tuple[int,bool]:
    import signal
    import keras
    from keras import layers, callbacks

    model:keras.models.Model = None
    if epoch > 0:
        model = keras.models.load_model(model_path)
    else:
        inputs = keras.Input(shape=train_dataset.shape, dtype='float32', name="input")
        n = inputs
        n = layers.TimeDistributed(layers.Dense(32, dtype='float32', activation='tanh'), name="dense-pre")(n)
        #skip = n
        #n = layers.TimeDistributed(layers.Dropout(DROPOUT), name="early-dropout", dtype='float32')(n)
        #n = layers.TimeDistributed(layers.Dense(32, dtype='float32', activation='relu'), name="dense-pre-2")(n)
        #n = layers.TimeDistributed(layers.Dropout(DROPOUT), name="early-dropout-maybe", dtype='float32')(n)
        #n = layers.TimeDistributed(layers.Dense(16, dtype='float32', activation='relu'), name="dense-pre-3")(n)
        #n = layers.Add()([n,skip])
        if RNN.lower() == "gru":
            n = layers.Bidirectional(layers.GRU(UNITS, dropout=DROPOUT, dtype='float32'), name="rnn")(n)
        elif RNN.lower() == 'lstm':
            n = layers.Bidirectional(layers.LSTM(UNITS, dropout=DROPOUT, dtype='float32'), name="rnn")(n)
            #n = layers.TimeDistributed(layers.Dense(UNITS, dtype='float32', activation='tanh'), name="dense-mid")(n)
            #n = layers.Bidirectional(layers.LSTM(UNITS, dropout=DROPOUT, dtype='float32'), name="MORE-rnn")(n)
        elif RNN.lower() == "c1d":
            n = layers.Conv1D(filters=32, 
                                kernel_size=5,
                                activation='relu',
                                padding='same',
                                name="conv1d_1")(n)
            n = layers.Conv1D(filters=32,
                                kernel_size=3,
                                activation='relu',
                                padding='same',
                                name="conv1d_2")(n)
            n = layers.GlobalAveragePooling1D(name="global_pool")(n)
        
        n = layers.Dense(32, dtype='float32', activation='relu', name="dense-post")(n)
        n = layers.Dense(16, dtype='float32', activation='relu', name="final")(n)
        outputs = layers.Dense(SceneType.count(), dtype='float32', activation='softmax', name="output")(n)
        model = keras.Model(inputs, outputs)
        model.summary()
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['categorical_accuracy', 'categorical_crossentropy'])
        model.save(model_path)

    cb = []

    cb.append(keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=10))
    cb.append(keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=10))
    
    class EpochCheckpoint(callbacks.ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            self.last_epoch = epoch
            return super().on_epoch_end(epoch, logs)

    ecp = EpochCheckpoint(model_path, verbose=1, monitor='val_categorical_accuracy', mode='auto', save_best_only=True)
    cb.append(ecp)

    class MemoryChecker(callbacks.Callback):
        def __init__(self, *args):
            super().__init__(*args)
            self.start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.exceeded = False

        def on_epoch_end(self, epoch, logs=None):
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            change = rss - self.start_rss
            if change > 6000000:
                print(f'Now using too much memory ({rss//1024}MB)! {change//1024}MB more than at start which was {self.start_rss//1024}MB')
                self.model.stop_training = True
                self.exceeded = True

    #cb.append(MemoryChecker())
    
    def handler(signum, frame):
        print("\nStopping (gracefully)...\n")
        model.stop_training = True
        signal.signal(signal.SIGINT, oldsint)
        signal.signal(signal.SIGTERM, oldterm)
        return
    oldsint = signal.signal(signal.SIGINT, handler)
    oldterm = signal.signal(signal.SIGTERM, handler)

    if not model.stop_training:
        model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, initial_epoch=epoch, callbacks=cb)

    #model.save(model_path) the checkpoint already saved the vest version
    return (ecp.last_epoch+1, model.stop_training or ecp.last_epoch+1 >= EPOCHS)

def predict(feature_log:str|TextIO|dict, opts:Any, write_log=None)->list:
    from .mythtv import set_job_status
    set_job_status(opts, "Inferencing...")

    import tensorflow as tf
    import keras

    flog = processor.read_feature_log(feature_log)
    duration = flog.get('duration', 0)
    frame_rate = flog.get('frame_rate', 29.97)
    
    assert(flog['frames'][-1][0] > frame_rate)

    (times,data,answers,_) = flog_to_vecs(flog, clean=False)
    assert(len(times) == len(data))

    mf = opts.model_file
    if not mf and opts:
        mf = f'{opts.models_dir or "."}{os.sep}model.h5'
    if not os.path.exists(mf):
        raise Exception(f"Model file '{mf}' does not exist")
    model:keras.models.Model = keras.models.load_model(mf)

    prediction = model.predict(np.array(data, dtype='float32'), verbose=True, batch_size=BATCH_SIZE)

    result = np.argmax(prediction, axis=1)
    results = [(0,(0,0))]
    for i in range(len(result)):
        when = times[i]
        ans = int(result[i])
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

    log.debug(f'Raw result n={len(results)}')

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
    
    results = _adjust_tags(results, spans.get('blank', []), spans.get('diff', []), duration)
    i = 1
    while i < len(results):
        if results[i][0] == results[i-1][0] and (results[i][1][0] - results[i-1][1][1]) < opts.show_min_len:
            results[i-1] = (results[i][0], (results[i-1][1][0], results[i][1][1]))
            del results[i]
        else:
            i += 1

    log.debug(f'Merge/Adjust n={len(results)}: {str(results)}')

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

    if orig_tags := flog.get('tags', []):
        log.debug(f'OLD tags n={len(orig_tags)} -> {str(orig_tags)}')
        log.debug(f'NEW tags n={len(results)} -> {str(results)}')
    else:
        log.debug(f'Final tags n={len(results)}: {str(results)}')

    flog['tags'] = results

    if write_log is not None:
        processor.write_feature_log(flog, write_log)

    return results

def eval_many(opts:Any):
    print("EVALUATE many", len(opts.eval))

    import tensorflow as tf
    import keras

    for f in opts.ml_data:
        if os.path.isdir(f):
            continue
        
        dataset = []
        gc.collect()

        try:
            if flog := processor.read_feature_log(f):
                (_,a,b,_) = flog_to_vecs(flog,True)
                dataset = DataGenerator(a,b)
            else:
                print('Load failed')
                continue
        except Exception as e:
            print(str(e))
            continue
        
        for mf in opts.eval:
            try:
                print(f,mf,keras.models.load_model(mf).evaluate(dataset))
            except Exception as e:
                print(f,mf,str(e))
        print()
    
    print()
    print("Done")

def eval(opts:Any):
    # yield CPU time to useful tasks, this is a background thing...
    try: os.nice(19)
    except: pass
    
    if len(opts.eval) > 1:
        return eval_many(opts)
    
    print("EVALUATE single")
    
    import tensorflow as tf
    import keras

    model:keras.models.Model = keras.models.load_model(opts.eval[0])
    
    for f in opts.ml_data:
        if os.path.isdir(f):
            continue
        gc.collect()
        print(f)
        try:
            if flog := processor.read_feature_log(f):
                (_,a,b,_) = flog_to_vecs(flog,True)
                print(model.evaluate(DataGenerator(a,b), verbose=0))
            else:
                print('Load failed')
        except Exception as e:
            print(str(e))
    
    print()
    print("Done")
