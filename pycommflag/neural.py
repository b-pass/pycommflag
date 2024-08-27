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
from typing import Any,TextIO,BinaryIO

import numpy as np
from keras.utils import Sequence

from .feature_span import *
from . import processor

WINDOW_BEFORE = 15
WINDOW_AFTER = 15
SUMMARY_RATE = 2

RATE = 29.97
RNN = 'lstm'
UNITS = 32
DROPOUT = 0.2
EPOCHS = 35
BATCH_SIZE = 1000

# clean up tags so they start/end exactly in a nearby blank block
# this is because the training data is supplied by humans, and might be off a bit
def _adjust_tags(tags:list,blanks:list,diffs:list,duration:float):
    filt = []
    for ti in range(len(tags)):
        (tt, (tb, te)) = tags[ti]
        tt = int(tt)

        bbest = (duration,)
        ebest = (duration,)
        left = bisect_left(blanks, tb-30, 0, len(blanks), key=lambda b: b[1][0])
        for bi in range(len(blanks)):
            (bv,(bs,be)) = blanks[bi]
            if not bv or bs < 1.0 or be < tb-10:
                continue
            if bs > te+10 or be >= duration-1:
                break
            
            for b in (bs,be):
                d = abs(tb - b)
                if bbest[0] > d: bbest = (d,bi)
            
                d = abs(te - b)
                if ebest[0] > d: ebest = (d,bi)
        
        maxdist = 10 if tt in [SceneType.SHOW.value, SceneType.COMMERCIAL.value] else 2
        
        if bbest[0] < maxdist and bbest[0] > 0:
            tags[ti] = (tt,(blanks[bbest[1]][1][1],te))
            #print(f'Adjust start of {ti} (type {tt}) from {tb} to blank at {tags[ti][1][0]}')
        if ebest[0] < maxdist and ebest[0] > 0:
            tags[ti] = (tt,(tags[ti][1][0],blanks[ebest[1]][1][0]))
            #print(f'Adjust end of {ti} (type {tt}) from {te} to blank at {tags[ti][1][1]}')
        
        bbest = (duration,)
        ebest = (duration,)
        left = bisect_left(diffs, tb-30, 0, len(diffs), key=lambda d: d[1][0])
        for di in range(left, len(diffs)):
            (dv, (db, de)) = diffs[di]
            if not dv or db < 1.0 or de < tb-10:
                continue
            if db > te+10 or de >= duration-1:
                break
            
            for b in (db,de):
                d = abs(tb - b)
                if bbest[0] > d: bbest = (d,di)
            
                d = abs(te - b)
                if ebest[0] > d: ebest = (d,di)
        
        if bbest[0] < 1 and bbest[0] > 0:
            tags[ti] = (tt,(diffs[bbest[1]][1][1],te))
            #print(f'Adjust start of {ti} (type {tt}) from {tb} to diff at {tags[ti][1][0]}')
        if ebest[0] < 1 and ebest[0] > 0:
            tags[ti] = (tt,(tags[ti][1][0],diffs[ebest[1]][1][0]))
            #print(f'Adjust end of {ti} (type {tt}) from {te} to diff at {tags[ti][1][1]}')
                
        if tags[ti][1][0] >= tags[ti][1][1]:
            filt.append(ti)
    
    for x in reversed(filt):
        #print('Tag evaporated:',tags[x])
        del tags[x]
    
    return tags

def condense(frames, step):
    if step > 1:
        # audio features are spread across time anyway so we can omit individual frames without much worry
        # but these video features can be important in a single frame, so make sure we capture that before dropping 
        old = frames
        frames = old[::step]
        if len(frames) == 1:
            frames[0][1] = np.average(old[:,1]) # logo
            frames[0][2] = np.average(old[:,2]) # blank
            frames[0][3] = np.count_nonzero(old[:,3] > 15) / len(old) # diff
        else:
            for i in range(len(frames)):
                frames[i][1] = np.average(old[i*step : (i+1)*step,1]) # logo
                frames[i][2] = np.average(old[i*step : (i+1)*step,2]) # blank
                frames[i][3] = np.count_nonzero(old[i*step : (i+1)*step,3] > 15) / step # diff
        old = None
    return frames

def flog_to_vecs(flog:dict, fitlerForTraining=False)->tuple[list[float], list[list[float]], list[float], list[float]]:
    version = flog.get('file_version', 10)
    frame_rate = flog.get('frame_rate', 29.97)
    endtime = flog.get('duration', 0)

    tags = flog.get('tags', [])
    
    if tags and fitlerForTraining:
        spans = processor.read_feature_spans(flog, 'blank', 'diff')
        tags = _adjust_tags(tags,spans.get('blank', []),spans.get('diff', []),endtime)
        ti = 0
        while ti < len(tags):
            if tags[ti][0] != SceneType.DO_NOT_USE and tags[ti][0] != SceneType.DO_NOT_USE.value:
                if tags[ti][1][1] - tags[ti][1][0] < 5:
                    del tags[ti]
                    continue
            ti += 1 
    
    frames_header = flog['frames_header']
    assert('time' in frames_header[0])
    assert('diff' in frames_header[3])

    frames = flog['frames']
    if frames[0] is None: 
        frames = flog['frames'][1:]

    # ok now we can numpy....
    frames = np.array(frames)
    
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

    # normalize frame diffs
    frames[...,3] = np.clip(frames[...,3] / 30, 0, 1.0)

    # +/- 1s is all frames, plus the WINDOW before/after which is condensed to SUMMARY_RATE 
    rate = round(RATE)
    summary = round(RATE/SUMMARY_RATE)
    wbefore = round(WINDOW_BEFORE * SUMMARY_RATE)
    wafter = round(WINDOW_AFTER * SUMMARY_RATE)
    data = []

    if len(frames) < (WINDOW_BEFORE + WINDOW_AFTER) * 2 * rate:
        return None
    
    # we do not use numpy for these because there is MUCH duplication, so we save memory by using pylists.
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
    bad = []
    answers = []
    prev = SceneType.COMMERCIAL
    weights = [1.0] * len(timestamps)
    tit = iter(tags)
    tag = next(tit, (SceneType.UNKNOWN, (0, endtime+1)))
    for ts in timestamps:
        while ts >= tag[1][1]:
            tag = next(tit, (SceneType.UNKNOWN, (0, endtime+1)))
        tt = tag[0] if ts >= tag[1][0] else SceneType.UNKNOWN
        if type(tt) is not int: tt = tt.value
        #if tt != SceneType.COMMERCIAL.value and tt != SceneType.SHOW.value:
        if tt == SceneType.DO_NOT_USE.value:
            bad.append(len(answers))
            answers.append(None)
            continue

        if fitlerForTraining and prev == tt and frames[len(answers)][2] <= 0 and frames[len(answers)][3] <= 10:
            bad.append(len(answers))
            answers.append(None)
            continue

        if prev != tt:
            prev = tt
            i = len(answers)
            for x in range(max(0,i-round(RATE)), min(i+1+round(RATE),len(weights))):
                weights[x] = 2.0

        x = [0] * SceneType.count()
        x[tt if type(tt) is int else tt.value] = 1
        answers.append(np.array(x))

    for i in reversed(bad):
        assert(answers[i] is None)
        del timestamps[i]
        del data[i]
        del answers[i]
        del weights[i]
    
    assert(len(timestamps) == len(data))
    assert(len(data) == len(answers))
    assert(len(answers) == len(weights))

    return (timestamps,data,answers,weights)

def load_data(opts)->tuple[list,list,list,list,list]:
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
            (_,a,b,c) = flog_to_vecs(flog)
            x += a
            y += b
            w += c
            
            a = None
            b = None
            c = None
            flog = None
            gc.collect()
    
    xt = []
    yt = []
    for f in test:
        if os.path.isdir(f):
            continue
        print("Loading",f)
        if flog := processor.read_feature_log(f):
            (_,a,b,_) = flog_to_vecs(flog)
            xt += a
            yt += b
            
            a = None
            b = None
            flog = None
            gc.collect()
    
    import random
    random.seed(time.time())
    need = len(x)//4 - len(xt)
    if need > len(x)/100:
        print('Need to move',need,'datum to the test/eval set')
        z = list(zip(x,y,w))
        n=x=y=None
        random.shuffle(z)
        (x,y,w) = zip(*z[need:])
        if len(xt) == 0:
            (xt,yt,_) = zip(*z[:need])
        else:
            (a,b,_) = zip(*z[:need])
            xt += a
            yt += b
    
    #x = np.array(x)
    #y = np.array(y)
    #w = np.array(w)
    #xt = np.array(xt)
    #yt = np.array(yt)
    gc.collect()

    return (x,y,w,xt,yt)

class DataGenerator(Sequence):
    def __init__(self, data, answers, weights=None):
        self.data = data
        self.answers = answers
        self.weights = weights
        self.len = ceil(len(self.answers) / BATCH_SIZE)
        self.shape = (len(data[0]), len(data[0][0]))
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        #gc.collect()
        index *= BATCH_SIZE
        if self.weights is not None:
            return np.array(self.data[index:index+BATCH_SIZE]), np.array(self.answers[index:index+BATCH_SIZE]), np.array(self.weights[index:index+BATCH_SIZE])
        else:
            return np.array(self.data[index:index+BATCH_SIZE]), np.array(self.answers[index:index+BATCH_SIZE])

def train(opts:Any=None):
    # yield CPU time to useful tasks, this is a background thing...
    
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

    #model_path = '/tmp/train-3koyrut2.pycf.model.h5'
    #epoch = 24

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
    
    print()

    return 0

def _train_proc(model_path, train_dataset, test_dataset, epoch, queue):
    queue.put( _train_some(model_path, train_dataset, test_dataset, epoch) )

def _train_some(model_path, train_dataset, test_dataset, epoch=0) -> tuple[int,bool]:
    import signal
    import keras
    from keras import layers, callbacks
    
    os.nice(10) # lower priority in case of other tasks on this server

    model:keras.models.Model = None
    if epoch > 0:
        model = keras.models.load_model(model_path)
    else:
        inputs = keras.Input(shape=train_dataset.shape, name="input")
        n = inputs
        #n = layers.TimeDistributed(layers.Dropout(DROPOUT), name="dropout")(n)
        n = layers.TimeDistributed(layers.Dense(32, activation='tanh'), name="dense-pre")(n)
        #n = layers.LSTM(UNITS, dropout=DROPOUT, name="rnn")(n)
        if RNN.lower() == "gru":
            n = layers.Bidirectional(layers.GRU(UNITS, dropout=DROPOUT), name="rnn")(n)
        else:
            n = layers.Bidirectional(layers.LSTM(UNITS, dropout=DROPOUT), name="rnn")(n)
        n = layers.Dense(UNITS//2, activation='relu', name="dense-post")(n)
        n = layers.Dense(16, activation='relu', name="final")(n)
        outputs = layers.Dense(SceneType.count(), activation='softmax', name="output")(n)
        model = keras.Model(inputs, outputs)
        model.summary()
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['categorical_accuracy', 'categorical_crossentropy'])
        model.save(model_path)

    cb = []

    #cb.append(keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=50))
    #if have_test:
    #    cb.append(keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=50))
    
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

    cb.append(MemoryChecker())
    
    def handler(signum, frame):
        print("\nStopping (gracefully)...\n")
        model.stop_training = True
        signal.signal(signal.SIGINT, oldsint)
        signal.signal(signal.SIGTERM, oldterm)
        return
    oldsint = signal.signal(signal.SIGINT, handler)
    oldterm = signal.signal(signal.SIGTERM, handler)

    #class_weights = {
    #    SceneType.SHOW.value : 0.75,
    #    SceneType.COMMERCIAL.value : 1.5,
    #    SceneType.CREDITS.value : 2,
    #    SceneType.INTRO.value : 2,
    #    SceneType.TRANSITION.value : 0, # unused?
    #}
    class_weights = None

    model.fit(train_dataset, epochs=EPOCHS, initial_epoch=epoch, shuffle=True, callbacks=cb, validation_data=test_dataset, class_weight=class_weights)

    #model.save(model_path) the checkpoint already saved the vest version
    return (ecp.last_epoch+1, ecp.last_epoch+1 >= EPOCHS)

def predict(feature_log:str|TextIO|dict, opts:Any)->list:
    import tensorflow as tf
    import keras

    flog = processor.read_feature_log(feature_log)
    duration = flog.get('duration', 0)
    frame_rate = flog.get('frame_rate', 29.97)
    spans = processor.read_feature_spans(flog, 'diff', 'blank')
    
    assert(flog['frames'][-1][0] > frame_rate)

    (times,data,_,_) = flog_to_vecs(flog)
    assert(len(times) == len(data))

    mf = opts.model_file
    if not mf and opts:
        mf = f'{opts.models_dir or "."}{os.sep}model.h5'
    if not os.path.exists(mf):
        raise Exception(f"Model file '{mf}' does not exist")
    model:keras.models.Model = keras.models.load_model(mf)

    prediction = model.predict(np.array(data), verbose=True)

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

    # clean up tiny gaps between identified breaks (including true 0-length gaps)
    # show must be at least 30 seconds long (opts.show_min_len), or we just combine it into the commercial break its in the middle of
    i = 1
    while i < len(results):
        if results[i][0] == results[i-1][0] and (results[i][1][0] - results[i-1][1][1]) < opts.show_min_len:
            results[i-1] = (results[i][0], (results[i-1][1][0], results[i][1][1]))
            del results[i]
        else:
            i += 1

    print(f'\n\nMerged n={len(results)}')
    print(results)

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
    
    print(f'Post filter n={len(results)}')
    print(results)

    results = _adjust_tags(results, spans.get('blank', []), spans.get('diff', []), duration)

    print(f'\n\nFinal n={len(results)}:')
    print(results)

    flog['tags'] = results
    processor.write_feature_log(flog, feature_log)

    return results
