import logging as log
import gc
import math
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

WINDOW_BEFORE = 0.5
WINDOW_AFTER = 2.5
UNITS = 64
DROPOUT = .1
EPOCHS = 15

# clean up tags so they start/end exactly in a nearby blank block
# this is because the training data is supplied by humans, and might be off a bit
def _adjust_tags(tags:list,blanks:list,duration:float):
    for ti in range(len(tags)):
        best = [(duration,)]*4
        (tt, (tb, te)) = tags[ti]
        
        for bi in range(len(blanks)):
            (bv,(bs,be)) = blanks[bi]
            if not bv or bs == 0.0 or be >= duration or (be-bs) > 5.0:
                continue
            n = 0
            for t in (tb,te):
                for b in (bs,be):
                    d = abs(t - b)
                    if best[n][0] > d: best[n] = (d,n,bi)
                    n += 1
        
        for (d,n,bi) in best:
            if d < 3:
                (bv,(bs,be)) = blanks[bi]
                if n == 0 or n == 1:
                    tags[ti] = (tt, (be,te))
                    #print(f'Adjust start of {ti} from {tb} to {be}')
                else:
                    tags[ti] = (tt, (tb,bs))
                    #print(f'Adjust end of {ti} from {te} to {bs}')

def flog_to_vecs(flog:dict)->tuple[list[float], list[list[float]], list[list[float]]]:
    version = flog.get('file_version', 10)
    frame_rate = flog.get('frame_rate', 29.97)
    endtime = flog.get('duration', 0)

    tags = flog.get('tags', [])
    
    if tags:
        blanks = processor.read_feature_spans(flog, 'blank').get('blank', [])
        if blanks:
            _adjust_tags(tags,blanks,endtime)
    
    frames_header = flog['frames_header']
    assert('time' in frames_header[0])
    assert('diff' in frames_header[3])

    frames = flog['frames']
    if frames[0] is None: 
        frames = flog['frames'][1:]

    # everything we do is for 29.97 or 30 (close enough); support 59.94 and 60 too... 
    if frame_rate >= 48:
        assert(frame_rate < 70) # sanity
        old = frames
        frames = []
        for i in range(1, len(old), 2):
            f = old[i]+[old[i][0]/endtime]
            # audio features are spread across time anyway so we can omit individual frames without much worry
            # but these video features can be important in a single frame, so make sure we capture that before dropping 
            f[1] = max(f[1], old[i-1][1]) # logo
            f[2] = max(f[1], old[i-1][1]) # blank
            f[3] = max(f[1], old[i-1][1]) # diff
            frames.append(f)
    else:
        # theres a way to do this in numpy with concatenate, does it matter?
        old = frames
        frames = []
        for i in range(len(old)):
            frames.append(old[i]+[old[i][0]/endtime])
    
    # ok now we can numpy....
    frames = np.array(frames)

    # copy the real timestamps
    timestamps = frames[...,0].tolist()

    # normalize the timestamps upto 30 minutes
    frames[...,0] = (frames[...,0] % 1800.0) / 1800.0

    # normalize frame diffs
    frames[...,3] = np.clip(frames[...,3] / 25, 0, 1.0)

    before = int(WINDOW_BEFORE * 29.97) + 1
    after = int(WINDOW_AFTER * 29.97) + 1
    
    if len(frames) < (before+1+after)*3:
        return ([],[],[])
    
    # there is a 59x duplication of the same data across sample sets here so BE CAREFUL with slices
    data = []
    i = 0

    # the leading frames can't use slices because the first frame repeats (for padding)
    while i <= before:
        d = ([frames[0]] * (before - i)) + frames[0 : 1 + i + after].tolist()
        data.append(np.array(d))
        #print(len(data[-1]), data[-1][0],data[-1][-1])
        i += 1

    # the majority of frames use a simple numpy slice        
    while i < (len(frames)-(after+1)):
        data.append(frames[i - before : 1 + i + after])
        #print(len(data[-1]), "!",data[-1][0],data[-1][-1])
        i += 1
    
    # the trailing frames can't use slices because the last frame repeats (for padding)
    while i < len(frames):
        d = frames[i - before : 1 + i + after].tolist()
        d += [frames[-1]] * (before+1+after - len(d))
        data.append(np.array(d))
        #print(len(data[-1]), "?",data[-1][0],data[-1][-1], "@", i, len(data[-1]))
        i += 1

    # convert tags to a one-hot per-frame
    bad = []
    answers = []
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

        x = [0] * SceneType.count()
        x[tt if type(tt) is int else tt.value] = 1
        answers.append(np.array(x))

    for i in reversed(bad):
        assert(answers[i] is None)
        del timestamps[i]
        del data[i]
        del answers[i]
    
    assert(len(timestamps) == len(data))
    assert(len(data) == len(answers))

    return (timestamps,data,answers)

def load_data(opts)->tuple[list,list,list,list]:
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
    for f in data:
        if os.path.isdir(f):
            continue
        print("Loading",f)
        if flog := processor.read_feature_log(f):
            (_,a,b) = flog_to_vecs(flog)
            x += a
            y += b
            flog = None
            a = None
            b = None
        gc.collect()
    
    xt = []
    yt = []
    for f in test:
        if os.path.isdir(f):
            continue
        print("Loading",f)
        if flog := processor.read_feature_log(f):
            (_,a,b) = flog_to_vecs(flog)
            xt += a
            yt += b
            flog = None
            a = None
            b = None
        gc.collect()
    
    import random
    random.seed(time.time())
    need = len(x)//4 - len(xt)
    if need > len(x)/100:
        print('Need to move',need,'datum to the test/eval set')
        z = list(zip(x,y))
        n=x=y=None
        random.shuffle(z)
        (x,y) = zip(*z[need:])
        if len(xt) == 0:
            (xt,yt) = zip(*z[:need])
        else:
            (a,b) = zip(*z[:need])
            xt += a
            yt += b
    
    #x = np.array(x)
    #y = np.array(y)
    #xt = np.array(xt)
    #yt = np.array(yt)

    return (x,y,xt,yt)

class DataGenerator(Sequence):
    def __init__(self, data, answers, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.answers = answers
        self.len = math.ceil(len(self.answers) / self.batch_size)
        self.shape = (len(data[0]), len(data[0][0]))
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        index *= self.batch_size
        return np.array(self.data[index:index+self.batch_size]), np.array(self.answers[index:index+self.batch_size])

def train(opts:Any=None):
    # yield CPU time to useful tasks, this is a background thing...
    
    (data,answers,test_data,test_answers) = load_data(opts)
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

    #train_dataset = tf.data.Dataset.from_tensor_slices((data, answers)).batch(batch_size)
    train_dataset = DataGenerator(data, answers, opts.tf_batch_size)
    data = None
    answers = None
    gc.collect()

    #test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_answers)).batch(batch_size)
    test_dataset = DataGenerator(test_data, test_answers, opts.tf_batch_size)
    have_test = test_answers is not None and len(test_answers) > 0
    test_data = None
    test_answers = None
    gc.collect()

    tfile = tempfile.NamedTemporaryFile(prefix='train-', suffix='.pycf.model.h5', )
    model_path = tfile.name
    
    stop = False
    epoch = 0
    while not stop:
        queue = Queue(1)
        sub = Process(target=_train_some, args=(model_path, train_dataset, test_dataset, epoch, queue))
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
        name = f'{opts.models_dir if opts and opts.models_dir else "."}{os.sep}pycf-{tmetrics[1]:.04f}-lstm{UNITS}-d{DROPOUT}-w{WINDOW_BEFORE}x{WINDOW_AFTER}-{int(time.time())}.h5'
        print()
        print('Saving as ' + name)

        import shutil
        shutil.copy(model_path, name)
    
    print()

    return 0

def _train_some(model_path, train_dataset, test_dataset, epoch, queue) -> tuple[int,bool]:
    import signal
    import keras
    from keras import layers, callbacks
    
    os.nice(10) # lower priority in case of other tasks on this server

    model:keras.models.Model = None
    if epoch > 0:
        model = keras.models.load_model(model_path)
    else:
        inputs = keras.Input(shape=train_dataset.shape)
        n = inputs
        #n = layers.TimeDistributed(layers.Dropout(DROPOUT))(n)
        n = layers.TimeDistributed(layers.Dense(32, activation='tanh'))(n)
        n = layers.LSTM(UNITS, dropout=DROPOUT)(n)
        #n = layers.LSTM(UNITS//2, dropout=DROPOUT, return_sequences=True)(n)
        #n = layers.LSTM(UNITS//2)(n)
        n = layers.Dense(64, activation='relu')(n)
        n = layers.Dense(32, activation='relu')(n)
        outputs = layers.Dense(SceneType.count(), activation='softmax')(n)
        model = keras.Model(inputs, outputs)
        model.summary()
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['categorical_accuracy', 'mean_squared_error'])
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
            if change > 7000000:
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

    class_weights = {
        SceneType.SHOW.value : 0.75,
        SceneType.COMMERCIAL.value : 1.5,
        SceneType.CREDITS.value : 2,
        SceneType.INTRO.value : 2,
        SceneType.TRANSITION.value : 0, # unused?
    }

    model.fit(train_dataset, epochs=EPOCHS, initial_epoch=epoch, shuffle=True, callbacks=cb, validation_data=test_dataset, class_weight=class_weights)

    #model.save(model_path) the checkpoint already saved the vest version
    queue.put( (ecp.last_epoch+1, ecp.last_epoch+1 >= EPOCHS) )
    return

def predict(feature_log:str|TextIO|dict, opts:Any)->list:
    import tensorflow as tf
    import keras

    flog = processor.read_feature_log(feature_log)
    duration = flog.get('duration', 0)
    frame_rate = flog.get('frame_rate', 29.97)
    spans = processor.read_feature_spans(flog, 'diff', 'blank')
    
    assert(flog['frames'][-1][0] > frame_rate)

    (times,data,answers) = flog_to_vecs(flog)
    assert(len(times) == len(data))

    mf = opts.model_file
    if not mf and opts:
        mf = f'{opts.models_dir or "."}{os.sep}model.h5'
    if not os.path.exists(mf):
        raise Exception(f"Model file '{mf}' does not exist")
    model:keras.models.Model = keras.models.load_model(mf)

    result = model.predict(np.array(data), verbose=True)

    result = np.argmax(result, axis=1)
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
                    results[i+1][1] = (nextstart, results[i+1][1][1])
                    break

        # its ok now, move on
        i += 1
    
    print('Post filter n=', len(results))
    print(results)

    _adjust_tags(results, spans.get('blank', []), duration)

    print(f'\n\nFinal n={len(results)}:')
    print(results)

    flog['tags'] = results
    processor.write_feature_log(flog, feature_log)

    return results
