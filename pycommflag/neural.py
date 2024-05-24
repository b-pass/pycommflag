import logging as log
import os
import numpy as np
import re
import time
import sys
from typing import Any,TextIO,BinaryIO

from .feature_span import *
from . import processor

TIME_WINDOW = 2.0
UNITS = 128
DROPOUT = .1

def flog_to_vecs(flog:dict)->tuple[list[float], list[list[float]], list[list[float]]]:
    version = flog.get('file_version', 10)
    frame_rate = flog.get('frame_rate', 29.97)
    endtime = flog.get('duration', 0)

    tags = flog.get('tags', [])
    
    # clean up tags so they start/end exactly in the middle of the blank block
    # this is because the training data is supplied by humans, and might be off a bit
    # TODO, do we need to handle all-blank scenes at the start/end of a block differently?
    # the reason would be because of inconsistent training data (sometimes the blank is part of the commercial, sometimes the show)
    # but there are blanks in the middle of both also so maybe that doesnt matter....?
    if tags:
        for (bv,(bs,be)) in processor.read_feature_spans(flog, 'blank'):
            if not bv or bs == 0.0 or be >= endtime:
                continue
            half = bs + (be - bs)
            for ti in range(len(tags)):
                (tt, (tb, te)) = tags[ti]
                fixb = bs <= tb and tb < be and abs(tb-half) >= 1.5/frame_rate
                fixe = bs <= te and te < be and abs(te-half) >= 1.5/frame_rate
                if fixe and fixb:
                    pass
                elif fixb and not fixe:
                    tags[ti] = (tt, (half, te))
                elif fixe and not fixb:
                    tags[ti] = (tt, (tags[ti][1][0], half))
    
    # copy the frames so we don't mess up "flog"
    frames = list(flog['frames'])

    frames_header = flog['frames_header']
    assert('time' in frames_header[0])
    assert('diff' in frames_header[3])

    if frames[0] is None: frames.pop(0)

    # everything we do is for 29.97 or 30 (close enough); support 59.94 and 60 too... 
    if frame_rate >= 58:
        frames = frames[::int(frame_rate/29)]

    # often the video doesn't start a PTS 0 because of sync issues, back-fill the first video frame
    if frames[0][0] > 0:
        while True:
            frames[0:0] = [list(frames[0])] # insert
            frames[0][0] = round(frames[0][0] - 1/frame_rate, 5)
            if frames[0][0] <= 0.0:
                frames[0][0] = 0.0
                break

    # convert tags to a one-hot per-frame
    answers = []
    tit = iter(tags)
    tag = next(tit, (SceneType.UNKNOWN, (0, endtime+1)))
    for f in frames:
        while f[0] >= tag[1][1]:
            tag = next(tit, (SceneType.UNKNOWN, (0, endtime+1)))
        tt = tag[0] if f[0] >= tag[1][0] else SceneType.UNKNOWN
        if type(tt) is not int: tt = tt.value
        if tt == SceneType.DO_NOT_USE.value:
            # truncate right here
            if len(answers) < len(frames):
                frames = frames[:len(answers)]
            break
        x = [0] * SceneType.count()
        x[tt if type(tt) is int else tt.value] = 1
        answers.append(x)
    
    # ok now we can numpy....
    frames = np.copy(frames)

    # copy the real timestamps
    timestamps = np.copy(frames[...,0])

    # normalize the timestamps upto 30 minutes
    frames[...,0] = (frames[...,0] % 1800.0) / 1800.0

    # normalize frame diffs
    frames[...,3] = np.clip(frames[...,3] / 30, 0, 1.0)

    need = int((TIME_WINDOW * 29) / 2)
        
    if len(frames) < need*5:
        return ([],[],[])

    data = []
    d = np.concatenate((np.tile(frames[0], (need,1)),frames[0:need+1]))
    for i in range(len(frames)-(need+1)):
        d = np.append(d[1:], [frames[i+need+1]], axis=0)
        data.append(d)
    for i in range(len(frames)-(need+1), len(frames)):
        d = np.append(d[1:], [frames[-1]], axis=0)
        data.append(d)

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
        #print(f)
        flog = processor.read_feature_log(f)
        if flog:
            (_,a,b) = flog_to_vecs(flog)
            x += a
            y += b
    
    xt = []
    yt = []
    for f in test:
        if os.path.isdir(f):
            continue
        #print(f)
        flog = processor.read_feature_log(f)
        if flog:
            (_,a,b) = flog_to_vecs(flog)
            xt += a
            yt += b
    
    import random
    random.seed(time.time())
    need = int(len(x)/4) - len(xt)
    if need > (len(x)/100):
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

    return (np.copy(x),np.copy(y),np.copy(xt),np.copy(yt))

def train(training:tuple, test_answers:list=None, opts:Any=None):
    # imort these here because their imports are slow 
    import tensorflow as tf
    import keras
    from keras import layers

    (data,answers,test_data,test_answers) = training

    sums = np.sum((np.sum(answers, axis=0), np.sum(test_answers, axis=0)), axis=0)
    weights = [1] * len(sums)
    weights[SceneType.SHOW.value] = sums[SceneType.COMMERCIAL.value] / sums[SceneType.SHOW.value]
    weights[SceneType.COMMERCIAL.value] = sums[SceneType.SHOW.value] / sums[SceneType.COMMERCIAL.value]
    print("Loss Weights",weights)

    nsteps = len(data[0])
    nfeat = len(data[0][0])
    print("Data (x):",len(data)," Test (y):", len(test_data), "; Samples=",nsteps,"; Features=",nfeat)
    inputs = keras.Input(shape=(nsteps,nfeat))
    n = inputs
    n = layers.LSTM(UNITS, dropout=DROPOUT)(n)
    #n = layers.LSTM(32, dropout=DROPOUT, return_sequences=True)(n)
    #n = layers.LSTM(16)(n)
    outputs = layers.Dense(SceneType.count(), activation='softmax')(n)
    model = keras.Model(inputs, outputs)
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", loss_weights=weights, metrics=['categorical_accuracy', 'mean_squared_error'])

    batch_size = opts.tf_batch_size if opts else 500
    train_dataset = tf.data.Dataset.from_tensor_slices((data, answers)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_answers)).batch(batch_size)
    
    class GracefulStop(keras.callbacks.Callback):
        def __init__(self):
            super(keras.callbacks.Callback, self).__init__()
            self._stop = False
            def handler(signum, frame):
                self._stop = True
                print("\nStopping!\n")
            import signal
            signal.signal(signal.SIGINT, handler)

        def on_epoch_end(self, epoch, logs={}):
            if self._stop:
                print("\nGraceful stop\n")
                self.model.stop_training = True

    callbacks = [
        GracefulStop(),
        # keras.callbacks.ModelCheckpoint('chk-'+name, monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True)
        #keras.callbacks.EarlyStopping(monitor='loss', patience=500),
        keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=500),
    ]
    if test_answers is not None and len(test_answers) > 0:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=500))
    
    model.fit(train_dataset, epochs=15, shuffle=False, callbacks=callbacks, validation_data=test_dataset)

    print()
    print("Done")
    dmetrics = model.evaluate(train_dataset, verbose=0)
    tmetrics = model.evaluate(test_dataset, verbose=0)

    print(dmetrics)
    print(tmetrics)

    dir = opts.models_dir if opts and opts.models_dir else '.'
    name = f'{dir}{os.sep}pycf-{tmetrics[1]:.04f}-lstm{UNITS}-d{DROPOUT}-w{TIME_WINDOW}-{int(time.time())}.h5'
    #if dmetrics[1] >= 0.85 and tmetrics[1] >= 0.85:
    print('Saving as ' + name)
    model.save(name)

    #for (f,d) in test_data.items():
    #    metrics = model.evaluate(d[0],d[1], verbose=0)
    #    print('%-30s = %8.04f' % (os.path.basename(f), metrics[1]))

    print()

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
    model:tf.keras.Model = keras.models.load_model(mf)

    results = [(0,(0,0))]
    result = model.predict(np.copy(data), verbose=True)
    result = np.argmax(result, axis=1)
    for i in range(len(result)):
        when = times[i]
        ans = int(result[i])
        results[-1] = (results[-1][0], (results[-1][1][0], when))
        if ans != results[-1][0]:
            results.append((ans, (when, when)))
    results[-1] = (results[-1][0], (results[-1][1][0], duration))

    i = 0
    while i < len(results):
        if results[i][0] != SceneType.COMMERCIAL.value:
            del results[i]
        else:
            i += 1
    
    print(f'\n\nInitial n={len(results)}:')
    print(results)

    i = 0
    while i < len(results):
        #print(i, results[i])
        # this also coalesces consecutive results into one larger range
        if results[i][0] == results[i-1][0] and (results[i][1][0] - results[i-1][1][1]) < opts.show_min_len:
            # tiny show gap, delete it by merging the commercials
            results[i-1] = (results[i][0], (results[i-1][1][0], results[i][1][1]))
            del results[i]
        else:
            i += 1

    # show must be at least 30 seconds long (opts.show_min_len), or we just combine it into the commercial break its in the middle of
    # commercials must be at least 60 (opts.comm_min_len) seconds long, if it's less, it is deleted
    # commercials must be less than 360 seconds long (opts.comm_max_len), if it's more then it is just show after that
    i = 1
    while i < len(results):
        #print(i, results[i])
        # this also coalesces consecutive results into one larger range
        if results[i][0] == results[i-1][0] and (results[i][1][0] - results[i-1][1][1]) < opts.show_min_len:
            # tiny show gap, delete it by merging the commercials
            results[i-1] = (results[i][0], (results[i-1][1][0], results[i][1][1]))
            del results[i]
        else:
            i += 1
    
    i = 0
    while i < len(results):
        clen = results[i][1][1] - results[i][1][0]
        if clen < opts.break_min_len and results[i][0] == SceneType.COMMERCIAL.value:
            if i+1 >= len(results) and clen >= 5 and results[i][1][1]+clen+5 > duration:
                # dont require full length if it goes over the end of the recording
                break
            elif i == 0 and clen >= 5 and results[i][1][0] < 5:
                # don't require full length at the beginning of the recording
                i += 1
                pass
            else:
                # tiny commercial, delete it
                del results[i]
            continue

        if clen > opts.break_max_len and results[i][0] == SceneType.COMMERCIAL.value:
            # huge commercial, truncate it
            results[i] = (results[i][0], (results[i][1][0], results[i][1][0] + opts.break_max_len))

        # its ok now, move on
        i += 1
    
    time_thresh = .5*frame_rate
    # now where there is a diff within a few frames of the start/end of a tag, move the tag
    # TODO also do this for audio diffs? or silence ranges?
    for (_,(b,e)) in spans.get('diff', []):
        for i in range(len(results)):
            if abs(results[i][1][0] - b) <= time_thresh:
               results[i] = (results[i][0], (b, results[i][1][1]))
            elif abs(results[i][1][0] - e) <= time_thresh:
               results[i] = (results[i][0], (e, results[i][1][1]))
            if abs(results[i][1][1] - b) <= time_thresh:
               results[i] = (results[i][0], (results[i][1][0], b))
            elif abs(results[i][1][1] - e) <= time_thresh:
               results[i] = (results[i][0], (results[i][1][0], e))
    
    # now where there is a blank within a few frames of the start/end of a tag, move the tag toward the middle of the blank
    for (v,(b,e)) in spans.get('blank', []):
        if not v:
            continue
        for i in range(len(results)):
            if b-time_thresh <= results[i][1][0] and results[i][1][0] <= e+time_thresh:
                results[i] = (results[i][0], (b+(e-b)/2, results[i][1][1]))
            elif b-time_thresh <= results[i][1][1] and results[i][1][1] <= e+time_thresh:
                results[i] = (results[i][0], (results[i][1][0], b+(e-b)/2))

    print(f'\n\nFinal n={len(results)}:')
    print(results)

    flog['tags'] = results
    processor.write_feature_log(flog, feature_log)

    '''
    this is broken because results[i] is a tuple of (t,(b,e)) and answers is a one-hot vector without time in it at all
    correct = 0
    incorrect = 0
    for i in range(min(len(results),len(answers))):
        actual = 0
        expected = 0
        for j in range(len(results[i])):
            print(i,j,actual,expected,results[i])
            if results[i][actual] < results[i][j]:
                actual = j
            if answers[i] is not None and answers[i][expected] < answers[i][j]:
                expected = j
        
        if actual != expected:
            print(f"{times[i]} MISMATCHED result type {actual} was not {expected} at strength {results[i][actual]} vs {results[i][expected]}")
            print(results[i])
            incorrect += 1
        else:
            #print(f"{times[i]} matched result type {actual} at strength {results[i][actual]}")
            correct += 1
    if correct+incorrect > 0:
        print(f"{round(correct*100/(correct+incorrect),2)}% accurate")
    '''
    return results
