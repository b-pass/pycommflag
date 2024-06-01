import logging as log
import os
import numpy as np
import re
import time
import sys
import gc
from typing import Any,TextIO,BinaryIO

from .feature_span import *
from . import processor

TIME_WINDOW = 2.0
UNITS = 128
DROPOUT = .2

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
        for (bv,(bs,be)) in processor.read_feature_spans(flog, 'blank').get('blank', []):
            if not bv or bs == 0.0 or be >= endtime or (be-bs) >= 4.0:
                continue
            half = bs + (be - bs)
            for ti in range(len(tags)):
                (tt, (tb, te)) = tags[ti]
                fixb = bs-.5 <= tb and tb <= be+.5
                fixe = bs-.5 <= te and te <= be+.5
                if fixb and not fixe:
                    tags[ti] = (tt, (half, te))
                elif fixe and not fixb:
                    tags[ti] = (tt, (tb, half))
    
    # copy the frames so we don't mess up "flog"
    frames = list(flog['frames'])

    frames_header = flog['frames_header']
    assert('time' in frames_header[0])
    assert('diff' in frames_header[3])

    if frames[0] is None: frames.pop(0)

    # everything we do is for 29.97 or 30 (close enough); support 59.94 and 60 too... 
    if frame_rate >= 58:
        frames = frames[::int(frame_rate/29)]
    
    # ok now we can numpy....
    frames = np.copy(frames)

    # copy the real timestamps
    timestamps = frames[...,0].tolist()

    # normalize the timestamps upto 30 minutes
    frames[...,0] = (frames[...,0] % 1800.0) / 1800.0

    # normalize frame diffs
    frames[...,3] = np.clip(frames[...,3] / 30, 0, 1.0)

    need = int((TIME_WINDOW * 29) / 2)
        
    if len(frames) < need*5:
        return ([],[],[])

    data = []
    # we don't use numpy here because it wants to make copies of all of these slices
    # but using a python list does not, and that is a SIGNIFICANT memory savings
    # that's because there is a 59x duplication of the same data across sample sets
    d = ([frames[0]]*need) + frames[0:need+1].tolist()
    for i in range(len(frames)-(need+1)):
        d = d[1:] + [frames[i+need+1]]
        data.append(d)
    for i in range(len(frames)-(need+1), len(frames)):
        d = d[1:] + [frames[-1]]
        data.append(d)

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
        answers.append(x)

    for i in reversed(bad):
        assert(answers[i] is None)
        del timestamps[i]
        del data[i]
        del answers[i]

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
        print('...done')
    
    #x = np.copy(x)
    #y = np.copy(y)
    #xt = np.copy(xt)
    #yt = np.copy(yt)

    return (x,y,xt,yt)

def train(opts:Any=None):
    # yield CPU time to useful tasks, this is a background thing...
    os.nice(10)

    (data,answers,test_data,test_answers) = load_data(opts)
    gc.collect()
    
    # imort these here because their imports are slow 
    import tensorflow as tf
    import keras
    from keras import layers

    print('Calculating loss weights')
    sums = np.sum((np.sum(answers, axis=0), np.sum(test_answers, axis=0)), axis=0)
    #weights = [1] * len(sums)
    #weights[SceneType.SHOW.value] = sums[SceneType.COMMERCIAL.value] / sums[SceneType.SHOW.value]
    #weights[SceneType.COMMERCIAL.value] = sums[SceneType.SHOW.value] / sums[SceneType.COMMERCIAL.value]
    sums += 1
    weights = (np.sum(sums)-sums)/np.sum(sums)
    print("Loss Weights",weights)
    
    nsteps = len(data[0])
    nfeat = len(data[0][0])
    print("Data (x):",len(data)," Test (y):", len(test_data), "; Samples=",nsteps,"; Features=",nfeat)
    
    batch_size = opts.tf_batch_size if opts else 500
    train_dataset = tf.data.Dataset.from_tensor_slices((data, answers)).batch(batch_size)
    data = None
    answers = None
    gc.collect()

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_answers)).batch(batch_size)
    have_test = test_answers is not None and len(test_answers) > 0
    test_data = None
    test_answers = None
    gc.collect()

    inputs = keras.Input(shape=(nsteps,nfeat))
    n = inputs
    n = layers.LSTM(UNITS, dropout=DROPOUT)(n)
    #n = layers.LSTM(UNITS, dropout=DROPOUT, return_sequences=True)(n)
    #n = layers.LSTM(UNITS//2)(n)
    outputs = layers.Dense(SceneType.count(), activation='softmax')(n)
    model = keras.Model(inputs, outputs)
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", loss_weights=weights, metrics=['categorical_accuracy', 'mean_squared_error'])

    import signal
    def handler(signum, frame):
        print("\nStopping (gracefully)...\n")
        model.stop_training = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGINT, handler)

    callbacks = []
    #callbacks.append(keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=50))
    #if have_test:
    #    callbacks.append(keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=50))
    
    gc.collect()
    
    model.fit(train_dataset, epochs=15, shuffle=True, callbacks=callbacks, validation_data=test_dataset)

    gc.collect()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
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

    result = model.predict(np.copy(data), verbose=True)

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
            nextstart = results[i][1][1] + opts.min_show_len
            # check to make sure we didn't somehow create a small gap, if we did then WIDEN it to be min_show_len
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

    # now where there is a diff within a few frames of the start/end of a tag, move the tag
    # TODO also do this for audio diffs? or silence ranges?
    for (_,(db,de)) in spans.get('diff', []):
        for ti in range(len(results)):
            (tt, (tb, te)) = results[ti]
            fixb = db-.1 < tb and tb < db+.1
            fixe = de-.1 < te and te < de+.1
            if fixb and not fixe:
                results[ti] = (tt, (db, te))
            elif fixe and not fixb:
                results[ti] = (tt, (tb, de))
    
    # now where there is a blank within a few frames of the start/end of a tag, move the tag toward the middle of the blank
    for (bv,(bs,be)) in spans.get('blank', []):
        if not bv or bs == 0.0 or be >= duration or (be-bs) >= 4.0:
            continue
        half = bs + (be - bs)
        for ti in range(len(results)):
            (tt, (tb, te)) = results[ti]
            fixb = bs-.5 <= tb and tb <= be+.5
            fixe = bs-.5 <= te and te <= be+.5
            if fixb and not fixe:
                results[ti] = (tt, (half, te))
            elif fixe and not fixb:
                results[ti] = (tt, (tb, half))
    
    print(f'\n\nFinal n={len(results)}:')
    print(results)

    flog['tags'] = results
    processor.write_feature_log(flog, feature_log)

    return results
