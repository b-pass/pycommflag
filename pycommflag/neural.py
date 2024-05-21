import logging as log
import os
import numpy as np
import re
import time
import sys
from typing import Any,TextIO,BinaryIO

from .feature_span import *
from . import processor, segmenter

def flog_to_vecs(flog:dict, seg:segmenter.SceneSegmenter=None)->tuple[list[list[float]], list[list[float]]]:
    version = flog.get('file_version', 10)
    frame_rate = flog.get('frame_rate', 29.97)
    endtime = flog.get('duration', 0)

    tags = flog.get('tags', [])
    
    # clean up tags so they start/end exactly in the middle of the blank block
    # this is because the training data is supplied by humans, and might be off a bit
    # TODO, do we need to handle all-blank scenes at the start/end of a block differently?
    # the reason would be because of inconsistent training data (sometimes the blank is part of the commercial, sometimes the show)
    # but there are blanks in the middle of both also so maybe that doesnt matter....?
    if tags and 'blank_span' in flog :
        blanks = flog.get('blank_span', [])
        for (bv,(bs,be)) in blanks:
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
    
    frames = flog['frames']
    if frames[0] is None: frames.pop(0)

    frames_header = flog['frames_header']
    assert(frames_header[0] == 'time')

    # often the video doesn't start a PTS 0 because of sync issues, back-fill the first video frame
    # note, assumes time is at frame[0]
    if frames[0][0] > 0:
        while True:
            frames[0:0] = [list(frames[0])] # insert
            frames[0][0] = round(frames[0][0] - 1/frame_rate, 5)
            if frames[0][0] <= 0.0:
                frames[0][0] = 0.0
                break
    
    # mark up each frame with the calculated audio type and volume level
    ######## if 'audio' not in frames_header:
    # normalize with the max volume of the whole recording
    if 'volume' in flog and 'volume' not in frames_header:
        vscale = np.max(np.array(flog['volume'])[...,1:3])
        vit = iter(flog['volume'])
        volume = next(vit, (0,0,0))
        
        for frame in frames:
            ft = frame[0]
            while ft >= volume[0]+0.00001:
                volume = next(vit, (endtime+1,0,0))
            frame += [volume[1]/vscale, volume[2]/vscale]
    
    # convert audio to one-hot and then add it
    ait = iter(flog['audio'])
    audio = next(ait, (0,(0,endtime+1)))
    for frame in frames:
        ft = frame[0]
        while ft >= audio[1][1]:
            audio = next(ait, (0,(0,endtime+1)))
        av = audio[0] if ft >= audio[1][0] else AudioSegmentLabel.SILENCE.value
        x = [0] * AudioSegmentLabel.count()
        x[int(av)] = 1
        frame += x

    # normalize time values on a 30-minute scale
    # save the real timestamps though
    timestamps = []
    for f in frames:
        timestamps.append(f[0])
        f[0] = (f[0] % 1800) / 1800.0
        
    # convert tag to a one-hot and then add it
    answers = []
    tit = iter(tags)
    tag = next(tit, (SceneType.UNKNOWN, (0, endtime+1)))
    for f in frames:
        while f[0] >= tag[1][1]:
            tag = next(tit, (SceneType.UNKNOWN, (0, endtime+1)))
        tt = tag[0] if f[0] >= tag[1][0] else SceneType.UNKNOWN
        x = [0] * SceneType.count()
        x[tt if type(tt) is int else tt.value] = 1
        answers.append(x)

    window = 1.0

    need = int((window * 29) / 2)
    skip = max(1,int(frame_rate / 30))
    data = []
    for i in range(len(frames)):
        d = []
        # near the beginning, fill with repeats from first frame in order to get to the correct size
        extra = need - i//skip 
        if extra > 0:
            d += [frames[0]] * extra
        d += frames[i - need : i+need+1 : skip]
        
        # TODO: should we include the answers from the previous frames? 
        # if we did, we'd need to carefully set the future one-hots to be all zero, i guess.

        extra = need*2+1 - len(d)
        if extra > 0:
            d += [frames[-1]] * extra
        
        data.append(d)
    
    return (timestamps,data,answers)

def load_data(opts)->tuple[list,list,list,list,list,list]:
    n = []
    x = []
    y = []

    seg = segmenter.parse(opts.segmeth)
    print(seg)

    data = opts.ml_data
    if not data:
        return []
    
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
    
    for f in data:
        if os.path.isdir(f):
            continue
        #print(f)
        flog = processor.read_feature_log(f)
        if flog:
            (t,a,b) = flog_to_vecs(flog, seg=seg)
            n += [os.path.basename(f)] * len(a)
            x += a
            y += b
    
    nt = []
    xt = []
    yt = []
    for f in test:
        if os.path.isdir(f):
            continue
        #print(f)
        flog = processor.read_feature_log(f)
        if flog:
            (t,a,b) = flog_to_vecs(flog, seg=seg)
            nt += [os.path.basename(f)] * len(a)
            xt += a
            yt += b
    
    import random
    random.seed(time.time())
    need = int(len(x)/5) - len(xt)
    if need > (len(x)/100):
        print('Need to move',need,'datum to the test/eval set')
        z = list(zip(n,x,y))
        n=x=y=None
        random.shuffle(z)
        (n,x,y) = zip(*z[need:])
        if len(xt) == 0:
            (nt,xt,yt) = zip(*z[:need])
        else:
            (a,b,c) = zip(*z[:need])
            nt += a
            xt += b
            yt += c

    return (n,x,y,nt,xt,yt)

def train(training:tuple, test_answers:list=None, opts:Any=None):
    # imort these here because their imports are slow 
    import tensorflow as tf
    import keras
    from keras import layers

    (names,data,answers,test_names,test_data,test_answers) = training

    DROPOUT = 0.15

    nsteps = len(data[0])
    nfeat = len(data[0][0])
    print("Data (x):",len(data)," Test (y):", len(test_data), "; Samples=",nsteps,"; Features=",nfeat)
    inputs = keras.Input(shape=(nsteps,nfeat))
    n = inputs
    n = layers.LSTM(64, dropout=DROPOUT)(n)
    #n = layers.LSTM(64, dropout=DROPOUT, return_sequences=True)(n)
    #n = layers.LSTM(32)(n)
    outputs = layers.Dense(SceneType.count(), activation='softmax')(n)
    model = keras.Model(inputs, outputs)
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['categorical_accuracy', 'mean_squared_error'])

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
    if test_answers:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=500))
    
    history = model.fit(train_dataset, epochs=5, shuffle=True, callbacks=callbacks, validation_data=test_dataset)

    print()
    print("Done")
    dmetrics = model.evaluate(train_dataset, verbose=0)
    tmetrics = model.evaluate(test_dataset, verbose=0)

    print(dmetrics)
    print(tmetrics)

    dir = opts.models_dir if opts and opts.models_dir else '.'
    name = f'{dir}{os.sep}pycf-{dmetrics[1]:.04f}-{tmetrics[1]:.04f}-mse{tmetrics[2]:.04f}-{int(time.time())}.h5'
    #if dmetrics[1] >= 0.95 and tmetrics[1] >= 0.95:
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
    (times,data,answers) = flog_to_vecs(flog, segmenter.parse(opts.segmeth))

    mf = opts.model_file
    if not mf and opts:
        mf = f'{opts.models_dir or "."}{os.sep}model.h5'
    if not os.path.exists(mf):
        raise Exception(f"Model file '{mf}' does not exist")
    model:tf.keras.Model = keras.models.load_model(mf)

    frame_rate = flog.get('frame_rate', 29.97)
    results = []
    prev = [0] * SceneType.count()
    prev[SceneType.COMMERCIAL.value] = 1.0
    prevtime = 0
    change = 0
    prevans = SceneType.COMMERCIAL.value
    for (when,entry) in zip(times,data):
        entry[-(SceneType.count()+1)] = min((prevtime - change)/300,1.0)
        entry[-len(prev):] = prev

        result = model.predict([entry], verbose=False)[0].tolist()
        ans = 0
        #print(result)
        for i in range(len(result)):
            if result[i] >= result[ans]:
                ans = i
        if ans != SceneType.UNKNOWN.value:
            results.append((ans, (prevtime,when)))
        if ans != prevans and prevtime:
            change = when
        
        prevtime = when
        prev = result
        prevans = ans
    
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
    print(results)
    
    i = 0
    while i < len(results):
        clen = results[i][1][1] - results[i][1][0]
        if clen < opts.break_min_len and results[i][0] == SceneType.COMMERCIAL.value:
            if i+1 >= len(results) and clen >= 5 and results[i][1][1]+clen+5 > flog.get('duration', 0):
                # dont require full length if it goes over the end of the recording
                break
            elif i == 0 and clen >= 5 and results[i][1][0] <= 0.1:
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
    
    # now where there is a diff within 2 frame of the start/end of a tag, move the tag
    # TODO also do this for audio diffs? or silence ranges?
    for (_,(b,e)) in flog.get('diff_span', []):
        for i in range(len(results)):
            if abs(results[i][1][0] - b) <= 2.5/frame_rate:
               results[i] = (results[i][0], (b, results[i][1][1]))
            elif abs(results[i][1][0] - e) <= 2.5/frame_rate:
               results[i] = (results[i][0], (e, results[i][1][1]))
            if abs(results[i][1][1] - b) <= 2.5/frame_rate:
               results[i] = (results[i][0], (results[i][1][0], b))
            elif abs(results[i][1][1] - e) <= 2.5/frame_rate:
               results[i] = (results[i][0], (results[i][1][0], e))
    
    # now where there is a blank within 2 frame of the start/end of a tag, move the tag toward the middle of the blank
    for (v,(b,e)) in flog.get('blank_span', []):
        if not v:
            continue
        for i in range(len(results)):
            if b-2/frame_rate <= results[i][1][0] and results[i][1][0] <= e+2/frame_rate:
                results[i] = (results[i][0], (b+(e-b)/2, results[i][1][1]))
            elif b-2/frame_rate <= results[i][1][1] and results[i][1][1] <= e+2/frame_rate:
                results[i] = (results[i][0], (results[i][1][0], b+(e-b)/2))

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
