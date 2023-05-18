import logging as log
import os
import numpy as np
import re
import time
import sys
from typing import Any,TextIO,BinaryIO

from .feature_span import *
from . import processor

def flog_to_vecs(flog:dict)->tuple[list[list[float]], list[list[float]]]:
    MAX_BLOCK_LEN = 300.0 # seconds
    x = []
    y = []

    # TODO, do we need to handle all-blank scenes at the start/end of a block differently?
    # the reason would be because of inconsistent training data (sometimes the blank is part of the commercial, sometimes the show)
    # but there are blanks in the middle of both also so maybe that doesnt matter....?

    frame_rate = flog['frame_rate']
    endtime = flog['duration']

    # clean up tags so they start/end exactly in the middle of the blank block
    tags = flog.get('tags', [])
    if tags and 'blank_span' in flog:
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


    feat = [ 
        flog.get('audio', []),
        flog.get('logo_span', []),
        flog.get('diff_span', []),
        flog.get('blank_span', [])
        ]
    fidx = [0]*len(feat)
    aidx = 0

    timestamps = []
    data = []
    answers = []
    aprev = [0] * SceneType.count()
    aprev[SceneType.SHOW.value] = 1.0
    alast_end = 0
    alast_val = SceneType.SHOW.value
    while True:
        now = None
        for (i,f) in zip(fidx,feat):
            if i < len(f):
                if now is None or abs(f[i][1][1] - now) < (2.0/frame_rate):
                    now = f[i][1][1]
        if now is None or now >= endtime:
            break

        entry = []
        entry.append((now%1800) / 1800.0)
        
        for mi in range(len(fidx)):
            i = fidx[mi]
            f = feat[mi]
            if i < len(f):
                if mi == 0:
                    # audio is a one-hot
                    oh = [0.0] * AudioSegmentLabel.count()
                    oh[f[i][0]] = 1.0
                    entry += oh
                else:
                    # everything else is a integerized-boolean
                    entry.append(int(f[i][0]))
                elapsed = now - f[i][1][0]
                entry.append(min(max(0,elapsed),MAX_BLOCK_LEN)/MAX_BLOCK_LEN)
                timeleft = f[i][1][1] - now
                if abs(timeleft) < 2.0/frame_rate:
                    timeleft = 0.0
                    fidx[mi] += 1
                entry.append(min(max(0,timeleft),MAX_BLOCK_LEN)/MAX_BLOCK_LEN)
            else:
                if mi == 0:
                    # audio is a one-hot
                    oh = [0.0] * AudioSegmentLabel.count()
                    entry += oh
                else:
                    # everything else is a integerized-boolean
                    entry.append(0.0)
                entry += [0.4, 0.6] # not really half way, not at the beginning or the end, just somewhere.
        
        entry.append(min(max(0,now - alast_end),MAX_BLOCK_LEN)/MAX_BLOCK_LEN)
        entry.append(alast_val)

        if aidx >= len(tags):
            anewval = SceneType.SHOW.value
            alast_val = anewval
            alast_end = tags[aidx-1][1][1] if tags else 0
        else:
            while aidx < len(tags):
                (t,(b,e)) = tags[aidx]
                if e <= now:
                    aidx += 1
                    continue
                if b <= now:
                    anewval = t
                    if anewval != alast_val:
                        alast_val = anewval
                        alast_end = b
                else:
                    anewval = SceneType.SHOW.value
                    alast_val = anewval
                    alast_end = tags[aidx-1][1][1]
                break
            
        if anewval == SceneType.DO_NOT_USE.value:
            continue
        
        timestamps.append(now)

        data.append(entry + aprev)
        #print(entry)
        #print(aprev)

        aprev = [0.0] * SceneType.count()
        aprev[anewval] = 1.0
        answers.append(aprev)

    return (timestamps,data,answers)

def load_data(opts)->tuple[list,list,list,list,list,list]:
    from . import processor
    n = []
    x = []
    y = []

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
        print(f)
        flog = processor.read_feature_log(f)
        if flog:
            (t,a,b) = flog_to_vecs(flog)
            n += [os.path.basename(f)] * len(a)
            x += a
            y += b
    
    nt = []
    xt = []
    yt = []
    for f in test:
        print(f)
        flog = processor.read_feature_log(f)
        if flog:
            (t,a,b) = flog_to_vecs(flog)
            nt += [os.path.basename(f)] * len(a)
            xt += a
            yt += b
    
    return (n,x,y,nt,xt,yt)

def train(training:tuple, test_answers:list=None, opts:Any=None):
    # imort these here because their imports are slow 
    import tensorflow as tf
    import keras
    from keras import layers

    (names,data,answers,test_names,test_data,test_answers) = training

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

    DROPOUT = 0.05

    inputs = keras.Input(shape=(len(data[0]),))
    n = inputs
    if DROPOUT > 0:
        n = layers.Dropout(DROPOUT)(n)
    n = layers.Dense(len(data[0]))(n)
    n = layers.Dense(len(data[0]))(n)
    n = layers.Dense(len(data[0]))(n)
    outputs = layers.Dense(SceneType.count(), activation='softmax')(n)
    model = keras.Model(inputs, outputs)
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['categorical_accuracy', 'mean_squared_error'])

    batch_size = opts.tf_batch_size if opts else 1000
    train_dataset = tf.data.Dataset.from_tensor_slices((data, answers)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_answers)).batch(batch_size)
    callbacks = [
        GracefulStop(),
        # keras.callbacks.ModelCheckpoint('chk-'+name, monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True)
        #keras.callbacks.EarlyStopping(monitor='loss', patience=100),
        keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=250),
    ]
    if test_answers:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=500))
    history = model.fit(train_dataset, epochs=5000, callbacks=callbacks, validation_data=test_dataset)

    print()
    print("Done")
    dmetrics = model.evaluate(train_dataset, verbose=0)
    tmetrics = model.evaluate(test_dataset, verbose=0)

    print(dmetrics)
    print(tmetrics)

    if opts and opts.models_dir:
        dir = opts.models_dir
        if dir and dir[-1] != '/':
            dir += '/'
    else:
        dir = './'
    name = f'{dir}pycf-{dmetrics[1]:.04f}-{tmetrics[1]:.04f}-mse{tmetrics[2]:.04f}-{int(time.time())}.h5'
    #if dmetrics[1] >= 0.95 and tmetrics[1] >= 0.95:
    print('Saving as ' + name)
    model.save(name)

    #for (f,d) in test_data.items():
    #    metrics = model.evaluate(d[0],d[1], verbose=0)
    #    print('%-30s = %8.04f' % (os.path.basename(f), metrics[1]))

    print()

def predict(feature_log:str|TextIO|dict, opts:Any=None)->list:
    import tensorflow as tf
    import keras

    flog = processor.read_feature_log(feature_log)
    (times,data,answers) = flog_to_vecs(flog)

    mf = opts.model_file
    if not mf and opts:
        mf = f'{opts.models_dir or "."}/model.h5'
    if not os.path.exists(mf):
        raise Exception(f"Model file '{mf}' does not exist")
    model:tf.keras.Model = keras.models.load_model(mf)

    results = []
    prev = [0] * SceneType.count()
    prev[SceneType.COMMERCIAL.value] = 1.0
    for entry in data:
        entry[-len(prev):] = prev
        result = model.predict([entry], verbose=False)[0].tolist()
        results.append(result)
        prev = result
    
    correct = 0
    incorrect = 0
    for i in range(len(results)):
        actual = 0
        expected = 0
        for j in range(len(results[i])):
            if results[i][actual] < results[i][j]:
                actual = j
            if  answers[i][expected] < answers[i][j]:
                expected = j
        
        if actual != expected:
            print(f"{times[i]} MISMATCHED result type {actual} was not {expected} at strength {results[i][actual]} vs {results[i][expected]}")
            print(results[i])
            incorrect += 1
        else:
            #print(f"{times[i]} matched result type {actual} at strength {results[i][actual]}")
            correct += 1
    print(f"{round(correct*100/(correct+incorrect),2)}% accurate")
    sys.exit(1)
