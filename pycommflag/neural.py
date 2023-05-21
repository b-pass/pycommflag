import logging as log
from math import ceil, floor
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

    audios = flog.get('audio', [])
    ai = 0
    ti = 0

    frames = []
    ftags = []
    ftimes = []
    for f in flog['frames']:
        if f is None or not f:
            continue
        ftime = f[0]
        
        while ai < len(audios) and ftime < audios[ai][1][0]:
            ai += 1
        av = [0] * AudioSegmentLabel.count() 
        if ai < len(audios) and ftime < audios[ai][1][1]:
            av[audios[ai][0]] = 1.0
        else:
            av[AudioSegmentLabel.SILENCE.value] = 1.0
        
        while ti < len(tags) and ftime < tags[ti][1][0]:
            ti += 1
        ftags.append(tags[ti][0] if ti < len(tags) and ftime < tags[ti][1][1] else SceneType.UNKNOWN.value)
        if ftags[-1] == SceneType.DO_NOT_USE:
            ftags.pop()
            continue
        
        f[0] = (f[0] % 1800) / 1800.0
        frames.append(f + av)
        ftimes.append(ftime)

    timestamps = []
    data = []
    answers = []
    for middle in range(len(frames)):
        timestamps.append(ftimes[middle])
        data.append(frames[max(0,middle - floor(30*frame_rate)) : min(middle + ceil(30*frame_rate), len(frames))])
        answers.append(ftags[middle])
    
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

    DROPOUT = 0 #0.05

    inputs = keras.Input(shape=(None,len(data[0][0])), dtype='float')
    n = inputs
    if DROPOUT > 0:
        n = layers.Dropout(DROPOUT)(n)
    
    n = layers.LSTM(64, return_sequences=True)(n)
    n = layers.LSTM(64)(n)
    n = layers.Dense(SceneType.count(), activation='elu')(n)
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
