import logging as log
import os
import numpy as np
import re
import time
import sys
from typing import Any,TextIO,BinaryIO

from .feature_span import *

def flog_to_vecs(flog:dict)->tuple[list[list[float]], list[list[float]]]:
    MAX_BLOCK_LEN = 600.0 # seconds
    x = []
    y = []

    # TODO, do we need to handle all-blank scenes at the start/end of a block differently?
    # the reason would be because of inconsistent training data (sometimes the blank is part of the commercial, sometimes the show)
    # but there are blanks in the middle of both also so maybe that doesnt matter....?

    # ok so how this is supposed to work:
    # each feature has 2 things
    # time since last change
    # time to next change, when active/true this will be zero

    frame_rate = flog['frame_rate']
    equal = lambda a,b, n=1.0: abs(a - b) < (n/frame_rate)
    less_equal = lambda a,b: a - b < 1.0/frame_rate

    tags = flog.get('tags', [])
    endtime = flog['duration']

    feat = [ 
        flog.get('audio', []),
        flog.get('logo_span', []),
        flog.get('diff_span', []),
        flog.get('blank_span', [])
        ]
    fidx = [0]*len(feat)
    aidx = 0

    data = []
    answers = []
    while True:
        now = None
        for (i,f) in zip(fidx,feat):
            if i < len(f):
                if now is None or abs(f[i][1][1] - now) < (2.0/frame_rate):
                    now = f[i][1][1]
        if now is None or now >= endtime:
            break

        a = SceneType.SHOW.value
        while aidx < len(tags):
            (t,(b,e)) = tags[aidx]
            if e < now:
                aidx += 1
                continue
            if b <= now:
                a = t
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
                    entry.append(f[i][0])
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
        
        if a == SceneType.DO_NOT_USE.value:
            continue

        data.append(entry)

        aoh = [0.0] * SceneType.count()
        aoh[a] = 1.0
        answers.append(aoh)

    return (data,answers)

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
            (a,b) = flog_to_vecs(flog)
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
            (a,b) = flog_to_vecs(flog)
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
    #if DROPOUT > 0:
    #    n = layers.Dropout(DROPOUT)(n)
    for i in range(10):
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

    name = 'blah'
    name = '%.04f-%.04f-mse%.04f-m%s'%(dmetrics[1],tmetrics[1],tmetrics[2],name)
    print()
    print(name)
    #if dmetrics[1] >= 0.95 and tmetrics[1] >= 0.95:
    #    print('Saving as ' + name)
    #    model.save(name)

    #for (f,d) in test_data.items():
    #    metrics = model.evaluate(d[0],d[1], verbose=0)
    #    print('%-30s = %8.04f' % (os.path.basename(f), metrics[1]))

    print()
    