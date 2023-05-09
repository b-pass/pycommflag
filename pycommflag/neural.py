import logging as log
import struct
import sys
import os
import re
import numpy as np
import math
import time

from typing import Any,BinaryIO
from .scene import *

def scenes_to_vecs(scenes:list[Scene])->tuple[list[list[float]], list[list[float]]]:
    MAX_SCENE_LEN = 300.0 # seconds
    x = []
    y = []

    # TODO, do we need to handle all-blank scenes at the start/end of a block differently?
    # the reason would be because of inconsistent training data (sometimes the blank is part of the commercial, sometimes the show)
    # but there are blanks in the middle of both also so maybe that doesnt matter....?

    block_len = 0.0
    prev = SceneType.COMMERCIAL
    first = True
    for s in scenes:
        if first:
            # don't use short scenes right at the beginning for training
            first = False
            if s.duration < 5:
                continue
        st = getattr(s, 'newtype', s.type)
        if st == SceneType.DO_NOT_USE:
            continue
        
        a = [0.0] * SceneType.count()
        a[st.value] = 1.0

        v = []
        v.append((s.start_time % 1800) / 1800.0)
        v.append(min(s.duration,MAX_SCENE_LEN)/MAX_SCENE_LEN)
        v.append(block_len/600.0)
        for ax in s.audio:
            v.append(ax)
        v.append(s.logo)
        v.append(s.blank)
        v.append(s.diff)
        # v += s.barcode.tolist()
        v += x[-1][0:len(v)] if x else [0.0]*(len(v)-1)+[1.0]
        #v += y[-1] if y else a

        x.append(v)
        y.append(a)
        if st != prev:
            block_len = 0.0
            prev = st
        else:
            block_len = min(block_len+s.duration, 600.0)
    
    return (x,y)

def load_data(opts)->tuple[list,list,list,list,list,list]:
    from . import processor
    n = []
    x = []
    y = []

    data = opts.ml_data
    #print(data)
    test = []

    if 'TEST' in data:
        i = data.index('TEST')
        test = data[i+1:]
        data = data[:i]
    
    for f in data:
        print(f)
        s = processor.read_scenes(f)
        if s:
            (a,b) = scenes_to_vecs(s)
            n += [os.path.basename(f)] * len(a)
            x += a
            y += b
    
    nt = []
    xt = []
    yt = []
    for f in test:
        s = processor.read_scenes(f)
        if s:
            (a,b) = scenes_to_vecs(s)
            nt += [os.path.basename(f)] * len(a)
            xt += a
            yt += b
    
    return (n,x,y,nt,xt,yt)

def train(training:tuple, test_answers:list=None, opts:Any=None):
    import numpy as np
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
        keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=100),
    ]
    if test_answers:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=100))
    history = model.fit(train_dataset, epochs=5000, callbacks=callbacks, validation_data=test_dataset)

    print()
    print("Done")
    dmetrics = model.evaluate(train_dataset, verbose=0)
    tmetrics = model.evaluate(test_dataset, verbose=0)

    print(dmetrics)
    print(tmetrics)

    name = 'blah'
    name = '%.04f-%.04f-mse%.04f-m%s'%(dmetrics[1],tmetrics[1],tmetrics[2],name)
    #if dmetrics[1] >= 0.95 and tmetrics[1] >= 0.95:
    #    print('Saving as ' + name)
    #    model.save(name)

    #for (f,d) in test_data.items():
    #    metrics = model.evaluate(d[0],d[1], verbose=0)
    #    print('%-30s = %8.04f' % (os.path.basename(f), metrics[1]))

    print()
    