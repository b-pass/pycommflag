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
    MAX_BLOCK_LEN = 300.0 # seconds
    x = []
    y = []

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

    feat_info = [('audio','faudio'),('logo_span','logo_present'),('blank_span','is_blank'),('diff_span','is_diff')]

    # put all the features in a flat array
    flat = []
    for (fn,pn) in feat_info:
        for (v,(b,e)) in flog.get(fn, []):
            flat.append({'ftime':b, pn:v})
    # sort if by time
    flat.sort(key=lambda x: x['ftime'])
    # combine features if they have the same timestamp
    i = 0
    while i+1 < len(flat):
        if flat[i]['ftime'] + (1.0/frame_rate) > flat[i+1]['ftime']:
            nope = False
            for k in flat[i].keys():
                if k != 'ftime' and k in flat[i+1]:
                    nope = True
                    break
            if not nope:
                del flat[i+1]['ftime']
                flat[i].update(flat[i+1])
                del flat[i+1]
                continue
        i += 1
    
    # now segment based on features and record the timestamps of each segment
    value = {'ftime':'0','faudio':AudioSegmentLabel.SILENCE,'logo_present':False,'is_blank':True,'is_diff':False}
    timestamps = []
    for x in flat:
        value.update(x)
        if seg.check(**value):
            timestamps.append(value['ftime'])
    
    # get rid of any empty 0 timestamps at the start
    while timestamps[0] == 0.0:
        timestamps.pop(0)
    # the upper bound
    timestamps.append(endtime)
    
    data = []
    for _ in range(len(timestamps)):
        data.append([0] * (AudioSegmentLabel.count() + 2))

    # Save duration of each audio type
    didx = 0
    for (v,(b,e)) in flog.get('audio',[]):
        while didx < len(timestamps) and b - 0.5/frame_rate > timestamps[didx]:
            didx += 1
        if didx >= len(timestamps): didx = len(timestamps)-1
        #print(v,b,e,didx,timestamps[didx])
        fi = v if type(v) is int else v.value
        data[didx][fi] += min(e,timestamps[didx]) - b
        for j in range(didx+1,len(timestamps)):
            if e <= timestamps[j-1]:
                break
            data[j][fi] += min(e,timestamps[j]) - timestamps[j-1]
            #print('X',timestamps[j-1],min(e,timestamps[j]),j,timestamps[j])

    # count the diffs
    didx = 0
    for (v,(b,e)) in flog.get('diff_span',[]):
        while didx < len(timestamps) and b - 0.5/frame_rate > timestamps[didx]:
            didx += 1
        if didx >= len(timestamps): didx = len(timestamps)-1
        data[didx][AudioSegmentLabel.count()+0] += 1
    
    # sum the logo time
    didx = 0
    for (v,(b,e)) in flog.get('logo_span',[]):
        if not v:
            continue
        while didx < len(timestamps) and b - 0.5/frame_rate > timestamps[didx]:
            didx += 1
        if didx >= len(timestamps): didx = len(timestamps)-1
        
        data[didx][AudioSegmentLabel.count()+1] += min(e,timestamps[didx]) - b
        for j in range(didx+1,len(timestamps)):
            if e <= timestamps[j-1]:
                break
            data[j][AudioSegmentLabel.count()+1] += min(e,timestamps[j]) - timestamps[j-1]

    # now associate labels/tags with the segments we have identified (based on timestamps)
    answers = []
    dstart = 0
    prev_tag = SceneType.COMMERCIAL.value
    delete_me = []
    for (dend,d) in zip(timestamps,data):
        ddur = dend - dstart
        d.append(min(ddur/300,1.0)) # length indicator
        d.append((dstart%1800)/1800.0) # position indicator
        d.append(max(d[-1],(dend%1800)/1800.0)) # end position indicator

        oh = [0]*SceneType.count()
        oh[prev_tag if type(prev_tag) is int else prev_tag.value] = 1.0
        d += oh # basic recurrent NN, this is the answer from the previous sample

        next_tag = None
        for (tt,(tb,te)) in tags:
            if dstart >= te or dend <= tb:
                continue # tag doesn't overlap segment
            
            if ddur > 1 and (dstart + ddur/2 < tb or dstart+ddur/2 > te):
                #print(dstart,dend,"mostly outside of tag",tt,"which is at",tb,te," -- ignoring the tag")
                continue
            if next_tag is not None:
                if dstart + ddur/2 >= te:
                    #print(dstart,dend,"straddles tag",tt,"which is at",tb,te," -- but not using it")
                    break
                #else: print(dstart,dend,"straddles tag",tt,"which is at",tb,te," -- overwriting", next_tag)
            next_tag = tt
            #print(dstart,dend,"matches tag",tt,"which is at",tb,te)
        if next_tag is None:
            next_tag = SceneType.UNKNOWN
            #print("missing next tag at",tstart,tend)
        
        if next_tag == SceneType.DO_NOT_USE or next_tag == SceneType.DO_NOT_USE.value:
            delete_me.append(dend)
            answers.append(None)
        else:
            oh = [0]*SceneType.count()
            oh[next_tag if type(next_tag) is int else next_tag.value] = 1.0
            answers.append(oh)
            prev_tag = next_tag
        dstart = dend
    
    for d in reversed(delete_me):
        for i in range(len(timestamps)):
            if timestamps[i] == d:
                del timestamps[i]
                del data[i]
                del answers[i]
                break

    pt = 0
    for (t,x,a) in zip(timestamps,data,answers):
        print(a,pt,t,x)
        pt = t
    
    return (timestamps,data,answers)

def load_data(opts)->tuple[list,list,list,list,list,list]:
    n = []
    x = []
    y = []

    seg = segmenter.parse(opts.segmeth)

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
            (t,a,b) = flog_to_vecs(flog, seg=seg)
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
            (t,a,b) = flog_to_vecs(flog, seg=seg)
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

    DROPOUT = 0.05

    nfeat = len(data[0])
    inputs = keras.Input(shape=(nfeat,))
    n = inputs
    if DROPOUT > 0:
        n = layers.Dropout(DROPOUT)(n)
    # don't know which activation to use? no problem, just use them all
    n = layers.Dense(nfeat, activation='relu')(n)
    n = layers.Dense(nfeat, activation='tanh')(n)
    n = layers.Dense(nfeat, activation='elu')(n)
    outputs = layers.Dense(SceneType.count(), activation='softmax')(n)
    model = keras.Model(inputs, outputs)
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['categorical_accuracy', 'mean_squared_error'])

    batch_size = opts.tf_batch_size if opts else 1000
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

    dir = opts.model_dir if opts and opts.models_dir else '.'
    name = f'{dir}{os.pathsep}pycf-{dmetrics[1]:.04f}-{tmetrics[1]:.04f}-mse{tmetrics[2]:.04f}-{int(time.time())}.h5'
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
        mf = f'{opts.models_dir or "."}{os.pathsep}model.h5'
    if not os.path.exists(mf):
        raise Exception(f"Model file '{mf}' does not exist")
    model:tf.keras.Model = keras.models.load_model(mf)

    results = []
    prev = [0] * SceneType.count()
    prev[SceneType.COMMERCIAL.value] = 1.0
    prevtime = 0
    for (when,entry) in zip(times,data):
        result = model.predict([entry+prev], verbose=False)[0].tolist()
        ans = 0
        for i in range(result):
            if result[i] >= result[ans]:
                ans = i
        if ans != SceneType.UNKNOWN.value:
            results.append((SceneType[ans], (prevtime,when)))
        prevtime = when
        prev = result
    
    flog['tags'] = results
    processor.write_feature_log(flog, feature_log)

    correct = 0
    incorrect = 0
    for i in range(min(len(results),len(answers))):
        actual = None
        expected = None
        for j in range(len(results[i])):
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
    
    return results
