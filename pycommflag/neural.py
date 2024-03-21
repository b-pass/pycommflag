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
        if (flat[i+1]['ftime'] - flat[i]['ftime']) <= (0.5/frame_rate):
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
    prev_end = 0
    delete_me = []
    for (dend,d) in zip(timestamps,data):
        ddur = dend - dstart
        time_since = dstart - prev_end
        for i in range(len(d)):
            d[i] = d[i] / ddur if d[i] > 0 else 0.0
        d.append(min(ddur/300,1.0)) # length indicator
        d.append((dstart%1800)/1800.0) # position indicator
        d.append((dend%1800)/1800.0) # end position indicator
        if d[-1] < d[-2]: d[-1] = 1.0 # position wrap
        d.append(min(time_since/300,1.0))

        oh = [0]*SceneType.count()
        oh[prev_tag if type(prev_tag) is int else prev_tag.value] = 1.0
        d += oh # basic recurrent NN, this is the answer from the previous sample

        next_tag = None
        for (tt,(tb,te)) in tags:
            if dstart >= te or dend <= tb:
                continue # tag doesn't overlap segment
            
            if ddur > 1 and (dstart + ddur/10 < tb or dend-ddur/10 > te):
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
        
        if prev_tag != next_tag and dstart:
            prev_end = dend

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

    # also, a paradoxical future NN, also include the next sample's information
    #for i in range(len(data)-1):
    #    data[i][-SceneType.count():-SceneType.count()] = data[i+1][:-SceneType.count()]
    # just repeat the last sample in this case to keep the tensor square
    #data[-1][-SceneType.count():-SceneType.count()] = data[-1][:-SceneType.count()]
    
    #pt = 0
    #for (t,x,a) in zip(timestamps,data,answers):
    #    print(a,pt,t,len(x),x)
    #    pt = t
    
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
    while need > 0:
        need -= 1
        i = random.randint(0,len(x)-1)
        nt.append(n[i])
        xt.append(x[i])
        yt.append(y[i])
        del n[i]
        del x[i]
        del y[i]

    return (n,x,y,nt,xt,yt)

def train(training:tuple, test_answers:list=None, opts:Any=None):
    # imort these here because their imports are slow 
    import tensorflow as tf
    import keras
    from keras import layers

    (names,data,answers,test_names,test_data,test_answers) = training

    print("Data (x):",len(data)," Test (y):", len(test_data))

    DROPOUT = 0.15

    nfeat = len(data[0])
    inputs = keras.Input(shape=(nfeat,))
    n = inputs
    if DROPOUT > 0: n = layers.Dropout(DROPOUT)(n)
    n = layers.Dense(nfeat, activation='linear')(n)
    # don't know which activation to use? no problem, just use them all
    n = layers.Dense(nfeat, activation='sigmoid')(n)
    #if DROPOUT > 0: n = layers.Dropout(DROPOUT)(n)
    #n = layers.Dense(nfeat, activation='tanh')(n)
    if DROPOUT > 0: n = layers.Dropout(DROPOUT)(n)
    n = layers.Dense(nfeat, activation='relu')(n)
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
        #keras.callbacks.EarlyStopping(monitor='loss', patience=500),
        keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=500),
    ]
    if test_answers:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=500))
    
    history = model.fit(train_dataset, epochs=5000, shuffle=True, callbacks=callbacks, validation_data=test_dataset)

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
