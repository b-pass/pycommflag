import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # shut up, tf

import logging as log
import gc
from math import ceil, floor, isqrt
from queue import Empty as QueueEmpty
from multiprocessing import Process, Queue

import resource
import sys
import tempfile
import time
from typing import Any, Iterator,TextIO,BinaryIO,List,Tuple

import numpy as np
import signal
import random

from .feature_span import *
from . import processor
from . import neural

random.seed(42)

# data params, both for train and for inference
WINDOW_BEFORE = 60
WINDOW_AFTER = 60
SUMMARY_RATE = 1
RATE = 29.97

# training params
RNN = 'conv'
DEPTH = 8
F = 48
K = 13
UNITS = 32
DROPOUT = 0.4
EPOCHS = 50
BATCH_SIZE = 64
TEST_PERC = 0.25
PATIENCE = 5

def build_model(input_shape=(121,17)):
    from keras import layers, regularizers, utils, Input, Model

    utils.set_random_seed(17)

    inputs = Input(shape=input_shape[-2:], dtype='float32', name="input")
    n = inputs

    l2reg = regularizers.l2(0.00002)
    residual = None

    for d in range(3):
        if d >= DEPTH:
            break
        n = layers.Conv1D(filters=F, kernel_size=K, padding='same', activation='relu', kernel_regularizer=l2reg)(n)
        n = layers.BatchNormalization()(n)
        n = layers.SpatialDropout1D(DROPOUT)(n)
        n = layers.MaxPooling1D(pool_size=2)(n)
    
    n = layers.Conv1D(filters=F, kernel_size=K, padding='same', activation='relu', kernel_regularizer=l2reg)(n)
    n = layers.BatchNormalization()(n)
    n = layers.SpatialDropout1D(DROPOUT)(n)
     
    residual = n

    attn = layers.GlobalAveragePooling1D()(n) # Squeeze
    attn = layers.Dense(n.shape[-1] // 8, activation='relu')(attn) # bottleneck
    attn = layers.Dense(n.shape[-1], activation='sigmoid')(attn) # Excite
    attn = layers.Reshape((1, n.shape[-1]))(attn) # fix dimensions
    n = layers.Multiply()([n, attn]) # apply SE

    n = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=F // 8,
        dropout=DROPOUT
    )(n, n)  # self-attention
    n = layers.LayerNormalization()(n)

    for Ks in [7, 5, 5, 3, 3]:
        if d >= DEPTH:
            break
        d += 1

        n = layers.Conv1D(filters=F, kernel_size=Ks, padding='same', activation='relu', kernel_regularizer=l2reg)(n)
        n = layers.BatchNormalization()(n)
        n = layers.SpatialDropout1D(DROPOUT)(n)

    n = layers.Add()([n, residual])

    x = layers.GlobalAveragePooling1D()(n)
    y = layers.GlobalMaxPooling1D()(n)
    n = layers.Concatenate()([x,y])
    #n = layers.Flatten()(n)

    n = layers.Dense(UNITS, dtype='float32', activation='relu', kernel_regularizer=l2reg)(n)
    n = layers.Dropout(DROPOUT)(n)

    n = layers.Dense(UNITS, dtype='float32', activation='relu', kernel_regularizer=l2reg)(n)
    n = layers.Dropout(DROPOUT)(n)
    
    outputs = layers.Dense(1, dtype='float32', activation='sigmoid', name="output")(n)
    
    return Model(inputs, outputs)

def _adjust_tags(tags: List[Tuple[int, Tuple[float, float]]], 
                 blanks: List[Tuple[bool, Tuple[float, float]]], 
                 diffs: List[Tuple[float, Tuple[float, float]]]) \
        -> List[Tuple[int, Tuple[float, float]]]:
    """
    Adjust tag boundaries to align with scene transitions using blank frames and diff values.
    This is because the training data is supplied by humans, and might be off a bit.

    Args:
        tags: List of (tag_type, (start_time, end_time))
        blanks: List of (is_blank, (start_time, end_time))
        diffs: List of (diff_value, (start_time, end_time))
    
    Returns:
        Adjusted tags list with updated boundaries
    """

    def find_highest_diff_boundary(target_time: float, 
                                 search_window: float, 
                                 diffs: List[Tuple[float, Tuple[float, float]]]) -> float:
        # Find diffs within window
        window_start = target_time - search_window
        window_end = target_time + search_window
        best = 0
        when = None

        for (val, (s,e)) in diffs:
            time = s + (e-s)/2
            if time < window_start or not val:
                continue
            if time > window_end:
                break
            if val > best:
                best = val
                when = time
        
        return when if when is not None else target_time

    def find_nearest_blank(target_time: float, 
                         blanks: List[Tuple[bool, Tuple[float, float]]], 
                         max_distance: float) -> float:

        best = None
        dist = max_distance
        
        for is_blank, (start, end) in blanks:
            if not is_blank:
                continue

            when = start + (end - start)/2
            
            if when > target_time + max_distance:
                break

            if abs(when - target_time) < dist:
                dist = abs(when - target_time)
                best = when
        
        return best if best is not None else target_time

    # Process each tag
    filtered_tags = []
    for tag_type, (start_time, end_time) in tags:
        if tag_type in (SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value):
            filtered_tags.append((tag_type, (start_time, end_time)))
            continue

        # Different max distances based on tag type
        max_distance = 10 if tag_type in {SceneType.SHOW, SceneType.SHOW.value, 
                                        SceneType.COMMERCIAL, SceneType.COMMERCIAL.value} else 2
        
        # First try to align with blank frames
        new_start = find_nearest_blank(start_time, blanks, max_distance)
        new_end = find_nearest_blank(end_time, blanks, max_distance)
        
        # If still at original positions, try aligning with diff boundaries
        if new_start == start_time:
            new_start = find_highest_diff_boundary(start_time, 2, diffs)
        #    if new_start != start_time:
        #        print(f"MOVED tag start {tag_type} from {start_time} {new_start}")
        #else:
        #    print(f"ALIGNED tag start {tag_type} from {start_time} {new_start}")
        if new_end == end_time:
            new_end = find_highest_diff_boundary(end_time, 2, diffs)
        #    if new_end != end_time:
        #        print(f"MOVED tag end {tag_type} from {end_time} {new_end}")
        #else:
        #    print(f"ALIGNED tag end {tag_type} from {end_time} {new_end}")
        
        # Only keep valid tags
        if new_start < new_end:
            filtered_tags.append((tag_type, (new_start, new_end)))
    
    return filtered_tags

def condense(frames: np.ndarray, step: int) -> np.ndarray:
    """
    Summarize video features by aggregating the specified step size.
    """
    if step <= 1:
        return frames

    def doit(a):
        # first, average everything
        res = []
        res.append(np.percentile(a[:, :, 3], 90, axis=1)) # diff 90th
        res.append(np.max(a[:, :, 3], axis=1)) # diff max
        for f in [4,5]: # fvol, rvol
            res.append(np.max(a[:, :, f], axis=1)) # vol max
            res.append(np.std(a[:, :, f], axis=1)) # vol std dev

        res = [np.average(a[:, :, 0:6], axis=1)] + [x.reshape(x.shape[0], 1) for x in res] + [np.average(a[:, :, 6:], axis=1)]

        res[0][:, 0] = a[:, a.shape[1]//2, 0] # Use the middle timestamp
        res[0][:, 3] = np.count_nonzero(a[:, :, 3] >= 0.5, axis=1) / a.shape[1]  # Diff count above 0.5
        
        res[-1][:, -2] = np.count_nonzero(a[:, :, -2] >= 0.5, axis=1) / a.shape[1] # answer is proportional 
        
        return np.concatenate([x.reshape((x.shape[0], 1)) if len(x.shape) == 1 else x for x in res], axis=1)
    
    n_frames = len(frames)
    remaining = n_frames % step
    if n_frames >= step:
        # Reshape the array to group frames by step size
        condensed = doit( frames[:(n_frames//step)*step].reshape(-1, step, frames.shape[1]) )
    else:
        condensed = None
    
    if remaining > 0:
        # Do the final, partial condensing
        partial = doit( frames[-remaining:].reshape(-1, remaining, frames.shape[1]) )

        if condensed is None:
            return partial
        return np.vstack((condensed, partial))
    return condensed

def load_nonpersistent(flog:dict, for_training=False)->np.ndarray:
    version = flog.get('file_version', 10)
    frame_rate = flog.get('frame_rate', 29.97)
    endtime = flog.get('duration', 0)

    if endtime < (WINDOW_BEFORE + 1 + WINDOW_AFTER) * 2:
        return None

    have_logo = not not flog.get('logo', None)

    tags = flog.get('tags', [])
    
    frames_header = flog['frames_header']
    assert('time' in frames_header[0])
    assert('diff' in frames_header[3])

    if tags and for_training:
        spans = processor.read_feature_spans(flog, 'blank')
        tags = _adjust_tags(tags, spans.get('blank', []), spans.get('diff', []))
        
        # clean up tiny gaps between identified breaks (including true 0-length gaps)
        i = 1
        while i < len(tags):
            if tags[i][0] == tags[i-1][0] and (tags[i][1][0] - tags[i-1][1][1]) < 15:
                tags[i-1] = (tags[i][0], (tags[i-1][1][0], tags[i][1][1]))
                del tags[i]
            else:
                i += 1
        
        i = 0
        while i < len(tags):
            clen = tags[i][1][1] - tags[i][1][0]
            if clen < 10 and not (tags[i][1][1]+clen+10 >= endtime or tags[i][0] in [SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value]):
                # delete the tiny segment
                del tags[i]
            else:
                i += 1
    
    frames = flog['frames']

    if frames and frames[0] is None: 
        frames = frames[1:]
    
    if len(frames) < frame_rate:
        return None
    
    if tags and tags[-1][0] in (SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value) and tags[-1][1][1]+10 >= endtime:
        flog['duration'] = endtime = min(endtime, tags[-1][1][0])
        e = len(frames) - 1
        while frames[e][0] >= endtime and e > 0:
            e -= 1
        if e > 0:
            del frames[e+1:]
        del tags[-1]
    
    if tags and tags[0][0] in (SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value) and tags[0][1][0] <= 5:
        b = 0
        while frames[b][0] < tags[0][1][1]:
            b += 1
        if b:
            del frames[:b]
        del tags[0]

    if len(frames) < (WINDOW_BEFORE + WINDOW_AFTER) * 2 * round(RATE):
        return None
    
    # ok now we can numpy....
    frames = np.array(frames, dtype='float32')

    if not have_logo:
        frames[..., 1] = 0

    # change the diff column to be normalized [0,30] -> [0,1]
    frames[...,3] = np.clip(frames[...,3] / 30, 0, 1.0)

    # add a column for time percentage [-4]
    frames = np.append(frames, (frames[...,0]/endtime)[:,np.newaxis], axis=1)

    # add a column for with the real timestamps [-3]
    frames = np.append(frames, frames[...,0].reshape((-1,1)), axis=1)

    # add a column for answers [-2]
    frames = np.append(frames, np.zeros((frames.shape[0],1), dtype=np.float32), axis=1)

    # add a column for weights [-1]
    frames = np.append(frames, np.ones((frames.shape[0],1), dtype=np.float32), axis=1)

    # change the first column to be normalized timestamps (30 minute segments)
    frames[...,0] = (frames[...,0] % 1800.0) / 1800.0

    wbefore = round(WINDOW_BEFORE * SUMMARY_RATE)
    wafter = round(WINDOW_AFTER * SUMMARY_RATE)

    timestamps = frames[:, -3]
    answers = frames[:, -2]
    weights = frames[:, -1]

    for (tt,(st,et)) in tags:
        if type(tt) is not int: tt = tt.value

        si = np.searchsorted(timestamps, st)
        ei = np.searchsorted(timestamps, et)

        if tt == SceneType.DO_NOT_USE.value:
            weights[si:ei] = 0 # ignore this entire section
        elif tt == SceneType.COMMERCIAL.value:
            answers[si:ei] = 1.0
        elif tt != SceneType.SHOW.value:
            weights[si:ei] = 0.75
    
    condensed = condense(frames, round(frame_rate/SUMMARY_RATE))
    
    #for x in [0,1]:
    #    print(f'{x}) {np.count_nonzero(answers == x)}')

    condensed = np.concatenate((
        np.tile(condensed[0], (wbefore,1)),
        condensed,
        np.tile(condensed[-1], (wafter,1)),
    ))

    return condensed

def load_persistent(flogname:str,for_training=True):
    fname = flogname
    if fname.endswith('.npy'):
        fname = fname[:-4]
    if fname.endswith('.gz'):
        fname = fname[:-3]
    if fname.endswith('.json'):
        fname = fname[:-5]
    
    if not os.path.exists(fname + '.data.npy'):
        condensed = load_nonpersistent(processor.read_feature_log(flogname), for_training)
        np.save(fname+'.data.npy', condensed)
    
    condensed = np.load(fname+'.data.npy', mmap_mode='r')
    return condensed

def make_data_generator(*args, **kwargs):
    from keras.utils import Sequence
    class DataGenerator(Sequence):
        def __init__(self, data, answers=None, weights=None):
            self.data = np.array(data, dtype='float32')
            self.answers = np.array(answers, dtype='float32') if answers else None
            self.weights = np.array(weights, dtype='float32') if weights else None
            self.len = ceil(len(self.data) / BATCH_SIZE)
            self.shape = (self.len, BATCH_SIZE, len(data[0]), len(data[0][0]))
        
        def __len__(self):
            return self.len
        
        def __getitem__(self, index):
            index *= BATCH_SIZE
            d = self.data[index:index+BATCH_SIZE]
            if self.answers is not None:
                a = self.answers[index:index+BATCH_SIZE]
                if self.weights is not None:
                    w = self.weights[index:index+BATCH_SIZE]
                    return d,a,w
                else:
                    return d,a
            else:
                return d
    return DataGenerator(*args, **kwargs)

def load_data_sliding_window(condensed:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if condensed is None:
        return ([],[],[],[])
    
    wbefore = round(WINDOW_BEFORE * SUMMARY_RATE)
    wafter = round(WINDOW_AFTER * SUMMARY_RATE)

    timestamps = condensed[wbefore:-wafter,-3]
    answers = condensed[wbefore:-wafter,-2]
    weights = condensed[wbefore:-wafter,-1]
    condensed = condensed[:, :-3]
    
    from numpy.lib.stride_tricks import sliding_window_view
    frames = sliding_window_view(condensed, (wbefore+1+wafter, condensed.shape[1],)).squeeze()

    #print(len(self.frames), len(self.timestamps))

    assert(np.shares_memory(condensed, frames))
    assert(len(frames) == len(timestamps))
    assert(len(frames) == len(answers))
    assert(len(frames) == len(weights))

    return frames, answers, weights, timestamps

def load_data(opts, do_not_test=False) -> tuple:
    datafiles = opts.ml_data
    if not datafiles:
        return None
    
    testfiles = []

    if 'TEST' in datafiles:
        i = datafiles.index('TEST')
        testfiles = datafiles[i+1:]
        datafiles = datafiles[:i]

        i = 0
        while i < len(datafiles):
            if not os.path.exists(datafiles[i]):
                print(datafiles[i], "does not exist!!")
                del datafiles[i]
            elif datafiles[i] in testfiles or not os.path.isfile(datafiles[i]):
                del datafiles[i]
            else:
                i += 1

    dlen = 0
    data = ([],[],[])
    tlen = 0
    test = ([],[],[])

    for f in datafiles:
        if os.path.isdir(f) or f.endswith('.npy'):
            continue
        print("Loading",f)
        stuff = load_data_sliding_window(load_persistent(f))
        if stuff is not None:
            dlen += len(stuff[0])
            for x in range(3):
                for i in range(len(stuff[x])):
                    data[x].append(stuff[x][i])
    
    for f in testfiles:
        if os.path.isdir(f) or f.endswith('.npy'):
            continue
        print("Loading test",f)
        stuff = load_data_sliding_window(load_persistent(f))
        if stuff is not None:
            tlen += len(stuff[0])
            for x in range(3):
                for i in range(len(stuff[x])):
                    test[x].append(stuff[x][i])
    stuff = None

    data = list(zip(*data))
    random.shuffle(data)

    if not do_not_test:
        need = int(dlen*TEST_PERC+1) - tlen
        if need > dlen/100 and tlen/(tlen+dlen) < 0.1:
            print(f'Need to move {need} of {dlen} elements to the test/eval set (have {tlen} will have ~{need+tlen})')
            for i in range(need):
                e = data[i]
                test[0].append(e[0])
                test[1].append(e[1])
                test[2].append(e[2])
            data = data[need:]
    
    data = make_data_generator(*zip(*data))
    test = make_data_generator(*test) if test else None
    
    return data,test

def train(opts:Any=None):
    # yield CPU time to useful tasks, this is a background thing...
    try: os.nice(19)
    except: pass

    (data,test) = load_data(opts)
    
    #print('Calculating loss weights')
    #sums = np.sum((np.sum(answers, axis=0), np.sum(test_answers, axis=0)), axis=0)
    ##weights = [1] * len(sums)
    ##weights[SceneType.SHOW.value] = sums[SceneType.COMMERCIAL.value] / sums[SceneType.SHOW.value]
    ##weights[SceneType.COMMERCIAL.value] = sums[SceneType.SHOW.value] / sums[SceneType.COMMERCIAL.value]
    #sums += 1
    #weights = (np.sum(sums)-sums)/np.sum(sums)
    #print("Loss Weights",weights)
    
    print(f"Data shape (x):{data.shape} - Test shape (y):{test.shape if test is not None else 'None'}")
    
    tfile = tempfile.NamedTemporaryFile(prefix='train-', suffix='.pycf.model.keras', )
    model_path = tfile.name
    
    stop = False
    epoch = 0

    #model_path = '/tmp/train-xyz.pycf.model.keras'
    #epoch = 10

    _train_some(model_path, data, test, epoch)
        
    print()
    print("Done")
    print()
    print('Final Evaluation...')

    import keras
    #dmetrics = model.evaluate(data, verbose=0)
    tmetrics = keras.models.load_model(model_path).evaluate(test, verbose=1)
    print()
    #print(dmetrics)
    print(tmetrics)

    if tmetrics[1] >= 0.80:
        name = f'{opts.models_dir if opts and opts.models_dir else "."}{os.sep}pycf-{tmetrics[1]:.04f}-c{F}x{K}x{DEPTH}-{RNN}{UNITS}-w{WINDOW_BEFORE}x{WINDOW_AFTER}-{int(time.time())}.keras'
        print()
        print('Saving as ' + name)

        import shutil
        shutil.copy(model_path, name)
        try: os.chmod(name, 0o644)
        except: pass
    
    print()

    return 0

def _train_some(model_path, train_dataset, test_dataset, epoch=0) -> tuple[int,bool]:
    import keras

    model:keras.models.Model = None
    if epoch > 0:
        model = keras.models.load_model(model_path)
    else:
        from keras.metrics import Recall, Precision
        model = build_model(train_dataset.shape)
        model.summary()
        model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy(label_smoothing=0), metrics=['accuracy', Recall(class_id=0), Precision(class_id=0)])
        model.save(model_path)
    
    gc.collect()

    cb = []

    from keras import callbacks

    cb.append(callbacks.EarlyStopping(monitor='loss', patience=PATIENCE))
    cb.append(callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE))

    def cosine_annealing_with_warmup(epoch, lr):
        WARMUP = 3
        FINETUNE = 25
        
        if epoch < WARMUP:
            # Warm up
            return 0.001 * (epoch + 1) / WARMUP
        
        # Cosine annealing 
        progress = (epoch - WARMUP) / (FINETUNE - WARMUP)
        return max( 0.0001 + (0.001 - 0.0001) * 0.5 * (1 + np.cos(np.pi * progress)), 0.00001 )
    
    cb.append(callbacks.LearningRateScheduler(cosine_annealing_with_warmup))
    #cb.append(callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=PATIENCE-1))
    
    class EpochModelCheckpoint(callbacks.ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            self.last_epoch = epoch
            return super().on_epoch_end(epoch, logs)

    ecp = EpochModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
    cb.append(ecp)

    def handler(signum, frame):
        print("\nStopping (gracefully)...\n")
        model.stop_training = True
        signal.signal(signal.SIGINT, oldsint)
        signal.signal(signal.SIGTERM, oldterm)
        return
    oldsint = signal.signal(signal.SIGINT, handler)
    oldterm = signal.signal(signal.SIGTERM, handler)

    model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, initial_epoch=epoch, callbacks=cb, class_weight={0:0.65, 1:1/0.65})

    #model.save(model_path) the checkpoint already saved the vest version
    return (ecp.last_epoch+1, model.stop_training or ecp.last_epoch+1 >= EPOCHS)

def predict(feature_log:str|TextIO|dict, opts:Any, write_log=None)->list:
    from .mythtv import set_job_status
    set_job_status(opts, "Inferencing...")

    import tensorflow as tf
    import keras

    flog = processor.read_feature_log(feature_log)
    frame_rate = flog.get('frame_rate', 29.97)
    
    assert(flog['frames'][-1][0] > frame_rate)

    mf = opts.model_file
    if not mf and opts:
        mf = f'{opts.models_dir or "."}{os.sep}model.keras'
    if not os.path.exists(mf):
        blah = mf
        mf = f'{opts.models_dir or "."}{os.sep}model.h5'
        if not os.path.exists(mf):
            raise Exception(f"Model files '{blah}' or '{mf}' do not exist")
    
    model:keras.models.Model = keras.models.load_model(mf)

    results = do_predict(flog, model, opts)
    if not results:
        results = []

    #if orig_tags := flog.get('tags', []):
    #    log.debug(f'OLD tags n={len(orig_tags)} -> {str(orig_tags)}')
    #    log.debug(f'NEW tags n={len(results)} -> {str(results)}')
    #else:

    log.debug(f'Final tags n={len(results)}: {str(results)}')

    flog['tags'] = results

    if write_log is not None:
        processor.write_feature_log(flog, write_log)
    
    return results

def do_predict(flog:dict, model, opts:Any):
    duration = flog.get('duration', 0)
    assert(model.output_shape[-1] == 1)
    
    data = load_nonpersistent(flog, False)
    data,_,_,times = load_data_sliding_window(data)

    prediction = model.predict(make_data_generator(data), verbose=True)

    results = [(0,(0,0))]
    for i in range(len(prediction)):
        when = float(times[i])
        ans = SceneType.COMMERCIAL.value if prediction[i] >= 0.5 else SceneType.SHOW.value
        
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

    #log.debug(f'Raw result n={len(results)}')

    # clean up tiny gaps between identified breaks (including true 0-length gaps)
    # show must be at least 30 seconds long (opts.show_min_len), or we just combine it into the commercial break its in the middle of
    i = 1
    while i < len(results):
        if results[i][0] == results[i-1][0] and (results[i][1][0] - results[i-1][1][1]) < opts.show_min_len:
            results[i-1] = (results[i][0], (results[i-1][1][0], results[i][1][1]))
            del results[i]
        else:
            i += 1

    spans = processor.read_feature_spans(flog, 'diff', 'blank')
    
    results = _adjust_tags(results, spans.get('blank', []), spans.get('diff', []))
    i = 1
    while i < len(results):
        if results[i][0] == results[i-1][0] and (results[i][1][0] - results[i-1][1][1]) < opts.show_min_len:
            results[i-1] = (results[i][0], (results[i-1][1][0], results[i][1][1]))
            del results[i]
        else:
            i += 1

    #log.debug(f'Merge/Adjust n={len(results)}: {str(results)}')

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
                    results[i+1] = (results[i+1][0], (nextstart, results[i+1][1][1]))
                    break

        # its ok now, move on
        i += 1
    
    #log.debug(f'Post n={len(results)}: {str(results)}')

    return results

def diff_tags(realtags, result) -> tuple[float,float,list]:
    # we create a list of tag pairs where they always exactly line up with boundaries in another list
    # this means we don't have to handle overlaps or tags spanning multiple other tags
    def split_upon(inlist, splitlist):
        sres = []
        for it,(ib,ie) in inlist:
            if it not in [SceneType.COMMERCIAL, SceneType.COMMERCIAL.value]:
                continue
            for st,(sb,se) in splitlist:
                if st not in [SceneType.COMMERCIAL, SceneType.COMMERCIAL.value,SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value]:
                    continue
                if ib < sb and sb < ie:
                    sres.append( (ib,sb) )
                    ib = sb
                if se > ib and se <= ie:
                    sres.append( (ib, se) )
                    ib = se
                if ib+1/30 > ie:
                    break
            if ib + 1/30 <= ie:
                sres.append( (ib, ie) )
        
        for st,(sb,se) in splitlist:
            if st in [SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value]:
                for i in range(len(sres)):
                    rb,re = sres[i]
                    if rb < se and re > sb:
                        del sres[i]
                        break

        #print("SPLIT", inlist, "into", sres)
        return sres

    orig = split_upon(realtags, result)
    result = split_upon(result, realtags)

    #print(orig)
    #print(result)

    missing = 0
    extra = 0
    rlist = []

    # now we have no overlaps, so all entries are one of: missing, extra, same
    ri = 0
    for ob,oe in orig:
        while ri < len(result):
            rb,re = result[ri]
            if ob < re:
                break
            else:
                extra += re - rb
                rlist.append( (1,rb,re) )
            ri += 1
        if ri >= len(result) or oe <= rb:
            missing += oe - ob
            rlist.append( (-1,ob,oe) )
        else:
            rlist.append( (0,ob,re) )
            assert(rb == ob and re == oe)
            ri += 1
    while ri < len(result):
        rb,re = result[ri]
        extra += re - rb
        rlist.append( (1,rb,re) )
        ri += 1
    
    #print("rlist:",rlist)
    
    return missing, extra, rlist


def eval(opts:Any):
    # yield CPU time to useful tasks, this is a background thing...
    try: os.nice(19)
    except: pass

    datafiles = []
    if opts.ml_data is None: opts.ml_data = []
    
    for f in opts.ml_data:
        if os.path.isdir(f) or f.endswith('.npy'):
            continue
        else:
            datafiles.append(f)
    
    print("EVALUATE", len(opts.eval), "on", len(datafiles))

    import keras

    models = {}
    total_time = 0
    all_missing = {}
    all_extra = {}
    for mf in opts.eval:
        try:
            models[mf] = keras.models.load_model(mf)
            print(mf)
            models[mf].summary()
            all_missing[mf] = 0
            all_extra[mf] = 0
        except Exception as e:
            log.exception(f"Unable to load MODEL {mf}")

    for f in datafiles:
        try:
            flog = processor.read_feature_log(f)
            if not flog:
                print('Load failed')
                continue

            spans = processor.read_feature_spans(flog, 'diff', 'blank')

            duration = flog.get('duration',0.00001)
            real_dur = duration
            realtags = []

            for (t,(b,e)) in flog.get('tags', []):
                if t in [SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value]:
                    if e+10 > real_dur and e < real_dur and b < real_dur:
                        e = real_dur
                    if b < 10 and e > 0:
                        b = 0
                    duration -= e-b
                elif t in [SceneType.COMMERCIAL, SceneType.COMMERCIAL.value]:
                    realtags.append( (t,(b,e)) )

            realtags = _adjust_tags(realtags, spans.get('blank', []), spans.get('diff', []))
            
            total_time += duration
        except Exception as e:
            log.exception(f"Unable to load {f}")
            continue
        
        for (mf, model) in models.items():
            try:
                result = do_predict(flog, model, opts)
                (missing,extra,_) = diff_tags(realtags, result)

                acc = 100 - 100*(missing+extra)/duration
                print(f'{f} @ {mf} -> Acc {round(acc,4)}% <- FN:{round(missing,3)} + FP:{round(extra,3)} = {round(missing+extra,3)} seconds WRONG')
                all_missing[mf] += missing
                all_extra[mf] += extra
                #print(f,mf,model.evaluate(dataset_cat if model.output_shape[-1] > 1 else dataset_bin))
            except Exception as e:
                log.exception(f"Unable to load MODEL {mf}")
        print()
    
    print()
    print()
    print("Results:\nName\t\tMissing\t\tExtra\t\tAbsDiffSum\t\tAcc")

    for mf in models.keys():
        total_wrong = all_extra[mf] + all_missing[mf]
        acc = '%.5f %%' % ((total_time - total_wrong) * 100.0 / total_time,)
        print(f'{mf}: \t-{all_missing[mf]} \t+{all_extra[mf]} \t{total_wrong} \t{acc}')
    print()

