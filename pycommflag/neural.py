import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # shut up, tf

import logging as log
import gc
from math import ceil, floor
from queue import Empty as QueueEmpty
from multiprocessing import Process, Queue

import resource
import sys
import tempfile
import time
from typing import Any, Iterator,TextIO,BinaryIO,List,Tuple

import numpy as np

from .feature_span import *
from . import processor
from . import neural

# data params, both for train and for inference
WINDOW_BEFORE = 30
WINDOW_AFTER = 30
SUMMARY_RATE = 2
RATE = 29.97

# training params
RNN = 'lstm'
UNITS = 32
DROPOUT = 0.4
EPOCHS = 40
BATCH_SIZE = 256
TEST_PERC = 0.25

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
        duration: Total duration of the video
    
    Returns:
        Adjusted tags list with updated boundaries
    """

    def find_highest_diff_boundary(target_time: float, 
                                 search_window: float, 
                                 diffs: List[Tuple[float, Tuple[float, float]]]) -> float:
        # Find diffs within window
        window_start = target_time - search_window
        window_end = target_time + search_window
        
        # Filter diffs within window
        candidates = [
            (diff_val, (start, end)) for diff_val, (start, end) in diffs 
            if window_start <= start <= window_end or 
               window_start <= end <= window_end
        ]
        
        if not candidates:
            return target_time
            
        # Find highest diff value and its corresponding boundary
        max_diff = max(candidates, key=lambda x: x[0])
        start, end = max_diff[1]
        
        # Return the closest boundary to target_time
        return start if abs(start - target_time) < abs(end - target_time) else end

    def find_nearest_blank(target_time: float, 
                         blanks: List[Tuple[bool, Tuple[float, float]]], 
                         max_distance: float,
                         left=True) -> float:

        nearest_time = None
        min_distance = max_distance
        
        for is_blank, (start, end) in blanks:
            if not is_blank:
                continue
            
            if start > target_time + max_distance:
                break

            if abs(start - target_time) < min_distance:
                min_distance = abs(start - target_time)
                nearest_time = (start,end)
            
            if abs(end - target_time) < min_distance:
                min_distance = abs(end - target_time)
                nearest_time = (start,end)
        
        # we exclude the blanks from the tag, so on the left side we use the end/nearest[1]
        # and on the right we use the begin/nearest[0]

        return nearest_time[int(left)] if nearest_time is not None else target_time

    # Process each tag
    filtered_tags = []
    for tag_type, (start_time, end_time) in tags:
        # Different max distances based on tag type
        max_distance = 10 if tag_type in {SceneType.SHOW, SceneType.SHOW.value, 
                                        SceneType.COMMERCIAL, SceneType.COMMERCIAL.value} else 2
        
        # First try to align with blank frames
        new_start = find_nearest_blank(start_time, blanks, max_distance, left=True)
        new_end = find_nearest_blank(end_time, blanks, max_distance, left=False)
        
        # If still at original positions, try aligning with diff boundaries
        if new_start == start_time:
            new_start = find_highest_diff_boundary(start_time, 2, diffs)
        if new_end == end_time:
            new_end = find_highest_diff_boundary(end_time, 2, diffs)
        
        # Only keep valid tags
        if new_start < new_end:
            filtered_tags.append((tag_type, (new_start, new_end)))
    
    return filtered_tags

def condense(frames: np.ndarray, step: int, summarize=True) -> np.ndarray:
    """
    Condense video frames by averaging specific features over specified step sizes.
    
    Args:
        frames: numpy array of shape (n_frames, n_features) where features at indices:
               1: logo
               2: blank
               3: diff
               Are summarized.
               All other features are preserved from the first frame of each group.
        step: number of frames to combine into one
        summarize: 
                True: summarize and aggregate feature values
                False: try to pick the most important value instead.
    
    Returns:
        Condensed numpy array with averaged features for logo, blank, diff
        and preserved values for other features
    """
    if step <= 1:
        return frames

    # audio features are spread across time anyway so we dont need to sumarize them unless the condense step is more than 0.5s
    
    n_frames = len(frames)
    remaining = n_frames % step
    if n_frames >= step:
        # Reshape the array to group frames by step size
        valid_frames = frames[:(n_frames//step)*step].reshape(-1, step, frames.shape[1])
        
        # Initialize condensed array with middle frame of each group
        # This preserves all features that don't need averaging
        condensed = valid_frames[:, step//2].copy()

        # Update only the features that need condensing
        if summarize:
            condensed[:, 1] = np.mean(valid_frames[:, :, 1], axis=1, dtype='float32')  # Average logo
            condensed[:, 2] = np.mean(valid_frames[:, :, 2], axis=1, dtype='float32')  # Average blank
            condensed[:, 3] = np.count_nonzero(valid_frames[:, :, 3] >= 0.5, axis=1) / step  # Diff ratio
        else:
            condensed[:, 1] = np.mean(valid_frames[:, :, 1], axis=1, dtype='float32')  # Logo percentage
            condensed[:, 2] = np.max(valid_frames[:, :, 2], axis=1)  # Blank present
            condensed[:, 3] = np.max(valid_frames[:, :, 3], axis=1)  # Diff max
    else:
        condensed = None
    
    if remaining > 0:
        # Do the final, partial condensing
        last_frames = frames[-remaining:]

        last_condensed = last_frames[remaining//2].copy()  # Keep all features from middle frame

        if summarize:
            last_condensed[1] = np.mean(last_frames[:, 1], dtype='float32')  # Average logo
            last_condensed[2] = np.mean(last_frames[:, 2], dtype='float32')  # Average blank
            last_condensed[3] = np.count_nonzero(last_frames[:, 3] >= 0.5) / remaining  # Diff ratio
        else:
            last_condensed[1] = np.mean(last_frames[:, 1], dtype='float32')  # Logo percentage
            last_condensed[2] = np.max(last_frames[:, 2])  # Blank present
            last_condensed[3] = np.max(last_frames[:, 3])  # Diff max
        
        if condensed is not None:
            condensed = np.vstack((condensed, last_condensed))
        else:
            condensed = last_condensed
    
    return condensed

def array2onehot(arr, nclasses, dtype='float32'):
    return np.eye(nclasses, dtype=dtype)[arr]

def flog_to_vecs(flog:dict, fitlerForTraining=False)->tuple[np.ndarray,np.ndarray]:
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

    if tags and fitlerForTraining:
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

    # add a column for time percentage
    frames = np.append(frames, (frames[...,0]/endtime)[:,np.newaxis], axis=1)

    # add a column for with the real timestamps [-3]
    frames = np.append(frames, frames[...,0].reshape((-1,1)), axis=1)

    # add a column for answers [-2]
    # and a column for weights [-1]
    frames = np.insert(frames, frames.shape[1], [[SceneType.SHOW.value], [1]], axis=1)

    # change the first column to be normalized timestamps (30 minute segments)
    frames[...,0] = (frames[...,0] % 1800.0) / 1800.0

    # normalize frame rate
    frames = condense(frames, round(frame_rate/RATE), summarize=False)

    # +/- 1s is all frames, plus the WINDOW before/after which is condensed to SUMMARY_RATE 
    rate = round(RATE)
    summary = round(RATE/SUMMARY_RATE)
    wbefore = round(WINDOW_BEFORE * SUMMARY_RATE)
    wafter = round(WINDOW_AFTER * SUMMARY_RATE)

    timestamps = frames[:, -3]
    answers = frames[:, -2]
    weights = frames[:, -1]

    # dont condense timestamps, answers, weights
    condensed = condense(frames[...,:-3], summary, summarize=True)

    for (tt,(st,et)) in tags:
        if type(tt) is not int: tt = tt.value

        si = np.searchsorted(timestamps, st)
        ei = np.searchsorted(timestamps, et)

        if tt == SceneType.DO_NOT_USE.value:
            weights[si:ei] = 0 # ignore this entire section
        else:
            answers[si:ei] = tt
            
            # higher weight near the transition
            for i in (si,ei):
                ws = max(0,i-rate*3)
                we = min(len(weights),i+rate*3)
                if ws > 0 and we < len(weights):
                    #weights[ws:we] = np.where(weights[ws:we] > 0, 2, 0)
                    dist = np.abs(np.arange(ws, we + 1) - i)
                    new_weights = 2.25 - ((dist / ((we - ws) / 2)) ** 2)  # Scale the distance by (rate * 3)
                    new_weights = np.clip(new_weights, 1.0, 2.0)  # Ensure weights are between 1.0 and 2.0
                    weights[ws:we + 1] = np.maximum(weights[ws:we + 1], new_weights, where=(weights[ws:we + 1] >= 0.5))

    #for w in range(3):
    #    print(f'{w}) {np.count_nonzero(weights == w)}')
    #for x in range(SceneType.count()+1):
    #    print(f'{SceneType(x)}) {np.count_nonzero(answers == x)}')
    
    frames = np.concatenate((
        np.tile(frames[0], (rate,1)),
        frames,
        np.tile(frames[-1], (rate,1)),
    ))

    condensed = np.concatenate((
        np.tile(condensed[0], (wbefore,1)),
        condensed,
        np.tile(condensed[-1], (wafter,1)),
    ))

    condensed = np.repeat(condensed, summary, axis=0)

    return (frames,condensed)

def load_persistent(flogname:str,for_training=True):
    fname = flogname
    if fname.endswith('.npy'):
        fname = fname[:-4]
    if fname.endswith('.gz'):
        fname = fname[:-3]
    if fname.endswith('.json'):
        fname = fname[:-5]
    
    if not os.path.exists(fname + ".frames.npy") or not os.path.exists(fname + '.summary.npy'):
        frames,condensed = flog_to_vecs(processor.read_feature_log(flogname), for_training)
        np.save(fname+'.frames.npy', frames)
        np.save(fname+'.summary.npy', condensed)
    
    frames = np.load(fname+'.frames.npy', mmap_mode='r')
    condensed = np.load(fname+'.summary.npy', mmap_mode='r')

    return (frames,condensed)

def load_nonpersistent(flog:dict,for_training=False):
    return flog_to_vecs(flog, for_training)

def sliding_window_skip(arr, window_size, repeat_skip):
    # Calculate the shape and strides for the sliding window view over an array that had np.repeat previously run on it
    from numpy.lib.stride_tricks import as_strided
    new_shape = (arr.shape[0] - (window_size-1)*repeat_skip, window_size, arr.shape[1])
    new_strides = (arr.strides[0], arr.strides[0] * repeat_skip, arr.strides[1])
    return as_strided(arr, shape=new_shape, strides=new_strides, writeable=False)

from keras.utils import Sequence
class WindowStackGenerator(Sequence):
    def __init__(self, stuff=None, ordered=True, categorical=True):
        super().__init__()
        if stuff is None:
            return
        
        # +/- 1s is all frames, plus the WINDOW before/after which is condensed to SUMMARY_RATE 
        rate = round(RATE)
        summary = round(RATE/SUMMARY_RATE)
        wbefore = round(WINDOW_BEFORE * SUMMARY_RATE)
        wafter = round(WINDOW_AFTER * SUMMARY_RATE)
        self.batch_size = BATCH_SIZE

        (frames,condensed) = stuff

        # frames has been tile with rate on each side and has timestamps,answers,weights concatted...
        # unpack all that
        self.timestamps = frames[rate:-rate,-3]
        if categorical:
            self.answers = array2onehot(frames[rate:-rate,-2].astype('uint32'), SceneType.count(), dtype='float32')
        else:
            self.answers = np.where(frames[rate:-rate,-2].astype('uint32') == SceneType.COMMERCIAL.value, 1.0, 0.0)
        self.weights = frames[rate:-rate,-1]
        frames = frames[:, :-3]
        
        from numpy.lib.stride_tricks import sliding_window_view
        self.frames = sliding_window_view(frames, (rate+1+rate, frames.shape[1],)).squeeze()
        self.before = sliding_window_skip(condensed[:-(wafter+1)*summary], wbefore, summary)
        self.after = sliding_window_skip(condensed[(wbefore+1)*summary:], wafter, summary)

        assert(np.shares_memory(frames, self.frames))
        assert(len(self.frames) == len(self.timestamps))
        assert(len(self.frames) == len(self.answers))
        assert(len(self.frames) == len(self.weights))

        # truncate these if they are a little over-size from being repeated
        assert(len(self.frames) <= len(self.before))
        assert(len(self.frames) <= len(self.after))
        self.before = self.before[:len(self.frames)]
        self.after = self.after[:len(self.frames)]
        assert(np.shares_memory(condensed, self.before))
        assert(np.shares_memory(condensed, self.after))

        self.stride = len(self.frames) // self.batch_size # we want every N to add up to about the Batch Size
        self.len = ceil(len(self.frames) / self.batch_size)
        self.shape = ( self.len, self.batch_size, self.before.shape[1] + self.frames.shape[1] + self.after.shape[1], self.frames.shape[2] )
        self.offset = 0

        self.set_ordered(ordered)

    def split(self, split=0.75):
        a = WindowStackGenerator()
        b = WindowStackGenerator()
        for (k,v) in self.__dict__.items():
            setattr(a, k, v)
            setattr(b, k, v)
        
        from random import randrange

        swap = randrange(2) == 0
        if swap:
            split = 1 - split

        a.set_ordered(False)
        a.offset = 0
        a.len = ceil( (len(self.frames) * split) / self.batch_size )
        a.shape = (a.len,) + a.shape[1:]

        b.set_ordered(False)
        b.offset = a.len
        b.len = self.len - a.len
        b.shape = (b.len,) + b.shape[1:]
        
        if swap:
            return (b,a)
        else:
            return (a,b)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.getter(index)
    
    def set_ordered(self, ordered=True):
        self.getter = self.get_ordered if ordered else self.get_unordered
    
    def get_ordered(self, index):
        index = (index + self.offset) * self.batch_size
        endex = index + self.batch_size
        data = np.concatenate((
            self.before[index:endex],
            self.frames[index:endex],
            self.after [index:endex],
        ), axis=1)
        return (data, self.answers[index:endex], self.weights[index:endex])
    
    def get_unordered(self, index):
        index += self.offset
        data = np.concatenate((
            self.before[index::self.stride],
            self.frames[index::self.stride],
            self.after [index::self.stride],
        ), axis=1)
        return (data, self.answers[index::self.stride], self.weights[index::self.stride])
    
class MultiGenerator(Sequence):
    def __init__(self, elements:list[WindowStackGenerator], **kwargs):
        super().__init__(**kwargs)
        self.elements = elements
        self.num_elements = len(self.elements)
        self._recalc()
    
    def _recalc(self):
        self.lengths = []
        self.offsets = []
        self.len = 0
        shape = self.elements[0].shape[1:]
        for e in self.elements:
            assert(e.shape[1:] == shape)
            #if e.shape[1:] != shape:
            #    print(e.shape[1:], "incompatible with", shape,"--",e.frames.shape, "vs", self.elements[0].frames.shape)
            
            x = len(e)
            self.lengths.append(x)
            self.offsets.append(self.len)
            self.len += x
        self.shape = (self.len,) + shape
        self.offsets = np.array(self.offsets, dtype='uint32')
        
    def shuffle(self):
        from random import shuffle
        shuffle(self.elements)
        for e in self.elements:
            e.set_ordered(False)
        self._recalc()
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        ei = np.searchsorted(self.offsets, index, 'right')-1
        return self.elements[ei][index - self.offsets[ei]]

def load_data(opts, do_not_test=False) -> tuple[MultiGenerator,MultiGenerator]:
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
    data = []
    tlen = 0
    test = []

    for f in datafiles:
        if os.path.isdir(f) or f.endswith('.npy'):
            continue
        print("Loading",f)
        if stuff := load_persistent(f, True):
            data.append( WindowStackGenerator(stuff, categorical=False) )
            dlen += len(data[-1])
    
    for f in testfiles:
        if os.path.isdir(f) or f.endswith('.npy'):
            continue
        print("Loading test",f)
        if stuff := load_persistent(f, True):
            test.append( WindowStackGenerator(stuff, categorical=False) )
            tlen += len(test[-1])
    
    need = int(dlen*TEST_PERC+1) - tlen
    if need > dlen/100 and not do_not_test and tlen < 10000:
        print(f'Need to move {need} of {dlen} batches to the test/eval set')
        datakeep = 1 - need/dlen
        orig = data
        data = []
        dlen = 0
        for o in orig:
            (d,t) = o.split(datakeep)
            dlen += len(d)
            tlen += len(t)
            data.append(d)
            test.append(t)
    
    return (MultiGenerator(data), MultiGenerator(test))

def train(opts:Any=None):
    # yield CPU time to useful tasks, this is a background thing...
    try: os.nice(19)
    except: pass

    (data,test) = load_data(opts)
    gc.collect()
    
    #print('Calculating loss weights')
    #sums = np.sum((np.sum(answers, axis=0), np.sum(test_answers, axis=0)), axis=0)
    ##weights = [1] * len(sums)
    ##weights[SceneType.SHOW.value] = sums[SceneType.COMMERCIAL.value] / sums[SceneType.SHOW.value]
    ##weights[SceneType.COMMERCIAL.value] = sums[SceneType.SHOW.value] / sums[SceneType.COMMERCIAL.value]
    #sums += 1
    #weights = (np.sum(sums)-sums)/np.sum(sums)
    #print("Loss Weights",weights)
    
    print(f"Data batches (x):{data.shape} - Test batches (y):{test.shape}")
    
    tfile = tempfile.NamedTemporaryFile(prefix='train-', suffix='.pycf.model.keras', )
    model_path = tfile.name
    
    stop = False
    epoch = 0

    #model_path = '/tmp/blah.h5'
    #epoch = 19

    if True:
        _train_some(model_path, data, test, epoch)
    else:
      while not stop:
        queue = Queue(1)
        sub = Process(target=_train_proc, args=(model_path, data, test, epoch, queue))
        sub.start()

        while sub.is_alive():
            try:
                epoch,stop = queue.get(timeout=0.1)
            except QueueEmpty:
                pass
            except KeyboardInterrupt:
                stop = True
                os.kill(sub.pid, 2)
        
        sub.join()
    
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
        name = f'{opts.models_dir if opts and opts.models_dir else "."}{os.sep}pycf-{tmetrics[1]:.04f}-{RNN}{UNITS}bin-d{DROPOUT}-w{WINDOW_BEFORE}x{WINDOW_AFTER}x{SUMMARY_RATE}-{int(time.time())}.keras'
        print()
        print('Saving as ' + name)

        import shutil
        shutil.copy(model_path, name)
        try: os.chmod(name, 0o644)
        except: pass
    
    print()

    return 0

def _train_proc(model_path, train_dataset, test_dataset, epoch, queue):
    queue.put( _train_some(model_path, train_dataset, test_dataset, epoch) )

def build_model(input_shape):
    from keras import layers, Input, Model

    inputs = Input(shape=input_shape[2:], dtype='float32', name="input")
    n = inputs

    n = layers.TimeDistributed(layers.Dense(32, dtype='float32', activation='tanh'), name="dense-pre")(n)
    n = layers.TimeDistributed(layers.Dropout(DROPOUT/2))(n)
    #n = layers.SpatialDropout1D(DROPOUT)(n)

    n = layers.Conv1D(filters=32, kernel_size=7, padding='same', activation='relu', name="conv_pre_a")(n)
    #n = layers.MaxPooling1D(pool_size=2, name="pool_a")(n)
    n = layers.SpatialDropout1D(DROPOUT)(n)

    n = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name="conv_pre_b")(n)
    n = layers.MaxPooling1D(pool_size=2, name="pool_b")(n)
    n = layers.SpatialDropout1D(DROPOUT)(n)

    # Squeeze-and-Excitation 
    se = layers.GlobalAveragePooling1D(name="squeeze")(n)
    se = layers.Dense(n.shape[-1] // 8, activation='relu', name='excite1')(se)
    se = layers.Dense(n.shape[-1], activation='sigmoid', name='excite2')(se)
    se = layers.Reshape((1,se.shape[-1]))(se)  # match shape for multiply
    n = layers.Multiply(name="se_apply")([n, se])

    if RNN.lower() == "gru":
        n = layers.GRU(UNITS, dropout=DROPOUT, name="rnn")(n)
    elif RNN.lower() == 'lstm':
        n = layers.LSTM(UNITS, dropout=0.5, name="rnn")(n)
    elif RNN.lower() == 'lstm-attn':
        n = layers.LSTM(UNITS, dropout=DROPOUT, return_sequences=True, name="rnn")(n)
        
        # MultiHeadAttention with 4 heads
        n = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=n.shape[-1] // 4,  # Split features across heads
            name="multi_head_attention"
        )(n, n) 
        
        # Global average pooling to reduce temporal dimension
        n = layers.GlobalAveragePooling1D()(n)  # (batch_size, features)
    
    n = layers.Dense(32, dtype='float32', activation='relu', name="dense-post")(n)
    n = layers.Dropout(0.2)(n)
    #n = layers.Dense(16, dtype='float32', activation='relu', name="final")(n)
    outputs = layers.Dense(1, dtype='float32', activation='sigmoid', name="output")(n)
    
    return Model(inputs, outputs)

def _train_some(model_path, train_dataset:MultiGenerator, test_dataset:MultiGenerator, epoch=0) -> tuple[int,bool]:
    import signal
    import keras

    train_dataset.shuffle()

    model:keras.models.Model = None
    if epoch > 0:
        model = keras.models.load_model(model_path)
    else:
        model = build_model(train_dataset.shape)
        model.summary()
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy', 'AUC'])
        model.save(model_path)
    
    cb = []

    from keras import callbacks

    cb.append(callbacks.EarlyStopping(monitor='accuracy', patience=3))
    cb.append(callbacks.EarlyStopping(monitor='val_accuracy', patience=3))
    cb.append(callbacks.ReduceLROnPlateau(patience=2))
    
    class EpochCheckpoint(callbacks.ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            self.last_epoch = epoch
            return super().on_epoch_end(epoch, logs)

    ecp = EpochCheckpoint(model_path, verbose=1, monitor='val_accuracy', mode='auto', save_best_only=True)
    cb.append(ecp)

    class MemoryChecker(callbacks.Callback):
        def __init__(self, *args):
            super().__init__(*args)
            self.start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.exceeded = False

        def on_epoch_end(self, epoch, logs=None):
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            change = rss - self.start_rss
            if change > 6000000:
                print(f'Now using too much memory ({rss//1024}MB)! {change//1024}MB more than at start which was {self.start_rss//1024}MB')
                self.model.stop_training = True
                self.exceeded = True
    #cb.append(MemoryChecker())
    
    def handler(signum, frame):
        print("\nStopping (gracefully)...\n")
        model.stop_training = True
        signal.signal(signal.SIGINT, oldsint)
        signal.signal(signal.SIGTERM, oldterm)
        return
    oldsint = signal.signal(signal.SIGINT, handler)
    oldterm = signal.signal(signal.SIGTERM, handler)

    model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, initial_epoch=epoch, callbacks=cb)

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
    stuff = load_nonpersistent(flog, False)
    if not stuff:
        return None
    
    duration = flog.get('duration', 0)
    categorical = model.output_shape[-1] > 1

    gen = WindowStackGenerator(stuff, ordered=True, categorical=categorical)
    times = gen.timestamps

    prediction = model.predict(gen, verbose=True)

    result = np.argmax(prediction, axis=1) if categorical else None
    results = [(0,(0,0))]
    for i in range(len(prediction)):
        when = float(times[i])
        if categorical:
            ans = int(result[i])
        else:
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

    print("EVALUATE", len(opts.eval), "on", len(opts.ml_data))


    import tensorflow as tf
    import keras

    models = {}
    total_time = 0
    all_missing = {}
    all_extra = {}
    for mf in opts.eval:
        try:
            models[mf] = keras.models.load_model(mf)
            all_missing[mf] = 0
            all_extra[mf] = 0
        except Exception as e:
            log.exception(f"Unable to load MODEL {mf}")

    for f in opts.ml_data:
        if os.path.isdir(f) or f.endswith('.npy'):
            continue
        
        try:
            flog = processor.read_feature_log(f)
            if not flog:
                print('Load failed')
                continue

            spans = processor.read_feature_spans(flog, 'diff', 'blank')
            realtags = _adjust_tags(flog.get('tags', []), spans.get('blank', []), spans.get('diff', []))
            duration = flog.get('duration',0.00001)
            real_dur = duration
            
            for (t,(b,e)) in realtags:
                if t in [SceneType.DO_NOT_USE, SceneType.DO_NOT_USE.value]:
                    if e+10 > real_dur and e < real_dur and b < real_dur:
                        e = real_dur
                    if b < 10 and e > 0:
                        b = 0
                    duration -= e-b
            
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

