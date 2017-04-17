# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from config import cfg
import numpy as np
import yaml
from handleData import get_blob
from timer import Timer
from multiprocessing import Process, Queue
import os
from multiprocessing.sharedctypes import Array as sharedArray
import ctypes
import atexit
import time
import sys

class faceDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_data_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._data['facePath'])))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.BATCHES_PER_STEP >= len(self._data):
            self._shuffle_data_inds()

        db_inds = self._perm[self._cur]
        self._cur += cfg.TRAIN.BATCHES_PER_STEP
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        # ti = Timer()
        # ti.tic()
        if cfg.TRAIN.USE_PREFETCH:
            # btime = ti.get_time()
            # print 'get blob begin ' + str(btime)

            while self._slots_filled.empty():
                self.check_prefetch_alive();
            deq_slot = self._slots_filled.get();
            blob = {}
            blob_keys = ['data','label']
            for i in range(self._num_tops):
                shared_mem = shared_mem_list[i][deq_slot];
                with shared_mem.get_lock():
                        shared_mem_arr = np.reshape(np.frombuffer(shared_mem.get_obj(),dtype=np.float32), self._data_shapes[i]);
                        blob[blob_keys[i]] = shared_mem_arr[...].astype(np.float32, copy=True); #copy since we will mark this slot as used
            # print 'fwd:: ', slot, im_datas[t].min(), im_datas[t].max(), im_datas[t].mean();
            self._slots_used.put(deq_slot);

            # ftime = ti.get_time()
            # print 'get blob finished ' + str(ftime)
            # onefetch = ti.toc()
            # print 'use_prefetch onefetch time '+str(onefetch)

            return blob

        else:
            db_inds = self._get_next_minibatch_inds()
            firstPath = self._data['facePath'][db_inds]
            rootPath = self._data['rootPath']
            label = self._data['label'][db_inds]
            blob = get_blob(firstPath, rootPath, label, cfg.TRAIN.IMS_PER_BATCH)
            # onefetch = ti.toc()
            # print 'no_prefetch onefetch time '+str(onefetch)
            return blob

    def set_data(self, data):
        """Set the data  to be used by this layer during training."""
        self._data = data
        self._shuffle_data_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self.setup_prefetch()

    def get_data(self):
        return self._data

    def check_prefetch_alive(self):
        try:
            for i in range(self._processNum):
                os.kill(self._prefetch_process_pid[i], 0) #not killing just poking to see if alive
        except err:
            #will raise exception if process is dead
            #can do something more intelligent here rather than raise the same error ...
            raise err

    def setup_prefetch(self):
        self._max_queue_size = 6
        self._slots_used = Queue(self._max_queue_size)
        self._slots_filled = Queue(self._max_queue_size)
        global shared_mem_list
        shared_mem_list = [[] for t in range(self._num_tops)]
        for t in range(self._num_tops):
            for c in range(self._max_queue_size):
                shared_mem = sharedArray(ctypes.c_float, self._blob_counts[t])
                with shared_mem.get_lock():
                    s = np.frombuffer(shared_mem.get_obj(), dtype=np.float32);
                    assert(s.size == self._blob_counts[t]), '{} {}'.format(s.size, self._blob_counts)
                shared_mem_list[t].append(shared_mem);
        self._shared_mem_shape = self._data_shapes;
        

        datalen = len(self._data['label'])
        self._processNum = 6
        partlen = datalen / self._processNum
        self._prefetch_process_id_q = Queue(self._processNum)
        self._prefetch_process_pid = []
        self._prefetch_process = {}
        for i in range(self._processNum):
            partdata = {}
            partdata['facePath'] = self._data['facePath'][i*partlen:(i+1)*partlen]
            partdata['rootPath'] = self._data['rootPath']
            partdata['label'] = self._data['label'][i*partlen:(i+1)*partlen]
            self._prefetch_process[i] = BlobFetcher(i, self._prefetch_process_id_q, self._slots_used, self._slots_filled, 
                                             partdata, self._shared_mem_shape, self._num_tops)
            self._prefetch_process[i].start()
            self._prefetch_process_pid.append(self._prefetch_process_id_q.get())
           # Terminate the child process when the parent exists
        for c in range(self._max_queue_size):
            self._slots_used.put(c)
        def cleanup():
            for i in range(self._processNum):
                print 'Terminating '+ str(i)+' BlobFetcher'
                self._prefetch_process[i].terminate()
                self._prefetch_process[i].join()
        atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the fcgtLayer."""

        # parse the layer parameter string, which must be valid YAML
        # layer_params = yaml.load(self.param_str_)

        # self._num_classes = layer_params['num_classes']

        self._cur = 0
        self._perm = None
        self._data = None
        self._num_tops = len(top)

        self._name_to_top_map = {
            'data': 0,
            'label': 1}

        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, 224, 224)

        # label blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[1].reshape(1)

        self._blob_counts = []
        self._data_shapes = []
        for i in range(self._num_tops):
            tempshape = top[i].data.shape
            self._data_shapes.append(tempshape)
            tempsize = 1
            for j in tempshape:
                tempsize = tempsize * j
            self._blob_counts.append(tempsize)



    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, selfid, process_pid_q, slots_used, slots_filled, partdata, shared_mem_shape, num_tops):
        super(BlobFetcher, self).__init__()
        self._id = selfid
        self._prefetch_process_id_q = process_pid_q;
        self._slots_used = slots_used
        self._slots_filled = slots_filled
        self._data = partdata
        self._shared_mem_shape = shared_mem_shape
        self._num_tops = num_tops
        self._perm = None
        self._cur = 0

        self._shuffle_data_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_data_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._data['facePath'])))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.BATCHES_PER_STEP >= len(self._data):
            self._shuffle_data_inds()

        db_inds = self._perm[self._cur]
        self._cur += cfg.TRAIN.BATCHES_PER_STEP
        return db_inds
        
    def self_cleanup(self):
        try:
            os.kill(self._parent_pid, 0) #not killing just poking to see if parent is alive
        except:
            #parent is dead and we are not, stop prefetching
            print 'prefetch %s (%d) : shutdown'%(self._process_name, self._self_pid)
            self.exit.set();
            sys.exit()
            
    def run(self):
        print 'BlobFetcher '+ str(self._id) + ' started'
        self._parent_pid = os.getppid()
        self._pid = os.getpid()
        self._prefetch_process_id_q.put(self._pid);
        global shared_mem_list

        while True:
            self.self_cleanup();
            if self._slots_used.empty():
                continue;
            slot = self._slots_used.get();
            
            # ti = Timer()
            # time = ti.get_time()
            # print 'process ' + str(self._id) +' blob prepare begin time: ' + str(time)

            db_inds = self._get_next_minibatch_inds()
            firstPath = self._data['facePath'][db_inds]
            rootPath = self._data['rootPath']
            label = self._data['label'][db_inds]
            blob = get_blob(firstPath, rootPath, label, cfg.TRAIN.IMS_PER_BATCH)

            for i,keys in enumerate(blob.keys()):
                shared_mem = shared_mem_list[i][slot];
                with shared_mem.get_lock():
                    s = np.frombuffer(shared_mem.get_obj(), dtype=np.float32);
                    # print s.size, self._shared_shapes[t];
                    shared_mem_arr = np.reshape(s, self._shared_mem_shape[i]);
                    shared_mem_arr[...] = blob[keys].astype(np.float32, copy=True);
                    # print 'helper:: ',im_datas[t].min(), im_datas[t].max(), im_datas[t].mean()
            self._slots_filled.put(slot);

            # time2 = ti.get_time()
            # print 'process ' + str(self._id) +' blob put in queue time: ' + str(time2)
