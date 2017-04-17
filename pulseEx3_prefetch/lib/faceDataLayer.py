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
        ti = Timer()
        ti.tic()
        if cfg.TRAIN.USE_PREFETCH:
            qlen = self._blob_queue.qsize()
            print 'queue size is '+ str(qlen)
            btime = ti.get_time()
            print 'get blob begin ' + str(btime)
            blob = self._blob_queue.get()
            ftime = ti.get_time()
            print 'get blob finished ' + str(ftime)
            onefetch = ti.toc()
            print 'use_prefetch onefetch time '+str(onefetch)
            return blob
        else:
            db_inds = self._get_next_minibatch_inds()
            firstPath = self._data['facePath'][db_inds]
            rootPath = self._data['rootPath']
            label = self._data['label'][db_inds]
            blob = get_blob(firstPath, rootPath, label, cfg.TRAIN.IMS_PER_BATCH)
            onefetch = ti.toc()
            print 'no_prefetch onefetch time '+str(onefetch)
            return blob

    def set_data(self, data):
        """Set the data  to be used by this layer during training."""
        self._data = data
        self._shuffle_data_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            datalen = len(self._data['label'])
            partnum = 10
            partlen = datalen / partnum
            self._prefetch_process = {}
            for i in range(partnum):
                partdata = {}
                partdata['facePath'] = self._data['facePath'][i*partlen:(i+1)*partlen]
                partdata['rootPath'] = self._data['rootPath']
                partdata['label'] = self._data['label'][i*partlen:(i+1)*partlen]
                self._prefetch_process[i] = BlobFetcher(self._blob_queue,
                                                 partdata, i)
                self._prefetch_process[i].start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                for i in range(10):
                    self._prefetch_process[i].terminate()
                    self._prefetch_process[i].join()
            import atexit
            atexit.register(cleanup)

    def get_data(self):
        return self._data

    def setup(self, bottom, top):
        """Setup the fcgtLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._cur = 0
        self._perm = None
        self._data = None

        self._name_to_top_map = {
            'data': 0,
            'label': 1}

        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(1, 3, 224, 224)

        # label blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[1].reshape(1)

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
    def __init__(self, queue, data, selfid):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._data = data
        self._perm = None
        self._cur = 0
        self._id = selfid
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

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            firstPath = self._data['facePath'][db_inds]
            rootPath = self._data['rootPath']
            label = self._data['label'][db_inds]
            blob = get_blob(firstPath, rootPath, label, cfg.TRAIN.IMS_PER_BATCH)
            ti = Timer()
            time = ti.get_time()
            print 'process ' + str(self._id) +' blob ready time: ' + str(time)
            self._queue.put(blob)
            time2 = ti.get_time()
            print 'process ' + str(self._id) +' blob put in queue time: ' + str(time2)
