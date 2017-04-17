import numpy as np
import sys
import os
import argparse
import pprint

import _init_paths
from handleData import get_from_h5
from config import cfg
from timer import Timer

import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, data, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_data(data)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)


    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        net  = self.solver.net
        iterNum = 0
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            print 'itersNum is ' + str(iterNum)
            print 'blobsConv5Data:'
            blobsConv5Data = net.blobs['conv5'].data
            print 'max: ' + str(np.max(blobsConv5Data))
            print 'min: ' + str(np.min(blobsConv5Data))
            sumnum = 1
            for  i in blobsConv5Data.shape:
                sumnum = sumnum * i
            above0Num = blobsConv5Data[blobsConv5Data>0].size
            print 'aboveNum: ' + str(above0Num) + ' in ' + str(sumnum) + ' of ' + str(blobsConv5Data.shape)

            print 'blobsConv5Diff:'
            blobsConv5Diff = net.blobs['conv5'].diff
            print 'max: ' + str(np.max(blobsConv5Diff))
            print 'min: ' + str(np.min(blobsConv5Diff))
            sumnum = 1
            for  i in blobsConv5Diff.shape:
                sumnum = sumnum * i
            above0Num = blobsConv5Diff[blobsConv5Diff>0].size
            print 'aboveNum: ' + str(above0Num) + ' in ' + str(sumnum) + ' of ' + str(blobsConv5Diff.shape)

            print 'paramsConv5Data:'
            paramsConv5Data = net.params['conv5'][0].data
            print 'max: ' + str(np.max(paramsConv5Data))
            print 'min: ' + str(np.min(paramsConv5Data))
            sumnum = 1
            for  i in paramsConv5Data.shape:
                sumnum = sumnum * i
            above0Num = paramsConv5Data[paramsConv5Data>0].size
            print 'aboveNum: ' + str(above0Num) + ' in ' + str(sumnum) + ' of ' + str(paramsConv5Data.shape)

            print 'paramsConv5Diff:'
            paramsConv5Diff = net.params['conv5'][0].diff
            print 'max: ' + str(np.max(paramsConv5Diff))
            print 'min: ' + str(np.min(paramsConv5Diff))
            sumnum = 1
            for  i in paramsConv5Diff.shape:
                sumnum = sumnum * i
            above0Num = paramsConv5Diff[paramsConv5Diff>0].size
            print 'aboveNum: ' + str(above0Num) + ' in ' + str(sumnum) + ' of ' + str(paramsConv5Diff.shape)

            iterNum  = iterNum + 1

            self.solver.step(1)
            timer.toc()

            fc7data = net.blobs['fc7s'].data

            maxind =  np.argmax(fc7data)

            print 'fc7 maxind: ' + str(maxind)

            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()


if __name__ == '__main__':
    datafile = '/net/liuwenran/datasets/DEAP/experiment/ex3_cnn_face/finalExData_shuffled/train_float.h5'
    face_rootPath = '/net/liuwenran/datasets/DEAP/experiment/ex2_fc_face/RoughFace/'

    solver_prototxt = '../proto/ex3_solver_loadconv.prototxt'
    output_dir = '../data/output_face_test/'
    max_iters = 60000
    # pretrained_model = '/net/liuwenran/fast-rcnn/data/fast_rcnn_models/caffenet_fast_rcnn_iter_40000.caffemodel'
    # pretrained_model = '../data/output_face_test/ex3_loadconv_Euc_iter_60000.caffemodel'
    pretrained_model = None 

    face_data = get_from_h5(datafile, face_rootPath)

    print('Using config:')
    pprint.pprint(cfg)

    caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    gpu_id = 2
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    sw = SolverWrapper(solver_prototxt, face_data, output_dir,
                    pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'
