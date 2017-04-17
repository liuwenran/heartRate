import numpy as np
import sys
import os
import argparse
import pprint

import _init_paths
from handleData import get_from_h5, get_blob
from config import cfg
from timer import Timer

import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2


def test_fcgt(net, data):
    """Test a Fast R-CNN network on an image database."""
    signalsum = data['facePath'].shape[0]
    result = []
    accuracy = 0
    for i in xrange(signalsum):
        blob = get_blob(data['facePath'][i], data['rootPath'], data['label'][i], cfg.TRAIN.IMS_PER_BATCH)
    	sig = np.zeros_like(blob['data'], dtype = np.float32)
    	sig[...] = blob['data'][...]
    	# label = data['label'][i]
        label = np.zeros((1,1,1,1), dtype = np.float32)
        label[0,0,0,0] = blob['label'][0]
    	net.blobs['data'].reshape(*(sig.shape))
    	net.blobs['label'].reshape(*(1,1,1,1))
    	blob_out = net.forward(data=sig.astype(np.float32, copy=False),
                               label=label.astype(np.float32, copy=False))
        net_output = blob_out[blob_out.keys()[0]]
        print blob_out.keys()[0]+' is '+str(net_output.shape)
        #class_sorted = np.argsort(-class_output)
        #if class_sorted[0,0] > (label[0,0,0,0] - 5) and class_sorted[0,0] < (label[0,0,0,0] + 5):
        #    accuracy += 1

    return accuracy


if __name__ == '__main__':
    datafile = '/net/liuwenran/datasets/DEAP/experiment/ex3_cnn_face/finalExData_shuffled/train.h5'
    test_prototxt = '../proto/ex3_test_loadconv.prototxt'
    rootPath = '/net/liuwenran/datasets/DEAP/experiment/ex2_fc_face/RoughFace/'
    pretrained_model = '/net/liuwenran/fast-rcnn/data/fast_rcnn_models/caffenet_fast_rcnn_iter_40000.caffemodel'
    # pretrained_model = None
    gpu_id = 2

    test_data = get_from_h5(datafile, rootPath)

    print('Using config:')
    pprint.pprint(cfg)

    print 'using model: ' + str(pretrained_model)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    if pretrained_model != None:
        net = caffe.Net(test_prototxt, pretrained_model, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(pretrained_model))[0]
    else:
        net = caffe.Net(test_prototxt, caffe.TEST)

    
    accuracy = test_fcgt(net, test_data)
    sumnum = test_data['data'].shape[0]
    goodrate = float(accuracy) / sumnum
    print 'accuracy is ' + str(accuracy)
    print 'goodrate is ' + str(goodrate)
 #   final_score = sum(accuracy) / mnist_test_data['data'].shape[0]
 #   print 'final_score is ' + str(final_score)