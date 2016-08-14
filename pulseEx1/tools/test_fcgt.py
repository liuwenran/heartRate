import numpy as np
import sys
import os
import argparse
import pprint

import _init_paths
from handleData import get_from_mat
from config import cfg
from timer import Timer

import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2


def test_fcgt(net, data):
    """Test a Fast R-CNN network on an image database."""
    signalsum = data['data'].shape[0]
    result = []
    accuracy = 0
    for i in xrange(signalsum):
    	sig = np.zeros((1,1,1,512), dtype = np.float32)
    	sig[0,0,0,:] = data['data'][i,0,0,:]
    	# label = data['label'][i]
        label = np.zeros((1,1,1,1), dtype = np.float32)
        label[0,0,0,0] = data['label'][i]
    	net.blobs['data'].reshape(*(sig.shape))
    	net.blobs['label'].reshape(*(1,1,1,1))
    	blob_out = net.forward(data=sig.astype(np.float32, copy=False),
                               label=label.astype(np.float32, copy=False))
        class_output = blob_out['fc2']
        class_sorted = np.argsort(-class_output)
        if class_sorted[0,0] > (label[0,0,0,0] - 5) and class_sorted[0,0] < (label[0,0,0,0] + 5):
            accuracy += 1

    return accuracy


if __name__ == '__main__':
    signalfile = '/net/liuwenran/datasets/DEAP/experiment/signal512/finalExData_shuffled/signal_train.mat'
    hrfile = '/net/liuwenran/datasets/DEAP/experiment/signal512/finalExData_shuffled/HeartRate_train.mat'
 #   signalfile = '/net/liuwenran/datasets/DEAP/experiment/ex1_fc_gt/finalExData_shuffled/signal_test.mat'
 #   hrfile = '/net/liuwenran/datasets/DEAP/experiment/ex1_fc_gt/finalExData_shuffled/HeartRate_test.mat'
 #   testfile = '/net/liuwenran/caffe_learn/data/lwr_test_img_mat.mat'
    test_prototxt = '/net/liuwenran/heartRate/pulseEx1/proto/lwr_ex1_test.prototxt'
    output_dir = '/net/liuwenran/heartRate/pulseEx1/data/output_tfc_512/'
    pretrained_model = '/net/liuwenran/heartRate/pulseEx1/data/output_tfc_512/fcgt_iter_10000.caffemodel'
    gpu_id = 3

    test_data = get_from_mat(signalfile, hrfile)

    print('Using config:')
    pprint.pprint(cfg)

    print 'using model: ' + pretrained_model

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(test_prototxt, pretrained_model, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(pretrained_model))[0]

    accuracy = test_fcgt(net, test_data)
    sumnum = test_data['data'].shape[0]
    goodrate = float(accuracy) / sumnum
    print 'accuracy is ' + str(accuracy)
    print 'goodrate is ' + str(goodrate)
 #   final_score = sum(accuracy) / mnist_test_data['data'].shape[0]
 #   print 'final_score is ' + str(final_score)