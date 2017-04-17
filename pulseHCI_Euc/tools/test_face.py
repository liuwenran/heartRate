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
        ti = Timer()
        ti.tic()

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
        # net_output = blob_out[blob_out.keys()[0]]
        # print blob_out.keys()[0]+' is '+str(net_output.shape)
        #class_sorted = np.argsort(-class_output)
        #if class_sorted[0,0] > (label[0,0,0,0] - 5) and class_sorted[0,0] < (label[0,0,0,0] + 5):
        #    accuracy += 1
        onetime = ti.toc()
        print 'iter '+str(i)+' time '+str(onetime)
        gt_label = blob_out['label']
        class_output = blob_out[blob_out.keys()[0]]
        class_sorted = np.argsort(-class_output)
        print 'out class is '+str(class_sorted[0,0])+' vs gt class is '+str(gt_label)
        if class_sorted[0,0] > (gt_label - 5) and class_sorted[0,0] < (gt_label + 5):
            print 'this is 1'
            accuracy += 1
        else:
            print 'this is 0'

    return accuracy

def test_face(net, data):
    signalsum = data['facePath'].shape[0]
    result = []
    originLabel = []
    accuracy = 0
    net.layers[0].set_data(data)
    for i in xrange(signalsum):
        ti = Timer()
        ti.tic()
        blob_out = net.forward()

        onetime = ti.toc()
        print 'iter '+str(i)+' time '+str(onetime)
        gt_label = blob_out['label']
        # class_output = blob_out[blob_out.keys()[0]]
        # class_sorted = np.argsort(-class_output)
        # print 'out class is '+str(class_sorted[0,0])+' vs gt class is '+str(gt_label)
        # if class_sorted[0,0] > (gt_label - 5) and class_sorted[0,0] < (gt_label + 5):
        #     print 'this is 1'
        #     accuracy += 1
        # else:
        #     print 'this is 0'
        classout = blob_out['sigScore'][0]
        print 'out class is '+str(classout)+' vs gt class is '+str(gt_label)
        result.append(classout[0])
        originLabel.append(gt_label[0])

        classoutRate = np.floor(classout * 100)
        gt_labelRate = np.floor(gt_label * 100)
        if classoutRate > (gt_labelRate - 5) and classoutRate < (gt_labelRate + 5):
            print 'this is 1'
            accuracy += 1
        else:
            print 'this is 0'

    return accuracy,result,originLabel


if __name__ == '__main__':
    datafile = '/net/liuwenran/datasets/HCI/experiment/ex3_cnn_face/finalExData_shuffled/test_float.h5'
    test_prototxt = '../proto/ex3_test_loadconv.prototxt'
    rootPath = '/net/liuwenran/datasets/HCI/experiment/videoFrame/RoughFace/'
    pretrained_model = '../data/output_face_test/ex3_HCI_loadconv_iter_60000.caffemodel'
    # pretrained_model = '/net/liuwenran/fast-rcnn/data/fast_rcnn_models/caffenet_fast_rcnn_iter_40000.caffemodel'
    # pretrained_model = '../data/output/ex3_iter_30000.caffemodel'
    # pretrained_model = None
    gpu_id = 1

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

    
    accuracy,result,originLabel = test_face(net, test_data)
    sumnum = test_data['label'].shape[0]
    goodrate = float(accuracy) / sumnum
    print 'sumnum is '  + str(sumnum)
    print 'accuracy is ' + str(accuracy)
    print 'goodrate is ' + str(goodrate)
    np.save('../data/output/HCI_result_loadconv_6w.npy', result)
    np.save('../data/output/HCI_label_loadconv_6w.npy', originLabel)
 #   final_score = sum(accuracy) / mnist_test_data['data'].shape[0]
 #   print 'final_score is ' + str(final_score)