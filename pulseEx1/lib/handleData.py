import scipy.io as sio
import os
import os.path as osp
import sys
import numpy as np
import h5py

def get_from_mat(signalfile, hrfile):
    assert os.path.exists(signalfile), \
            'signal file not found at: {}'.format(signalfile)
    assert os.path.exists(hrfile), \
            'heartrate file not found at: {}'.format(hrfile)
    signal = h5py.File(signalfile)
    signalKeys = signal.keys()
    signalData = signal[signalKeys[0]]
    dataSize = signalData.shape

    label = h5py.File(hrfile)
    labelKeys = label.keys()
    labelData = label[labelKeys[0]]
    labelSize = labelData.shape

    blob_signal = np.zeros((dataSize[1],1, 1, dataSize[0]), dtype = np.float32)
    blob_label = np.zeros(labelSize[1],dtype = np.float32)
    imagecount = 0
    for i in xrange(labelSize[1]):
        blob_signal[i,0,0,:] = signalData[:,i]
        blob_label[i] = labelData[0,i]
    

    result = {}
    result['data'] = blob_signal
    result['label'] = blob_label
    return result

if __name__ == '__main__':
    signalfile = '/net/liuwenran/datasets/DEAP/experiment/ex1_fc_gt/finalExData/signal_train.mat'
    hrfile = '/net/liuwenran/datasets/DEAP/experiment/ex1_fc_gt/finalExData/HeartRate_train.mat'
    data = get_from_mat(signalfile, hrfile)
    perm = np.random.permutation(np.arange(len(data['data'])))
    inds = perm[0:99]
    data['data'] = data['data'][inds,:,:,:]
    data['label'] = data['label'][inds]

