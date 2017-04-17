import scipy.io as sio
import os
import os.path as osp
import sys
import numpy as np
import h5py
import cv2

def get_from_h5(dataPath, rootPath):
    assert os.path.exists(dataPath), \
            'signal file not found at: {}'.format(dataPath)
    dataFile = h5py.File(dataPath,'r')
    dataKeys = dataFile.keys()
    facePath = dataFile[dataKeys[1]].value
    label = dataFile[dataKeys[2]].value
    result = {}
    result['facePath'] = facePath
    result['label'] = label
    result['rootPath'] = rootPath
    return result

def extendPath(firstPath, rootPath, imNum):
    imlist = ['' for i in range(imNum)]
    rind = firstPath.rindex('.')
    lind = firstPath.rindex('_')
    frameNo = int(firstPath[lind+1:rind])
    headPart = firstPath[:lind+1]
    tailPart = firstPath[rind:]
    for i in range(imNum):
        imlist[i] = rootPath+headPart+str(frameNo + i)+tailPart
    return imlist

def imlist_to_blob(imlist):
    imnum = len(imlist)
    blob = np.zeros((imnum,3,224,224))
    for i,imName in enumerate(imlist):
        im = cv2.imread(imName)
        im = cv2.resize(im,(224,224))
        im = (im.astype('float') - 0) / 255
        for channel in range(3):
            blob[i,channel,:,:] = im[:,:,channel] - sum(im[:,:,channel].ravel())/im[:,:,channel].size

    return blob

def get_blob(firstPath, rootPath, label, imNum):
    imlist = extendPath(firstPath, rootPath, imNum)
    blob = {}
    blob['data'] = imlist_to_blob(imlist)
    blob['label'] = np.zeros(1)
    blob['label'][0] = label
    return blob


if __name__ == '__main__':
    dataPath = '/net/liuwenran/datasets/DEAP/experiment/ex3_cnn_face/finalExData_shuffled/train.h5'
    rootPath = '/net/liuwenran/datasets/DEAP/experiment/ex2_fc_face/RoughFace/'
    data = get_from_h5(dataPath, rootPath)
    perm = np.random.permutation(np.arange(len(data['facePath'])))
    inds = perm[0:99]
    data['facePath'] = data['facePath'][inds]
    data['label'] = data['label'][inds]
    imlist = extendPath(data['facePath'][0], rootPath, 400)
    blob = imlist_to_blob(imlist)
