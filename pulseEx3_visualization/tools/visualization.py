import numpy as np
import matplotlib.pyplot as plt
import os
import _init_paths
import caffe
import Image
from handleData import get_from_h5, get_blob

def norm(x, s=1.0):
    x -= x.min()
    x /= x.max()
    return x*s

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    global imcount
    imname = output_path + 'im_' + str(imcount) + '.jpg'
    plt.imsave(imname, data)
    imcount += 1

    plt.axis('off')
    plt.imshow(data)

def switch_channel(im):
    channel1 = im[:,:,0]
    channel2 = im[:,:,1]
    channel3 = im[:,:,2]
    newim = np.zeros(im.shape)
    newim[:,:,0] = channel3
    newim[:,:,1] = channel2
    newim[:,:,2] = channel1
    return newim

def channel_addMean(im, channelMean):
    for i in range(3):
        im[:,:,i] = im[:,:,i] + channelMean[0,i]
    return im

caffe_root = '/net/liuwenran/caffe/caffe/'
output_path = '../data/output/'
global imcount 
imcount = 1

caffe.set_mode_cpu()
net = caffe.Net('../proto/ex3_test_loadconv.prototxt',
                '/net/liuwenran/heartRate/pulseEx3_sharedPre/data/output_face_test/ex3_load_pretrain_iter_60000.caffemodel',
                caffe.TEST)

invnet = caffe.Net('../proto/ex3_inv.prototxt',caffe.TEST)

datafile = '/net/liuwenran/datasets/DEAP/experiment/ex3_cnn_face/finalExData_shuffled/test_float.h5'
rootPath = '/net/liuwenran/datasets/DEAP/experiment/ex2_fc_face/RoughFace/'
test_data = get_from_h5(datafile, rootPath)

i = 0
imnum_perbatch = 1
blob, channelMean = get_blob(test_data['facePath'][i], test_data['rootPath'], test_data['label'][i], imnum_perbatch)
sig = np.zeros_like(blob['data'], dtype = np.float32)
sig[...] = blob['data'][...]
# label = data['label'][i]
label = np.zeros((1,1,1,1), dtype = np.float32)
label[0,0,0,0] = blob['label'][0]
net.blobs['data'].reshape(*(sig.shape))
net.blobs['label'].reshape(*(1,1,1,1))
blob_out = net.forward(data=sig.astype(np.float32, copy=False),
                       label=label.astype(np.float32, copy=False))


im = blob['data']
im = np.transpose(im , (2,3,1,0))
im = np.squeeze(im)
im = channel_addMean(im, channelMean)
# im = switch_channel(im)
imname =  output_path + 'face_origin.jpg'
plt.imsave(imname, im)
plt.imshow(im)
plt.show()
# plt.clf()

#feat = net.blobs['conv1'].data[0,:36]
#vis_square(feat, padval=1)
#plt.savefig('conv1_butterfly.png', dpi = 400, bbox_inches='tight', transparent=True)

#plt.clf()
#feat = net.blobs['pool5'].data[0]
#vis_square(feat, padval=1)
#plt.savefig('pool5_butterfly.png', dpi = 400, bbox_inches='tight', transparent=True)

#plt.clf()
#feat = net.blobs['prob'].data[0]
#plt.plot(feat.flat)
#plt.savefig('prob_butterfly.png', dpi = 400, bbox_inches='tight', transparent=True)


for b in invnet.params:
    invnet.params[b][0].data[...] = net.params[b][0].data.reshape(invnet.params[b][0].data.shape)
#    print invnet.params[b][0].data.shape, net.params[b][0].data.shape
#    print invnet.params[b][1].data.shape, net.params[b][1].data.shape
#    invnet.params[b][1].data[...] = net.params[b][1].data.reshape(invnet.params[b][1].data.shape)

feat = net.blobs['pool5'].data
threshold = np.max(feat) / 2
# feat[0][feat[0] < threshold] = 0
vis_square(feat[0], padval=1)
plt.show()



invnet.blobs['pooled'].data[...] = feat
invnet.blobs['switches5'].data[...] = net.blobs['switches5'].data
invnet.blobs['switches2'].data[...] = net.blobs['switches2'].data
invnet.blobs['switches1'].data[...] = net.blobs['switches1'].data
invnet.forward()


plt.clf()
feat = norm(invnet.blobs['conv1'].data[0],255.0)
# imback = transformer.deprocess('data', feat)
# imback = switch_channel(imback)
imback = np.transpose(feat, (1,2,0))
plt.imshow(imback)
imname = output_path + 'im_' + str(imcount) + '.jpg'
plt.imsave(imname, imback)
imcount +=1
plt.show()

#vis_square(feat, padval=1)
#plt.savefig('test_deconv.png', dpi = 400, bbox_inches='tight', transparent=True)


#features = np.zeros((50000,4096))
#i = 0

# for img in os.listdir('ILSVRC2012_img_val'):
#     net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('ILSVRC2012_img_val/'+img))
#     out = net.forward()
#     feat = net.blobs['fc7'].data[0]
#     features[i,:] = feat
#     i += 1

# np.save('features', features)
