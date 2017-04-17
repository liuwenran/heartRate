import caffe
import numpy as np

class ShiftDimLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        self._bottomDim = bottom[0].data.shape
        self._topDim = (1,self._bottomDim[0],self._bottomDim[1],self._bottomDim[2] * self._bottomDim[3])

        top[0].reshape(*self._topDim)

    def forward(self, bottom, top):

        self._bottomDim = bottom[0].data.shape

        print bottom[0].data.shape
        print top[0].data.shape

        for i in range(self._bottomDim[0]):
        	for j in range(self._bottomDim[1]):
        		for m in range(self._bottomDim[2]):
        			for n in range(self._bottomDim[3]):
        				top[0].data[0,i,j, m * self._bottomDim[3] + n] = bottom[0].data[i,j,m,n]



    def backward(self, top, propagate_down, bottom):
        for i in range(self._bottomDim[0]):
        	for j in range(self._bottomDim[1]):
        		for m in range(self._bottomDim[2]):
        			for n in range(self._bottomDim[3]):
        				bottom[0].diff[i,j,m,n] = top[0].diff[0,i,j, m + self._bottomDim[3] + n]

    def reshape(self, bottom, top):
        # check input dimensions match

        self._bottomDim = bottom[0].data.shape
        self._topDim = (1,self._bottomDim[0],self._bottomDim[1],self._bottomDim[2] * self._bottomDim[3])

        top[0].reshape(*self._topDim)