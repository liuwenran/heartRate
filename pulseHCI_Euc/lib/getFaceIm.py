import cv2
import os
from os import listdir
from os.path import isfile,join
import h5py

allFacePath = '/net/liuwenran/datasets/DEAP/experiment/ex2_fc_face/RoughFace'
allLabelPath = '/net/liuwenran/datasets/DEAP/experiment/ex1_fc_gt/GTHR_norm_sigPeak1024_2'
allFaceFlist = listdir(allFacePath)
allLabelFlist = listdir(allLabelPath)

correspondPath = '/net/liuwenran/datasets/DEAP/correspond.mat'
correspond = h5py.File(correspondPath)
correspond = correspond[correspond.keys()[0]]
frameNum = 3000

faceIm = []
label = []
for i in allFaceFlist:
	personLabelPath = join(allLabelPath, i)
	personFacePath = join(allFacePath, i)
	personLabelFlist = listdir(personLabelPath)
	personFaceFlist = listdir(personFacePath)

	personLabelName = join(personLabelPath, personLabelFlist[0])
	personLabel = h5py.File(personLabelName)
	personLabel = personLabel[personLabel.keys()[0]]

	for j in personFaceFlist:
		videoPath = join(personFacePath, j)
		videoNo = int(j[1:2])
		


		


		



