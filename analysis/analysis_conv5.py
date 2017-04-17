import numpy as np

fastrcnn_conv5 = np.load('fastrcnn-con5data.npy')

datashape = fastrcnn_conv5.shape

print 'datashape is ' + str(datashape)
count = 0
for i in range(datashape[0]):
	for j in range(datashape[1]):
		for  m in range(datashape[2]):
			for n in range(datashape[3]):
				if fastrcnn_conv5[i][j][m][n] > 0:
					count = count + 1

sumnum = 1
for t in datashape:
	sumnum = sumnum * t

print 'not 0 num :'+str(count) + ' in ' + str(sumnum)


