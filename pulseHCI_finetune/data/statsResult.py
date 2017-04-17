import numpy as np
import h5py

result = np.load('output/HCI_result_loadppretrain_6w.npy')

# datafile = '/net/liuwenran/datasets/DEAP/experiment/ex3_cnn_face/finalExData_shuffled/test.h5'

# originFile = h5py.File(datafile,'r')
# keys = originFile.keys()
# originLabel = originFile[keys[2]].value
originLabel = np.load('output/HCI_label_loadppretrain_6w.npy')

num = result.shape[0]

result = 200 * result + 108
result = 60 / (result / 256)
originLabel = 200 * originLabel + 108
originLabel = 60 / (originLabel / 256)


diff = result - originLabel
diffMean = np.mean(diff)
diffStd = np.std(diff)
diffabs = np.absolute(diff)

RMSE = diffabs
RMSE = RMSE * RMSE
RMSE = np.sqrt(np.sum(RMSE) / diffabs.shape[0])

MERP = diffabs / originLabel
MERP = np.sum(MERP) / diffabs.shape[0]

resultMean = np.mean(result)
originLabelMean = np.mean(originLabel)
cov = np.sum((result - resultMean) * (originLabel - originLabelMean))
resultVar = np.var(result)
originLabelVar = np.var(originLabel)
COR = cov / np.sqrt(resultVar * num * originLabelVar * num)

print 'diffMean is ' + str(diffMean)
print 'diffStd is ' + str(diffStd)
print 'RMSE is ' + str(RMSE)
print 'MERP is ' + str(MERP)
print 'COR is ' + str(COR)
