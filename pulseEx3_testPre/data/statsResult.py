import numpy as np
import h5py

result = np.load('output/test_result_1w.npy')

# datafile = '/net/liuwenran/datasets/DEAP/experiment/ex3_cnn_face/finalExData_shuffled/test.h5'

# originFile = h5py.File(datafile,'r')
# keys = originFile.keys()
# originLabel = originFile[keys[2]].value
originLabel = np.load('output/test_label_1w.npy')

num = result.shape[0]

result = 100 * result + 54
result = 60 / (result / 128)
originLabel = 100 * originLabel + 54
originLabel = 60 / (originLabel / 128)


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
