import numpy as np
import h5py

# datafile = '/net/liuwenran/datasets/DEAP/experiment/ex3_cnn_face/finalExData_shuffled/test.h5'
# originFile = h5py.File(datafile,'r')
# keys = originFile.keys()
# originLabel = originFile[keys[2]].value

result = np.load('output/HCI_result_seperate6w.npy')
originLabel = np.load('output/HCI_label_seperate6w.npy')
firstPath = np.load('output/HCI_firstPath_seperate6w.npy')

num = result.shape[0]
session = []
for i in range(num):
    sessionNum = len(session)
    sampleNow = firstPath[i]
    ind = sampleNow.index('/')
    sessionNow = sampleNow[:ind]
    flag = 0
    for j in range(sessionNum):
        if sessionNow == session[j]:
            flag = 1
            break
    if flag == 1:
        continue
    session.append(sessionNow)

sessionCount = len(session)
personLabel = {}
personResult = {}
for i in range(sessionCount):
    personLabel[session[i]] = []
    personResult[session[i]] = []

for i in range(num):
    sampleNow = firstPath[i]
    ind = sampleNow.index('/')
    sessionNow = sampleNow[:ind]
    personLabel[sessionNow].append(originLabel[i])
    personResult[sessionNow].append(result[i])
    
result = np.zeros(sessionCount)
originLabel = np.zeros(sessionCount)
for i, name in enumerate(session):
    result[i] = np.mean(personLabel[name])
    originLabel[i] = np.mean(personResult[name])

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

num = len(result)
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
