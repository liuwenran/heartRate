from easydict import EasyDict as edict

__C = edict()

cfg = __C

#
# Training options
#

__C.TRAIN = edict()


# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 488

__C.TRAIN.BATCHES_PER_STEP = 1

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 10000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = 'HCI_loadconv'


__C.TRAIN.USE_PREFETCH = True
#
# Testing options
#

__C.TEST = edict()



#
# MISC
#
__C.RNG_SEED = 3
