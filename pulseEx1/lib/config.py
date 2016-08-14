from easydict import EasyDict as edict

__C = edict()

cfg = __C

#
# Training options
#

__C.TRAIN = edict()


# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 100

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 1000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''
#
# Testing options
#

__C.TEST = edict()



#
# MISC
#
__C.RNG_SEED = 3
