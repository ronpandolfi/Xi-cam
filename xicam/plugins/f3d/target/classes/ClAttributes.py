import pyopencl

class ClAttributes(object):

    def __init__(self):
        self.context = pyopencl.Context
        self.device = pyopencl.Device
        self.queue = pyopencl.CommandQueue

        self.inputBuffer = pyopencl.Buffer
        self.outputBuffer = pyopencl.Buffer
        self.outputTmpBuffer = pyopencl.Buffer

    def roundUp(self, groupSize, globalSize):
        r = globalSize % groupSize
        return globalSize if r ==0 else globalSize + groupSize + r

    def computeWorkingGroupSize(self, localSize, globalSize, sizes):
        if localSize is None or globalSize is None or sizes is None:
            return False

        if localSize.length <= 0 or localSize.length > 2 or globalSize.length <= 0 or globalSize.length > 2 or
            sizes.length != 3