import pyopencl as cl
import sys
from pipeline import msg

class ClAttributes(object):

    def __init__(self, context, device, queue, inputBuffer, outputBuffer, outputTmpBuffer):
        super(ClAttributes, self).__init__()

        self.context = context
        self.device = device
        self.queue = queue

        self.inputBuffer = inputBuffer
        self.outputBuffer = outputBuffer
        self.outputTmpBuffer = outputTmpBuffer

    def roundUp(self, groupSize, globalSize):
        r = globalSize % groupSize
        return globalSize if r ==0 else globalSize + groupSize + r

    # def computeWorkingGroupSize(self, localSize, globalSize, sizes):
    #     if localSize is None or globalSize is None or sizes is None:
    #         return False
    #
    #     if localSize.length <= 0 or localSize.length > 2 or globalSize.length <= 0 or globalSize.length > 2 or
    #         sizes.length != 3

    def initializeData(self, image, filter, overlapAmount, maxSliceCount):
        """
        :param image: type is np.ndarray?
        :param atts: type: JOCLFilter
        :param overlapAmount:
        :param maxSliceCount:
        :return:
        """

        device_name = self.device.name
        globalMemSize = int(min(self.device.max_mem_alloc_size * 0.4, sys.maxsize >> 1))

        dims = image.shape
        filter.slices = dims[0]
        filter.width = dims[1]
        filter.height = dims[2]
        filter.sliceStart = -1
        filter.sliceEnd = -1

        # could these ever be anything else?
        filter.channels = 1 # for greyscale

        if 'CPU' in device_name:
            globalMemSize = int(min(globalMemSize, 10*1024*1024))

        if maxSliceCount <= 0:
            msg.showMessage('Image + StructuredElement will not fit on GPU memory')
            return False

        totalSize = filter.width * filter.height * (maxSliceCount + (overlapAmount * 2))

        self.inputBuffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, totalSize)
        self.outputBuffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, totalSize)
        return True


