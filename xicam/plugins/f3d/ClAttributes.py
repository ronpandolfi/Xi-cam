import pyopencl as cl
import sys
from pipeline import msg
import numpy as np

class ClAttributes(object):

    def __init__(self, context, device, queue, inputBuffer, outputBuffer, outputTmpBuffer):
        super(ClAttributes, self).__init__()

        self.context = context
        self.device = device
        self.queue = queue

        self.inputBuffer = inputBuffer
        self.outputBuffer = outputBuffer
        self.outputTmpBuffer = outputTmpBuffer

        # multiply by 8 for 8bit images?
        self.globalMemSize = int(min(self.device.max_mem_alloc_size * 0.5, sys.maxsize >> 1))
        # self.globalMemSize = int(min(self.device.max_mem_alloc_size * 0.5, sys.maxsize >> 1))
        if 'CPU' in self.device.name:
            self.globalMemSize = int(min(self.globalMemSize, 10*1024*1024))

        self.maxSliceCount = 0

    def roundUp(self, groupSize, globalSize):
        r = globalSize % groupSize
        return globalSize if r ==0 else globalSize + groupSize - r

    def computeWorkingGroupSize(self, localSize, globalSize, sizes):
        if not localSize or not globalSize or not sizes:
            return False
        elif len(localSize) <= 0 or len(localSize) > 2 or len(globalSize) <= 0 or len(globalSize) > 2 or len(
                sizes) != 3:
            return False

        name = self.device.name

        # set working group sizes
        dimensions = len(globalSize)
        if dimensions == 1:
            localSize[0] = self.device.max_work_group_size
            globalSize[0] = self.roundUp(localSize[0], sizes[0]*sizes[1]*sizes[2])
        elif dimensions == 2:
            localSize[0] = min(int(np.sqrt(self.device.max_work_group_size)), 16)
            globalSize[0] = self.roundUp(localSize[0], sizes[0])

            localSize[1] = min(int(np.sqrt(self.device.max_work_group_size)), 16)
            globalSize[1] = self.roundUp(localSize[1], sizes[1])

        if 'CPU' in name:
            for i in range(dimensions):
                globalSize[i] = localSize[i]
                localSize[i] = 1

        return True

    def setMaxSliceCount(self, image):

        dim = image.shape
        maxSliceCount = int(self.globalMemSize/(dim[1]*dim[2]*8))

        # TODO: where does maxOverlap value come from?
        # maxSliceCount -= maxOverlap

        if maxSliceCount > dim[0]:
            maxSliceCount = dim[0]
            maxOverlap = 0

        self.maxSliceCount = maxSliceCount

    def initializeData(self, image, atts, overlapAmount, maxSliceCount):
        """
        :param image: type is np.ndarray?
        :param atts: type: POCLFilter
        :param overlapAmount:
        :param maxSliceCount:
        :return:
        """

        dims = image.shape
        # atts.width = dims[0]
        # atts.height = dims[1]
        # atts.slices = dims[2]
        atts.width = dims[1]
        atts.height = dims[2]
        atts.slices = dims[0]
        atts.sliceStart = -1
        atts.sliceEnd = -1

        # could these ever be anything else?
        atts.channels = 1 # for greyscale

        if maxSliceCount <= 0:
            msg.showMessage('Image + StructuredElement will not fit on GPU memory')
            return False

        totalSize = (atts.width * atts.height * (maxSliceCount + (overlapAmount * 2)))*8

        self.inputBuffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, totalSize)
        self.outputBuffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, totalSize)
        return True


    def loadNextData(self, image, atts, startRange, endRange, overlap):

        minIndex = max(0, startRange - overlap)
        maxIndex = min(atts.slices, endRange + overlap)

        cl.enqueue_copy(self.queue, self.inputBuffer, image[minIndex:maxIndex,:,:])
        return True

    def writeNextData(self, atts, startRange, endRange, overlap):
        # startIndex = 0 if startRange==0 else overlap
        length = endRange - startRange
        # size = atts.height*atts.width*length # for 8bit images?
        output = np.empty(length*atts.width*atts.height).astype(np.int8)
        cl.enqueue_copy(self.queue, output, self.outputBuffer)
        output = output.reshape(length, atts.width, atts.height)
        # image = np.append(image, output, axis=0)
        # return image
        return output

    def swapBuffers(self):
        pass

    def convertToFloatBuffer(self, buffer):
        pass

    def convertToByteBuffer(self, buffer):
        pass


