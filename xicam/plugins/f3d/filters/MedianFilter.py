from pyopencl import Program
from pyopencl import Kernel
import collections
import JOCLFilter
import time
import pyopencl as cl
import pkg_resources as pkg

class MedianFilter:

    # program = Program()
    # kernel = Kernel()

    def __init__(self):
        self.name = 'MedianFilter'
        self.index = -1

        # load clattr from RunnableJOCLFilter
        self.clattr = None
        self.atts = None #load attributes from RunnableJOCLFilter

    def toJSONString(self):
        result = "{ \"Name\" : \"" + self.getName() + "\", " + "\" }"
        return result

    def getInfo(self):
        info = JOCLFilter.FilterInfo()
        info.name = self.getName()
        info.memtype = JOCLFilter.Type.Byte
        info.overlapX = info.overlapY = info.overlapZ = 0
        return info

    def getName(self):
        return "MedianFilter"

    def loadKernel(self):
        # median_comperror = ""
        try:
            filename = "xicam/plugins/f3d/OpenCL/MedianFilter.cl"
            program = cl.Program(self.clattr.context, pkg.resource_string(__name__, filename)).build()
        except Exception as e:
            raise e
            # other stuff to log errors

        kernel = cl.Kernel(program, "MedianFilter")
        return kernel

    def runFilter(self):
        start_time = time.time()

        if self.atts.height == 1 and self.atts.slices == 1:
            mid = 1
        elif self.atts.slices == 1:
            mid = 4
        else: mid = 13

        globalSize = [0, 0]
        localSize = [0, 0]

        try:
            cl.enqueue_copyself.clattr.queue

