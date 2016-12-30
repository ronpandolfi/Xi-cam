from JOCLFilter import JOCLFilter

from pyopencl import Program
from pyopencl import Kernel
from pyopencl import

class MedianFilter(JOCLFilter):

    program = Program()
    kernel = Kernel()

    def toJSONString(self):
        result = "{ \"Name\" : \"" + self.getName() + "\", " + "\" }"
        return result

    def getInfo(self):
        info = self.FilterInfo()
        info.name = self.getName()
        info.memtype = JOCLFilter.Type.Byte
        info.overlapX = info.overlapY = info.overlapZ = 0
        return info

    def getName(self):
        return "MedianFilter"

    def loadKernel(self):
        median_comperror = ""
        try:
            filename = "/OpenCL/MedianFilter.cl"
            program =
        except Exception as e:
            raise e