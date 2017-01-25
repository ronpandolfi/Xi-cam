from enum import Enum
from xicam.widgets import featurewidgets as fw

class JOCLFilter(fw.FeatureWidget):


    class Type(Enum):
        Byte = bytes
        Float = float

    class FilterInfo(object):

        def __init__(self):
            self.name = ""
            self.overlapX = 0
            self.overlapY = 0
            self.overlapZ = 0
            self.memtype = JOCLFilter.Type.Byte
            self.useTempBuffer = False

    class FilterPanel(object):

        int L = -1
        maskImage = ""
        maskImages = None #need to set as empty list?

        def toJSONString(self):
            result = "[{"  + "\"maskImage\" : \"" + self.maskImage + "\""
            if self.maskImage.startswith("StructuredElementL"):
                result += ", \"maskLen\" : \"" + self.L + "\""
            else:
                result += ""
            result += "}]"

            return result


        def fromJSONString(self, str):
            pass
            # parser = JSONParser()
            #
            # try:
            #     objFilter = parser.parser(str)
            #     jsonFilterObject = objFilter
            #
            #     maskArray = jsonFilterObject.get("Mask")
            #     jsonMaskObject = maskArray.get(0)
            #
            #     maskImage = jsonMaskObject.get("MaskImage")
            #     if None!=jsonMaskObject.get("maskLen"):
            #         L = int(jsonMaskObject.get("maskLen"))
            #     else:
            #         L = -1
            # except

        def setupInterface(self):
            pass

    def getInfo(self):
        pass

    def getName(self):
        pass

    def loadKernel(self):
        pass

    def runFilter(self):
        pass

    def releaseKernel(self):
        pass

    def getFilterwindowComponent(self):
        pass

    def processFilterWindowComponent(self):
        pass

    def newInstance(self):
        pass

    def toJSONString(self):
        pass

    def fromJSONString(self):
        pass

    def setAttributes(self, CLAttributes, FilterAttributes, F3DMonitor, idx):
        self.clattr = CLAttributes
        self.atts = FilterAttributes
        self.index = idx
        self.monitor = F3DMonitor

    def clone(self):
        filter = self.newInstance()

        filter.fromJSONString(self.toJSONString())
        filter.processFilterWindowComponent()

        return filter