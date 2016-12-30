import abc
from enum import Enum

class JOCLFilter(object):

    __metaclass__ = abc.ABCMeta

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
            result = "[{" \
                     + "\"maskImage\" : \"" + maskImage + "\"" \
                + (maskImage.startsWith("StructuredElementL") ? ", \"maskLen\" : \"" + L +
                     + ", \"maskLen\" if maskimage.startswith("StructuredElementL") else
			result += ", \"maskLen\"" if maskimage.startswith("StructuredElementL") else
            maskImage.startsWith("StructuredElementL") ", \"maskLen\" : \"" + L + "\"" : "")
                    + "}]";

        def fromJSONString(self, str):
            parser = JSONParser()

            try:
                objFilter = parser.parser(str)
                jsonFilterObject = objFilter

                maskArray = jsonFilterObject.get("Mask")
                jsonMaskObject = maskArray.get(0)

                maskImage = jsonMaskObject.get("MaskImage")
                if None!=jsonMaskObject.get("maskLen"):
                    L = int(jsonMaskObject.get("maskLen"))
                else:
                    L = -1
            except

        def setupInterface(self):





    @abc.abstractmethod
    def