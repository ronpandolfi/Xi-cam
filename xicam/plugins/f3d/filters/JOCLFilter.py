from PySide import QtGui, QtCore
from enum import Enum
from xicam.widgets import featurewidgets as fw
from xicam.plugins.f3d import importer
from pyqtgraph.parametertree import Parameter, ParameterTree

class JOCLFilter(fw.FeatureWidget):

    def __init__(self, name, checkable=True, closeable=True,parent=None):
        super(JOCLFilter, self).__init__(name, checkable=checkable, closeable=closeable, parent=parent)
        self.parent = parent
        self.details = importer.filters[name]
        self.info = self.FilterInfo()
        self.info.name = name


        try:
            self.params = Parameter.create(name=name, children=self.details['Parameters'], type='group')
            self.form = ParameterTree(showHeader=False)
            self.form.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.form.customContextMenuRequested.connect(self.paramMenuRequested)
            self.form.setParameters(self.params, showTop=True)

        except KeyError:
            self.params = None
            self.form.clear()

        # set parameter defaults if there are any
        for item in self.details['Parameters']:
            if 'default' in item.iterkeys():
                self.params.child(item['name']).setValue(item['default'])
                self.params.child(item['name']).setDefault(item['default'])

            # connect structuredElementL to show extra parameter
            if 'Mask' in item.itervalues():
                self.params.child(item['name']).sigValueChanged.connect(self.hideL)
                self.params.child('L').setLimits([1,20])
                self.params.child('L').sigValueChanged.connect(self.setL)


        self.previewButton.customContextMenuRequested.connect(self.menuRequested)
        self.menu = QtGui.QMenu()

        # change on/off icons
        icon = QtGui.QIcon()
        if checkable:
            icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_51.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_45.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
            self.previewButton.setCheckable(True)
        else:
            icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_45.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.previewButton.setCheckable(False)
            self.previewButton.setChecked(True)

        self.previewButton.setIcon(icon)
        self.previewButton.setFlat(True)
        self.previewButton.setChecked(True)
        self.previewButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)


    def menuRequested(self):
        pass

    def paramMenuRequested(self, pos):
        """
        Menus when a parameter in the form is right clicked
        """
        if self.form.currentItem().parent():
            self.parammenu.exec_(self.form.mapToGlobal(pos))

    def hideL(self, parameter):

        if parameter.value() == 'StructuredElementL':
            self.params.child('L').show()
        else:
            self.params.child('L').hide()

    def setL(self, parameter):

        self.info.L = parameter.value()

    class Type(Enum):
        Byte = bytes
        Float = float

    class FilterInfo(object):

        def __init__(self):
            self.name = ""
            self.L = -1
            self.overlapX = 0
            self.overlapY = 0
            self.overlapZ = 0
            self.memtype = JOCLFilter.Type.Byte
            self.useTempBuffer = False



    # class FilterPanel(object):
    #
    #     L = -1
    #     maskImage = ""
    #     maskImages = None #need to set as empty list?

        # def toJSONString(self):
        #     result = "[{"  + "\"maskImage\" : \"" + self.maskImage + "\""
        #     if self.maskImage.startswith("StructuredElementL"):
        #         result += ", \"maskLen\" : \"" + self.L + "\""
        #     else:
        #         result += ""
        #     result += "}]"
        #
        #     return result
        #
        #
        # def fromJSONString(self, str):
        #     pass
        #     # parser = JSONParser()
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

    # def clone(self):
    #     filter = self.newInstance()
    #
    #     filter.fromJSONString(self.toJSONString())
    #     filter.processFilterWindowComponent()
    #
    #     return filter