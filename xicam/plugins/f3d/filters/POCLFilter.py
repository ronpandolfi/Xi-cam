from PySide import QtGui, QtCore
from enum import Enum
from xicam.widgets import featurewidgets as fw
from xicam.plugins.f3d import importer
from pyqtgraph.parametertree import Parameter, ParameterTree
from MedianFilter import MedianFilter as mf
from BilateralFilter import BilateralFilter as bf
from MMFilterClo import MMFilterClo as mmclo
from FFTFilter import FFTFilter as fft
from MaskFilter import MaskFilter as mskf
from MMFilterDil import MMFilterDil as mmdil
from MMFilterEro import MMFilterEro as mmero
from MMFilterOpe import MMFilterOpe as mmope

class POCLFilter(fw.FeatureWidget):

    filter_types = {'MedianFilter': mf, 'BilateralFilter': bf, 'MMFilterClo': mmclo, 'FFTFilter': fft,
                    'MaskFilter': mskf, 'MMFilterDil': mmdil, 'MMFilterEro': mmero, 'MMFilterOpe': mmope}

    def __init__(self, name, checkable=True, closeable=True,parent=None):
        super(POCLFilter, self).__init__(name, checkable=checkable, closeable=closeable, parent=parent)
        self.parent = parent
        self.details = importer.filters[name]
        self.info = self.FilterInfo()
        self.info.name = name
        self.name = name

        self.width = 0
        self.height = 0
        self.channels = 0
        self.slices = 0
        self.sliceStart = -1
        self.sliceEnd = -1

        # filter package (?) will go here. Its processing methods will be inherited by the POCLFilter
        self.filter = self.filter_types[self.name]()

        try:
            self.params = Parameter.create(name=name, children=self.details['Parameters'], type='group')
            self.form = ParameterTree(showHeader=False)
            self.form.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.form.customContextMenuRequested.connect(self.paramMenuRequested)
            self.form.setParameters(self.params, showTop=True)


        except KeyError:
            self.params = Parameter.create(name=name)
            self.form.clear()

        self.reconnectDefaults()


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

    # for eventual use with previews
    @property
    def stack_dict(self):
        stack_dict = {'Name': self.info.name}

        if self.params:
            for param in self.params.children():
                stack_dict[param.name()] = param.value()
            try:
                if stack_dict['Mask'] != 'StructuredElementL':
                    stack_dict.pop('L')
            except KeyError:
                pass
        return stack_dict

    def reconnectDefaults(self):
        # set parameter defaults if there are any
        try:
            for item in self.details['Parameters']:
                if 'default' in item.iterkeys():
                    self.params.child(item['name']).setValue(item['default'])
                    self.params.child(item['name']).setDefault(item['default'])

                # connect structuredElementL to show extra parameter
                if 'Mask' in item.itervalues():
                    self.params.child(item['name']).sigValueChanged.connect(self.hideL)
                    self.params.child('L').setLimits([1,20])
                    self.params.child('L').sigValueChanged.connect(self.setL)
        except KeyError:
            pass
    def menuRequested(self):
        pass

    def paramMenuRequested(self, pos):
        """
        Menus when a parameter in the form is right clicked
        """
        pass

        # if self.form.currentItem().parent():
        #     self.parammenu.exec_(self.form.mapToGlobal(pos))

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
            self.memtype = POCLFilter.Type.Byte
            self.useTempBuffer = False

    def getInfo(self):
        return self.filter.getInfo()

    def getName(self):
        return self.name

    def loadKernel(self):
        self.filter.loadKernel()

    def runFilter(self):
        self.filter.runFilter()

    def releaseKernel(self):
        self.filter.releaseKernel()

    #
    # def processFilterWindowComponent(self):
    #     pass

    def toJSONString(self):
        pass

    def fromJSONString(self):
        pass

    def clone(self):

        return self.filter_types[self.name]()

class FilteringAttributes:

    def __init__(self):
        self.width = 0
        self.height = 0
        self.channels = 0
        self.slices = 0
        self.sliceStart = 0
        self.sliceEnd = 0
        self.maxOverlap = 0
        self.intermediateSteps = False
        self.preview = False
        self.chooseConstantDevices = False
        self.inputDeviceLength = 1
