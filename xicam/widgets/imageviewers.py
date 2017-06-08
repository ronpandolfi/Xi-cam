
__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


from PySide import QtGui, QtCore
from xicam.widgets.customwidgets import ImageView, histDialogButton
import numpy as np


class StackViewer(ImageView):
    """
    PG ImageView subclass to view 3D datasets as image stacks. Removes Menu and ROI buttons from imageview and replaces
    it with a spinbox with the current frame index and a label.
    """

    def __init__(self, data=None, view_label=None, *args, **kwargs):
        super(StackViewer, self).__init__(*args, **kwargs)

        if view_label is None:
            view_label = QtGui.QLabel(self)
            view_label.setText('No: ')
        self.view_spinBox = QtGui.QSpinBox(self)
        self.view_spinBox.setKeyboardTracking(False)

        if data is not None:
            self.setData(data)

        # push button for setting the max/min values of a histogram widget
        self.setButton = histDialogButton('Set', parent=self)
        self.setButton.connectToHistWidget(self.getHistogramWidget())

        l = QtGui.QHBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(view_label)
        l.addWidget(self.view_spinBox)
        l.addStretch(1)
        w = QtGui.QWidget()
        w.setLayout(l)
        self.ui.gridLayout.addWidget(self.setButton, 1, 1, 1, 2)
        self.ui.gridLayout.addWidget(view_label, 2, 1, 1, 1)
        self.ui.gridLayout.addWidget(self.view_spinBox, 2, 2, 1, 1)
        self.ui.menuBtn.setParent(None)
        self.ui.roiBtn.setParent(None)

        self.sigTimeChanged.connect(self.indexChanged)
        self.view_spinBox.valueChanged.connect(self.setCurrentIndex)

        self.label = QtGui.QLabel(parent=self)
        self.ui.gridLayout.addWidget(self.label, 2, 0, 1, 1)

    def setData(self, data):
        self.data = data
        self.setImage(self.data)
        self.autoLevels()
        self.view_spinBox.setRange(0, self.data.shape[0] - 1)
        self.getImageItem().setRect(QtCore.QRect(0, 0, self.data.rawdata.shape[0], self.data.rawdata.shape[1]))

    def indexChanged(self, ind, time):
        self.view_spinBox.setValue(ind)

    def setIndex(self, ind):
        self.setCurrentIndex(ind)
        self.view_spinBox.setValue(ind)

    @property
    def currentdata(self):
        return self.data[self.data.currentframe].transpose()  # Maybe we need to transpose this

    def resetImage(self):
        self.setImage(self.data, autoRange=False)
        self.setIndex(self.currentIndex)

    def connectImageToName(self, image_names):

        self.image_names = image_names
        self.sigTimeChanged.connect(self.indexAndNameChanged)
        self.label.setText(str(self.image_names[self.currentIndex]))


    def indexAndNameChanged(self, ind, time):
        self.setCurrentIndex(ind)
        self.view_spinBox.setValue(ind)
        self.label.setText(str(self.image_names[ind]))

class ArrayViewer(StackViewer):
    """
    Subclass of StackViewer to allow for more general data inputs
    """

    def __init__(self, data=None, view_label=None, flipAxes=False, *args, **kwargs):

        super(StackViewer, self).__init__(*args, **kwargs)

        if view_label is None:
            view_label = QtGui.QLabel(self)
            view_label.setText('No: ')
        self.view_spinBox = QtGui.QSpinBox(self)
        self.view_spinBox.setKeyboardTracking(False)

        # push button for setting the max/min values of a histogram widget
        self.setButton = histDialogButton('Set', parent=self)
        self.setButton.connectToHistWidget(self.getHistogramWidget())

        l = QtGui.QHBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(view_label)
        l.addWidget(self.view_spinBox)
        l.addStretch(1)
        w = QtGui.QWidget()
        w.setLayout(l)
        self.ui.gridLayout.addWidget(self.setButton, 1, 1, 1, 2)
        self.ui.gridLayout.addWidget(view_label, 2, 1, 1, 1)
        self.ui.gridLayout.addWidget(self.view_spinBox, 2, 2, 1, 1)
        self.ui.menuBtn.setParent(None)
        self.ui.roiBtn.setParent(None)

        self.sigTimeChanged.connect(self.indexChanged)
        self.view_spinBox.valueChanged.connect(self.setCurrentIndex)

        self.label = QtGui.QLabel(parent=self)
        self.ui.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        if data:
            self.setData(data)


    def setData(self, data):
        if type(data) is dict or type(data).__bases__[0] is dict:
            self.keys = data.keys()
            data = np.array(data.values())
        else:
            self.keys = None
        self.data = data
        self.setImage(self.data)
        self.autoLevels()
        self.view_spinBox.setRange(0, self.data.shape[0] - 1)
        self.getImageItem().setRect(QtCore.QRect(0, 0, self.data.shape[1], self.data.shape[2]))
        if self.keys:
            self.connectImageToName(self.keys)

