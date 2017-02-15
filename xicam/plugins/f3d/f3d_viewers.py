from PySide import QtGui, QtCore
from xicam.widgets.imageviewers import StackViewer
from pipeline.loader import StackImage
from xicam.plugins.tomography.viewers import PreviewViewer
import numpy as np

class F3DViewer(QtGui.QWidget):

    def __init__(self, files, *args, **kwargs):
        super(F3DViewer, self).__init__(*args, **kwargs)

        self.viewer = QtGui.QStackedWidget(self)
        self.tabs = QtGui.QTabBar(self)
        self.tabs.addTab('Projection View')
        self.tabs.addTab('Preview')
        self.tabs.setShape(QtGui.QTabBar.TriangularSouth)

        self.data = StackImage(filepath=files, uchar8=True)
        self.rawdata = self.data.fabimage.rawdata
        self.imageviewer = StackViewer(self.data)
        self.imageviewer.setMinimumHeight(100)
        self.imageviewer.setMinimumWidth(20)
        self.viewer.addWidget(self.imageviewer)

        # TODO: fill in proper signals/slots when adding previews
        self.previews = PreviewViewer(self.data.shape[1], parent=self)
        # self.previewViewer.sigSetDefaults.connect(self.sigSetDefaults.emit)
        self.viewer.addWidget(self.previews)

        fullLayout = QtGui.QVBoxLayout()
        fullLayout.setContentsMargins(0, 0, 0, 0)
        fullLayout.addWidget(self.viewer)
        fullLayout.addWidget(self.tabs)

        self.setLayout(fullLayout)

        self.tabs.currentChanged.connect(self.viewer.setCurrentIndex)
        self.viewer.currentChanged.connect(self.tabs.setCurrentIndex)

    def addPreview(self, image, pipeline, slice_no):
        self.previews.addPreview(image, pipeline, slice_no)



