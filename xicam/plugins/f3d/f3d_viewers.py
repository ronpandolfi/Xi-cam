from PySide import QtGui, QtCore
from xicam.widgets.imageviewers import StackViewer
from pipeline.loader import StackImage
import numpy as np

class F3DViewer(QtGui.QWidget):

    def __init__(self, files, *args, **kwargs):
        super(F3DViewer, self).__init__(*args, **kwargs)
        self.data = StackImage(filepath=files)
        self.imageviewer = StackViewer(self.data)
        self.imageviewer.setMinimumHeight(100)
        self.imageviewer.setMinimumWidth(20)

        fullPlugin = QtGui.QSplitter(QtCore.Qt.Vertical)
        fullPlugin.addWidget(self.imageviewer)

        fullLayout = QtGui.QVBoxLayout()
        fullLayout.setContentsMargins(0, 0, 0, 0)
        fullLayout.addWidget(fullPlugin)

        self.setLayout(fullLayout)




