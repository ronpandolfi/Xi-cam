

__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


import numpy as np
import pyqtgraph as pg
import explorer
from PySide import QtGui, QtCore


class DataTreeWidget(QtGui.QTreeWidget):
    """
    Widget for displaying hierarchical python data structures
    (eg, nested dicts, lists, and arrays), adapted from pyqtgraph datatree.
    """

    def __init__(self, parent=None, data=None):
        QtGui.QTreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.setData(data)
        self.setColumnCount(2)
        self.setHeaderLabels(['Parameter', 'value'])

    def setData(self, data, hideRoot=False):
        """data should be a dictionary."""
        self.clear()
        self.buildTree(data, self.invisibleRootItem(), hideRoot=hideRoot)
        self.expandToDepth(3)
        self.resizeColumnToContents(0)

    def buildTree(self, data, parent, name='', hideRoot=False):
        if hideRoot:
            node = parent
        else:
            node = QtGui.QTreeWidgetItem([name, ""])
            parent.addChild(node)

        if isinstance(data, dict):
            for k in data.keys():
                self.buildTree(data[k], node, str(k))
        elif isinstance(data, list) or isinstance(data, tuple):
            for i in range(len(data)):
                self.buildTree(data[i], node, str(i))
        else:
            node.setText(1, str(data))


class ImageView(pg.ImageView):
    """
    Subclass of PG ImageView to correct z-slider signal behavior, and add coordinate label.
    See pygtgraph.ImageView for documentation
    """
    sigDeletePressed = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super(ImageView, self).__init__(*args, **kwargs)
        self.scene.sigMouseMoved.connect(self.mouseMoved)

        self.coordsLabel = QtGui.QLabel(' ', parent=self)
        self.coordsLabel.setMinimumHeight(16)
        self.layout().addWidget(self.coordsLabel)
        self.coordsLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
        self.setStyleSheet("background-color: rgba(0,0,0,0%)")


    def buildMenu(self):
        super(ImageView, self).buildMenu()
        self.menu.removeAction(self.normAction)

    def keyPressEvent(self, ev):
        super(ImageView, self).keyPressEvent(ev)
        if ev.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
            self.timeLineChanged()
        elif ev.key() == QtCore.Qt.Key_Delete or ev.key() == QtCore.Qt.Key_Backspace:
            self.sigDeletePressed.emit()

    def timeIndex(self, slider):
        ## Return the time and frame index indicated by a slider
        if self.image is None:
            return (0,0)

        t = slider.value()

        xv = self.tVals
        if xv is None:
            ind = int(t)
        else:
            if len(xv) < 2:
                return (0,0)
            totTime = xv[-1] + (xv[-1]-xv[-2])
            inds = np.argwhere(xv <= t)
            if len(inds) < 1:
                return (0,t)
            ind = inds[-1,0]
        return ind, t

    def mouseMoved(self, ev):
        pos = ev
        viewBox = self.imageItem.getViewBox()
        try:
            if viewBox.sceneBoundingRect().contains(pos):
                mousePoint = viewBox.mapSceneToView(pos)
                x, y = map(int, (mousePoint.x(), mousePoint.y()))
                if (0 <= x < self.imageItem.image.shape[0]) & (0 <= y < self.imageItem.image.shape[1]):  # within bounds
                    self.coordsLabel.setText(u"<div style='font-size: 12pt;background-color:#111111;'>x={0},"
                                             u"   <span style=''>y={1}</span>,   <span style=''>I={2}</span>"\
                                             .format(x, y, self.imageItem.image[x, y]))
                else:
                    self.coordsLabel.setText(u"<div style='font-size: 12pt;background-color:#111111;'>x= ,"
                                             u"   <span style=''>y= </span>,   <span style=''>I= </span>")
        except AttributeError:
            pass

class dataDialog(QtGui.QDialog):

    """
    Subclass of QDialog to allow for inputs
    """

    def __init__(self, parent=None, msg=None):
        super(dataDialog, self).__init__(parent)

        # text at top of box
        text = QtGui.QLabel(msg)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(text)
        layout.addSpacing(5)

        # dialog for searching through files
        dialog = explorer.MultipleFileExplorer(self)
        layout.addWidget(dialog)

        horiz_layout = QtGui.QHBoxLayout()
        # horiz_layout.setContentsMargins(0,0,0,0)

        # ok and cancel buttons for user interaction
        button_box = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)

        # file name viewer
        file_label = QtGui.QLabel("File name: ")
        file_name = QtGui.QLineEdit()
        file_name.setReadOnly(True)
        file_name.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)

        horiz_layout.addWidget(file_label)
        horiz_layout.addWidget(file_name)
        horiz_layout.addWidget(button_box)

        layout.addSpacing(5)
        layout.addLayout(horiz_layout)
        self.setLayout(layout)
        self.exec_()



class sliceDialog(QtGui.QDialog):

    """
    Subclass of QDialog to allow for more diverse returns from input dialog popups, specifically
    relating to tomography slice previews

    Attributes
    ----------
    field1: int
        integer representing lowest slice number to reconstruct
    field2: int
        integer representing highest slice number to reconstruct
    """

    def __init__(self, parent=None, val1=0, val2=0, maximum=1000):

        super(sliceDialog, self).__init__(parent)

        text = QtGui.QLabel("Which slice(s) would you like to reconstruct?")
        layout = QtGui.QVBoxLayout()
        layout.addWidget(text)

        label1 = QtGui.QLabel("Start slice: ")
        label2 = QtGui.QLabel("End slice: ")
        self.field1 = QtGui.QSpinBox()
        self.field1.setRange(0,maximum)
        self.field1.setValue(val1)
        self.field2 = QtGui.QSpinBox()
        self.field2.setRange(0,maximum)
        self.field2.setValue(val2)


        h1 = QtGui.QHBoxLayout()
        h1.addWidget(label1)
        h1.addWidget(self.field1)
        h2 = QtGui.QHBoxLayout()
        h2.addWidget(label2)
        h2.addWidget(self.field2)

        valueButton = QtGui.QPushButton("Ok")
        valueButton.setDefault(True)
        cancelButton = QtGui.QPushButton("Cancel")
        valueButton.clicked.connect(self.return_vals)
        cancelButton.clicked.connect(self.cancel)
        h3 = QtGui.QHBoxLayout()
        h3.addWidget(cancelButton)
        h3.addWidget(valueButton)

        layout.addLayout(h1)
        layout.addLayout(h2)
        layout.addLayout(h3)
        self.setLayout(layout)
        self.setWindowTitle("Slice Preview")
        self.exec_()

    def return_vals(self):
        try:
            val1 = int(self.field1.text())
            val2 = int(self.field2.text())
            if val1>val2:
                self.value = [val2,val1]
            else:
                self.value = [val1,val2]
        except ValueError:
            self.value = "Slice values must be integers"

        self.accept()


    def cancel(self):
        self.value = None
        self.reject()
