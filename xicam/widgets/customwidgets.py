

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