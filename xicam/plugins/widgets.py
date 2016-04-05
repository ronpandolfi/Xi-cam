# --coding: utf-8 --

__author__ = "Ronald J Pandolfi"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"

import os

import numpy as np
import pyqtgraph as pg
from PySide import QtGui, QtCore

from xicam import config
from pipeline import loader


class OOMTabItem(QtGui.QWidget):
    sigLoaded = QtCore.Signal()

    def __init__(self, itemclass=None, *args, **kwargs):
        """
        A collection of references that can be used to reconstruct an object dynamically and dispose of it when unneeded
        :type paths: list[str]
        :type experiment: config.experiment
        :type parent: main.MyMainWindow
        :type operation:
        :return:
        """
        super(OOMTabItem, self).__init__()

        self.itemclass = itemclass
        self.args = args
        self.kwargs = kwargs

        self.isloaded = False
        self.setLayout(QtGui.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

    def load(self):
        """
        load this tab; rebuild the viewer
        """
        if not self.isloaded:
            if 'operation' in self.kwargs:
                if self.kwargs['operation'] is not None:
                    print self.kwargs['paths']
                    imgdata = [loader.loadimage(path) for path in self.kwargs['paths']]
                    imgdata = self.kwargs['operation'](imgdata)
                    dimg = loader.diffimage(filepath=self.kwargs['paths'][0], data=imgdata)
                    self.kwargs['dimg'] = dimg

            self.widget = self.itemclass(*self.args, **self.kwargs)

            self.layout().addWidget(self.widget)

            self.isloaded = True

            self.sigLoaded.emit()
            print 'Loaded:', self.itemclass


    def unload(self):
        """
        orphan the tab widget and queue them for deletion. Mwahahaha.
        """
        if self.isloaded:
            self.widget.deleteLater()
            self.widget = None

        self.isloaded = False



class ImageView(pg.ImageView):
    sigKeyRelease = QtCore.Signal()

    def buildMenu(self):
        super(ImageView, self).buildMenu()
        self.menu.removeAction(self.normAction)

    def keyReleaseEvent(self, ev):
        super(ImageView, self).keyReleaseEvent(ev)
        if ev.key() in [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            ev.accept()
            self.sigKeyRelease.emit()


class pluginModeWidget(QtGui.QWidget):
    def __init__(self, plugins):
        super(pluginModeWidget, self).__init__()
        self.setLayout(QtGui.QHBoxLayout())

        self.font = QtGui.QFont()
        self.font.setPointSize(16)
        self.plugins = plugins

        self.reload()

    def reload(self):
        w = self.layout().takeAt(0)
        while w:
            w.widget().deleteLater()
            del w
            w = self.layout().takeAt(0)

        for key, plugin in self.plugins.items():
            if plugin.enabled:
                if plugin.instance.hidden:
                    continue

                button = QtGui.QPushButton(plugin.name)
                button.setFlat(True)
                button.setFont(self.font)
                button.setProperty('isMode', True)
                button.setAutoFillBackground(False)
                button.setCheckable(True)
                button.setAutoExclusive(True)
                button.clicked.connect(plugin.activate)
                if plugin is self.plugins.values()[0]:
                    button.setChecked(True)
                self.layout().addWidget(button)
                label = QtGui.QLabel('|')
                label.setFont(self.font)
                label.setStyleSheet('background-color:#111111;')
                self.layout().addWidget(label)
        try: # hack for tomo branch
            self.layout().takeAt(self.layout().count() - 1).widget().deleteLater()  # Delete the last pipe symbol
        except AttributeError:
            pass

class previewwidget(pg.GraphicsLayoutWidget):
    """
    top-left preview
    """

    def __init__(self, tree):
        super(previewwidget, self).__init__()
        self.tree = tree
        self.model = tree.model()
        self.view = self.addViewBox(lockAspect=True)

        self.imageitem = pg.ImageItem()
        self.view.addItem(self.imageitem)
        self.imgdata = None
        self.setMinimumHeight(250)

        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)


    def loaditem(self, current, previous):

        try:
            path = self.model.filePath(current)
            if os.path.isfile(path):
                self.imgdata = loader.loadimage(path)
                self.imageitem.setImage(np.rot90(np.log(self.imgdata * (self.imgdata > 0) + (self.imgdata < 1)), 3),
                                        autoLevels=True)
        except TypeError:
            self.imageitem.clear()

