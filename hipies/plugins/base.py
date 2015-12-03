from PySide import QtCore, QtGui
from pyqtgraph.parametertree import ParameterTree
from hipies import config
from hipies import models
import widgets

activeplugin = None


class plugin(QtCore.QObject):
    name = 'Unnamed Plugin'
    sigUpdateExperiment = QtCore.Signal()

    def __init__(self, placeholders):
        super(plugin, self).__init__()

        self.placeholders = placeholders

        if not hasattr(self, 'centerwidget'):
            self.centerwidget = None

        if not hasattr(self, 'rightwidget'):
            w = QtGui.QWidget()
            l = QtGui.QVBoxLayout()
            l.setContentsMargins(0, 0, 0, 0)

            configtree = ParameterTree()
            configtree.setParameters(config.activeExperiment, showTop=False)
            config.activeExperiment.sigTreeStateChanged.connect(self.sigUpdateExperiment)
            l.addWidget(configtree)

            self.imagePropModel = models.imagePropModel(self.currentImage)
            propertytable = QtGui.QTableView()
            propertytable.setModel(self.imagePropModel)
            l.addWidget(propertytable)

            w.setLayout(l)
            self.rightwidget = w


        if not hasattr(self, 'bottomwidget'):
            self.bottomwidget = None

        if not hasattr(self, 'leftwidget'):
            w = QtGui.QSplitter()
            w.setOrientation(QtCore.Qt.Vertical)
            w.setContentsMargins(0, 0, 0, 0)

            self.filetree = widgets.fileTreeWidget()
            w.addWidget(self.filetree)

            preview = widgets.previewwidget(self.filetree)
            w.insertWidget(0, preview)

            self.filetree.currentChanged = preview.loaditem

            w.setSizes([250, w.height() - 250])

            self.leftwidget = w

        if not hasattr(self, 'toolbar'):
            self.toolbar = None

        for widget, placeholder in zip(
                [self.centerwidget, self.rightwidget, self.bottomwidget, self.toolbar, self.leftwidget],
                self.placeholders):
            if widget is not None and placeholder is not None:
                placeholder.addWidget(widget)

    def openfiles(files, operation=None):
        pass

    def opendirectory(files, operation=None):
        pass

    def addfiles(files, operation=None):
        pass

    def calibrate(self):
        self.centerwidget.currentWidget().widget.calibrate()

    def activate(self):

        for widget, placeholder in zip(
                [self.centerwidget, self.rightwidget, self.bottomwidget, self.toolbar, self.leftwidget],
                self.placeholders):
            if widget is not None and placeholder is not None:
                placeholder.setCurrentWidget(widget)
                placeholder.show()
            if widget is None and placeholder is not None:
                placeholder.hide()

        global activeplugin
        activeplugin = self

    def currentImage(self):
        pass