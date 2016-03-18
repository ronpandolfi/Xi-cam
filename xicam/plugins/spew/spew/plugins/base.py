# -*- coding: utf-8 -*-
"""
@author: rpandolfi
"""

from PySide import QtCore, QtGui
from pyqtgraph.parametertree import ParameterTree
from widgets import explorer, toolbars

activeplugin = None

# DEFAULTS
fileexplorer = explorer.MultipleFileExplorer()
explorertoolbar = toolbars.ExplorerToolbar()
explorertoolbar.connect_triggers(fileexplorer.openDataset, fileexplorer.deleteFile, fileexplorer.uploadFile,
                                 fileexplorer.downloadFile, fileexplorer.transferFile)
w = QtGui.QWidget()
w.setContentsMargins(0, 0, 0, 0)
l = QtGui.QVBoxLayout()
l.setContentsMargins(0, 0, 0, 0)
l.setSpacing(0)
l.addWidget(fileexplorer)
l.addWidget(explorertoolbar)
w.setLayout(l)

leftwidget = w

class Plugin(QtCore.QObject):
    name = 'Base Plugin'
    #sigUpdateExperiment = QtCore.Signal()
    hidden = False

    def __init__(self, placeholders, parent=None):
        super(Plugin, self).__init__()

        self.placeholders = placeholders

        if not hasattr(self, 'centerwidget'):
            self.centerwidget = None

        if not hasattr(self, 'rightwidget'):
            self.rightwidget = None

        if not hasattr(self, 'bottomwidget'):
            self.bottomwidget = None

        if not hasattr(self, 'leftwidget'):
            self.leftwidget = leftwidget
            self.fileexplorer = fileexplorer

        if not hasattr(self, 'toolbar'):
            self.toolbar = None

        for widget, placeholder in zip(
                [self.centerwidget, self.rightwidget, self.bottomwidget, self.toolbar, self.leftwidget],
                self.placeholders):
            if widget is not None and placeholder is not None:
                placeholder.addWidget(widget)


    def openSelected(self, operation=None, operationname=None):
        indices = self.filetree.selectedIndexes()
        paths = [self.filetree.filetreemodel.filePath(index) for index in indices]

        self.openfiles(paths, operation, operationname)

    def openfiles(self, files, operation=None, operationname=None):
        pass

    @property
    def isActive(self):
        return activeplugin == self

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
