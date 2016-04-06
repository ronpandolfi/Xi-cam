#! /usr/bin/env python

__author__ = "Ronald J Pandolfi"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"

# Use NSURL as a workaround to pyside/Qt4 behaviour for dragging and dropping on OSx
import platform
op_sys = platform.system()
if op_sys == 'Darwin':
    from Foundation import NSURL

import os
import numpy as np
import pipeline
from pipeline import loader
from PySide import QtCore, QtGui
from xicam import xglobals
import widgets as twidgets
from xicam.plugins import widgets, explorer
from xicam.plugins import base
import toolbar
from PySide import QtUiTools
import featuremanager
from pyqtgraph.parametertree import ParameterTree
from xicam import models


class plugin(base.plugin):
    name = "Stack"

    def __init__(self, *args, **kwargs):

        self.toolbar = toolbar.tomotoolbar()

        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)






        self.bottomwidget = None

        self.featurewidget = QtUiTools.QUiLoader().load('gui/tomographyleft.ui')

        self.leftwidget =  QtGui.QSplitter(QtCore.Qt.Vertical)
        self.leftwidget.addWidget(self.featurewidget)
        self.fileexplorer = explorer.MultipleFileExplorer()
        self.leftwidget.addWidget(self.fileexplorer)




        w = QtGui.QWidget()
        l = QtGui.QVBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)

        configtree = ParameterTree()
        #configtree.setParameters(config.activeExperiment, showTop=False)
        #config.activeExperiment.sigTreeStateChanged.connect(self.sigUpdateExperiment)
        l.addWidget(configtree)

        propertytable = QtGui.QTableView()
        self.imagePropModel = models.imagePropModel(self.currentImage, propertytable)
        propertytable.verticalHeader().hide()
        propertytable.horizontalHeader().hide()
        propertytable.setModel(self.imagePropModel)
        propertytable.horizontalHeader().setStretchLastSection(True)
        l.addWidget(propertytable)
        w.setLayout(l)
        self.rightwidget = w


        # SETUP FEATURES
        featuremanager.layout = self.featurewidget.featuresList
        featuremanager.load()

        super(plugin, self).__init__(*args, **kwargs)
        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if op_sys == 'Darwin':
                fname = str(NSURL.URLWithString_(str(url.toString())).filePathURL().path())
            else:
                fname = str(url.toLocalFile())
            if os.path.isfile(fname):
                print(fname)
                self.openfiles([fname])
            e.accept()

    def dragEnterEvent(self, e):
        print(e)
        e.accept()
        # if e.mimeData().hasFormat('text/plain'):
        # e.accept()
        # else:
        #     e.accept()

    def currentChanged(self, index):
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        self.centerwidget.currentWidget().load()
        self.imagePropModel.widgetchanged()

    def tabCloseRequested(self, index):
        self.centerwidget.widget(index).deleteLater()

    def openfiles(self, paths,*args,**kwargs):
        self.activate()
        if type(paths) is list:
            paths = paths[0]

        widget = widgets.OOMTabItem(itemclass=twidgets.tomoWidget, paths=paths)
        self.centerwidget.addTab(widget, os.path.basename(paths))
        self.centerwidget.setCurrentWidget(widget)