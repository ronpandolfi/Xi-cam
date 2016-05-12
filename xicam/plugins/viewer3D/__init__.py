#! /usr/bin/env python

import os
import numpy as np
import pipeline
import xicam.plugins.viewer
from pipeline import loader
from PySide import QtCore, QtGui
from xicam import xglobals
import widgets as twidgets
from xicam.plugins import widgets

class plugin(xicam.plugins.viewer.plugin):
    name = "3D Viewer"

    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        super(plugin, self).__init__(*args, **kwargs)

        self.sigUpdateExperiment.connect(self.redrawcurrent)
        self.sigUpdateExperiment.connect(self.replotcurrent)
        self.sigUpdateExperiment.connect(self.invalidatecache)

        self.toolbar = None
        self.bottomwidget = None
        self.rightwidget = None

        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent

    def openfiles(self, paths,*args,**kwargs):
        self.activate()
        if type(paths) is list:
            paths = paths[0]

        widget = widgets.OOMTabItem(itemclass=twidgets.VolumeViewer, path=paths)
        self.centerwidget.addTab(widget, os.path.basename(paths))
        self.centerwidget.setCurrentWidget(widget)