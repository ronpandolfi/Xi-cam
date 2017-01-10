#! /usr/bin/env python

import os
from PySide import QtGui
from  xicam.plugins import base
from viewer import ThreeDViewer

import platform
op_sys = platform.system()
# if op_sys == 'Darwin':
#     from Foundation import NSURL

class Viewer3DPlugin(base.plugin):
    name = "3D Viewer"

    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent
        self.rightwidget = None

        super(Viewer3DPlugin, self).__init__(*args, **kwargs)

    def openfiles(self, paths):
        print paths
        self.activate()
        widget = ThreeDViewer(paths=paths)
        self.centerwidget.addTab(widget, os.path.basename(paths[0]))
        self.centerwidget.setCurrentWidget(widget)

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if op_sys == 'Darwin':
                fname = str(NSURL.URLWithString_(str(url.toString())).filePathURL().path())
            else:
                fname = str(url.toLocalFile())
            if os.path.isfile(fname):
                self.openfiles([fname])
            e.accept()

    def dragEnterEvent(self, e):
        print(e)
        e.accept()

    def currentChanged(self, index):
        pass

    def tabCloseRequested(self, index):
        self.centerwidget.widget(index).deleteLater()