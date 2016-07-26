#! /usr/bin/env python

import os
from PySide import QtCore, QtGui
from .. import base
from .. import widgets
import widgets as twidgets

import platform
op_sys = platform.system()
# if op_sys == 'Darwin':
#     from Foundation import NSURL

class plugin(base.plugin):
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

        super(plugin, self).__init__(*args, **kwargs)

    def openfiles(self, paths):
        print paths
        self.activate()
        widget = widgets.OOMTabItem(itemclass=twidgets.ThreeDViewer, paths=paths)
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
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        self.centerwidget.currentWidget().load()

    def tabCloseRequested(self, index):
        self.centerwidget.widget(index).deleteLater()