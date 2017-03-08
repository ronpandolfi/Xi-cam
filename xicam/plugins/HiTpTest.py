import os
from PySide import QtGui
from xicam.plugins import base
from xicam import config
import pyqtgraph as pg
from pyqtgraph import parametertree as pt
from fabio import tifimage
from pipeline import loader, hig, msg
import numpy as np
import subprocess
import xicam.RmcView as rmc


class plugin(base.plugin):
    name = "HiTpWAXS"

    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)
        self.rightwidget = None

        super(plugin, self).__init__(*args, **kwargs)

    def openfiles(self, paths):
        self.activate()
        view_widget = inOutViewer(paths = paths)
        self.centerwidget.addTab(view_widget, os.path.basename(paths[0]))
        self.centerwidget.setCurrentWidget(view_widget)
        view_widget.drawCameraLocation(view_widget.view_stack,view_widget.cameraLocation)

    def tabCloseRequested(self,index):
        self.centerwidget.widget(index).deleteLater()