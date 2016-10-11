import os

from xicam.plugins import base
from PySide import QtGui, QtCore, QtUiTools

from xicam import config
from xicam import xglobals
from pipeline import msg
from slacxbase.core import slacximgman
from slacxbase.core.operations import slacxopman
from slacxbase.core.workflow import slacxwfman

#import pyqtgraph as pg
#import pyqtgraph.parametertree.parameterTypes as pTypes
#from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
#from pipeline import loader, writer
#import re
#import numpy as np
#import widgets

class SlacxPlugin(base.plugin):
    # The display name in the xi-cam plugin bar
    name = 'Slacx'

    def __init__(self, *args, **kwargs):
        
        # Start up Slacx model managers
        imgman = slacximgman.ImgManager()
        opman = slacxopman.OpManager()
        wfman = slacxwfman.WfManager(imgman=imgman)

        # Plugins may have a centerwidget, leftwidget, 
        # rightwidget, bottomwidget, and toolbar (top).

        # Load the slacx UI
        ui_file = QtCore.QFile(os.getcwd()+"/xicam/plugins/slacx/slacxbase/ui/basic.ui")
        ui_file.open(QtCore.QFile.ReadOnly)
        slacxui = QtUiTools.QUiLoader().load(ui_file)
        ui_file.close()

        #self.centerwidget = QtGui.QFrame()
        #self.centerwidget.setLayout(QtGui.QGridLayout())
        #self.centerwidget.layout().addItem(slacxui.image_viewer,0,0)
        self.centerwidget = slacxui.center_frame
        self.leftwidget = slacxui.left_frame
        self.rightwidget = slacxui.right_frame
        #self.centerwidget.layout().addItem(slacxui.image_viewer,0,0)
        #self.fileslistwidget = widgets.filesListWidget()
        #self.centerwidget.setLayout(QtGui.QVBoxLayout())
        #self.centerwidget.layout().addWidget(self.fileslistwidget)
        #self.processButton.sigActivated.connect(self.processfiles)

        super(SlacxPlugin, self).__init__(*args, **kwargs)

