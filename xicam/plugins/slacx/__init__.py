import os

import base
from PySide import QtGui
from PySide import QtCore

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
        ui_file = QtCore.QFile(os.getcwd()+"xicam/plugins/slacx/slacxbase/ui/basic.ui")
        ui_file.open(QtCore.QFile.ReadOnly)
        slacxui = QtUiTools.QUiLoader().load(ui_file)
        ui_file.close()

        #self.centerwidget = QtGui.QFrame()
        #self.centerwidget.setLayout(QtGui.QGridLayout())
        #self.centerwidget.layout().addItem(slacxui.image_viewer,0,0)
        self.centerwidget = slacxui.image_viewer

        self.leftwidget = QtGui.QFrame()
        self.leftwidget.setLayout(slacxui.workflow_layout)
        #self.centerwidget.layout().addItem(slacxui.image_viewer,0,0)
        #self.fileslistwidget = widgets.filesListWidget()
        #self.centerwidget.setLayout(QtGui.QVBoxLayout())
        #self.centerwidget.layout().addWidget(self.fileslistwidget)



        self.rightwidget = ParameterTree()
        self.remeshOption = pTypes.SimpleParameter(type='bool', name='GIXS remeshing', value=False)
        self.integrateOption = pTypes.SimpleParameter(type='bool', name='Azimuthal integration', value=True)
        self.roiOption = pTypes.SimpleParameter(type='bool', name='Integrate last ROI', value=True)
        self.logOption = pTypes.SimpleParameter(type='bool', name='Log scale image', value=False)
        self.cakeOption = pTypes.SimpleParameter(type='bool', name='Cake (q/chi)', value=False)
        self.exportformat = pTypes.ListParameter(type='list', name='Image export format', value=0, values=['EDF (.edf)','TIFF (.tif)','JPEG (.jpg)'])
        self.processButton = pTypes.ActionParameter(name='Process')
        # self.abortButton = pTypes.ActionParameter(name='Abort')
        params = [self.remeshOption, self.cakeOption, self.integrateOption, self.roiOption, self.logOption, self.exportformat, self.processButton]
        paramgroup = Parameter.create(name='params', type='group', children=params)
        self.rightwidget.setParameters(paramgroup, showTop=False)

        self.processButton.sigActivated.connect(self.processfiles)

        super(SlacxPlugin, self).__init__(*args, **kwargs)

