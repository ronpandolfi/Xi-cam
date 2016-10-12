import os

from xicam.plugins import base
from PySide import QtGui, QtCore, QtUiTools

from xicam import config
from xicam import xglobals
from pipeline import msg
from slacxbase.slacxui import slacxuiman
from slacxbase.slacxcore import slacximgman
from slacxbase.slacxcore.operations import slacxopman
from slacxbase.slacxcore.workflow import slacxwfman

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

        # start slacx core objects    
        imgman = slacximgman.ImgManager()
        opman = slacxopman.OpManager()
        wfman = slacxwfman.WfManager(imgman=imgman)

        # start slacx ui objects
        root_qdir = QtCore.QDir(__file__)
        rootdir = os.path.split( root_qdir.absolutePath() )[0]+'/slacxbase'
        uiman = slacxuiman.UiManager(rootdir)

        # set up ui-core refs    
        uiman.imgman = imgman
        uiman.opman = opman
        uiman.wfman = wfman

        # Make the slacx title box
        uiman.make_title()    

        # Connect the menu actions to UiManager functions
        uiman.connect_actions()

        # Take care of remaining details
        uiman.final_setup()

        # Set the widgets in base.plugin containers
        self.centerwidget = uiman.ui.center_frame
        self.leftwidget = uiman.ui.left_frame
        self.rightwidget = uiman.ui.right_frame

        super(SlacxPlugin, self).__init__(*args, **kwargs)

        # Load the slacx UI
        #ui_file = QtCore.QFile(os.getcwd()+"/xicam/plugins/slacx/slacxbase/slacxui/basic.ui")
        #ui_file.open(QtCore.QFile.ReadOnly)
        #slacxui = QtUiTools.QUiLoader().load(ui_file)
        #ui_file.close()

