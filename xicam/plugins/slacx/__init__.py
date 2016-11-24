import os

from xicam.plugins import base
from PySide import QtGui, QtCore, QtUiTools

from slacx.slacxui import slacxuiman
from slacx.slacxcore.operations import slacxopman
from slacx.slacxcore.workflow import slacxwfman

class SlacxPlugin(base.plugin):
    # The display name in the xi-cam plugin bar
    name = 'Workflow'

    def __init__(self, *args, **kwargs):

        # start slacx core objects    
        opman = slacxopman.OpManager()
        wfman = slacxwfman.WfManager()
        # start slacx ui objects
        uiman = slacxuiman.UiManager(opman,wfman)
        wfman.logmethod = uiman.msg_board_log
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
        self.bottomwidget = uiman.ui.message_board

        # There seems to be a problem with this plugin loading approach,
        # where the frames, *in some circumstances, not always*, 
        # mysteriously fail to bring their children with them.
        # Adding these calls to findChildren() 
        # seems to force the frames to find their children.
        # Curious. TODO: Sort this out. -LAP
        uiman.ui.left_frame.findChildren(object)
        uiman.ui.right_frame.findChildren(object)
        uiman.ui.center_frame.findChildren(object)
        #import pdb
        #pdb.set_trace()

        super(SlacxPlugin, self).__init__(*args, **kwargs)


