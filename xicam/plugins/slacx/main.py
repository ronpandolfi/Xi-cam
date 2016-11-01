import os
import sys

from PySide import QtGui, QtCore
import qdarkstyle

from slacx.slacxui import slacxuiman
from slacx.slacxcore.operations import slacxopman
from slacx.slacxcore.workflow import slacxwfman
from slacx.slacxcore import slacxtools

"""
slacx main module.
"""

def main():
    """
    slacx main execution method.
    """

    # TODO: parse sys.argv for an input file.
    # Input images, operations, workflows to load,  
    # as well as flags for batch mode, real-time mode, gui mode.
    
    # Instantiate QApplication, pass in cmd line args sys.argv.
    try:
        app = QtGui.QApplication(sys.argv)
    except RuntimeError:
        app = QtCore.QCoreApplication.instance()

    # If running with gui, load dark style:
    app.setStyleSheet(qdarkstyle.load_stylesheet() + app.styleSheet())

    #root_qdir = QtCore.QDir(__file__)
    #rootdir = os.path.split( root_qdir.absolutePath() )[0]+'/slacx'

    # TODO: give kwargs to these init routines to rebuild saved jobs?
    # Start a UiManager to create and manage a QMainWindow.
    uiman = slacxuiman.UiManager()
    # Start an OpManager to manage operations.
    opman = slacxopman.OpManager()
    # Start a WfManager to manage workflows.
    wfman = slacxwfman.WfManager(logmethod=uiman.msg_board_log)

    # UiManager needs to store references to the QAbstractItemModel objects
    # that interact with the features of the gui
    # TODO: make this part of the UiManager constructor?
    uiman.opman = opman
    uiman.wfman = wfman

    # Make the slacx title box
    uiman.make_title()    

    # Connect the menu actions to UiManager functions
    uiman.connect_actions()

    # Take care of remaining details
    uiman.final_setup()

    ### LAUNCH ###
    # Show uiman.ui (a QMainWindow)
    uiman.ui.show()
    # sys.exit gracefully after app.exec_() returns its exit code
    sys.exit(app.exec_())
    
# Run the main() function if this module is invoked 
if __name__ == '__main__':
    main()

