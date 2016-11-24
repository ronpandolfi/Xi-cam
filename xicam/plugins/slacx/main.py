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
    opman = slacxopman.OpManager()
    wfman = slacxwfman.WfManager(app=app)
    uiman = slacxuiman.UiManager(opman,wfman)
    wfman.logmethod = uiman.msg_board_log

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

