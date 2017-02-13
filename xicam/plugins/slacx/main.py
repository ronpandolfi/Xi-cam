import os
import sys

from PySide import QtCore
import qdarkstyle

from slacx.slacxcore.operations import slacxopman
from slacx.slacxcore.workflow import slacxwfman
from slacx.slacxcore import slacxtools

# TODO: Only do the following imports if we are using a gui
from slacx.slacxui import slacxuiman, uitools
from PySide import QtGui

"""
slacx main module.
"""

def main():
    """
    slacx main execution method.
    """

    # TODO: parse sys.argv for whatever inputs 
    
    # Instantiate QApplication, pass in cmd line args sys.argv.
    try:
        app = QtGui.QApplication(sys.argv)
    except RuntimeError:
        # An application already exists, probably. Get a reference to it.
        app = QtCore.QCoreApplication.instance()

    # If running with gui, load dark style:
    app.setStyleSheet(qdarkstyle.load_stylesheet() + app.styleSheet())

    # TODO: give kwargs to these init routines to rebuild saved jobs?
    opman = slacxopman.OpManager()
    wfman = slacxwfman.WfManager(app=app)
    uiman = slacxuiman.UiManager(opman,wfman)
    wfman.logmethod = uiman.msg_board_log
    opman.logmethod = uiman.msg_board_log

    # Make the slacx title box
    uiman.make_title()    

    # Connect the menu actions to UiManager functions
    uiman.connect_actions()

    # Take care of remaining details
    uiman.final_setup()

    ### LAUNCH ###
    # show uiman.ui (a QMainWindow)
    uiman.ui.show()
    # app.exec_() returns an exit code when it is done
    ret = app.exec_()
    # save config files as needed
    opman.save_config()
    sys.exit(ret)
    
# Run the main() function if this module is invoked 
if __name__ == '__main__':
    main()

