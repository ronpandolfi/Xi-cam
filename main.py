__author__ = "Ronald J Pandolfi"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"

import os
import sys

from PySide import QtGui, QtCore

from xicam import xglobals
from xicam.splash import SplashScreen


def main():
    xglobals.load()
    sys.path.append(os.path.join(os.getcwd(), 'lib/python2.7/lib-dynload'))
    for path in sys.path:
        print 'path:', path
    app=QtGui.QApplication(sys.argv)
    xglobals.app = app
    pixmap = QtGui.QPixmap(os.path.join(os.getcwd(), "gui/splash.gif"))
    print 'CWD:', os.getcwd()
    splash = SplashScreen(pixmap, f=QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.SplashScreen)
    splash.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    splash.setMask(pixmap.mask())
    splash.show()
    splash.raise_()
    splash.activateWindow()
    app.setActiveWindow(splash)
    app.processEvents()

    sys.exit(app.exec_())



if __name__ == '__main__':
    main()