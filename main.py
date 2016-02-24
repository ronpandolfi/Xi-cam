from PySide import QtGui, QtCore
import hipies
import sys
import os
from hipies import globals
from splash import SplashScreen



if __name__ == '__main__':
    globals.load()
    sys.path.append(os.path.join(os.getcwd(), 'lib/python2.7/lib-dynload'))
    for path in sys.path:
        print 'path:', path
    app=QtGui.QApplication(sys.argv)
    globals.app = app
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
