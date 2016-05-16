import os
import sys

from PySide import QtGui, QtCore

from xicam import xglobals
from xicam import threads
from xicam.splash import SplashScreen
import xicam

def main():
    os.chdir(os.path.abspath(os.path.join(xicam.__file__,'../..')))
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
    app.lastWindowClosed.connect(threads.worker_thread.exit)

    sys.exit(app.exec_())



if __name__ == '__main__':
    main()