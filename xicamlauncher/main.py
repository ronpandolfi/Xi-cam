import os
import sys

from PySide import QtGui, QtCore

if sys.platform == 'win32':
    sys.stdout = open(os.path.join(os.path.expanduser('~'),'out.log'),'w')
    sys.stderr = open(os.path.join(os.path.expanduser('~'),'err.log'),'w')

from splash import SplashScreen

def main():
    sys.path.append(os.path.join(os.getcwd(), 'lib/python2.7/lib-dynload'))
    for path in sys.path:
        print 'path:', path
    app=QtGui.QApplication(sys.argv)

    pixmap = QtGui.QPixmap(os.path.join(os.getcwd(), "gui/splash.gif"))
    print 'CWD:', os.getcwd()
    if True:  # Disable to bypass splashscreen for testing on windows
        splash = SplashScreen(pixmap, f=QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.SplashScreen)
        splash.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        splash.setMask(pixmap.mask())
        splash.show()
        splash.raise_()
        splash.activateWindow()
        app.setActiveWindow(splash)

    else:
        import xicam
        from xicam import xglobals

        xglobals.window = xicam.xicamwindow.MyMainWindow(xglobals.app)

        xglobals.window.ui.show()
        xglobals.window.ui.raise_()
        xglobals.window.ui.activateWindow()
        xglobals.app.setActiveWindow(xglobals.window.ui)

    app.processEvents()

    sys.exit(app.exec_())



if __name__ == '__main__':
    main()