import os
import sys

from PySide import QtGui, QtCore

if sys.platform == 'win32':
    sys.stdout = open(os.path.join(os.path.expanduser('~'),'out.log'),'w')
    sys.stderr = open(os.path.join(os.path.expanduser('~'),'err.log'),'w')

from splash import SplashScreen

def main():
    sys.path.append(os.path.join(os.getcwd(), 'lib/python2.7/lib-dynload'))

    # set QApplication working dir, so that relative paths resolve properly across different install types
    try:
        d=QtCore.QDir(__file__)
        d.cdUp()
        d.cdUp()
        d.setCurrent(d.path())
        print 'QApp root:',QtCore.QDir().current()
    except NameError:
        print 'Could not set QApp root.' # Hopefully this is run as an executable, and this is unnecessary anyway

    for path in sys.path:
        print 'path:', path
    import xicam  # IMPORTANT! DO NOT REMOVE! Xicam must be loaded early to avoid graphical bugs on mac (?!)
    app=QtGui.QApplication(sys.argv)

    pixmap = QtGui.QPixmap("xicam/gui/splash.gif")
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

        xglobals.window = xicam.xicamwindow.MyMainWindow(app)

        xglobals.window.ui.show()
        xglobals.window.ui.raise_()
        xglobals.window.ui.activateWindow()
        app.setActiveWindow(xglobals.window.ui)

    app.processEvents()

    sys.exit(app.exec_())



if __name__ == '__main__':
    main()