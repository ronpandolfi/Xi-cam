from PySide import QtCore, QtGui
import hipies
from hipies import globals


class SplashScreen(QtGui.QSplashScreen):
    def __init__(self, pixmap, f=None):
        super(SplashScreen, self).__init__(pixmap, f)
        print 'WHY?!:', pixmap.size()
        self.pixmap = pixmap
        self.timer = QtCore.QTimer(self)
        globals.window = hipies.hipies.MyMainWindow(globals.app)
        self.timer.singleShot(3000, self.launchwindow)
        self.timer.start(3000)

    def launchwindow(self):
        print 'WHY?!:', self.pixmap.size()
        self.timer.stop()

        globals.window.ui.show()
        globals.window.ui.raise_()
        globals.window.ui.activateWindow()
        globals.app.setActiveWindow(globals.window.ui)
        self.hide()
        # self.finish(window.ui)

