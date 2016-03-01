from PySide import QtCore, QtGui


class SplashScreen(QtGui.QSplashScreen):
    def __init__(self, pixmap, f=None):
        super(SplashScreen, self).__init__(pixmap, f)
        self.pixmap = pixmap
        self.timer = QtCore.QTimer(self)
        self.timer.singleShot(1000, self.launchwindow)
        self._launching = False

    def mousePressEvent(self, *args, **kwargs):
        self.timer.stop()
        self.launchwindow()



    def launchwindow(self):
        if not self._launching:
            self._launching = True
            import hipies
            from hipies import xglobals

            xglobals.window = hipies.hipies.MyMainWindow(xglobals.app)
            self.timer.stop()

            xglobals.window.ui.show()
            xglobals.window.ui.raise_()
            xglobals.window.ui.activateWindow()
            xglobals.app.setActiveWindow(xglobals.window.ui)
            self.hide()
            # self.finish(window.ui)

