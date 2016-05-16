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
        self.startworker()
        self.launchwindow()

    def startworker(self):
        from xicam import threads
        threads.worker_thread.start()

    def launchwindow(self):
        if not self._launching:
            self._launching = True
            import xicam
            from xicam import xglobals

            xglobals.window = xicam.xicamwindow.MyMainWindow(xglobals.app)
            self.timer.stop()

            xglobals.window.ui.show()
            xglobals.window.ui.raise_()
            xglobals.window.ui.activateWindow()
            xglobals.app.setActiveWindow(xglobals.window.ui)
            self.hide()
            # self.finish(window.ui)

