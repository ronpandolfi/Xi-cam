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
            from hipies import globals

            globals.window = hipies.hipies.MyMainWindow(globals.app)
            self.timer.stop()

            globals.window.ui.show()
            globals.window.ui.raise_()
            globals.window.ui.activateWindow()
            globals.app.setActiveWindow(globals.window.ui)
            self.hide()
            # self.finish(window.ui)

