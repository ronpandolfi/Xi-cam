import base
from PySide import QtGui, QtCore
from xicam import xglobals
from pipeline import msg

colors = {msg.DEBUG: QtCore.Qt.gray, msg.ERROR: QtCore.Qt.darkRed, msg.CRITICAL: QtCore.Qt.red,
          msg.INFO: QtCore.Qt.white, msg.WARNING: QtCore.Qt.yellow}


class plugin(base.plugin):
    name = 'Log'

    def __init__(self, *args, **kwargs):
        self.centerwidget = QtGui.QListWidget()
        self.rightwidget = None
        self.featureform = None
        self.bottomwidget = None
        self.leftwidget = None
        self.toolbar = None

        msg.guilogcallable = self.log
        msg.flushbacklog()

        super(plugin, self).__init__(*args, **kwargs)

    def log(self, level, timestamp, s, icon=None):  # We can have icons!
        item = QtGui.QListWidgetItem(s)
        item.setForeground(QtGui.QBrush(colors[level]))
        item.setToolTip(timestamp)
        self.centerwidget.addItem(item)

    # Defer methods to other plugins

    def openSelected(self, *args, **kwargs):
        xglobals.plugins['Timeline'].instance.activate()
        xglobals.plugins['Timeline'].instance.openSelected(*args, **kwargs)

    def openfiles(self, *args, **kwargs):
        xglobals.plugins['Viewer'].instance.activate()
        xglobals.plugins['Viewer'].instance.openfiles(*args, **kwargs)

    def calibrate(self):
        xglobals.plugins['Viewer'].instance.activate()
        xglobals.plugins['Viewer'].instance.calibrate()

    def currentImage(self):
        xglobals.plugins['Viewer'].instance.activate()
        xglobals.plugins['Viewer'].instance.currentImage()
