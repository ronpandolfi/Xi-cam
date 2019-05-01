from __future__ import absolute_import
from __future__ import unicode_literals
from . import base
from PySide import QtGui
import os
from xicam import xglobals

from . import widgets


class MOTDPlugin(base.plugin):
    name = 'MOTD'
    hidden = True


    def __init__(self, *args, **kwargs):
        self.rightwidget = None

        try:
            from PySide import QtWebKit
        except ImportError:
            pass
        else:
            self.centerwidget = webview = QtWebKit.QWebView()
            webview.load('MOTD.html')

        super(MOTDPlugin, self).__init__(*args, **kwargs)

    # Defer methods to other plugins

    def openSelected(self, *args, **kwargs):
        xglobals.plugins['Timeline'].instance.activate()
        xglobals.plugins['Timeline'].instance.openSelected(*args, **kwargs)

    def openfiles(self, *args, **kwargs):
        xglobals.plugins['Viewer'].instance.activate()
        xglobals.plugins['Viewer'].instance.openfiles(*args, **kwargs)

    def appendfiles(self, files):
        xglobals.plugins['Viewer'].instance.activate()
        xglobals.plugins['Viewer'].instance.openfiles(files)

    def calibrate(self):
        xglobals.plugins['Viewer'].instance.activate()
        xglobals.plugins['Viewer'].instance.calibrate()

    def currentImage(self):
        xglobals.plugins['Viewer'].instance.activate()
        xglobals.plugins['Viewer'].instance.currentImage()
