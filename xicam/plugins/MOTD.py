import base
from PySide import QtGui#,QtWebKit
import os
from xicam import xglobals

import widgets


class MOTDPlugin(base.plugin):
    name = 'MOTD'
    hidden = True


    def __init__(self, *args, **kwargs):
        #self.centerwidget = webview = QtWebKit.QWebView()
        self.rightwidget = None

        #webview.load('MOTD.html')

        super(MOTDPlugin, self).__init__(*args, **kwargs)

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
