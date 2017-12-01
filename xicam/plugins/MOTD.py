import base
from PySide import QtGui
import os
from xicam import xglobals

import widgets


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
        xglobals.plugins['Tomography'].instance.activate()
        xglobals.plugins['Tomography'].instance.openSelected(*args, **kwargs)

    def openfiles(self, *args, **kwargs):
        xglobals.plugins['Tomography'].instance.activate()
        xglobals.plugins['Tomography'].instance.openfiles(*args, **kwargs)

    def calibrate(self):
        xglobals.plugins['Tomography'].instance.activate()
        xglobals.plugins['Tomography'].instance.calibrate()

    def currentImage(self):
        xglobals.plugins['Tomography'].instance.activate()
        xglobals.plugins['Tomography'].instance.currentImage()
