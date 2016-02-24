import base
from PySide import QtGui
import os
from hipies import globals

import widgets


class plugin(base.plugin):
    name = 'MOTD'
    hidden = True

    MOTD = """<div align='center'><img src='{}/gui/camera.jpg' width='200'/><h1 style='font-family:Zero Threes;'>Welcome to Xi-cam</h1><br />Please cite Xi-cam in published work: <br />Pandolfi, R., Kumar, D., Venkatakrishnan, S., Hexemer, A.
(under preparation)</div>""".format(os.getcwd())

    def __init__(self, *args, **kwargs):
        self.centerwidget = QtGui.QLabel(self.MOTD)
        self.rightwidget = None

        super(plugin, self).__init__(*args, **kwargs)

    # Defer methods to other plugins

    def openSelected(self, *args, **kwargs):
        globals.plugins['Timeline'].instance.activate()
        globals.plugins['Timeline'].instance.openSelected(*args, **kwargs)

    def openfiles(self, *args, **kwargs):
        globals.plugins['Viewer'].instance.activate()
        globals.plugins['Viewer'].instance.openfiles(*args, **kwargs)

    def calibrate(self):
        globals.plugins['Viewer'].instance.activate()
        globals.plugins['Viewer'].instance.calibrate()

    def currentImage(self):
        globals.plugins['Viewer'].instance.activate()
        globals.plugins['Viewer'].instance.currentImage()