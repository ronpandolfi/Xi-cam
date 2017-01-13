import base
from PySide import QtGui
import os
from xicam import xglobals

import widgets


class MOTDPlugin(base.plugin):
    name = 'MOTD'
    hidden = True

    MOTD = """
    <div align='center'>
        <img src='xicam/gui/camera.jpg' width='200'/>
        <h1 style='font-family:Zero Threes;'>
            Welcome to Xi-cam
        </h1>
        <br />
        Please cite Xi-cam in published work: <br />
        Pandolfi, R., Kumar, D., Venkatakrishnan, S., Krishnan, H., Hexemer, A.
        (under preparation)
    </div>"""

    def __init__(self, *args, **kwargs):
        self.centerwidget = QtGui.QLabel(self.MOTD)
        self.rightwidget = None

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