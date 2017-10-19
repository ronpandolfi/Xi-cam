__author__ = "Dinesh Kumar"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar"] 
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.govr"
__status__ = "Alpha"


import platform
op_sys = platform.system()

import os
import numpy as np
from PySide import QtGui, QtCore
from collections import OrderedDict
from xicam.plugins import base
#from .viewer import Viewer
from . import ui
import dxchange
from pipeline import msg
from pyqtgraph import parametertree as pt
from pyqtgraph import ImageView



class MSMCam(base.plugin):
    """
    MSMCam plugin class

    Attributes
    ----------
    """

    name = "MSMCam"

    def __init__(self, *args, **kwargs):

        _ui = ui.UI()
        self.centerwidget = _ui.centerwidget
        self.toolbar = _ui.toolbar
        self.bottomwidget = None
        self.rightwidget = _ui.rightwidget
        self.rightwidget.show()
        self.params = _ui.params
        super(MSMCam, self).__init__(*args, **kwargs)

    def openfiles(self, paths):
        """
        Overrides openfiles in base.plugin. Loads tomography
        data from known file formats, i.e. tiff and hdf5.

        Parameters
        ----------
        paths: str/list
            Path to file with multiple frames or list of multiple frames.
        """

        self.activate()
        self.path = paths
        data = self.loaddata(self.path)
        try:
            viewer = ImageView()
            viewer.setImage(data)
        except Exception as e:
            msg.showMessage('Unable to load data. Check log for details', timeout=10)
            raise e

        self.centerwidget.addTab(viewer, 'Image')
        self.centerwidget.setCurrentWidget(viewer)


    @staticmethod
    def loaddata(path):
        return np.array(dxchange.reader.read_tiff(path[0]))
