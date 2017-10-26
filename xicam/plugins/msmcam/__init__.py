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

import numpy as np
from PySide import QtGui, QtCore
from xicam.plugins import base
from xicam import threads
import ui
from workflow import Workflow
import dxchange
from pipeline import msg



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

        # connects
        self.params.sigTreeStateChanged.connect(self.updateparams)
        self.toolbar.actionFilter.triggered.connect(self.filter)
        self.toolbar.actionSegment.triggered.connect(self.run)

        super(MSMCam, self).__init__(*args, **kwargs)

        # set up workflow
        self.path = None
        self.data = None
        self.in_memory = True
        self.wf = Workflow()

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
        if self.path is not None and self.path == paths:
            return
        self.path = paths
        if len(paths) == 1:
            self.wf.input_settings['InputType'] = 1
        else:
            self.wf.input_settings['InputType'] = 0

        self.data = self.loaddata(self.path)
        try:
            self.centerwidget.tab['image'].setImage(self.data)
        except Exception as e:
            msg.showMessage('Unable to load data. Check log for details', timeout=10)
            raise e
        self.centerwidget.setCurrentWidget(self.centerwidget.tab['image'])
        self.toolbar.actionFilter.setEnabled(True)

    def updateparams(self):
        self.wf.update_preproc_settings(self.params)
        self.wf.update_input_settings(self.params)
        self.in_memory = self.wf.input_settings['InMemory']
        self.wf.update_segmentaion_settings(self.params)

    def filter(self):
        self.wf.update_preproc_settings(self.params)
        self.wf.update_input_settings(self.params)
        msg.showBusy()    
        if self.in_memory:
            threads.method(callback_slot=self.showFiltered)(self.wf.filter)(self.data)
        else:
            threads.method(callback_slot=self.showFiltered)(self.wf.filter)(self.path[0])

    def run(self):
        self.wf.update_preproc_settings(self.params)
        self.wf.update_input_settings(self.params)
        self.wf.update_segmentaion_settings(self.params)
        msg.showBusy()
        threads.method(callback_slot=self.showSegmented)(self.wf.run)()
            
    def showFiltered(self, *args):
        msg.hideBusy()
        self.toolbar.actionSegment.setEnabled(True)
        if self.in_memory:
            self.centerwidget.tab['filtered'].setImage(self.wf.filtered)
        else:
            data = self.loaddata(self.wf.filtered)
            self.centerwidget.tab['filtered'].setImage(data)
        self.centerwidget.setCurrentWidget(self.centerwidget.tab['filtered'])

    def showSegmented(self, *args):
        msg.hideBusy()
        if self.in_memory:
            self.centerwidget.tab['segmented'].setImage(self.wf.segmented['kmeans'])
        else:
            data = self.loaddata(self.wf.segmented['kmeans'])
            self.centerwidget.tab['segmented'].setImage(data)
        self.centerwidget.setCurrentWidget(self.centerwidget.tab['segmented'])

    @staticmethod
    def loaddata(path, ibeg=0, iend=None):
        if len(path) > 1:
            if iend is None:
                iend = len(path)
            slc = range(ibeg, iend)
            return dxchange.reader.read_tiff_stack(path[0], slc)
        else:
            data = dxchange.reader.read_tiff(path[0])
            if iend is None:
                iend = data.shape[0]
            slc = range(ibeg, iend)
            return np.array(data[slc,:,:])
