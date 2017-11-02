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
        self.toolbar.actionROI.triggered.connect(self.centerwidget.setROI)
        self.toolbar.actionSaveCfg.triggered.connect(self.saveConfig)
        self.toolbar.viewSelect.currentIndexChanged.connect(self.updateView)
        self.toolbar.boxT.stateChanged.connect(self.viewTranspose)

        super(MSMCam, self).__init__(*args, **kwargs)

        # set up workflow
        self.path = None
        self.data = None
        self.in_memory = True
        self.wf = Workflow()
        self.segmented = None

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

        if self.toolbar.boxT.checkState():
            self.toolbar.boxT.setChecked(False)
        self.wf.input_settings['InputDir'] = os.path.dirname(self.path[0])
        self.centerwidget.setCurrentWidget(self.centerwidget.tab['image'])
        self.centerwidget.tab['filtered'].clear()
        self.centerwidget.tab['segmented'].clear()
        self.toolbar.actionFilter.setEnabled(True)
        self.toolbar.actionROI.setEnabled(True)

    def updateView(self, idx):
        if self.segmented is None: return
        if idx < 0: return
        key = self.segmented[idx]
        if self.toolbar.boxT.checkState():
            self.toolbar.boxT.setChecked(False)
        self.centerwidget.tab['segmented'].setImage(self.wf.segmented[key])

    def protectparams(self, idx):
        if idx < 0: return
        if idx == 0: self.rightwidget.unlock()
        else: self.rightwidget.lock()

    def viewTranspose(self, state):
        idx = self.centerwidget.currentIndex()
        if idx == 0:
            if state == 0:
                self.centerwidget.tab['image'].setImage(self.data)
            else:
                self.centerwidget.tab['image'].setImage(self.data.T)

        if idx == 1:
            if state == 0:
                self.centerwidget.tab['filtered'].setImage(self.wf.filtered)
            else:
                self.centerwidget.tab['filtered'].setImage(self.wf.filtered.T)

        if idx == 2:
            j = self.toolbar.viewSelect.currentIndex()
            key = self.segmented[j] 
            if state == 0:
                self.centerwidget.tab['segmented'].setImage(self.wf.segmented[key])
            else:
                self.centerwidget.tab['segmented'].setImage(self.wf.segmented[key].T)


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
            if self.centerwidget.ROISET:
                img = self.centerwidget.currentWidget().getImageItem()
                slc = self.data[0,:,:]
                roi = self.centerwidget.roi
                _,idx = roi.getArrayRegion(slc, img, returnMappedCoords=True)
                rows = idx[0].astype(int)
                cols = idx[1].astype(int)
                inp = self.data[:,rows, cols]
            else:
                inp = self.data
        else:
            inp = self.path[0]
        threads.method(callback_slot=self.showFiltered)(self.wf.filter)(inp)

    def run(self):
        self.wf.update_preproc_settings(self.params)
        self.wf.update_input_settings(self.params)
        self.wf.update_segmentaion_settings(self.params)
        msg.showBusy()
        threads.method(callback_slot=self.showSegmented)(self.wf.run)()
            
    def showFiltered(self, *args):
        msg.hideBusy()
        self.toolbar.actionSegment.setEnabled(True)
        if self.toolbar.boxT.checkState():
            self.toolbar.boxT.setChecked(False)
        if self.in_memory:
            self.centerwidget.tab['filtered'].setImage(self.wf.filtered)
        else:
            data = self.loaddata(self.wf.filtered)
            self.centerwidget.tab['filtered'].setImage(data)
        self.centerwidget.setCurrentWidget(self.centerwidget.tab['filtered'])

    def showSegmented(self, *args):
        msg.hideBusy()
        res = []
   
        for key, val in self.wf.segmented.items():
            if val is not None:
                res.append(key)
        self.toolbar.viewSelect.clear()
        self.toolbar.viewSelect.addItems(res)
        self.segmented = res
        if self.toolbar.boxT.checkState():
            self.toolbar.boxT.setChecked(False)

        if self.in_memory:
            self.centerwidget.tab['segmented'].setImage(self.wf.segmented['k-means'])
        else:
            data = self.loaddata(self.wf.segmented['k-means'])
            self.centerwidget.tab['segmented'].setImage(data)
        self.centerwidget.setCurrentWidget(self.centerwidget.tab['segmented'])

    def saveConfig(self):
        dirname = os.path.expanduser('~')
        filename, _ = QtGui.QFileDialog.getSaveFileName(caption='Select output file', dir=dirname)
        if filename:
            self.wf.writeConfig(filename) 
         
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
