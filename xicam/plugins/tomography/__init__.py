#! /usr/bin/env python

__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"

# Use NSURL as a workaround to pyside/Qt4 behaviour for dragging and dropping on OSx
import platform
op_sys = platform.system()
# if op_sys == 'Darwin':
#     from Foundation import NSURL

import os
import time
from PySide import QtGui
import yamlmod
from xicam.plugins import base
from pipeline import msg
import widgets as twidgets
import ui
import config
from fncwidgets import FunctionManager

DEFAULT_PIPELINE_YAML = 'yaml/tomography/default_pipeline.yml'

class plugin(base.plugin):
    """
    Tomography plugin class
    """
    name = "Tomography"
    def __init__(self, *args, **kwargs):

        self.ui = ui.UIform()
        self.ui.setupUi()
        self.centerwidget = self.ui.centerwidget
        self.toolbar = self.ui.toolbar
        self.leftmodes = self.ui.leftmodes
        self.rightmodes = self.ui.rightmodes
        self.bottomwidget = self.ui.bottomwidget
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        # Setup FunctionManager
        self.manager = FunctionManager(self.ui.functionwidget.functionsList, self.ui.param_form,
                                       blank_form='Select a function from\n below to set parameters...')
        config.load_pipeline(DEFAULT_PIPELINE_YAML, self.manager, setdefaults=True)

        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent

        self.toolbar.connecttriggers(self.previewSlice, self.preview3D, self.fullReconstruction, self.manualCenter)
        self.ui.connectTriggers(self.loadPipeline, self.savePipeline, self.resetPipeline,
                        lambda: self.manager.swapFeatures(self.manager.selectedFeature, self.manager.previousFeature),
                        lambda: self.manager.swapFeatures(self.manager.selectedFeature, self.manager.nextFeature),
                                self.clearPipeline)

        super(plugin, self).__init__(*args, **kwargs)

        self.recon_start_time = 0

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if op_sys == 'Darwin':
                fname = str(NSURL.URLWithString_(str(url.toString())).filePathURL().path())
            else:
                fname = str(url.toLocalFile())
            if os.path.isfile(fname):
                self.openfiles([fname])
            e.accept()

    def dragEnterEvent(self, e):
        e.accept()

    def currentChanged(self, index):
        self.toolbar.actionCenter.setChecked(False)
        try:
            current_dataset = self.currentDataset()
            if current_dataset is not None:
                current_dataset.sigReconFinished.connect(self.fullReconstructionFinished)
                current_dataset.wireupCenterSelection(self.manager.recon_function)
                self.setPipelineValues(current_dataset)
        except AttributeError as e:
            msg.logMessage(e, level=40)

    def openfiles(self, paths, *args, **kwargs):
        msg.showMessage('Loading file...', timeout=10)
        self.activate()
        if type(paths) is list:
            paths = paths[0]

        widget = twidgets.TomoViewer(paths=paths)
        self.centerwidget.addTab(widget, os.path.basename(paths))
        self.centerwidget.setCurrentWidget(widget)

    def loadPipeline(self):
        open_file = QtGui.QFileDialog.getOpenFileName(None, 'Open tomography pipeline file',
                                                      os.path.expanduser('~'), selectedFilter='*.yml')[0]
        if open_file != '':
            config.load_pipeline(open_file, self.manager)

    def savePipeline(self):
        save_file = QtGui.QFileDialog.getSaveFileName(None, 'Save tomography pipeline file as',
                                                      os.path.expanduser('~'), selectedFilter='*.yml')[0]

        save_file = save_file.split('.')[0] + '.yml'
        with open(save_file, 'w') as yml:
            yamlmod.ordered_dump(self.manager.pipeline_dict, yml)

    def clearPipeline(self):
        value = QtGui.QMessageBox.question(None, 'Delete functions', 'Are you sure you want to clear ALL functions?',
                                           (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
        if value is QtGui.QMessageBox.Yes:
            self.manager.removeAllFeatures()

    def resetPipeline(self):
        value = QtGui.QMessageBox.question(None, 'Reset functions', 'Do you want to reset to default functions?',
                                           (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
        if value is QtGui.QMessageBox.Yes:
            config.load_pipeline(DEFAULT_PIPELINE_YAML, self.manager, setdefaults=True)

    def setPipelineValues(self, widget):
        self.ui.property_table.setData(widget.data.header.items())
        self.ui.property_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.ui.property_table.show()
        ui.setconfigparams(int(widget.data.header['nslices']),
                           int(widget.data.header['nangles']))
        # manager.set_function_defaults(widget.data.header, funcs=manager.functions)
        # manager.update_function_parameters(funcs=manager.functions)
        # recon = manager.recon_function
        # if recon is not None:
        #     recon.setCenterParam(self.currentDataset().cor)

    def tabCloseRequested(self, index):
        self.ui.property_table.clear()
        self.ui.property_table.hide()
        self.centerwidget.widget(index).deleteLater()

    def currentDataset(self):
        try:
            return self.centerwidget.currentWidget()
        except AttributeError:
            return None

    def previewSlice(self):
        msg.showMessage('Computing slice preview...', timeout=0)
        self.currentDataset().runSlicePreview()

    def preview3D(self):
        msg.showMessage('Computing 3D preview...', timeout=0)
        self.currentDataset().run3DPreview()

    def fullReconstruction(self):
        if not self._recon_running:
            self._recon_running = True
            self.bottomwidget.local_console.clear()
            start = ui.configparams.child('Start Sinogram').value()
            end = ui.configparams.child('End Sinogram').value()
            step =  ui.configparams.child('Step Sinogram').value()
            msg.showMessage('Computing reconstruction...', timeout=0)
            self.currentDataset().runFullRecon((ui.configparams.child('Start Projection').value(),
                                                ui.configparams.child('End Projection').value(),
                                                ui.configparams.child('Step Projection').value()),
                                               (start, end, step),
                                               ui.configparams.child('Sinograms/Chunk').value(),
                                               ui.configparams.child('CPU Cores').value(),
                                               update_call=self.bottomwidget.log2local,
                                               interrupt_signal=self.bottomwidget.local_cancelButton.clicked)
            self.recon_start_time = time.time()
        else:
            print 'Beep'
            # r = QtGui.QMessageBox.warning(self, 'Reconstruction running', 'A reconstruction is currently running.\n'
            #                                                               'Are you sure you want to start another one?',
            #                               (QtGui.QMessageBox.Yes | QtGui.QMessageBox.No))
            # if r is QtGui.QMessageBox.Yes:
            #     QtGui.QMessageBox.information(self, 'Reconstruction request',
            #                                   'Then you should wait until the first one finishes.')

    def fullReconstructionFinished(self):
        run_time = time.time() - self.recon_start_time
        self.bottomwidget.log2local('Reconstruction complete. Run time: {:.2f} s'.format(run_time))
        self._recon_running = False

    def manualCenter(self, value):
        self.currentDataset().onManualCenter(value)
