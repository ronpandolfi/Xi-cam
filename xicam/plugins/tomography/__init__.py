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
from functools import partial
from PySide import QtGui
import yamlmod
from xicam.plugins import base
from pipeline import msg
from xicam import threads
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

        # Connect toolbar signals and ui button signals
        self.toolbar.connecttriggers(self.slicePreviewAction, self.preview3DAction, self.fullReconstruction,
                                     self.manualCenter)
        self.ui.connectTriggers(self.loadPipeline, self.savePipeline, self.resetPipeline,
                        lambda: self.manager.swapFeatures(self.manager.selectedFeature, self.manager.previousFeature),
                        lambda: self.manager.swapFeatures(self.manager.selectedFeature, self.manager.nextFeature),
                                self.clearPipeline)
        ui.build_function_menu(self.ui.addfunctionmenu, config.funcs['Functions'],
                               config.names, self.manager.addFunction)


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

    def openfiles(self, paths, *args, **kwargs):
        msg.showMessage('Loading file...', timeout=10)
        self.activate()
        if type(paths) is list:
            paths = paths[0]

        widget = twidgets.TomoViewer(paths=paths)
        self.centerwidget.addTab(widget, os.path.basename(paths))
        self.centerwidget.setCurrentWidget(widget)

    def currentDataset(self):
        try:
            return self.centerwidget.currentWidget()
        except AttributeError:
            return None

    def currentChanged(self, index):
        self.toolbar.actionCenter.setChecked(False)
        try:
            current_dataset = self.currentDataset()
            if current_dataset is not None:
                current_dataset.sigReconFinished.connect(self.fullReconstructionFinished)
                current_dataset.wireupCenterSelection(self.manager.recon_function)
        except AttributeError as e:
            msg.logMessage(e, level=40)
        self.setPipelineValues()

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
            pipeline = config.extract_pipeline_dict(self.manager.features)
            print pipeline
            yamlmod.ordered_dump(pipeline, yml)

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

    def setPipelineValues(self):
        widget = self.currentDataset()
        if widget is not None:
            self.ui.property_table.setData(widget.data.header.items())
            self.ui.property_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
            self.ui.property_table.show()
            self.ui.setConfigParams(widget.data.shape[0], widget.data.shape[2])
            config.set_als832_defaults(widget.data.header, funcs=self.manager.features)
        # manager.update_function_parameters(funcs=manager.functions)
        # recon = manager.recon_function
        # if recon is not None:
        #     recon.setCenterParam(self.currentDataset().cor)

    def tabCloseRequested(self, index):
        self.ui.property_table.clear()
        self.ui.property_table.hide()
        self.centerwidget.widget(index).deleteLater()

    def manualCenter(self, value):
        self.currentDataset().onManualCenter(value)

    def checkPipeline(self):
        if len(self.manager.features) < 1 or self.currentDataset() is None:
            return False
        elif 'Reconstruction' not in [func.func_name for func in self.manager.features]:
            QtGui.QMessageBox.warning(None, 'Reconstruction method required',
                                      'You have to select a reconstruction method to run a preview')
            return False
        return True

    def slicePreviewAction(self):
        if self.checkPipeline():
            msg.showMessage('Computing slice preview...', timeout=0)
            self.processFunctionStack(callback=lambda x: self.runSlicePreview(*x))

    def runSlicePreview(self, partial_stack, stack_dict):
        initializer = self.currentDataset().getsino()
        slice_no = self.currentDataset().sinogramViewer.currentIndex
        callback = partial(self.currentDataset().addSlicePreview, stack_dict, slice_no=slice_no)
        message = 'Unable to compute slice preview. Check log for details.'
        self.foldPreviewStack(partial_stack, initializer, callback, message)

    def preview3DAction(self):
        if self.checkPipeline():
            msg.showMessage('Computing 3D preview...', timeout=0)
            slc = (slice(None), slice(None, None, 8), slice(None, None, 8))
            self.processFunctionStack(callback=lambda x: self.run3DPreview(*x), slc=slc)

    def run3DPreview(self, partial_stack, stack_dict):
        slc = (slice(None), slice(None, None, 8), slice(None, None, 8))
        initializer = self.currentDataset().getsino(slc)
        self.manager.cor_scale = lambda x: x // 8
        callback = partial(self.currentDataset().add3DPreview, stack_dict)
        message = 'Unable to compute 3D preview. Check log for details.'
        self.foldPreviewStack(partial_stack, initializer, callback, message)

    def processFunctionStack(self, callback, finished=None, slc=None):
        bg_functionstack = threads.method(callback_slot=callback, finished_slot=finished,
                                          lock=threads.mutex)(self.manager.previewFunctionStack)
        bg_functionstack(self.currentDataset(), slc=slc, ncore=self.ui.config_params.child('CPU Cores').value())

    def foldPreviewStack(self, partial_stack, initializer, callback, error_message):
        except_slot = lambda: msg.showMessage(error_message)
        bg_fold = threads.method(callback_slot=callback, finished_slot=msg.clearMessage, lock=threads.mutex,
                                 except_slot=except_slot)
        bg_fold(self.manager.foldFunctionStack)(partial_stack, initializer)

    def fullReconstruction(self):
        self.bottomwidget.local_console.clear()
        start = self.ui.config_params.child('Start Sinogram').value()
        end = self.ui.config_params.child('End Sinogram').value()
        step =  self.ui.config_params.child('Step Sinogram').value()
        self.currentDataset().runFullRecon((self.ui.config_params.child('Start Projection').value(),
                                            self.ui.config_params.child('End Projection').value(),
                                            self.ui.config_params.child('Step Projection').value()),
                                           (start, end, step),
                                           self.ui.config_params.child('Sinograms/Chunk').value(),
                                           self.ui.config_params.child('CPU Cores').value(),
                                           update_call=self.bottomwidget.log2local,
                                           interrupt_signal=self.bottomwidget.local_cancelButton.clicked)
        msg.showMessage('Computing reconstruction...', timeout=0)
        self.recon_start_time = time.time()
        # else:
        #     print 'Beep'
        #     # r = QtGui.QMessageBox.warning(self, 'Reconstruction running', 'A reconstruction is currently running.\n'
        #     #                                                               'Are you sure you want to start another one?',
        #     #                               (QtGui.QMessageBox.Yes | QtGui.QMessageBox.No))
        #     # if r is QtGui.QMessageBox.Yes:
        #     #     QtGui.QMessageBox.information(self, 'Reconstruction request',
        #     #                                   'Then you should wait until the first one finishes.')

    def fullReconstructionFinished(self):
        run_time = time.time() - self.recon_start_time
        self.bottomwidget.log2local('Reconstruction complete. Run time: {:.2f} s'.format(run_time))
        self._recon_running = False

