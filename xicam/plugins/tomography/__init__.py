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
from modpkgs import yamlmod
from xicam.plugins import base
from pipeline import msg
from xicam import threads
from viewers import TomoViewer
import ui
import config
from functionwidgets import FunctionManager

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

        # Keep a timer for reconstructions
        self.recon_start_time = 0

        # Setup FunctionManager
        self.manager = FunctionManager(self.ui.functionwidget.functionsList, self.ui.param_form,
                                       blank_form='Select a function from\n below to set parameters...')
        self.manager.setPipelineFromYAML(config.load_pipeline(DEFAULT_PIPELINE_YAML))
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
        self.manager.sigTestRange.connect(self.slicePreviewAction)
        ui.build_function_menu(self.ui.addfunctionmenu, config.funcs['Functions'],
                               config.names, self.manager.addFunction)
        super(plugin, self).__init__(*args, **kwargs)

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

        widget = TomoViewer(paths=paths)
        widget.sigSetDefaults.connect(self.manager.setPipelineFromDict)
        widget.wireupCenterSelection(self.manager.recon_function)
        self.centerwidget.addTab(widget, os.path.basename(paths))
        self.centerwidget.setCurrentWidget(widget)

    def currentWidget(self):
        try:
            return self.centerwidget.currentWidget()
        except AttributeError:
            return None

    def currentChanged(self, index):
        try:
            self.setPipelineValues()
            self.manager.updateParameters()
            self.toolbar.actionCenter.setChecked(False)
        except (AttributeError, RuntimeError) as e:
            msg.logMessage(e.message, level=msg.ERROR)

    def loadPipeline(self):
        open_file = QtGui.QFileDialog.getOpenFileName(None, 'Open tomography pipeline file',
                                                      os.path.expanduser('~'), selectedFilter='*.yml')[0]
        if open_file != '':
            self.manager.setPipelineFromYAML(config.load_pipeline(open_file))

    def savePipeline(self):
        save_file = QtGui.QFileDialog.getSaveFileName(None, 'Save tomography pipeline file as',
                                                      os.path.expanduser('~'), selectedFilter='*.yml')[0]

        save_file = save_file.split('.')[0] + '.yml'
        with open(save_file, 'w') as yml:
            pipeline = config.extract_pipeline_dict(self.manager.features)
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
            self.manager.setPipelineFromYAML(config.load_pipeline(DEFAULT_PIPELINE_YAML))

    def setPipelineValues(self):
        widget = self.currentWidget()
        if widget is not None:
            self.ui.property_table.setData(widget.data.header.items())
            self.ui.property_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
            self.ui.property_table.show()
            self.ui.setConfigParams(widget.data.shape[0], widget.data.shape[2])
            config.set_als832_defaults(widget.data.header, funcwidget_list=self.manager.features)
            recon_funcs = [func for func in self.manager.features if func.func_name == 'Reconstruction']
            for rfunc in recon_funcs:
                rfunc.params.child('center').setValue(widget.data.shape[1]/2)
                rfunc.input_functions['theta'].params.child('nang').setValue(widget.data.shape[0])

    def tabCloseRequested(self, index):
        self.ui.setConfigParams(0, 0)
        self.ui.property_table.clear()
        self.ui.property_table.hide()
        self.centerwidget.widget(index).deleteLater()

    def manualCenter(self, value):
        self.currentWidget().onManualCenter(value)

    def checkPipeline(self):
        if len(self.manager.features) < 1 or self.currentWidget() is None:
            return False
        elif 'Reconstruction' not in [func.func_name for func in self.manager.features]:
            QtGui.QMessageBox.warning(None, 'Reconstruction method required',
                                      'You have to select a reconstruction method to run a preview')
            return False
        return True

    def slicePreviewAction(self, message='Computing slice preview...', fixed_func=None):
        if self.checkPipeline():
            msg.showMessage(message, timeout=0)
            self.processFunctionStack(callback=lambda x: self.runSlicePreview(*x), fixed_func=fixed_func)

    def runSlicePreview(self, partial_stack, stack_dict):
        initializer = self.currentWidget().getsino()
        slice_no = self.currentWidget().sinogramViewer.currentIndex
        callback = partial(self.currentWidget().addSlicePreview, stack_dict, slice_no=slice_no)
        message = 'Unable to compute slice preview. Check log for details.'
        self.foldPreviewStack(partial_stack, initializer, callback, message)

    def preview3DAction(self):
        if self.checkPipeline():
            msg.showMessage('Computing 3D preview...', timeout=0)
            slc = (slice(None), slice(None, None, 8), slice(None, None, 8))
            self.manager.cor_scale = lambda x: x // 8
            self.processFunctionStack(callback=lambda x: self.run3DPreview(*x), slc=slc)

    def run3DPreview(self, partial_stack, stack_dict):
        slc = (slice(None), slice(None, None, 8), slice(None, None, 8))
        initializer = self.currentWidget().getsino(slc)  # this step takes quite a bit, think of running a thread
        self.manager.updateParameters()
        callback = partial(self.currentWidget().add3DPreview, stack_dict)
        err_message = 'Unable to compute 3D preview. Check log for details.'
        self.foldPreviewStack(partial_stack, initializer, callback, err_message)

    def processFunctionStack(self, callback, finished=None, slc=None, fixed_func=None):
        bg_functionstack = threads.method(callback_slot=callback, finished_slot=finished,
                                          lock=threads.mutex)(self.manager.previewFunctionStack)
        bg_functionstack(self.currentWidget(), slc=slc, ncore=self.ui.config_params.child('CPU Cores').value(),
                         fixed_func=fixed_func)

    def foldPreviewStack(self, partial_stack, initializer, callback, error_message):
        except_slot = lambda: msg.showMessage(error_message)
        bg_fold = threads.method(callback_slot=callback, finished_slot=msg.clearMessage, lock=threads.mutex,
                                 except_slot=except_slot)
        bg_fold(self.manager.foldFunctionStack)(partial_stack, initializer)

    def fullReconstruction(self):
        if self.checkPipeline():
            name = self.centerwidget.tabText(self.centerwidget.currentIndex())
            msg.showMessage('Computing reconstruction for {}...'.format(name), timeout=0)
            self.bottomwidget.local_console.clear()
            self.manager.updateParameters()
            recon_iter = threads.iterator(callback_slot=self.bottomwidget.log2local,
                                          interrupt_signal=self.bottomwidget.local_cancelButton.clicked,
                                          finished_slot=self.reconstructionFinished)(self.manager.functionStackGenerator)
            pstart = self.ui.config_params.child('Start Projection').value()
            pend = self.ui.config_params.child('End Projection').value()
            pstep = self.ui.config_params.child('Step Projection').value()
            sstart = self.ui.config_params.child('Start Sinogram').value()
            send = self.ui.config_params.child('End Sinogram').value()
            sstep =  self.ui.config_params.child('Step Sinogram').value()
            recon_iter(self.currentWidget(), (pstart, pend, pstep), (sstart, send, sstep),
                       self.ui.config_params.child('Sinograms/Chunk').value(),
                       ncore=self.ui.config_params.child('CPU Cores').value())
            self.recon_start_time = time.time()

    def reconstructionFinished(self):
        run_time = time.time() - self.recon_start_time
        self.bottomwidget.log2local('Reconstruction complete. Run time: {:.2f} s'.format(run_time))
        msg.showMessage('Reconstruction complete.', timeout=10)

