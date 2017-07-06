#! /usr#! /usr/bin/env python


__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.govr"
__status__ = "Beta"


# Use NSURL as a workaround to pyside/Qt4 behaviour for dragging and dropping on OSx
import platform
op_sys = platform.system()
# if op_sys == 'Darwin':
#     from Foundation import NSURL

import os
import time
import numpy as np
from functools import partial
from PySide import QtGui, QtCore
from collections import OrderedDict
from modpkgs import yamlmod
from xicam.plugins import base
from xicam.widgets.customwidgets import sliceDialog
from pipeline import msg
from xicam import threads
from viewers import TomoViewer
import ui
import config
from functionmanager import FunctionManager
from psutil import cpu_count

# YAML file specifying the default workflow pipeline
DEFAULT_PIPELINE_YAML = 'xicam/yaml/tomography/als_default_pipeline.yml'
APS_PIPELINE_YAML = 'xicam/yaml/tomography/aps_default_pipeline.yml'


class TomographyPlugin(base.plugin):
    """
    Tomography plugin class


    Attributes
    ----------
    ui : UIform
        Class with the ui setup for plugin
    centerwidget : QtGui.QTabWidget
        Standard centerwidget overriding base.plugin. QTabWidget that holds instances of viewer.TomoViewer for open
        datasets
    toolbar : QtGui.QToolbar
        Standard toolbar overriding base.plugin
    leftmodes : list of tuples
        Standard left modes list [(widget, icon),...]. Current leftmodes include the standard file explorer and the
        function pipeline workflow editor
    rightmodes : list of tuples
        Standard left modes list [(widget, icon),...]. Currently only one rightmode (this is equivalent to defining
        rightwidget). The current widget is the configuration parameters + metadata table
    bottomwidget : viewers.RunConsole
        Standard bottomwidget overriding base.plugin. Console for viewing status message from a full tomography
        reconstruction


    Parameters
    ----------
    placeholders : QtGui Containers
        Containers with method addWidget. Standard containers are managed/defined in base.plugin
    args
        Additional arguments. Not really used
    kwargs
        Additional keyword arguments. Not really used
    """

    name = "Tomography"

    def __init__(self, placeholders, *args, **kwargs):
        self.ui = ui.UIform()
        self.ui.setupUi()
        self.centerwidget = self.ui.centerwidget
        self.toolbar = self.ui.toolbar
        self.leftmodes = self.ui.leftmodes
        self.rightwidget = None
        self.bottom = self.ui.bottomwidget
        # self.bottomwidget = self.ui.bottomwidget
        # self.bottomwidget.hide()
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)


        # Keep a timer for reconstructions
        self.recon_start_time = 0

        # Flag to set if reconstruction is running
        self.recon_running = False

        # Setup FunctionManager
        self.manager = FunctionManager(self.ui.functionwidget.functionsList, self.ui.param_form,
                                       blank_form='Select a function from\n below to set parameters...')
        self.manager.setPipelineFromYAML(config.load_pipeline(DEFAULT_PIPELINE_YAML))
        self.manager.sigPipelineChanged.connect(self.reconnectTabs)
        self.manager.sigFuncAdded.connect(self.setPipelineValues)

        # queue for recon jobs
        self.queue_widget = self.ui.queue
        # self.queue_widget.sigReconSwapped.connect(self.manager.swapQueue)
        # self.queue_widget.sigReconDeleted.connect(self.manager.delQueueJob)

        # DRAG-DROP
        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent

        # Connect toolbar signals and ui button signals
        self.toolbar.connectTriggers(self.slicePreviewAction, self.multiSlicePreviewAction, self.preview3DAction,
                                            self.loadFullReconstruction, self.manualCenter,  self.roiSelection,
                                            self.mbir, self.openFlats, self.openDarks)

        self.ui.connectTriggers(self.loadPipeline, self.savePipeline, self.resetPipeline,
                        lambda: self.manager.swapFeatures(self.manager.selectedFeature, self.manager.previousFeature),
                        lambda: self.manager.swapFeatures(self.manager.selectedFeature, self.manager.nextFeature),
                                self.clearPipeline)
        self.manager.sigTestRange.connect(self.slicePreviewAction)
        self.bottom.local_cancelButton.clicked.connect(self.freeRecon)
        ui.build_function_menu(self.ui.addfunctionmenu, config.funcs['Functions'],
                               config.names, self.manager.addFunction)
        super(TomographyPlugin, self).__init__(placeholders, *args, **kwargs)

        self.leftwidget.resize(300, 480)

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


    def openfiles(self, paths):
        """
        Override openfiles method in base plugin. Used to open a tomography dataset from the recognized file formats
        and instantiate a viewer.TomoViewer tab. This function takes quite a bit, consider running this in a background
        thread

        Parameters
        ----------
        paths : str/list
            Path to file. Currently only one file is supported. Multiple paths (ie stack of tiffs should be easy to
            implement using the formats.StackImage class.

        """
        msg.showMessage('Loading file...', timeout=10)
        self.activate()

        try:
            if type(paths) is list:
                paths = paths[0]
            widget = TomoViewer(toolbar=self.toolbar, paths=paths)
        except Exception as e:
            msg.showMessage('Unable to load file. Check log for details.', timeout=10)
            raise e

        # connect signals
        widget.sigSetDefaults.connect(self.manager.setPipelineFromDict)
        widget.projectionViewer.sigROIWidgetChanged.connect(self.manager.connectReaderROI)
        self.wireCORWidgets(widget)

        self.centerwidget.addTab(widget, os.path.basename(paths))
        self.centerwidget.setCurrentWidget(widget)

    def wireCORWidgets(self, widget):
        widget.wireupCenterSelection(self.manager.recon_function)


        # add COR func to pipeline to connect signals
        self.manager.updateCORFunc("Phase Correlation", widget.projectionViewer.auto_cor_widget)

        # turns on/off COR function in pipeline from COR widget's signal
        widget.projectionViewer.sigCORChanged.connect(self.manager.updateCORChoice)

        # updates the COR function in pipeline based on function chosen in COR widget
        widget.projectionViewer.auto_cor_widget.sigCORFuncChanged.connect(self.manager.updateCORFunc)

        # updates the COR widget based on the COR function chosen
        self.manager.sigCORDetectChanged.connect(widget.projectionViewer.updateCORChoice)


    def opendirectory(self, files, operation=None):
        msg.showMessage('Loading directory...', timeout=10)
        self.activate()
        try:
            if type(files) is list:
                files = files[0]

            widget = TomoViewer(paths=files)
        except Exception as e:
            msg.showMessage('Unable to load directory. Check log for details.', timeout= 10)
            raise e
        widget.sigSetDefaults.connect(self.manager.setPipelineFromDict)
        widget.wireupCenterSelection(self.manager.recon_function)
        self.centerwidget.addTab(widget, os.path.basename(files))
        self.centerwidget.setCurrentWidget(widget)

    def openFlats(self):

        currentWidget = self.centerwidget.widget(self.currentIndex())
        currentWidget.openFlats()

    def openDarks(self):

        currentWidget = self.centerwidget.widget(self.currentIndex())
        currentWidget.openDarks()

    def currentIndex(self):

        try:
            return self.centerwidget.currentIndex()
        except AttributeError:
            raise

    def currentWidget(self):
        """
        Return the current widget (viewer.TomoViewer) from the centerwidgets tabs
        """

        try:
            return self.centerwidget.currentWidget()
        except AttributeError:
            return None

    def currentChanged(self, index):
        """
        Slot to recieve centerwidgets currentchanged signal when a new tab is selected
        """
        try:
            self.setPipelineValues()
            self.manager.updateParameters()
            self.toolbar.actionCenter.setChecked(False)
        except (AttributeError, RuntimeError) as e:
            msg.logMessage(e.message, level=msg.ERROR)

    def reconnectTabs(self):
        """
        Reconnect TomoViewers when the pipeline is reset
        """
        # TODO: change order of functions in manager.features (see functionmanager)
        for idx in range(self.centerwidget.count()):
            self.centerwidget.widget(idx).wireupCenterSelection(self.manager.recon_function)
            self.centerwidget.widget(idx).sigSetDefaults.connect(self.manager.setPipelineFromDict)

    def loadPipeline(self):
        """
        Load a workflow pipeline yaml file
        """

        open_file = QtGui.QFileDialog.getOpenFileName(None, 'Open tomography pipeline file',
                                                      os.path.expanduser('~'), selectedFilter='*.yml')[0]
        if open_file != '':
            self.manager.setPipelineFromYAML(config.load_pipeline(open_file))

    def savePipeline(self):
        """
        Save a workflow pipeline from UI as a yaml file
        """

        save_file = QtGui.QFileDialog.getSaveFileName(None, 'Save tomography pipeline file as',
                                                      os.path.expanduser('~'), selectedFilter='*.yml')[0]

        save_file = save_file.split('.')[0] + '.yml'
        with open(save_file, 'w') as yml:
            pipeline = config.extract_pipeline_dict(self.manager.features)
            yamlmod.ordered_dump(pipeline, yml)

    def clearPipeline(self):
        """
        Clears the current workflow pipeline in UI
        """

        value = QtGui.QMessageBox.question(None, 'Delete functions', 'Are you sure you want to clear ALL functions?',
                                           (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
        if value is QtGui.QMessageBox.Yes:
            self.manager.removeAllFeatures()

    def resetPipeline(self):
        """
        Resets the workflow pipeline to defaults specified by DEFAULT_PIPELINE_YAML file
        """

        value = QtGui.QMessageBox.question(None, 'Reset functions', 'Do you want to reset to default functions?',
                                           (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
        if value is QtGui.QMessageBox.Yes:
            self.manager.setPipelineFromYAML(config.load_pipeline(DEFAULT_PIPELINE_YAML))
        self.setPipelineValues()
        self.manager.updateParameters()

    # def setPipelineValues(self):
    #     """
    #     Sets up the metadata table and default values in configuration parameters and functions based on the selected
    #     dataset
    #     """
    #
    #     widget = self.currentWidget()
    #     if widget is not None:
    #         self.ui.property_table.setData(widget.data.header.items())
    #         self.ui.property_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
    #         self.ui.property_table.show()
    #         self.ui.setConfigParams(widget.data.shape[0], widget.data.shape[2])
    #         config.set_als832_defaults(widget.data.header, funcwidget_list=self.manager.features)
    #         recon_funcs = [func for func in self.manager.features if func.func_name == 'Reconstruction']
    #         for rfunc in recon_funcs:
    #             rfunc.params.child('center').setValue(widget.data.shape[1]/2)
    #             rfunc.input_functions['theta'].params.child('nang').setValue(widget.data.shape[0])

    def setPipelineValues(self):
        """
        Sets up the metadata table and default values in configuration parameters and functions based on the selected
        dataset
        """
        widget =  self.centerwidget.widget(self.currentIndex())
        if widget is not None:
            self.ui.property_table.setData(widget.data.header.items())
            # self.ui.setMBIR(widget.data.header.items())
            self.ui.property_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
            self.ui.property_table.show()
            # self.ui.setConfigParams(widget.data.shape[0], widget.data.shape[2])
            if widget.data.fabimage.classname == 'ALS832H5image':
                config.set_als832_defaults(widget.data.header, funcwidget_list=self.manager.features,
                        path = widget.path, shape=widget.data.shape)
            # if widget.data.fabimage.classname == 'GeneralAPSH5image':
            else:
                self.manager.setPipelineFromYAML(config.load_pipeline(APS_PIPELINE_YAML))
                config.set_aps_defaults(widget.data.header, funcwidget_list=self.manager.features,
                        path = widget.path, shape=widget.data.shape)
            recon_funcs = [func for func in self.manager.features if func.func_name == 'Reconstruction']
            for rfunc in recon_funcs:
                if not rfunc.params.child('center').value():
                    rfunc.params.child('center').setValue(widget.data.shape[1]/2)
                rfunc.input_functions['theta'].params.child('nang').setValue(widget.data.shape[0])


    def loadPipelineDictionary(self):
        """
        Loads a pipeline dictionary containing information relevant to reconstruction, including parameters and
        arguments that are held by FunctionWidgets. This information can be updated on the  widgets in the middle of a
        run, so the reconstruction should refer to this dictionary for relevant parameters
        """

        currentWidget = self.centerwidget.widget(self.currentIndex())

        for function in self.manager.features:
            currentWidget.pipeline[function.name] = OrderedDict()
            for (key,val) in function.param_dict.iteritems():
                currentWidget.pipeline[function.name][key] = val
            currentWidget.pipeline[function.name]['func_name'] = str(function.package) + "." + \
                                                                 str(function._function.__name__)
            currentWidget.pipeline[function.name]['enabled'] = function.enabled


            lst = []
            for item in function.missing_args:
                lst.append(item)
            currentWidget.pipeline[function.name]['missing_args'] = lst

            input_dict = OrderedDict()
            for key,val in function.input_functions.iteritems():
                # print key, ",", val.params.children()
                dict = OrderedDict()
                dict['func'] = val
                dict['enabled'] = val.enabled
                dict['subfunc_name'] = val.subfunc_name
                dict['vals'] = OrderedDict()
                for child in val.params.children():
                    dict['vals'][child.name()] = child.value()
                input_dict[key] = dict
            currentWidget.pipeline[function.name]["input_functions"] = input_dict
        currentWidget.pipeline['pipeline_for_yaml'] = config.extract_pipeline_dict(self.manager.features)


    def tabCloseRequested(self, index):
        """
        Slot to receive signal when a tab is closed. Simply resets configuration parameters and clears metadata table

        Parameters
        ----------
        index : int
            Index of tab that is being closed.
        """

        # self.ui.setConfigParams(0, 0)
        self.ui.property_table.clear()
        self.ui.property_table.hide()
        self.centerwidget.widget(index).deleteLater()

    def roiSelection(self):
        """
        Slot to receive signal from roi button in toolbar. Simply calls onROIselection from current widget
        """

        self.centerwidget.widget(self.currentIndex()).onROIselection()

    def manualCenter(self, value):
        """
        Slot to receive signal from center detection button in toolbar. Simply calls onManualCenter(value) from current
        widgetpipe
        """

        if self.ui.toolbar.actionMBIR.isChecked():
            self.ui.toolbar.actionMBIR.setChecked(False)

        self.ui.toolbar.actionCenter.setChecked(value)
        self.centerwidget.widget(self.currentIndex()).onManualCenter(value)

    def mbir(self, value):

        if self.ui.toolbar.actionCenter.isChecked():
            self.ui.toolbar.actionCenter.setChecked(False)

        self.ui.toolbar.actionMBIR.setChecked(value)
        self.centerwidget.widget(self.currentIndex()).onMBIR(value)



            # if self.checkPipeline():
        #     msg.showMessage('Computing MBIR preview...', timeout=0)


    def checkPipeline(self):
        """
        Checks the current workflow pipeline to ensure a reconstruction function is included. More checks should
        eventually be added here to ensure the wp makes sense.
        """

        if len(self.manager.features) < 1 or self.centerwidget.widget(self.currentIndex()) == -1:
            return False
        elif 'Reconstruction' not in [func.func_name for func in self.manager.features]:
            QtGui.QMessageBox.warning(None, 'Reconstruction method required',
                                      'You have to select a reconstruction method to run a preview')
            return False
        return True

    def slicePreviewAction(self, message='Computing slice preview...', fixed_func=None, prange=None):
        """
        Called when a reconstruction preview is requested either by the toolbar button or by the test parameter range
        from a parameter.

        Parameters
        ----------
        message : str, optional
            Message to log. Test Parameters log a different message than the default
        fixed_func : type class
            A dynamic class with only the necessary attributes to be run in a workflow pipeline. This is used for
            parameter range tests to create the class with the parameter to be run and send it to a background thread.
            See FunctionManager.testParameterRange for more details
        """
        if self.checkPipeline():
            msg.showMessage(message, timeout=0)
            prev_slice = self.centerwidget.currentWidget().sinogramViewer.currentIndex
            dims = self.get_reader_dims(sino=(prev_slice, prev_slice + 1, 1))
            dims += (0,)
            self.preview_slices = self.centerwidget.widget(self.currentIndex()).sinogramViewer.currentIndex
            self.processFunctionStack(callback=lambda x: self.runSlicePreview(*x), dims=dims,
                                      fixed_func=fixed_func, prange=prange)

    def multiSlicePreviewAction(self, message='Computing multi-slice preview...', fixed_func=None):

        slice_no = self.centerwidget.widget(self.currentIndex()).sinogramViewer.currentIndex
        maximum = self.centerwidget.widget(self.currentIndex()).sinogramViewer.data.shape[0]-1
        dialog = sliceDialog(parent=self.centerwidget, val1=slice_no, val2=slice_no+20,maximum=maximum)
        try:
            value = dialog.value

            if value is None:
                pass
            elif type(value) == str:
                msg.showMessage(value,timeout=0)
            else:
                if self.checkPipeline():
                    msg.showMessage(message, timeout=0)
                    if value[0] == value[1]:
                        dims = self.get_reader_dims(sino = (value[1], value[1] + 1, 1))
                        dims += (0,)
                        self.preview_slices = value[1]
                        self.centerwidget.widget(self.currentIndex()).sinogramViewer.setIndex(self.preview_slices)
                        self.processFunctionStack(callback=lambda x: self.runSlicePreview(*x), dims=dims,
                                                  fixed_func=fixed_func)
                    else:
                        dims = self.get_reader_dims(sino=(value[0], value[1] + 1, 1))
                        dims += (0,)
                        self.preview_slices = [value[0],value[1]]
                        self.processFunctionStack(callback=lambda x: self.runSlicePreview(*x), dims=dims,
                                                  fixed_func=fixed_func)
        except AttributeError:
            pass

    def preview3DAction(self):
        """
        Called when a reconstruction 3D preview is requested either by the toolbar button.
        The process is almost equivalent to running a slice preview except a different slice object is passed to extract
        a subsampled array from the raw tomographic array
        """

        if self.checkPipeline():

            window = QtGui.QInputDialog(self.centerwidget)
            window.setIntValue(8)
            window.setWindowTitle('3D Preview subsample factor')
            window.setLabelText('Choose a subsample factor for the 3D preview: ')
            window.exec_()
            val = window.intValue()

            msg.showMessage('Computing 3D preview...', timeout=0)
            dims = self.get_reader_dims(sino = (None, None, val), width=(None, None, val))
            dims += (0,)
            self.processFunctionStack(callback=lambda x: self.run3DPreview(*x), dims=dims)


    def runSlicePreview(self, datawidget, func_dict, theta, center, stack_dict, prange=None, dims=None):
        """
        Callback function that receives the partial stack and corresponding dictionary required to run a preview and
        add it to the viewer.TomoViewer.previewViewer

        Parameters
        ----------
        partial_stack : list of functools.partial
            List of partials that require only the input array to run.
        stack_dict : dict
            Dictionary describing the workflow pipeline being run. This is displayed to the left of the preview image in
            the viewer.TomoViewer.previewViewer
        """

        callback = partial(self.centerwidget.currentWidget().addSlicePreview, stack_dict, slice_no=self.preview_slices,
                           prange=prange)
        err_message = 'Unable to compute slice preview. Check log for details.'

        except_slot = lambda: msg.showMessage(err_message)
        bg_fold = threads.iterator(callback_slot=callback, finished_slot=msg.clearMessage, lock=threads.mutex,
                                 except_slot=except_slot)
        bg_fold(self.manager.reconGenerator)(datawidget, func_dict, theta, center, None, None, dims, None, 'Slice')


    def run3DPreview(self, datawidget, func_dict, theta, center, stack_dict, prange=None, dims=None):
        """
        Callback function that receives the partial stack and corresponding dictionary required to run a preview and
        add it to the viewer.TomoViewer.preview3DViewer

        Parameters
        ----------
        partial_stack : list of functools.partial
            List of partials that require only the input array to run.
        stack_dict : dict
            Dictionary describing the workflow pipeline being run. This is displayed to the left of the preview image in
            the viewer.TomoViewer.previewViewer
        data_dict: dict
            Dictionary of data to be reconstructed
        prange: list
            list of values to be iterated over in reconstruction preview. Not used in 3D previews
        """

        self.manager.updateParameters()
        callback = partial(self.centerwidget.widget(self.currentIndex()).add3DPreview, stack_dict)
        err_message = 'Unable to compute 3D preview. Check log for details.'
        except_slot = lambda: msg.showMessage(err_message)
        bg_fold = threads.iterator(callback_slot=callback, finished_slot=msg.clearMessage, lock=threads.mutex,
                                   except_slot=except_slot)
        bg_fold(self.manager.reconGenerator)(datawidget, func_dict, theta, center, None, None, dims, None, '3D')


    def processFunctionStack(self, callback, finished=None, dims=None, fixed_func=None, prange=None):
        """
        Runs the FunctionManager's loadPreviewData on a background thread to create the partial function stack and
        corresponding dictionary for running slice previews and 3D previews.

        Parameters
        ----------
        callback : function
            function to be called with the return values of manager.loadPreviewData: partial_stack, stack_dict
            This function is either self.run3DPreview or self.runSlicePreview
        finished : function/QtCore.Slot, optional
            Slot to receive the background threads finished signal
        slc : slice
            slice object specifying the slices to take from the input tomographic array
        fixed_func : type class
            A dynamic class with only the necessary attributes to be run in a workflow pipeline. This is used for
            parameter range tests to create the class with the parameter to be run and send it to a background thread.
            See FunctionManager.testParameterRange for more details
        """
        bg_functionstack = threads.method(callback_slot=callback, finished_slot=finished,
                                          lock=threads.mutex)(self.manager.loadPreviewData)
        bg_functionstack(self.centerwidget.widget(self.currentIndex()), dims=dims,
                         ncore=cpu_count(), fixed_func=fixed_func, prange=prange)

    def runFullReconstruction(self):
        """
        Sets up a full reconstruction to be run in a background thread for the current dataset based on the current
        workflow pipeline and configuration parameters. Called when the corresponding toolbar button is clicked.

        Deprecated by loadFullReconstruction and runReconstruction
        """
        if not self.checkPipeline():
            return

        value = QtGui.QMessageBox.question(None, 'Run Full Reconstruction',
                                           'You are about to run a full reconstruction. '
                                           'This step can take some minutes. Do you want to continue?',
                                   (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
        if value is QtGui.QMessageBox.Cancel:
            return

        name = self.centerwidget.tabText(self.centerwidget.currentIndex())
        msg.showMessage('Computing reconstruction for {}...'.format(name),timeout = 0)
        self.bottom.local_console.clear()
        self.manager.updateParameters()
        recon_iter = threads.iterator(callback_slot=self.bottom.log2local,
                                    interrupt_signal=self.bottom.local_cancelButton.clicked,
                                    finished_slot=self.reconstructionFinished)(self.manager.functionStackGenerator)

    def freeRecon(self):
        """
        Frees plugin to run reconstruction and run next in queue when job is canceled
        """
        msg.showMessage("Reconstruction interrupted.", timeout=0)
        self.bottom.log2local('---------- RECONSTRUCTION INTERRUPTED ----------')
        self.queue_widget.removeRecon(0)
        self.recon_running = False
        self.runReconstruction()


    def loadFullReconstruction(self):
        """
        Sets up a full reconstruction for the current dataset based on the current workflow pipeline and configuration
        parameters. Does not run reconstruction if there is one already running. Called when the corresponding toolbar
        button is clicked.
        """
        if not self.checkPipeline():
            return

        value = QtGui.QMessageBox.question(None, 'Run Full Reconstruction',
                                           'You are about to run a full reconstruction. '
                                           'This step can take some minutes. Do you want to continue?',
                                   (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
        if value is QtGui.QMessageBox.Cancel:
            return

        currentWidget = self.centerwidget.widget(self.currentIndex())
        self.manager.updateParameters()

        func_dict, theta, center, pipeline_dict, run_dict = self.manager.saveState(currentWidget)
        recon_iter = threads.iterator(callback_slot=self.bottom.log2local, except_slot=self.reconstructionFinished,
                            interrupt_signal=self.bottom.local_cancelButton.clicked,
                            finished_slot=self.reconstructionFinished)(self.manager.reconGenerator)

        proj, sino, width, proj_chunk, sino_chunk = self.get_reader_dims()

        dims = (proj, sino, width, sino_chunk, proj_chunk, cpu_count())
        args = (currentWidget, func_dict, theta, center, pipeline_dict, run_dict, dims)

        self.queue_widget.recon_queue.append([recon_iter, args])
        self.queue_widget.addRecon(args)

        if self.recon_running:
            name = self.centerwidget.tabText(self.centerwidget.currentIndex())
            msg.showMessage('Queued reconstruction for {}.'.format(name), timeout=0)

        self.runReconstruction()

    def runReconstruction(self):
        """
        Takes reconstruction job from self.queue_widget.recon_queue and runs it on background thread. Saves function
        pipeline as python runnable after reconstruction is finished.
        """
        if (not len(self.queue_widget.recon_queue)==0) and (not self.recon_running):
            self.recon_running = True
            self.queue_widget.manager.features[0].closeButton.hide()
            recon_job = self.queue_widget.recon_queue.popleft()
            args = recon_job[1]
            name = self.centerwidget.tabText(self.centerwidget.indexOf(args[0]))
            msg.showMessage('Computing reconstruction for {}...'.format(name), timeout=0)

            recon_job[0](*args)


    @QtCore.Slot()
    def reconstructionFinished(self):
        """
        Slot to revieve the reconstruction background threads finished signal. Runs another reconstruction is there is
        one on the queue
        """

        msg.showMessage('Reconstruction complete.', timeout=10)

        self.queue_widget.removeRecon(0)
        if len(self.queue_widget.recon_queue) > 0:
            self.bottom.log2local('------- Beginning next reconstruction -------')
        self.recon_running = False
        self.runReconstruction()


    def get_reader_dims(self, proj=None, sino=None, width=None):
        """
        Get dimensions for data to be reconstructed based on ReaderWidget (or other) input
        Proj, sino, width are all three-tuples in the form some combination of int and None values, ex:
        (int, int, int), (None, None, int), etc.
        """

        sino_chunk = None
        proj_chunk = None
        reader = None
        for f in self.manager.features:
            if 'Reader' in f.name:
                reader = [f.projections, f.width, f.sinograms]
                if not proj: proj = reader[0]
                if not sino: sino = reader[2]
                if not width: width = reader[1]
                sino_chunk = f.sino_chunk
                proj_chunk = f.proj_chunk

        dims = [proj, width, sino]
        for i, dim in enumerate(dims):
            if not dim:
                dim = (0, self.centerwidget.currentWidget().data.shape[i])
            else:
                if not dim[0]: dim = (0 if not reader else reader[i][0], dim[1], dim[2])
                if not dim[1]: dim = (dim[0], self.centerwidget.currentWidget().data.shape[i] if not reader
                                      else reader[i][1], dim[2])
                if not dim[2]: dim = (dim[0], dim[1], 1 if not reader else reader[i][2])
            dims[i] = dim
        proj, width, sino = dims

        if not sino_chunk:
            sino_chunk = cpu_count()*5
        if not proj_chunk:
            proj_chunk = cpu_count()*5

        return proj, sino, width, proj_chunk, sino_chunk





