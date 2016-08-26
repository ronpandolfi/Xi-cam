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
from PySide import QtGui, QtCore
from modpkgs import yamlmod
from xicam.plugins import base
from pipeline import msg
from xicam import threads
from viewers import TomoViewer
import ui
import Queue
import config
from functionwidgets import FunctionManager

# YAML file specifying the default workflow pipeline
DEFAULT_PIPELINE_YAML = 'yaml/tomography/default_pipeline.yml'


class plugin(base.plugin):
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
        self.rightmodes = self.ui.rightmodes
        self.bottomwidget = self.ui.bottomwidget
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        # Keep a timer for reconstructions
        self.recon_start_time = 0

        # Queue for reconstruction
        # for recon on multiple nodes, implement the threads.Worker class
        self.recon_queue = Queue.Queue()
        self.recon_running = False

        # Setup FunctionManager
        self.manager = FunctionManager(self.ui.functionwidget.functionsList, self.ui.param_form,
                                       blank_form='Select a function from\n below to set parameters...')
        self.manager.setPipelineFromYAML(config.load_pipeline(DEFAULT_PIPELINE_YAML))
        self.manager.sigPipelineChanged.connect(self.reconnectTabs)

        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent

        # Connect toolbar signals and ui button signals
        self.toolbar.connectTriggers(self.slicePreviewAction, self.preview3DAction, self.loadFullReconstruction,
                                     self.manualCenter, self.roiSelection)
        self.ui.connectTriggers(self.loadPipeline, self.savePipeline, self.resetPipeline,
                        lambda: self.manager.swapFeatures(self.manager.selectedFeature, self.manager.previousFeature),
                        lambda: self.manager.swapFeatures(self.manager.selectedFeature, self.manager.nextFeature),
                                self.clearPipeline)
        self.manager.sigTestRange.connect(self.slicePreviewAction)
        ui.build_function_menu(self.ui.addfunctionmenu, config.funcs['Functions'],
                               config.names, self.manager.addFunction)
        super(plugin, self).__init__(placeholders, *args, **kwargs)

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
        if type(paths) is list:
            paths = paths[0]

        # create file name to pass to manager (?)
        file_name = paths.split("/")[-1]
        body = paths.split(file_name)[0]
        self.path = body + "RECON_" + file_name.split(".")[0] + "/RECON_" + file_name.split(".")[0]

        widget = TomoViewer(paths=paths)
        widget.sigSetDefaults.connect(self.manager.setPipelineFromDict)
        widget.wireupCenterSelection(self.manager.recon_function)
        self.centerwidget.addTab(widget, os.path.basename(paths))
        self.centerwidget.setCurrentWidget(widget)

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

    def setPipelineValues(self):
        """
        Sets up the metadata table and default values in configuration parameters and functions based on the selected
        dataset
        """

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
        """
        Slot to receive signal when a tab is closed. Simply resets configuration parameters and clears metadata table

        Parameters
        ----------
        index : int
            Index of tab that is being closed.
        """

        self.ui.setConfigParams(0, 0)
        self.ui.property_table.clear()
        self.ui.property_table.hide()
        self.centerwidget.widget(index).deleteLater()

    def roiSelection(self):
        """
        Slot to receive signal from roi button in toolbar. Simply calls onROIselection from current widget
        """

        self.currentWidget().onROIselection()

    def manualCenter(self, value):
        """
        Slot to receive signal from center detection button in toolbar. Simply calls onManualCenter(value) from current
        widget

        Parameters
        ----------
        index : bool
            Boolean for toggling center detection on/off
        """

        self.currentWidget().onManualCenter(value)

    def checkPipeline(self):
        """
        Checks the current workflow pipeline to ensure a reconstruction function is included. More checks should
        eventually be added here to ensure the wp makes sense.
        """

        if len(self.manager.features) < 1 or self.currentWidget() is None:
            return False
        elif 'Reconstruction' not in [func.func_name for func in self.manager.features]:
            QtGui.QMessageBox.warning(None, 'Reconstruction method required',
                                      'You have to select a reconstruction method to run a preview')
            return False
        return True

    def slicePreviewAction(self, message='Computing slice preview...', fixed_func=None):
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
            self.processFunctionStack(callback=lambda x: self.runSlicePreview(*x), fixed_func=fixed_func)

    def runSlicePreview(self, partial_stack, stack_dict):
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

        initializer = self.currentWidget().getsino()
        slice_no = self.currentWidget().sinogramViewer.currentIndex
        callback = partial(self.currentWidget().addSlicePreview, stack_dict, slice_no=slice_no)
        message = 'Unable to compute slice preview. Check log for details.'
        self.foldPreviewStack(partial_stack, initializer, callback, message)

    def preview3DAction(self):
        """
        Called when a reconstruction 3D preview is requested either by the toolbar button.
        The process is almost equivalent to running a slice preview except a different slice object is passed to extract
        a subsampled array from the raw tomographic array
        """

        if self.checkPipeline():
            msg.showMessage('Computing 3D preview...', timeout=0)
            slc = (slice(None), slice(None, None, 8), slice(None, None, 8))
            self.manager.cor_scale = lambda x: x // 8
            self.processFunctionStack(callback=lambda x: self.run3DPreview(*x), slc=slc)

    def run3DPreview(self, partial_stack, stack_dict):
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
        """

        slc = (slice(None), slice(None, None, 8), slice(None, None, 8))
        initializer = self.currentWidget().getsino(slc)  # this step takes quite a bit, think of running a thread
        self.manager.updateParameters()
        callback = partial(self.currentWidget().add3DPreview, stack_dict)
        err_message = 'Unable to compute 3D preview. Check log for details.'
        self.foldPreviewStack(partial_stack, initializer, callback, err_message)

    def processFunctionStack(self, callback, finished=None, slc=None, fixed_func=None):
        """
        Runs the FunctionManager's previewFunctionStack on a background thread to create the partial function stack and
        corresponding dictionary for running slice previews and 3D previews.

        Parameters
        ----------
        callback : function
            function to be called with the return values of manager.previewFunctionStack: partial_stack, stack_dict
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
                                          lock=threads.mutex)(self.manager.previewFunctionStack)
        bg_functionstack(self.currentWidget(), slc=slc, ncore=self.ui.config_params.child('CPU Cores').value(),
                         fixed_func=fixed_func)

    def foldPreviewStack(self, partial_stack, initializer, callback, error_message):
        """
        Calls the managers foldFunctionStack on a background thread. This is what tells the manager to compute a
        slice preview or a 3D preview from a specified workflow pipeline

        Parameters
        ----------
        partial_stack : list of functools.partial
            List of partials that require only the input array to run.
        initializer : ndarray
            Array to use as initializer for folding operation
        callback : function
            function to be called with the return value of the fold (ie the resulting reconstruction).
            This is the current TomoViewers addSlicePreview or add3DPreview methods
        error_message : str
            Message to log/display if the fold process raises an exception
        """

        except_slot = lambda: msg.showMessage(error_message)
        bg_fold = threads.method(callback_slot=callback, finished_slot=msg.clearMessage, lock=threads.mutex,
                                 except_slot=except_slot)
        bg_fold(self.manager.foldFunctionStack)(partial_stack, initializer)

    def loadFullReconstruction(self):
        """
        Sets up a full reconstruction to be run in a background thread for the current dataset based on the current
        workflow pipeline and configuration parameters. Called when the corresponding toolbar button is clicked.
        """
        if not self.checkPipeline():
            return

        value = QtGui.QMessageBox.question(None, 'Run Full Reconstruction',
                                           'You are about to run a full reconstruction.'
                                           'This step can take some minutes. Do you want to continue?',
                                   (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
        if value is QtGui.QMessageBox.Cancel:
            return

        name = self.centerwidget.tabText(self.centerwidget.currentIndex())
        msg.showMessage('Computing reconstruction for {}...'.format(name), timeout=0)
        self.bottomwidget.local_console.clear()
        self.manager.updateParameters()

        # self.manager.getFunctionPipeline or something

        recon_iter = threads.iterator(callback_slot=self.bottomwidget.log2local,
                                      interrupt_signal=self.bottomwidget.local_cancelButton.clicked,
                                      finished_slot=self.reconstructionFinished)(self.manager.functionStackGenerator)
        pstart = self.ui.config_params.child('Start Projection').value()
        pend = self.ui.config_params.child('End Projection').value()
        pstep = self.ui.config_params.child('Step Projection').value()
        sstart = self.ui.config_params.child('Start Sinogram').value()
        send = self.ui.config_params.child('End Sinogram').value()
        sstep =  self.ui.config_params.child('Step Sinogram').value()

        # recon_iter(datawidget = self.currentWidget(), proj = (pstart, pend, pstep), sino = (sstart, send, sstep),
        #            sino_p_chunk = self.ui.config_params.child('Sinograms/Chunk').value(),
        #            ncore=self.ui.config_params.child('CPU Cores').value())

        args = (self.currentWidget(), (pstart, pend, pstep),(sstart, send, sstep),
                self.ui.config_params.child('Sinograms/Chunk').value(), self.currentWidget().path,
                self.ui.config_params.child('CPU Cores').value())


        # recon_iter = threads.RunnableIterator(iterator = self.manager.functionStackGenerator,iterator_args = args,
        #                                 finished_slot=self.reconstructionFinished,
        #                                 callback_slot=self.bottomwidget.log2local,
        #                                 interrupt_signal = self.bottomwidget.local_cancelButton.clicked)

        self.recon_queue.put([recon_iter, args])
        self.runReconstruction()

    def runReconstruction(self):
        if not self.recon_queue.empty() and not self.recon_running:
            self.recon_running = True
            recon_job = self.recon_queue.get()
            recon_job[0](*recon_job[1])




    @QtCore.Slot()
    def reconstructionFinished(self):
        """
        Slot to revieve the reconstruction background threads finished signal. The total time the reconstruction took is
        displayed in the console.
        """

        # save function pipeline as yml when reconstruction is run

        # for feature in self.manager.features:
        #     print feature.func_name, ",", feature.subfunc_name,","
        #     for p in feature.params.children():
        #         print p.name(), ",",p.value()
        #     print "-----------------"


        save_file = self.currentWidget().path + ".yml"
        with open(save_file, 'w') as yml:
            pipeline = config.extract_pipeline_dict(self.manager.features)
            yamlmod.ordered_dump(pipeline, yml)

        run_time = time.time() - self.recon_start_time
        self.bottomwidget.log2local('Reconstruction complete. Run time: {:.2f} s'.format(run_time))
        msg.showMessage('Reconstruction complete.', timeout=10)

        self.recon_running = False
        self.runReconstruction()



