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
from functools import partial
from PySide import QtGui, QtCore
from collections import OrderedDict
from modpkgs import yamlmod
from xicam.plugins import base
from pipeline import msg
from xicam import threads
from viewers import TomoViewer
import ui
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
        self.rightwidget = None
        self.bottomwidget = self.ui.bottomwidget
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
        self.bottomwidget.local_cancelButton.clicked.connect(self.freeRecon)
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

        # # create file name to pass to manager (?)
        # file_name = paths.split("/")[-1]
        # self.working_dir = paths.split(file_name)[0]


        widget = TomoViewer(paths=paths)
        widget.sigSetDefaults.connect(self.manager.setPipelineFromDict)
        widget.wireupCenterSelection(self.manager.recon_function)
        self.centerwidget.addTab(widget, os.path.basename(paths))
        self.centerwidget.setCurrentWidget(widget)

    # def currentWidget(self):
    #     """
    #     Return the current widget (viewer.TomoViewer) from the centerwidgets tabs
    #     """
    #
    #     try:
    #         return self.centerwidget.currentWidget()
    #     except AttributeError:
    #         return None



    def currentWidget(self):

        try:
            return self.centerwidget.currentIndex()
        except AttributeError:
            raise

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


    def freeRecon(self):
        """
        Frees plugin to run reconstruction and run next in queue when job is canceled
        """
        self.recon_running = False
        self.runReconstruction()

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

        widget =  self.centerwidget.widget(self.currentWidget())
        if widget is not None:
            self.ui.property_table.setData(widget.data.header.items())
            self.ui.property_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
            self.ui.property_table.show()
            self.ui.setConfigParams(widget.data.shape[0], widget.data.shape[2])
            config.set_als832_defaults(widget.data.header, funcwidget_list=self.manager.features,
                    path = widget.path)
            recon_funcs = [func for func in self.manager.features if func.func_name == 'Reconstruction']
            for rfunc in recon_funcs:
                rfunc.params.child('center').setValue(widget.data.shape[1]/2)
                rfunc.input_functions['theta'].params.child('nang').setValue(widget.data.shape[0])


    def loadPipelineDictionary(self):
        """
        Loads a pipeline dictionary containing information relevant to reconstruction, including parameters and
        arguments that are held by FunctionWidgets. This information can be updated on the  widgets in the middle of a
        run, so the reconstruction should refer to this dictionary for relevant parameters
        """

        currentWidget = self.centerwidget.widget(self.currentWidget())

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

        self.ui.setConfigParams(0, 0)
        self.ui.property_table.clear()
        self.ui.property_table.hide()
        self.centerwidget.widget(index).deleteLater()

    def roiSelection(self):
        """
        Slot to receive signal from roi button in toolbar. Simply calls onROIselection from current widget
        """

        self.centerwidget.widget(self.currentWidget()).onROIselection()

    def manualCenter(self, value):
        """
        Slot to receive signal from center detection button in toolbar. Simply calls onManualCenter(value) from current
        widgetpipe
        """

        self.centerwidget.widget(self.currentWidget()).onManualCenter(value)

    def checkPipeline(self):
        """
        Checks the current workflow pipeline to ensure a reconstruction function is included. More checks should
        eventually be added here to ensure the wp makes sense.
        """

        if len(self.manager.features) < 1 or self.centerwidget.widget(self.currentWidget()) == -1:
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

    def runSlicePreview(self, partial_stack, stack_dict, data_dict):
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

        initializer = self.centerwidget.widget(self.currentWidget()).getsino()
        slice_no = self.centerwidget.widget(self.currentWidget()).sinogramViewer.currentIndex
        callback = partial(self.centerwidget.widget(self.currentWidget()).addSlicePreview, stack_dict, slice_no=slice_no)
        message = 'Unable to compute slice preview. Check log for details.'
        self.foldPreviewStack(partial_stack, initializer, data_dict, callback, message)

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

    def run3DPreview(self, partial_stack, stack_dict, data_dict):
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

        # this step takes quite a bit, think of running a thread
        initializer = self.centerwidget.widget(self.currentWidget()).getsino(slc)
        self.manager.updateParameters()
        callback = partial(self.centerwidget.widget(self.currentWidget()).add3DPreview, stack_dict)
        err_message = 'Unable to compute 3D preview. Check log for details.'
        # self.foldPreviewStack(partial_stack, initializer, callback, err_message)
        self.foldPreviewStack(partial_stack, initializer, data_dict, callback, err_message)

    def processFunctionStack(self, callback, finished=None, slc=None, fixed_func=None):
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
        bg_functionstack(self.centerwidget.widget(self.currentWidget()), slc=slc,
                         ncore=self.ui.config_params.child('CPU Cores').value(), fixed_func=fixed_func)

    def foldPreviewStack(self, partial_stack, initializer, data_dict, callback, error_message):
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
        # bg_fold(self.manager.foldFunctionStack)(partial_stack, initializer)
        bg_fold(self.manager.foldSliceStack)(partial_stack, data_dict)

    def runFullReconstruction(self):
        """
        Sets up a full reconstruction to be run in a background thread for the current dataset based on the current
        workflow pipeline and configuration parameters. Called when the corresponding toolbar button is clicked.

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

        recon_iter(datawidget = self.currentWidget(), proj = (pstart, pend, pstep), sino = (sstart, send, sstep),
                   sino_p_chunk = self.ui.config_params.child('Sinograms/Chunk').value(),
                   ncore=self.ui.config_params.child('CPU Cores').value())





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

        currentWidget = self.centerwidget.widget(self.currentWidget())
        self.manager.updateParameters()

        run_state = self.manager.saveState(currentWidget)



        recon_iter = threads.iterator(callback_slot=self.bottomwidget.log2local,
                            interrupt_signal=self.bottomwidget.local_cancelButton.clicked,
                            finished_slot=self.reconstructionFinished)(self.manager.reconGenerator)
        pstart = self.ui.config_params.child('Start Projection').value()
        pend = self.ui.config_params.child('End Projection').value()
        pstep = self.ui.config_params.child('Step Projection').value()
        sstart = self.ui.config_params.child('Start Sinogram').value()
        send = self.ui.config_params.child('End Sinogram').value()
        sstep =  self.ui.config_params.child('Step Sinogram').value()

        args = (currentWidget, run_state,
                (pstart, pend, pstep),(sstart, send, sstep), self.ui.config_params.child('Sinograms/Chunk').value(),
                self.ui.config_params.child('CPU Cores').value())

        self.manager.recon_queue.put([recon_iter, args])

        if self.recon_running:
            name = self.centerwidget.tabText(self.centerwidget.currentIndex())
            msg.showMessage('Queued reconstruction for {}.'.format(name), timeout=0)

        self.runReconstruction()

    def runReconstruction(self):
        """
        Takes reconstruction job from self.manager.recon_queue and runs it on background thread. Saves function
        pipeline as python runnable after reconstruction is finished.
        """
        if (not self.manager.recon_queue.empty()) and (not self.recon_running):
            self.recon_running = True
            recon_job = self.manager.recon_queue.get()
            args = recon_job[1]
            name = self.centerwidget.tabText(self.centerwidget.indexOf(args[0]))
            msg.showMessage('Computing reconstruction for {}...'.format(name), timeout=0)

            recon_job[0](*args)

            run_state = args[1]
            proj = args[2]; sino = args[3]
            sino_p_chunk = args[4]; ncore = args[5]
            path = args[0].path






    @QtCore.Slot()
    def reconstructionFinished(self):
        """
        Slot to revieve the reconstruction background threads finished signal. Runs another reconstruction is there is
        one on the queue
        """

        msg.showMessage('Reconstruction complete.', timeout=10)

        self.recon_running = False
        self.runReconstruction()



