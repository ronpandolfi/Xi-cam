from PySide import QtGui, QtCore
from pyqtgraph.parametertree import ParameterTree, Parameter
from functools import partial
from PySide.QtUiTools import QUiLoader
from psutil import cpu_count
import pyqtgraph as pg
import os
from modpkgs import yamlmod
from collections import OrderedDict
from pipeline import msg
import inspect
import importlib
import time
from copy import deepcopy

class workflowEditorWidget(QtGui.QSplitter):
    # override this to set your default workflow

    sigExecute = QtCore.Signal()



    def __init__(self, DEFAULT_PIPELINE_YAML, module):  # TODO: make modules a list

        self.DEFAULT_PIPELINE_YAML = DEFAULT_PIPELINE_YAML
        self.manifest = yamlmod.ordered_load(module.functionManifest)

        super(workflowEditorWidget, self).__init__()
        self.setOrientation(QtCore.Qt.Vertical)

        self.functionwidget = QUiLoader().load('/home/rp/PycharmProjects/xicam/gui/tomographyleft.ui')
        self.functionwidget.functionsList.setAlignment(QtCore.Qt.AlignBottom)
        self.functionwidget.addFunctionButton.setToolTip('Add function to pipeline')
        self.functionwidget.clearButton.setToolTip('Clear pipeline')
        self.functionwidget.fileButton.setToolTip('Save/Load pipeline')
        self.functionwidget.moveDownButton.setToolTip('Move selected function down')
        self.functionwidget.moveUpButton.setToolTip('Move selected function up')

        self.addfunctionmenu = QtGui.QMenu()
        self.functionwidget.addFunctionButton.setMenu(self.addfunctionmenu)
        self.functionwidget.addFunctionButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.functionwidget.addFunctionButton.setArrowType(QtCore.Qt.NoArrow)

        filefuncmenu = QtGui.QMenu()
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_55.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.openaction = QtGui.QAction(icon, 'Open', filefuncmenu)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_59.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.saveaction = QtGui.QAction(icon, 'Save', filefuncmenu)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_56.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.refreshaction = QtGui.QAction(icon, 'Reset', filefuncmenu)
        filefuncmenu.addActions([self.openaction, self.saveaction, self.refreshaction])

        self.functionwidget.fileButton.setMenu(filefuncmenu)
        self.functionwidget.fileButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.functionwidget.fileButton.setArrowType(QtCore.Qt.NoArrow)

        paramtree = ParameterTree()
        self.nodeEditor = QtGui.QStackedWidget()
        self.nodeEditor.addWidget(paramtree)
        self.addWidget(self.nodeEditor)
        self.addWidget(self.functionwidget)

        #
        self.manager = FunctionManager(self.functionwidget.functionsList, self.nodeEditor,
                                       blank_form='Select a function from\n below to set parameters...')
        self.manager.setPipelineFromYAML(load_pipeline(self.DEFAULT_PIPELINE_YAML))

        self.build_function_menu()

        # self.openaction.triggered.connect(open)
        # self.saveaction.triggered.connect(save)
        # self.refreshaction.triggered.connect(reset)
        # self.functionwidget.moveDownButton.clicked.connect(moveup)
        # self.functionwidget.moveUpButton.clicked.connect(movedown)
        # self.functionwidget.clearButton.clicked.connect(clear)
        self.functionwidget.runWorkflowButton.clicked.connect(self.sigExecute)

    def build_function_menu(
            self):  # , menu, functree, functiondata, actionslot):#, self.addfunctionmenu, self.manager.addFunction):
        """
        Builds the function menu's and submenu's anc connects them to the corresponding slot to add them to the workflow
        pipeline

        Parameters
        ----------
        menu : QtGui.QMenu
            Menu object to populate with submenu's and actions
        functree : dict
            Dictionary specifying the depth levels of functions. See functions.yml entry "Functions"
        functiondata : dict
            Dictionary with function information. See function_names.yml
        actionslot : QtCore.Slot
            slot where the function action triggered signal shoud be connected
        """

        for group, items in self.manifest.iteritems():
            funcmenu = QtGui.QMenu(group)
            self.addfunctionmenu.addMenu(funcmenu)
            for item in items:

                if 'displayName' in item:  # The subgroup is a function, not a group!
                    # try:
                    self.buildFunction(group, item, funcmenu)
                    # except KeyError:
                    #    pass
                else:
                    funcsubmenu = QtGui.QMenu(group)
                    funcmenu.addMenu(funcsubmenu)
                    for group, function in item.iteritems():
                        # try:
                        self.buildFunction(group, function, funcsubmenu)
                        # except KeyError:
                        #    pass

    def buildFunction(self, group, function, menu):
        funcAction = QtGui.QAction(function['displayName'], self.addfunctionmenu)
        funcAction.triggered.connect(partial(self.manager.addFunction, function))
        menu.addAction(funcAction)

    def loadPipeline(self):
        """
        Load a workflow pipeline yaml file
        """

        open_file = QtGui.QFileDialog.getOpenFileName(None, 'Open tomography pipeline file',
                                                      os.path.expanduser('~'), selectedFilter='*.yml')[0]
        if open_file != '':
            self.manager.setPipelineFromYAML(load_pipeline(open_file))

    def savePipeline(self):
        """
        Save a workflow pipeline from UI as a yaml file
        """

        save_file = QtGui.QFileDialog.getSaveFileName(None, 'Save tomography pipeline file as',
                                                      os.path.expanduser('~'), selectedFilter='*.yml')[0]

        save_file = save_file.split('.')[0] + '.yml'
        with open(save_file, 'w') as yml:
            pipeline = extract_pipeline_dict(self.manager.features)
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
            self.manager.setPipelineFromYAML(load_pipeline(self.DEFAULT_PIPELINE_YAML))
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

    def checkPipeline(self):
        """
        Checks the current workflow pipeline to ensure a reconstruction function is included. More checks should
        eventually be added here to ensure the wp makes sense.
        """
        return True

    def runWorkflow(self, **workspace):
        from xicam import threads
        workflowIterator = threads.iterator(callback_slot=self.callback,
                                            interrupt_signal=None,
                                            finished_slot=self.finished)(self.manager.workflowExectutionGenerator)
        workspace['ncore'] = 1
        workflowIterator(**workspace)

    def callback(self, msg):
        if type(msg) is unicode:
            print msg
        elif type(msg) is dict:
            print 'Finished workspace:', msg

    def finished(self):
        print 'Finished!'


def load_pipeline(yaml_file):
    """
    Load a workflow pipeline from a yaml file
    """

    with open(yaml_file, 'r') as y:
        pipeline = yamlmod.ordered_load(y)
    if pipeline: return pipeline
    return {}


def save_function_pipeline(pipeline, file_name):
    """
    Save a workflow pipeline from dict

    Parameters
    ----------
    pipeline : dict
        dictionary specifying the workflow pipeline
    file_name : str
        file name to save as yml
    """

    if file_name != '':
        file_name = file_name.split('.')[0] + '.yml'
        with open(file_name, 'w') as y:
            yamlmod.ordered_dump(pipeline, y)


def extract_pipeline_dict(funwidget_list):
    """
    Extract a dictionary from a FunctionWidget list in the appropriate format to save as a yml file

    Parameters
    ----------
    funwidget_list : list of FunctionWidgets
        list of FunctionWidgets exposed in the UI workflow pipeline

    Returns
    -------
    dict
        dictionary specifying the workflow pipeline
    """

    d = OrderedDict()
    for f in funwidget_list:
        d[f.func_name] = {f.subfunc_name: {'Parameters': {p.name(): p.value() for p in f.params.children()}}}
        d[f.func_name][f.subfunc_name]['Enabled'] = f.enabled
        if f.func_name == 'Reconstruction':
            d[f.func_name][f.subfunc_name].update({'Package': f.packagename})
        for param, ipf in f.input_functions.iteritems():
            if 'Input Functions' not in d[f.func_name][f.subfunc_name]:
                d[f.func_name][f.subfunc_name]['Input Functions'] = {}
            id = {
                ipf.func_name: {ipf.subfunc_name: {'Parameters': {p.name(): p.value() for p in ipf.params.children()}}}}
            d[f.func_name][f.subfunc_name]['Input Functions'][param] = id
    return d


class FeatureWidget(QtGui.QWidget):
    """
    Widget that stands for a feature or function with a a preview icon and a delete icon. These are added to a layout
    to represent a set of features (as in higgisaxs) or functions (as in tomography). FeatureWidgets are normally
    managed by an instance of the FeatureManager Class


    Attributes
    ----------
    name : str
        Name to be shown in the FeatureWidget's GUI
    form
        Widget that portrays information about the widget (ie a textlabel, spinbox, ParameterTree, etc)
    subfeatures : list of FeatureWidgets/QWidgets
        Subfeatures of the widget
    previewButton : QtGui.QPushButton
        Button to call the preview action associated with this widget
    closeButton : QtGui.QPushButton
        Button that emits sigDelete

    Signals
    -------
    sigClicked(FeatureWidget)
        Signal emitted when the widget is clicked, self is emitted so that a FeatureManager has access to the sender
        Qt has a sender method which could also be used instead...
    sigDelete(FeatureWidget)
        Signal emitted when the widget's closeButton is clicked, self is emitted
    sigSubFeature(FeatureWidget)
        Signal emitted when a subfeature is added to the current widget. Emits the subfeature.


    Parameters
    ----------
    name : str
        Name to be given to widget
    checkable : bool, optional
        Boolean specifying if the previewButton is checkable (for toggling)
    closeable : bool, optional
        Boolean specifying whether the widget can be closed/deleted
    subfeatures : list of FeatureWidgets/QWidgets
        Initialization list of subfeatures. New subfeatures can be added with the addSubFeatureMethod
    parent
        Parent widget, normally the manager
    """

    sigClicked = QtCore.Signal(QtGui.QWidget)
    sigDelete = QtCore.Signal(QtGui.QWidget)
    sigSubFeature = QtCore.Signal(QtGui.QWidget)

    def __init__(self, name='', checkable=True, closeable=True, subfeatures=None, parent=None):
        super(FeatureWidget, self).__init__(parent=parent)

        self.name = name
        self.form = QtGui.QLabel(self.name)  # default form
        self.subfeatures = []

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)

        self.frame = QtGui.QFrame(self)
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.frame)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.previewButton = QtGui.QPushButton(parent=self)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.previewButton.sizePolicy().hasHeightForWidth())
        self.previewButton.setSizePolicy(sizePolicy)
        self.previewButton.setStyleSheet("margin:0 0 0 0;")
        self.previewButton.setText("")
        icon = QtGui.QIcon()

        if checkable:
            icon.addPixmap(QtGui.QPixmap("gui/icons_48.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("gui/icons_47.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
            self.previewButton.setCheckable(True)
        else:
            icon.addPixmap(QtGui.QPixmap("gui/icons_47.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.previewButton.setCheckable(False)
            self.previewButton.setChecked(True)

        self.previewButton.setIcon(icon)
        self.previewButton.setFlat(True)
        self.previewButton.setChecked(True)
        self.horizontalLayout_2.addWidget(self.previewButton)
        self.line = QtGui.QFrame(self.frame)
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.horizontalLayout_2.addWidget(self.line)
        self.txtName = ROlineEdit(self.frame)
        self.horizontalLayout_2.addWidget(self.txtName)
        self.txtName.setText(name)
        self.line_3 = QtGui.QFrame(self.frame)
        self.line_3.setFrameShape(QtGui.QFrame.VLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.horizontalLayout_2.addWidget(self.line_3)
        self.closeButton = QtGui.QPushButton(self.frame)
        self.closeButton.setStyleSheet("margin:0 0 0 0;")
        self.closeButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("gui/icons_46.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.closeButton.setIcon(icon1)
        self.closeButton.setFlat(True)
        self.closeButton.clicked.connect(self.delete)
        self.horizontalLayout_2.addWidget(self.closeButton)
        self.verticalLayout.addWidget(self.frame)
        self.txtName.sigClicked.connect(self.mouseClicked)
        self.frame.setFrameShape(QtGui.QFrame.Box)
        self.frame.setCursor(QtCore.Qt.ArrowCursor)

        self.subframe = QtGui.QFrame(self)
        self.subframe.setFrameShape(QtGui.QFrame.StyledPanel)
        self.subframe.setFrameShadow(QtGui.QFrame.Raised)
        self.subframe_layout = QtGui.QVBoxLayout(self.subframe)  # QtGui.QGridLayout(self.subframe)
        self.subframe_layout.setContentsMargins(0, 0, 0, 0)
        self.subframe_layout.setSpacing(0)
        self.verticalLayout.addWidget(self.subframe)
        self.subframe.hide()

        if not closeable:
            self.closeButton.hide()

        if subfeatures is not None:
            for subfeature in subfeatures:
                self.addSubFeature(subfeature)

        self.collapse()

    def addSubFeature(self, subfeature):
        """
        Adds a subfeature to the widget

        Parameters
        ----------
        subfeature : FeatureWidget/QWidget
            Widget to add as a subfeature
        """

        h = QtGui.QHBoxLayout()
        indent = QtGui.QLabel('  -   ')
        h.addWidget(indent)
        subfeature.destroyed.connect(indent.deleteLater)
        subfeature.destroyed.connect(h.deleteLater)
        if isinstance(subfeature, QtGui.QLayout):
            h.addLayout(subfeature)
        elif isinstance(subfeature, QtGui.QWidget):
            h.addWidget(subfeature)
        self.subframe_layout.addLayout(h)
        try:
            subfeature.sigDelete.connect(self.removeSubFeature)
        except AttributeError:
            pass

        self.sigSubFeature.emit(subfeature)
        self.subfeatures.append(subfeature)

    def removeSubFeature(self, subfeature):
        """
        Removes a subfeature

        Parameters
        ----------
        subfeature : FeatureWidget/QWidget
            Feature to remove
        """

        self.subfeatures.remove(subfeature)
        subfeature.deleteLater()
        del subfeature

    def delete(self):
        """
        Emits delete signal with self. Connected to deleteButton's clicked
        """

        self.sigDelete.emit(self)

    def collapse(self):
        """
        Collapses all expanded subfeatures
        """

        if self.subframe is not None:
            self.subframe.hide()

    def expand(self):
        """
        Expands subfeatures
        """

        if self.subframe is not None:
            self.subframe.show()

    def mouseClicked(self):
        """
        Slot to handle when a feature is clicked
        """

        self.sigClicked.emit(self)
        self.setFocus()
        self.previewButton.setFocus()


class FunctionWidget(FeatureWidget):
    """
    Subclass of FeatureWidget that defines attributes to show parameters to a given function and run the function
    with the given parameters. These should be used with the corresponding FunctionManager to run Tomography pipeline
    workflows


    Attributes
    ----------
    func_name : str
        Function name
    subfunc_name : str
        Specific function name
    input_functions : dict
        dictionary with keys being parameters of this function to be overriden, and values being a FunctionWidget
        whose function will override said parameter
    exposedParameters : dict
        Dictionary with parameter names and values
    _function : function
        Function object corresponding to the function represented by widget
    params : pyqtgraph.Parameter
        Parameter instance with function parameter exposed in UI
    missing_args : list of str
        Names of missing arguments not contained in param_dict

    Signals
    -------
    sigTestRange(QtGui.QWidget, str, tuple)
        Emitted when parameter range test is requested. Emits the sending widget, a string with a message to log, and
        a tuple with the range values for the parameter


    Parameters
    ----------
    name : str
        generic name of function
    subname : str
        specific name of function under the generic name category
    package : python package
        package
    input_functions : dict, optional
        dictionary with keys being parameters of this function to be overriden, and values being a FunctionWidget
        whose function will override said parameter
    checkable : bool, optional
        bool to set the function to be toggled on and of when running constructed workflows
    closeable : bool, optional
        bool to set if the function can be deleted from the pipeline editor
    parent : QWidget
        parent of this FunctionWidget
    """

    INPUT = 'INPUT'
    PROCESS = 'PROCESS'
    OUTPUT = 'OUTPUT'
    VISUALIZE = 'VISUALIZE'

    # TODO perhaps its better to not pass in the package object but only a string, package object can be retrived from reconpkgs.packages dict
    def __init__(self, function, input_functions=None, checkable=True, closeable=True,
                 parent=None):
        self.name = function['displayName']
        self.subname = function['displayName']
        self.functionType = function['functionType']
        name = self.name
        subname = self.subname
        package = importlib.import_module('pipeline.workflowfunctions.' + function['moduleName'])
        funcname = function['functionName']
        if name != subname:
            self.name += ' (' + subname + ')'
        super(FunctionWidget, self).__init__(self.name, checkable=checkable, closeable=closeable, parent=parent)

        self.func_name = name
        self.subfunc_name = subname
        self.input_functions = {}
        if 'parameters' in function:
            params = function['parameters']
        else:
            params = []
        self._function = getattr(package, funcname)

        # TODO have the children kwarg be passed to __init__
        self.params = Parameter.create(name=self.name, children=params, type='group')  #

        self.form = ParameterTree(showHeader=False)
        self.form.setParameters(self.params, showTop=True)

        # Initialize parameter dictionary with keys and default values
        # self.updateParamsDict()
        # argspec = inspect.getargspec(self._function)
        # default_argnum = len(argspec[3])
        # self.param_dict.update({key: val for (key, val) in zip(argspec[0][-default_argnum:], argspec[3])})
        # for key, val in self.param_dict.iteritems():
        #     if key in [p.name() for p in self.params.children()]:
        #         self.params.child(key).setValue(val)
        #         self.params.child(key).setDefault(val)
        #
        # # Create a list of argument names (this will most generally be the data passed to the function)
        # self.missing_args = [i for i in argspec[0] if i not in self.param_dict.keys()]

        self.previewButton.customContextMenuRequested.connect(self.menuRequested)
        self.menu = QtGui.QMenu()

        if input_functions is not None:
            for param, ipf in input_functions.iteritems():
                self.addInputFunction(param, ipf)

        # wire up param changed signals
        for param in self.params.children():
            param.sigValueChanged.connect(self.paramChanged)

    @property
    def enabled(self):
        """
        Boolean showing if the function widget is enabled (eye open/closed)
        """
        if self.previewButton.isChecked() or not self.previewButton.isCheckable():
            return True
        return False

    @enabled.setter
    def enabled(self, val):
        """
        Set enabled value by toggling the previewButton only if the widget is checkable
        """
        if val and self.previewButton.isCheckable():
            self.previewButton.setChecked(True)
        else:
            self.previewButton.setChecked(False)

    @property
    def exposed_param_dict(self):
        """
        Parameter dictionary with only the parameters that are shown in GUI
        """
        param_dict = {param.name(): param.value() for param in self.params.childs}
        return param_dict

    @property
    def partial(self):
        """
        Package up all parameters into a functools.partial
        """
        return partial(self._function, *self.exposed_param_dict.values())

    @property
    def func_signature(self):
        """
        String for function signature. Hopefully this can eventually be used to save workflows as scripts :)
        """
        signature = str(self._function.__name__) + '('
        for arg in self.missing_args:
            signature += '{},'.format(arg)
        for param, value in self.exposedParameters.iteritems():
            signature += '{0}={1},'.format(param, value) if not isinstance(value, str) else \
                '{0}=\'{1}\','.format(param, value)
        return signature[:-1] + ')'

    def updateParamsDict(self):
        """
        Update the values of the parameter dictionary with the current values in UI
        """
        self.exposedParameters.update({param.name(): param.value() for param in self.params.children()})
        for p, ipf in self.input_functions.iteritems():
            ipf.updateParamsDict()

    def addInputFunction(self, parameter, functionwidget):
        """
        Add an input function widget

        Parameters
        ----------
        parameter : str
            Parameter name that will be overriden by return value of the input function
        functionwidget : FunctionWidget
            FunctionWidget representing the input function

        """

        if parameter in self.input_functions:  # Check to see if parameter already has input function
            if functionwidget.subfunc_name == self.input_functions[parameter].subfunc_name:
                raise AttributeError('Input function already exists')  # skip if the input function already exists
            self.removeInputFunction(parameter)  # Remove it if it will be replaced
        self.input_functions[parameter] = functionwidget
        self.addSubFeature(functionwidget)
        functionwidget.sigDelete.connect(lambda: self.removeInputFunction(parameter))

    def removeInputFunction(self, parameter):
        """
        Remove the input function for the given parameter

        Parameters
        ----------
        parameter : str
            Parameter name that will be overriden by return value of the input function

        """
        function = self.input_functions.pop(parameter)
        self.removeSubFeature(function)

    def paramChanged(self, param):
        """
        Slot connected to a pg.Parameter.sigChanged signal
        """
        pass

    def allReadOnly(self, boolean):
        """
        Make all parameter read only
        """
        for param in self.params.children():
            param.setReadonly(boolean)

    def menuRequested(self):
        """
        Context menu for functionWidget. Default is not menu.
        """
        pass


class ROlineEdit(QtGui.QLineEdit):
    """
    Subclass of QlineEdit used for labels in FeatureWidgets
    """

    sigClicked = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super(ROlineEdit, self).__init__(*args, **kwargs)
        self.setReadOnly(True)
        self.setFrame(False)

    def focusOutEvent(self, *args, **kwargs):
        super(ROlineEdit, self).focusOutEvent(*args, **kwargs)
        self.setCursor(QtCore.Qt.ArrowCursor)

    def mousePressEvent(self, *args, **kwargs):
        super(ROlineEdit, self).mousePressEvent(*args, **kwargs)
        self.sigClicked.emit()

    def mouseDoubleClickEvent(self, *args, **kwargs):
        super(ROlineEdit, self).mouseDoubleClickEvent(*args, **kwargs)
        self.setFrame(True)
        self.setFocus()
        self.selectAll()


class FeatureManager(QtCore.QObject):
    """
    Feature Manager class to manage a list of FeatureWidgets and show the list in an appropriate layout and their
    corresponding forms in another layout. list layout must have an addWidget, removeWidget methods. Form layout
    must in addition have a setCurrentWidget method

    Attributes
    ----------
    features : list of FeatureWidgets
        List of the FeatureWidgets managed
    selectedFeature : FeatureWidget
        The currently selected feature

    Parameters
    ----------
    list_layout : QtGui.QLayout
        Layout to display the list of FeatureWidgets
    form_layout : QtGui.QLayout
        Layout to display the FeaturenWidgets form (pyqtgraph.Parameter)
    feature_widgets : list of FeatureWidgets, optional
        List with feature widgets for initialization
    blank_form : QtGui.QWidget, optional
        Widget to display in form_layout when not FunctionWidget is selected
    """

    def __init__(self, list_layout, form_layout, feature_widgets=None, blank_form=None):
        self._llayout = list_layout
        self._flayout = form_layout
        self.features = []
        self.selectedFeature = None

        if feature_widgets is not None:
            for feature in feature_widgets:
                self.addFeature(feature)

        if blank_form is not None:
            if isinstance(blank_form, str):
                self.blank_form = QtGui.QLabel(blank_form)
            else:
                self.blank_form = blank_form
        else:
            self.blank_form = QtGui.QLabel('Select a feature to view its form')
        self.blank_form.setAlignment(QtCore.Qt.AlignCenter)
        self.blank_form.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)

        self._flayout.addWidget(self.blank_form)
        self.showForm(self.blank_form)
        super(FeatureManager, self).__init__()

    @property
    def count(self):
        """
        Number of features managed
        """
        return len(self.features)

    @property
    def nextFeature(self):
        """
        The feature after the selectedFeature
        """
        try:
            index = self.features.index(self.selectedFeature)
        except (ValueError, IndexError):
            return None
        if index > 0:
            return self.features[index - 1]
        else:
            return self.selectedFeature

    @property
    def previousFeature(self):
        """
        The feature before the selectedFeature
        """
        if self.selectedFeature is None:
            return None
        index = self.features.index(self.selectedFeature)
        if index < self.count - 1:
            return self.features[index + 1]
        else:
            return self.selectedFeature

    @QtCore.Slot(QtGui.QWidget)
    def featureClicked(self, feature):
        """
        Slot used to receive features sigClicked
        """
        if feature in self.features:
            self.collapseAllFeatures()
        self.showForm(feature.form)
        feature.expand()
        self.selectedFeature = feature

    @QtCore.Slot(QtGui.QWidget)
    def removeFeature(self, feature):
        """
        Slot used to receive features sigDelete
        """
        self.features.remove(feature)
        feature.deleteLater()
        del feature
        self.showForm(self.blank_form)

    @QtCore.Slot(QtGui.QWidget)
    def subFeatureAdded(self, subfeature):
        """
        Slot used to receive features sigSubFeature
        """
        try:
            subfeature.sigClicked.connect(self.featureClicked)
            subfeature.sigSubFeature.connect(self.subFeatureAdded)
            self._flayout.addWidget(subfeature.form)
        except AttributeError:
            pass

    def addFeature(self, feature):
        """
        Adds a subfeature to the given feature
        """
        self.features.append(feature)
        self._llayout.addWidget(feature)
        self._flayout.addWidget(feature.form)
        feature.sigClicked.connect(self.featureClicked)
        feature.sigDelete.connect(self.removeFeature)
        feature.sigSubFeature.connect(self.subFeatureAdded)
        if feature.subfeatures is not None:
            for subfeature in feature.subfeatures:
                self.subFeatureAdded(subfeature)

    def collapseAllFeatures(self):
        """
        Collapses all features with subfeatures
        """
        for feature in self.features:
            feature.collapse()
            if feature.subfeatures is not None:
                for subfeature in feature.subfeatures:
                    subfeature.collapse()

    def showForm(self, form):
        """
        Shows the current features form
        """
        self._flayout.setCurrentWidget(form)

    def removeAllFeatures(self):
        """
        Deletes all features
        """
        for feature in self.features:
            feature.deleteLater()
            del feature
        self.features = []
        self.showForm(self.blank_form)

    def clearLayouts(self):
        """
        Removes all features and forms from layouts
        """
        for feature in self.features:
            self._flayout.removeWidget(feature)
            self._llayout.removeWidget(feature)

    def update(self):
        """
        Updates the layouts to show the current list of features
        """
        self.clearLayouts()
        for feature in self.features:
            self._llayout.addWidget(feature)
            self._flayout.addWidget(feature.form)
        self.showForm(self.selectedFeature.form)

    def swapFeatures(self, f1, f2):
        """
        Swaps the location of two features
        """
        idx_1, idx_2 = self.features.index(f1), self.features.index(f2)
        self.features[idx_1], self.features[idx_2] = self.features[idx_2], self.features[idx_1]
        self.update()


class FunctionManager(FeatureManager):
    """
    Subclass of FeatureManager used to manage tomography workflow/pipeline FunctionWidgets

    Attributes
    ----------
    cor_offest : function/lambda
        function to correct for an offset in the COR location. As when padding the input array
    corr_scale : function/lambda
        function to correct for a scaling in the COR location. As when subsampling the input array

    recon_function : FunctionWidget
        FunctionWidget representing the Reconstruction Function in worflow pipeline

    Parameters
    ----------
    list_layout : QtGui.QLayout
        Layout to display the list of FunctionWidgets
    form_layout : QtGui.QLayout
        Layout to display the FunctionWidgets form (pyqtgraph.Parameter)
    function_widgets : list of FunctionWidgets, optional
        List with functionwidgets for initialization
    blank_form : QtGui.QWidget, optional
        Widget to display in form_layout when not FunctionWidget is selected

    Signals
    -------
    sigTestRange(str, object)
    sigPipelineChanged()
        Emitted when the pipeline changes or the reconstruction function is changed
    """

    sigPipelineChanged = QtCore.Signal()

    center_func_slc = {'Phase Correlation': (0, -1)}  # slice parameters for center functions

    def __init__(self, list_layout, form_layout, function_widgets=None, blank_form=None):
        super(FunctionManager, self).__init__(list_layout, form_layout, feature_widgets=function_widgets,
                                              blank_form=blank_form)
        self.cor_offset = lambda x: x  # dummy
        self.cor_scale = lambda x: x  # dummy
        self.recon_function = None

    # TODO fix this astra check raise error if package not available
    def addFunction(self, function):
        """
        Adds a Function to the workflow pipeline

        Parameters
        ----------
        function : str
            generic name of function
        subfunction : str
            specific name of function under the generic name category
        package : python package
            package where function is defined
        """

        func_widget = FunctionWidget(function)
        self.addFeature(func_widget)
        return func_widget

    def addInputFunction(self, funcwidget, parameter, function, subfunction, package, **kwargs):
        """
        Adds an input function to the give function widget

        Parameters
        ----------
        funcwidget : FunctionWidget
            Widget to add subfunction to
        parameter : str
            Parameter name that will be overriden by return value of input function
        function : str
            generic name of function
        subfunction : str
            specific name of function under the generic name category
        package : python package
            package where function is defined
        kwargs
            Additional keyword arguments
        """
        try:
            ipf_widget = FunctionWidget(function, subfunction, package, **kwargs)
            funcwidget.addInputFunction(parameter, ipf_widget)
        except AttributeError:
            ipf_widget = funcwidget.input_functions[parameter]
        return ipf_widget

    def updateParameters(self):
        """
        Updates all parameters for the current function list
        """
        for function in self.features:
            function.updateParamsDict()

    def lockParams(self, boolean):
        """
        Locks all parameters for the current function list
        """
        for func in self.features:
            func.allReadOnly(boolean)

    def resetCenterCorrection(self):
        """
        Resets the center correction functions to dummy lambdas
        """
        self.cor_offset = lambda x: x  # dummy
        self.cor_scale = lambda x: x  # dummy

    def setCenterCorrection(self, name, param_dict):
        """
        Sets the center correction lambda's according to the effect of function given to the input array

        Parameters
        ----------
        name : str
            Name of function that has an effect on the COR value
        param_dict : dict
            Parameter dictionary of the function give
        """

        if 'Padding' in name and param_dict['axis'] == 2:
            n = param_dict['npad']
            self.cor_offset = lambda x: x + n
        elif 'Downsample' in name and param_dict['axis'] == 2:
            s = param_dict['level']
            self.cor_scale = lambda x: x / 2 ** s
        elif 'Upsample' in name and param_dict['axis'] == 2:
            s = param_dict['level']
            self.cor_scale = lambda x: x * 2 ** s

    def updateFunctionPartial(self, funcwidget, datawidget, stack_dict=None, slc=None):
        """
        Updates the given FunctionWidget's partial

        Parameters
        ----------
        funcwidget : FunctionWidget
            Widget whos partial is to be updated
        datawidget
            Class holding the input dataset
        stack_dict : dict, optional
            Copy FunctionWidget's param_dict
        slc : slice
            Slice object to extract flat/dark fields when appropriate

        Returns
        -------
        functools.partial
            partial object with updated keywords
        """

        fpartial = funcwidget.partial
        return fpartial

    def previewFunctionStack(self, datawidget, slc=None, ncore=None, skip_names=['Write'], fixed_func=None):
        """
        Create the function stack and summary dictionary used for running slice previews and 3D previews

        Parameters
        ----------
        datawidget
            Class holding the input dataset
        slc slice
            Slice object to extract tomography/flat/dark data when appropriate
        ncore : int
            number of cores to set the appropriate functions to run on
        skip_names : list of str, optional
            Names of functions to skip when running but still add to the dict representing the pipeline to run.
            Currently only the Writing functions are skipped as writing is not necessary in previews.
        fixed_func : type class
            A dynamic class with only the necessary attributes to be run in a workflow pipeline. This is used for
            parameter range tests to create the class with the parameter to be run and send it to a background thread.
            See testParameterRange for more details

        Returns
        -------
        list of partials:
            List with function partials needed to run preview
        dict
            Dictionary summarizing functions and parameters representing the pipeline (used for the list of partials)
        """

        stack_dict = OrderedDict()
        partial_stack = []
        self.lockParams(True)
        for func in self.features:
            if not func.enabled:
                continue
            elif func.func_name in skip_names:
                stack_dict[func.func_name] = {func.subfunc_name: deepcopy(func.exposed_param_dict)}
                continue
            elif fixed_func is not None and func.func_name == fixed_func.func_name:
                func = fixed_func  # replace the function with the fixed function
            stack_dict[func.func_name] = {func.subfunc_name: deepcopy(func.exposed_param_dict)}
            p = self.updateFunctionPartial(func, datawidget, stack_dict[func.func_name][func.subfunc_name], slc)
            if 'ncore' in p.keywords:
                p.keywords['ncore'] = ncore
            partial_stack.append(p)
            for param, ipf in func.input_functions.iteritems():
                if ipf.enabled:
                    if 'Input Functions' not in stack_dict[func.func_name][func.subfunc_name]:
                        stack_dict[func.func_name][func.subfunc_name]['Input Functions'] = {}
                    ipf_dict = {param: {ipf.func_name: {ipf.subfunc_name: ipf.exposed_param_dict}}}
                    stack_dict[func.func_name][func.subfunc_name]['Input Functions'].update(ipf_dict)
        self.lockParams(False)
        return partial_stack, stack_dict

    @staticmethod
    def foldFunctionStack(partial_stack, initializer):
        """
        Static class method to fold a partial function stack given an initializer

        Parameters
        ----------
        partial_stack : list of functools.partial
            List of partials that require only the input array to run.
        initializer : ndarray
            Array to use as initializer for folding operation

        Returns
        -------
        Return value of last partial in stack
            Result of folding operation
        """
        return reduce(lambda f1, f2: f2(f1), partial_stack, initializer)

    def functionStackGenerator(self, datawidget, proj, sino, sino_p_chunk, ncore=None):
        """
        Generator for running full reconstruction. Yields messages representing the status of reconstruction
        This is ideally used as a threads.method or the corresponding threads.RunnableIterator.

        Parameters
        ----------
        datawidget
        proj : tuple of int
            Projection range indices (start, end, step)
        sino : tuple of int
            Sinogram range indices (start, end, step)
        sino_p_chunk : int
            Number of sinograms per chunk
        ncore : int
            Number of cores to run functions

        Yields
        -------
        str
            Message of current status of function
        """

        write_start = sino[0]
        nchunk = ((sino[1] - sino[0]) // sino[2] - 1) // sino_p_chunk + 1
        total_sino = (sino[1] - sino[0] - 1) // sino[2] + 1
        if total_sino < sino_p_chunk:
            sino_p_chunk = total_sino

        for i in range(nchunk):
            init = True
            start, end = i * sino[2] * sino_p_chunk + sino[0], (i + 1) * sino[2] * sino_p_chunk + sino[0]
            end = end if end < sino[1] else sino[1]

            for function in self.features:
                if not function.enabled:
                    continue
                ts = time.time()
                yield 'Running {0} on slices {1} to {2} from a total of {3} slices...'.format(function.name, start,
                                                                                              end, total_sino)
                fpartial = self.updateFunctionPartial(function, datawidget,
                                                      slc=(slice(*proj), slice(start, end, sino[2]),
                                                           slice(None, None, None)))
                if init:
                    tomo = datawidget.getsino(slc=(slice(*proj), slice(start, end, sino[2]),
                                                   slice(None, None, None)))
                    init = False
                elif 'Tiff' in function.name:
                    fpartial.keywords['start'] = write_start
                    write_start += tomo.shape[0]
                # elif 'Reconstruction' in fname:
                #     # Reset input_partials to None so that centers and angle vectors are not computed in every iteration
                #     # and set the reconstruction partial to the updated one.
                #     if ipartials is not None:
                #         ind = next((i for i, names in enumerate(fpartials) if fname in names), None)
                #         fpartials[ind][0], fpartials[ind][4] = fpartial, None
                #     tomo = fpartial(tomo)
                tomo = fpartial(tomo)
                yield ' Finished in {:.3f} s\n'.format(time.time() - ts)

    def setPipelineFromYAML(self, pipeline, setdefaults=False, config_dict={}):
        """
        Sets the managers function/feature list from a dictionary from a YAML file.

        Parameters
        ----------
        pipeline : dict
            Dict extracted from YAML file
        setdefaults : bool
            Set the given parameter values as defaults
        config_dict : dict, optional
            Dictionary with configuration specifications/function parameter details
        """

        self.removeAllFeatures()
        for func in pipeline:
            self.addFunction(func)

        self.sigPipelineChanged.emit()

    def setPipelineFromDict(self, pipeline, config_dict={}):
        """
        Sets the managers function/feature list from a dictionary extracted from a summary dictionary as the ones
        displayed in previews

        Parameters
        ----------
        pipeline : dict
            Dict representing the workflow pipeline
        config_dict : dict, optional
            Dictionary with configuration specifications/function parameter details
        """

        self.removeAllFeatures()
        for func, subfuncs in pipeline.iteritems():
            for subfunc in subfuncs:
                funcWidget = self.addFunction(func, subfunc, package=reconpkg.packages[config_dict[subfunc][1]])
                for param, value in subfuncs[subfunc].iteritems():
                    if param == 'Package':
                        continue
                    elif param == 'Input Functions':
                        for param, ipfs in value.iteritems():
                            for ipf, sipf in ipfs.iteritems():
                                ifwidget = self.addInputFunction(funcWidget, param, ipf, sipf.keys()[0],
                                                                 package=reconpkg.packages[
                                                                     config_dict[sipf.keys()[0]][1]])
                                for p, v in sipf[sipf.keys()[0]].items():
                                    ifwidget.params.child(p).setValue(v)
                                ifwidget.updateParamsDict()
                    else:
                        funcWidget.params.child(param).setValue(value)
                    funcWidget.updateParamsDict()
        self.sigPipelineChanged.emit()

    def workflowExectutionGenerator(self, **workspace):
        """
        Generator for running full reconstruction. Yields messages representing the status of reconstruction
        This is ideally used as a threads.method or the corresponding threads.RunnableIterator.

        Parameters
        ----------
        ncore : int
            Number of cores to run functions

        Yields
        -------
        str
            Message of current status of function
        """

        stack_dict = OrderedDict()
        partial_stack = []

        for func in self.features:
            if not func.enabled:
                continue

            ts = time.time()
            yield 'Running {0}...'.format(func.name)

            stack_dict[func.func_name] = {func.subfunc_name: deepcopy(func.exposed_param_dict)}
            p = self.updateFunctionPartial(func, stack_dict[func.func_name][func.subfunc_name])
            partial_stack.append(p)

            if func.functionType == func.PROCESS:
                workspace, updates = p(**workspace)
            elif func.functionType == func.OUTPUT:
                p(updates, **workspace)
        # self.lockParams(False)
        yield 'Finished in {:.3f} s\n'.format(time.time() - ts)
        yield workspace


def map_loc(slc, loc):
    """
    Does a linear mapping of the indices in loc from a range given by slc start and stop with step of one to a new
    range given by len(range(slc.start, slc.stop, slc.step))

    Parameters
    ----------
    slc : slice
    loc : list
        list of indices assumed to span from slc.start to slc.stop

    Returns
    -------
    list
        mapped indices to new range

    """

    step = slc.step if slc.step is not None else 1
    ind = range(slc.start, slc.stop, step)
    loc = np.array(loc)
    low, upp = ind[0], ind[-1]
    buff = (loc[-1] - loc[0]) / len(loc)
    min_loc = low - buff
    max_loc = upp + buff
    loc = np.intersect1d(loc[loc > min_loc], loc[loc < max_loc])
    new_upp = len(ind)
    loc = (new_upp * (loc - low)) // (upp - low)
    if loc[0] < 0:
        loc[0] = 0

    return np.ndarray.tolist(loc)

if __name__ == '__main__':

    app = QtGui.QApplication([])
    import numpy

    import imp

    module = imp.load_source('saxsfunctions',
                             '/home/rp/PycharmProjects/xicam/pipeline/workflowfunctions/testfunctions.py')

    w = workflowEditorWidget('/home/rp/PycharmProjects/xicam/pipeline/workflowfunctions/testworkflow.yml', module)
    w.show()

    w.sigExecute.connect(lambda: w.runWorkflow(**{'testvar': 'THIS IS TEST INPUT'}))



    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
