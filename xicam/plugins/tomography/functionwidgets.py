# -*- coding: utf-8 -*-


__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


import inspect
import time
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from modpkgs import yamlmod
import numpy as np
from PySide import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import config
import reconpkg
import ui
from xicam.widgets import featurewidgets as fw


class FunctionWidget(fw.FeatureWidget):
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
    param_dict : dict
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

    sigTestRange = QtCore.Signal(QtGui.QWidget, str, tuple)

    # TODO perhaps its better to not pass in the package object but only a string, package object can be retrived from reconpkgs.packages dict
    def __init__(self, name, subname, package, input_functions=None, checkable=True, closeable=True, parent=None):
        self.name = name
        if name != subname:
            self.name += ' (' + subname + ')'
        super(FunctionWidget, self).__init__(self.name, checkable=checkable, closeable=closeable, parent=parent)

        self.func_name = name
        self.subfunc_name = subname
        self.input_functions = {}
        self.param_dict = {}
        self._function = getattr(package, config.names[self.subfunc_name][0])

        # TODO have the children kwarg be passed to __init__
        self.params = Parameter.create(name=self.name, children=config.parameters[self.subfunc_name], type='group')

        self.form = ParameterTree(showHeader=False)
        self.form.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.form.customContextMenuRequested.connect(self.paramMenuRequested)
        self.form.setParameters(self.params, showTop=True)

        # Initialize parameter dictionary with keys and default values
        self.updateParamsDict()
        argspec = inspect.getargspec(self._function)
        default_argnum = len(argspec[3])
        self.param_dict.update({key : val for (key, val) in zip(argspec[0][-default_argnum:], argspec[3])})
        for key, val in self.param_dict.iteritems():
            if key in [p.name() for p in self.params.children()]:
                self.params.child(key).setValue(val)
                self.params.child(key).setDefault(val)

        # Create a list of argument names (this will most generally be the data passed to the function)
        self.missing_args = [i for i in argspec[0] if i not in self.param_dict.keys()]

        self.parammenu = QtGui.QMenu()
        action = QtGui.QAction('Test Parameter Range', self)
        action.triggered.connect(self.testParamTriggered)
        self.parammenu.addAction(action)

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
        param_dict = {key: val for (key, val) in self.param_dict.iteritems()
                      if key in [param.name() for param in self.params.children()]}
        return param_dict

    @property
    def partial(self):
        """
        Package up all parameters into a functools.partial
        """
        return partial(self._function, **self.param_dict)

    @property
    def func_signature(self):
        """
        String for function signature. Hopefully this can eventually be used to save workflows as scripts :)
        """
        signature = str(self._function.__name__) + '('
        for arg in self.missing_args:
            signature += '{},'.format(arg)
        for param, value in self.param_dict.iteritems():
            signature += '{0}={1},'.format(param, value) if not isinstance(value, str) else \
                '{0}=\'{1}\','.format(param, value)
        return signature[:-1] + ')'

    def updateParamsDict(self):
        """
        Update the values of the parameter dictionary with the current values in UI
        """
        self.param_dict.update({param.name(): param.value() for param in self.params.children()})
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
        self.param_dict.update({param.name(): param.value()})

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

    def paramMenuRequested(self, pos):
        """
        Menus when a parameter in the form is right clicked
        """
        if self.form.currentItem().parent():
            self.parammenu.exec_(self.form.mapToGlobal(pos))

    def testParamTriggered(self):
        """
        Slot when a parameter range is clicked. Will emit the parameter name and the chosen range
        """
        param = self.form.currentItem().param
        if param.type() == 'int' or param.type() == 'float':
            start, end, step = None, None, None
            if 'limits' in param.opts:
                start, end = param.opts['limits']
                step = (end - start) / 3 + 1
            elif param.value() is not None:
                start, end, step = param.value() / 2, 4 * (param.value()) / 2, param.value() / 2
            test = TestRangeDialog(param.type(), (start, end, step))
        elif param.type() == 'list':
            test = TestListRangeDialog(param.opts['values'])
        else:
            return
        if test.exec_():
            self.sigTestRange.emit(self, param.name(), test.selectedRange())


class TomoPyReconFunctionWidget(FunctionWidget):
    """
    Subclass of FunctionWidget used for Tomopy recon functions. Allows adding input functions for center of rotation
    detection. And has a default Projection Angles function to provide the theta parameter to the function.

    Parameters
    ----------
    name : str
        generic name of function
    subname : str
        specific name of function under the generic name category
    package : python package
        package
    """

    def __init__(self, name, subname, package):

        self.packagename = package.__name__
        self.input_functions = {'theta': FunctionWidget('Projection Angles', 'Projection Angles', closeable=False,
                                                  package=reconpkg.packages['tomopy'], checkable=False)}
        super(TomoPyReconFunctionWidget, self).__init__(name, subname, package, input_functions=self.input_functions,
                                                        checkable=False)
        # Fill in the appropriate 'algorithm' keyword
        self.param_dict['algorithm'] = subname.lower()
        self.submenu = QtGui.QMenu('Input Function')
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_39.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.submenu.setIcon(icon)
        ui.build_function_menu(self.submenu, config.funcs['Input Functions'][name], config.names,
                               self.addCenterDetectFunction)
        self.menu.addMenu(self.submenu)

    @property
    def partial(self):
        """
        Overrides partial property to do some cleanup before creating the partial
        """
        kwargs = deepcopy(self.param_dict)
        # 'cutoff' and 'order' are not passed into the tomopy recon function as {'filter_par': [cutoff, order]}
        if 'cutoff' in kwargs.keys() and 'order' in kwargs.keys():
            kwargs['filter_par'] = list((kwargs.pop('cutoff'), kwargs.pop('order')))
        self._partial = partial(self._function, **kwargs)
        return self._partial

    def addCenterDetectFunction(self, name, subname, package=reconpkg.packages['tomopy']):
        """
        Used to add a center detection input function from the FunctionWidgets context menu

        Parameters
        ----------
        name : str
            generic name of function
        subname : str
            specific name of function under the generic name category
        package : python package
            package where function is defined
        """
        try:
            self.addInputFunction('center', FunctionWidget(name, subname, package=package))
        except AttributeError:
            pass

    def setCenterParam(self, value):
        """
        Sets the center parameter for the FunctionWidget to give value
        """
        self.params.child('center').setValue(value)
        self.params.child('center').setDefault(value)

    def menuRequested(self, pos):
        """
        Slot when menu is requested from FeatureWidget ui (ie to add input functions)
        """
        self.menu.exec_(self.previewButton.mapToGlobal(pos))


class AstraReconFuncWidget(TomoPyReconFunctionWidget):
    """
    Subclass of FunctionWidget used for Astra recon functions using Tomopy's astra wrapper
    """
    def __init__(self, name, subname, package):
        super(AstraReconFuncWidget, self).__init__(name, subname, reconpkg.packages['tomopy'])
        self.param_dict['algorithm'] = reconpkg.packages['astra']
        self.param_dict['options'] = {}
        self.param_dict['options']['method'] = subname.replace(' ', '_')
        if 'CUDA' in subname:
            self.param_dict['options']['proj_type'] = 'cuda'
        else:
            self.param_dict['options']['proj_type'] = 'linear'

    @property
    def partial(self):
        """
        Return the base FunctionWidget partial property
        """
        return FunctionWidget.partial(self)


class WriteFunctionWidget(FunctionWidget):
    """
    Subclass of FunctionWidget for write functions to have a Browse button to use a QFileDialog for setting 'fname'
    parameter

    Parameters
    ----------
    name : str
        generic name of function
    subname : str
        specific name of function under the generic name category
    package : python package
        package where function is defined
    """
    def __init__(self, name, subname, package):
        super(WriteFunctionWidget, self).__init__(name, subname, package)
        self.params.child('Browse').sigActivated.connect(
            lambda: self.params.child('fname').setValue( str(QtGui.QFileDialog.getSaveFileName(None,
            'Save reconstruction as', self.params.child('fname').value())[0])))

    def updateParamsDict(self):
        """
        Overrides updating the parameter_dict to avoid adding the 'Browse' action
        """
        self.param_dict.update({param.name(): param.value() for param in self.params.children()
                                if param.name() != 'Browse'})  # skip the Browse parameter!
        for p, ipf in self.input_functions.iteritems():
            ipf.updateParamsDict()


class TestRangeDialog(QtGui.QDialog):
    """
    Simple QDialog subclass with three spinBoxes to inter start, end, step for a range to test a particular function
    parameter
    """

    def __init__(self, dtype, prange, **opts):
        super(TestRangeDialog, self).__init__(**opts)
        SpinBox = QtGui.QSpinBox if dtype == 'int' else QtGui.QDoubleSpinBox
        self.gridLayout = QtGui.QGridLayout(self)
        self.spinBox = SpinBox(self)
        self.gridLayout.addWidget(self.spinBox, 1, 0, 1, 1)
        self.spinBox_2 = SpinBox(self)
        self.gridLayout.addWidget(self.spinBox_2, 1, 1, 1, 1)
        self.spinBox_3 = SpinBox(self)
        self.gridLayout.addWidget(self.spinBox_3, 1, 2, 1, 1)
        self.label = QtGui.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtGui.QLabel(self)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.label_2, 0, 1, 1, 1)
        self.label_3 = QtGui.QLabel(self)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.label_3, 0, 2, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.gridLayout.addWidget(self.buttonBox, 0, 3, 2, 1)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        if prange is not (None, None, None):
            self.spinBox.setMaximum(9999)  # 3 * prange[2])
            self.spinBox_2.setMaximum(9999)  # 3 * prange[2])
            self.spinBox_3.setMaximum(9999)  # prange[2])

            self.spinBox.setValue(prange[0])
            self.spinBox_2.setValue(prange[1])
            self.spinBox_3.setValue(prange[2])

        self.setWindowTitle("Set parameter range")
        self.label.setText("Start")
        self.label_2.setText("End")
        self.label_3.setText("Step")

    def selectedRange(self):
        # return the end as selected end + step so that the range includes the end
        return np.arange(self.spinBox.value(), self.spinBox_2.value(), self.spinBox_3.value())


class TestListRangeDialog(QtGui.QDialog):
    """
    Simple QDialog subclass with comboBox and lineEdit to choose from a list of available function parameter keywords
    in order to test the different function parameters.
    """

    def __init__(self, options, **opts):
        super(TestListRangeDialog, self).__init__(**opts)
        self.gridLayout = QtGui.QGridLayout(self)
        self.comboBox = QtGui.QComboBox(self)
        self.gridLayout.addWidget(self.comboBox, 1, 0, 1, 1)
        self.lineEdit = QtGui.QLineEdit(self)
        self.lineEdit.setReadOnly(True)
        self.gridLayout.addWidget(self.lineEdit, 2, 0, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.gridLayout.addWidget(self.buttonBox, 1, 1, 2, 1)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.setWindowTitle('Set parameter range')

        self.options = options
        self.comboBox.addItems(options)
        self.comboBox.activated.connect(self.addToList)
        self.lineEdit.setText(' '.join(options))
        self.lineEdit.keyPressEvent = self.keyPressEvent

    def addToList(self, option):
        self.lineEdit.setText(str(self.lineEdit.text()) + ' ' + self.options[option])

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Backspace or ev.key() == QtCore.Qt.Key_Delete:
            self.lineEdit.setText(' '.join(str(self.lineEdit.text()).split(' ')[:-1]))
        elif ev.key() == QtCore.Qt.Key_Enter or ev.key() == QtCore.Qt.Key_Return:
            self.addToList(self.comboBox.currentIndex())
        ev.accept()

    def selectedRange(self):
        return str(self.lineEdit.text()).split(' ')


class FunctionManager(fw.FeatureManager):
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

    sigTestRange = QtCore.Signal(str, object)
    sigPipelineChanged = QtCore.Signal()

    center_func_slc = {'Phase Correlation': (0, -1)}  # slice parameters for center functions

    def __init__(self, list_layout, form_layout, function_widgets=None, blank_form=None):
        super(FunctionManager, self).__init__(list_layout, form_layout, feature_widgets=function_widgets,
                                              blank_form=blank_form)
        self.cor_offset = lambda x: x  # dummy
        self.cor_scale = lambda x: x  # dummy
        self.recon_function = None

    # TODO fix this astra check raise error if package not available
    def addFunction(self, function, subfunction, package):
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

        if function == 'Reconstruction':
            if 'astra' in reconpkg.packages and package == reconpkg.packages['astra']:
                func_widget = AstraReconFuncWidget(function, subfunction, package)
            else:
                func_widget = TomoPyReconFunctionWidget(function, subfunction, package)
            self.recon_function = func_widget
            self.sigPipelineChanged.emit()
        elif function == 'Write':
            func_widget = WriteFunctionWidget(function, subfunction, package)
        else:
            func_widget = FunctionWidget(function, subfunction, package)
        func_widget.sigTestRange.connect(self.testParameterRange)
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

    def updateFunctionPartial(self, funcwidget, datawidget, function_dict, stack_dict=None, slc=None):
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
        func_params = function_dict[funcwidget.name]
        missing_args = func_params["missing_args"]
        for argname in missing_args: # find a more elegant way to point to the flats and darks
            if argname in 'flats':
                fpartial.keywords[argname] = datawidget.getflats(slc=slc)
            if argname in 'darks':
                fpartial.keywords[argname] = datawidget.getdarks(slc=slc)
            if argname in 'flat_loc': # I don't like this at all
                if slc is not None and slc[0].start is not None:
                    slc_ = (slice(slc[0].start, datawidget.data.shape[0] - 1, slc[0].step) if slc[0].stop is None
                            else slc[0])
                    fpartial.keywords[argname] = map_loc(slc_, datawidget.data.fabimage.flatindices())
                else:
                    fpartial.keywords[argname] = datawidget.data.fabimage.flatindices()

        input_funcs = func_params['input_functions']
        for param, ipf_dict in input_funcs.iteritems():
            args = []
            if not ipf_dict['enabled']:
                continue
            if param == 'center':
                if ipf_dict['subfunc_name'] in FunctionManager.center_func_slc:
                    map(args.append, map(datawidget.data.fabimage.__getitem__,
                                         FunctionManager.center_func_slc[ipf_dict['subfunc_name']]))
                else:
                    args.append(datawidget.getsino())

                if ipf_dict['subfunc_name'] == 'Nelder Mead':
                    ipf_dict['func'].partial.keywords['theta'] = funcwidget.input_functions['theta'].partial()

            fpartial.keywords[param] = ipf_dict['func'].partial(*args)

        if funcwidget.func_name in ('Padding', 'Downsample', 'Upsample'):
            self.setCenterCorrection(funcwidget.func_name, fpartial.keywords)
        elif 'Reconstruction' in funcwidget.func_name:
            fpartial.keywords['center'] = self.cor_offset(self.cor_scale(fpartial.keywords['center']))
            self.resetCenterCorrection()
        elif 'Write' in funcwidget.func_name:
            fpartial.keywords['fname'] = func_params['fname']

        return fpartial



        # for param, ipf in funcwidget.input_functions.iteritems():
        #     # print param,",", ipf.name
        #     args = []
        #     if not ipf.enabled:
        #         continue
        #     if param == 'center':  # Need to find a cleaner solution to this
        #         if ipf.subfunc_name in FunctionManager.center_func_slc:
        #             map(args.append, map(datawidget.data.fabimage.__getitem__,
        #                                  FunctionManager.center_func_slc[ipf.subfunc_name]))
        #         else:
        #             args.append(datawidget.getsino())
        #         if ipf.subfunc_name == 'Nelder Mead':  # Also needs a cleaner solution
        #             ipf.partial.keywords['theta'] = funcwidget.input_functions['theta'].partial()
        #     fpartial.keywords[param] = ipf.partial(*args)
        #     # these lines don't seem to do anything - Holden
        #     # if stack_dict and param in stack_dict:  # update the stack dict with new kwargs
        #     #     stack_dict[param] = fpartial.keywords[param]
        # if funcwidget.func_name in ('Padding', 'Downsample', 'Upsample'):
        #     self.setCenterCorrection(funcwidget.func_name, fpartial.keywords)
        # elif 'Reconstruction' in funcwidget.func_name:
        #     fpartial.keywords['center'] = self.cor_offset(self.cor_scale(fpartial.keywords['center']))
        #     self.resetCenterCorrection()
        # elif 'Write' in funcwidget.func_name:
        #     fpartial.keywords['fname'] = save_name
        # return fpartial

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

        self.stack_dict = OrderedDict()
        partial_stack = []
        self.lockParams(True)
        for func in self.features:
            if not func.enabled:
                continue
            elif func.func_name in skip_names:
                self.stack_dict[func.func_name] = {func.subfunc_name: deepcopy(func.exposed_param_dict)}
                continue
            elif fixed_func is not None and func.func_name == fixed_func.func_name:
                func = fixed_func  # replace the function with the fixed function
            self.stack_dict[func.func_name] = {func.subfunc_name: deepcopy(func.exposed_param_dict)}
            p = self.updateFunctionPartial(func, datawidget, self.stack_dict[func.func_name][func.subfunc_name], slc)
            if 'ncore' in p.keywords:
                p.keywords['ncore'] = ncore
            partial_stack.append(p)
            for param, ipf in func.input_functions.iteritems():
                if ipf.enabled:
                    if 'Input Functions' not in self.stack_dict[func.func_name][func.subfunc_name]:
                        self.stack_dict[func.func_name][func.subfunc_name]['Input Functions'] = {}
                    ipf_dict = {param: {ipf.func_name: {ipf.subfunc_name: ipf.exposed_param_dict}}}
                    self.stack_dict[func.func_name][func.subfunc_name]['Input Functions'].update(ipf_dict)
        self.lockParams(False)
        print self.stack_dict
        return partial_stack, self.stack_dict

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

    def functionStackGenerator(self, datawidget, pipeline_dict, proj, sino, sino_p_chunk, ncore=None):
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

        start_time = time.time()
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
                # if not function.enabled:
                if not pipeline_dict[function.name]['enabled']:
                    continue
                ts = time.time()
                yield 'Running {0} on slices {1} to {2} from a total of {3} slices...'.format(function.name, start,
                                                                                              end, total_sino)

                fpartial = self.updateFunctionPartial(function, datawidget, pipeline_dict,
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

        # save yaml in reconstruction folder
        for key in pipeline_dict.iterkeys():
            if 'Write' in key:
                save_file = pipeline_dict[key]['fname'] + '.yml'
        try:
            with open(save_file, 'w') as yml:
                yamlmod.ordered_dump(pipeline_dict['pipeline_for_yaml'], yml)
        except NameError:
            print "Error: function pipeline yaml not written - path could not be found"

        # print final 'finished with recon' message
        yield 'Reconstruction complete. Run time: {:.2f} s'.format(time.time()-start_time)


    def testParameterRange(self, function, parameter, prange):
        """
        Used to set off parameter range tests. Emits sigTestRange with message and a fixed_func object representing the
        function who's parameter is to be changed

        Parameters
        ----------
        function : FunctionWidget
            Widget containing the parameter to be evaluated
        parameter : str
            Parameter name
        prange : tuple/list
            Range of parameters to be evaluated
        """

        self.updateParameters()
        for i in prange:
            function.param_dict[parameter] = i
            # Dynamic FixedFunc "dummed down" FuncWidget class. cool.
            fixed_func = type('FixedFunc', (), {'func_name': function.func_name, 'subfunc_name': function.subfunc_name,
                                                'missing_args': function.missing_args,
                                                'param_dict': function.param_dict,
                                                'exposed_param_dict': function.exposed_param_dict,
                                                'partial': function.partial,
                                                'input_functions': function.input_functions})
            self.sigTestRange.emit('Computing preview for {} parameter {}={}...'.format(function.name, parameter, i),
                                   fixed_func)

    def setPipelineFromYAML(self, pipeline, setdefaults=False, config_dict=config.names):
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
        # Way too many for loops, oops... may want to restructure the yaml files
        for func, subfuncs in pipeline.iteritems():
            for subfunc in subfuncs:
                funcWidget = self.addFunction(func, subfunc, package=reconpkg.packages[config_dict[subfunc][1]])
                if 'Enabled' in subfuncs[subfunc] and not subfuncs[subfunc]['Enabled']:
                    funcWidget.enabled = False
                if 'Parameters' in subfuncs[subfunc]:
                    for param, value in subfuncs[subfunc]['Parameters'].iteritems():
                        child = funcWidget.params.child(param)
                        child.setValue(value)
                        if setdefaults:
                            child.setDefault(value)
                if 'Input Functions' in subfuncs[subfunc]:
                    for param, ipfs in subfuncs[subfunc]['Input Functions'].iteritems():
                        for ipf, sipfs in ipfs.iteritems():
                            for sipf in sipfs:
                                if param in funcWidget.input_functions:
                                    ifwidget = funcWidget.input_functions[param]
                                else:
                                    ifwidget = self.addInputFunction(funcWidget, param, ipf, sipf,
                                                                     package=reconpkg.packages[config_dict[sipf][1]])
                                if 'Enabled' in sipfs[sipf] and not sipfs[sipf]['Enabled']:
                                    ifwidget.enabled = False
                                if 'Parameters' in sipfs[sipf]:
                                    for p, v in sipfs[sipf]['Parameters'].iteritems():
                                        ifwidget.params.child(p).setValue(v)
                                        if setdefaults:
                                            ifwidget.params.child(p).setDefault(v)
                                ifwidget.updateParamsDict()
                funcWidget.updateParamsDict()
        self.sigPipelineChanged.emit()

    def setPipelineFromDict(self, pipeline, config_dict=config.names):
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
                                                                 package=reconpkg.packages[config_dict[sipf.keys()[0]][1]])
                                for p, v in sipf[sipf.keys()[0]].items():
                                    ifwidget.params.child(p).setValue(v)
                                ifwidget.updateParamsDict()
                    else:
                        funcWidget.params.child(param).setValue(value)
                    funcWidget.updateParamsDict()
        self.sigPipelineChanged.emit()

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
