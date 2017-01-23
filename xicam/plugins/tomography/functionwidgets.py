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
import os
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
import Queue
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
    package : str
        name of package to which function belongs
    params : pyqtgraph.Parameter
        Parameter instance with function parameter exposed in UI
    missing_args : list of str
        Names of missing arguments not contained in param_dict

    Signals
    -------
    sigTestRange(QtGui.QWidget, str, tuple, dict)
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

        #perhaps unnecessary
        self.package = package.__name__

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

        # change on/off icons
        icon = QtGui.QIcon()
        if checkable:
            icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_51.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_45.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
            self.previewButton.setCheckable(True)
        else:
            icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_45.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.previewButton.setCheckable(False)
            self.previewButton.setChecked(True)

        self.previewButton.setIcon(icon)
        self.previewButton.setFlat(True)
        self.previewButton.setChecked(True)
        self.previewButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        self.defaults = config.function_defaults['Other']
        self.allowed_types = {'str': str, 'int': int, 'float': float, 'bool': bool, 'unicode': unicode}

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
        return partial(self._function, **self.updated_param_dict)

    @property
    def updated_param_dict(self):
        """
        Return the param dict of the FunctionWidget updated with proper types
        """

        if self.defaults:
            param_dict = {}
            for key, val in self.param_dict.iteritems():
                if type(val) is str and 'None' in val:
                    pass
                elif key in self.defaults.iterkeys():
                    arg_type = self.defaults[key]['type']
                    try:
                        param_dict[key] = arg_type(val)
                    except ValueError:
                        param_dict[key] = None
                else:
                    param_dict[key] = val
            return param_dict
        else:
            return self.param_dict


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
        if self.defaults:
            try:
                arg_type = self.defaults[param.name()]['type']
                try:
                    self.allowed_types[arg_type](param.value())
                    self.param_dict.update({param.name(): param.value()})
                except ValueError:
                    if param.value() == "None":
                        self.param_dict.update({param.name(): param.value()})
                    else:
                        param.setValue(self.param_dict[param.name()])
            except KeyError:
                self.param_dict.update({param.name(): param.value()})
        else:
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
                                                  package=reconpkg.packages['tomopy'], checkable=False),
                                'center': FunctionWidget('Center Detection', 'Phase Correlation', closeable=True,
                                                  package=reconpkg.packages['tomopy'])}
        super(TomoPyReconFunctionWidget, self).__init__(name, subname, package, input_functions=self.input_functions,
                                                        checkable=False)
        # Fill in the appropriate 'algorithm' keyword
        self.param_dict['algorithm'] = subname.lower()
        self.submenu = QtGui.QMenu('Input Function')
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_39.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.submenu.setIcon(icon)
        ui.build_function_menu(self.submenu, config.funcs['Input Functions'][name], config.names,
                               self.addCenterDetectFunction)
        self.menu.addMenu(self.submenu)

        defaults = config.function_defaults['Tomopy']
        if subname in defaults:
            self.defaults = defaults[subname]
            for key in self.defaults.iterkeys():
                val = self.defaults[key]['default']
                self.param_dict.update({key: val})
                if key in [p.name() for p in self.params.children()]:
                    self.params.child(key).setValue(val)
                    self.params.child(key).setDefault(val)




    @property
    def partial(self):
        """
        Package up all parameters into a functools.partial
        """
        return partial(self._function, **self.updated_param_dict)

    @property
    def updated_param_dict(self):
        """
        Return the param dict of the FunctionWidget updated with proper types
        """
        param_dict = {}
        for key, val in self.param_dict.iteritems():
            if key in self.defaults.iterkeys():
                arg_type = self.defaults[key]['type']
                try:
                    param_dict[key] = self.allowed_types[arg_type](val)
                except ValueError:
                    param_dict[key] = None
            else:
                param_dict[key] = val
        # 'cutoff' and 'order' are not passed into the recon function in the correct format
        if 'cutoff' in param_dict.keys() and 'order' in param_dict.keys():
            param_dict['filter_par'] = list((param_dict.pop('cutoff'), param_dict.pop('order')))
        return param_dict

    # @property
    # def partial(self):
    #     """
    #     Overrides partial property to do some cleanup before creating the partial
    #     """
    #     kwargs = deepcopy(self.param_dict)
    #     # 'cutoff' and 'order' are not passed into the tomopy recon function as {'filter_par': [cutoff, order]}
    #     if 'cutoff' in kwargs.keys() and 'order' in kwargs.keys():
    #         kwargs['filter_par'] = list((kwargs.pop('cutoff'), kwargs.pop('order')))
    #     self._partial = partial(self._function, **kwargs)
    #     return self._partial

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


class TomoCamReconFuncWidget(TomoPyReconFunctionWidget):
    """
    Subclass of tomopy FunctionWidget used for TomoCam recon functions
    """

    @property
    def partial(self):
        return partial(self._function, **self.reorganized_param_dict)

    @property
    def reorganized_param_dict(self):
        param_dict = dict(self.param_dict)
        input_params = {}
        input_params['gpu_device'] = 0
        input_params['oversamp_factor'] = param_dict['oversamp_factor']
        param_dict.pop('oversamp_factor')

        if 'gridrec'in param_dict['algorithm']:
            input_params['fbp_filter_param'] = param_dict['cutoff']
            param_dict.pop('cutoff')
        else:
            input_params['num_iter'] = param_dict['num_iter']
            param_dict.pop('num_iter')

        param_dict['input_params'] = input_params
        return param_dict




class AstraReconFuncWidget(TomoPyReconFunctionWidget):
    """
    Subclass of FunctionWidget used for Astra recon functions using Tomopy's astra wrapper
    """
    def __init__(self, name, subname, package):
        super(AstraReconFuncWidget, self).__init__(name, subname, reconpkg.packages['tomopy'])
        # self.tomopy_args = {'center':'center', 'sinogram_order': False, 'nchunk': None,
        #                     'init_recon': None, 'tomo': 'tomo', 'theta': 'theta'}
        self.tomopy_args = ['center', 'sinogram_order', 'nchunk', 'init_recon', 'tomo', 'theta',
                            'ncore', 'algorithm']


        self.param_dict['algorithm'] = reconpkg.packages['tomopy'].astra
        self.param_dict['options'] = {}
        self.param_dict['options']['method'] = subname.replace(' ', '_')
        if 'CUDA' in subname:
            self.param_dict['options']['proj_type'] = 'cuda'
        else:
            self.param_dict['options']['proj_type'] = 'linear'

        defaults = config.function_defaults['Astra']
        if subname in defaults:
            self.defaults = defaults[subname]
            for key in self.defaults.iterkeys():
                val = self.defaults[key]['default']
                self.param_dict.update({key: val})
                if key in [p.name() for p in self.params.children()]:
                    self.params.child(key).setValue(val)
                    self.params.child(key).setDefault(val)




    @property
    def updated_param_dict(self):
        """
        Return the param dict of the FunctionWidget reorganized for proper execution as tomopy function
        """

        # copy optional params to the options/extra_options subdictionaries
        param_dict = {}
        param_dict['options'] = self.param_dict['options']
        param_dict['options']['extra_options']={}
        for key, val in self.param_dict.iteritems():
            if key in self.tomopy_args:
                if type(val) == str and 'None' in val:
                    param_dict[key] = None
                elif key in self.defaults.iterkeys():
                    arg_type = self.defaults[key]['type']
                    param_dict[key] = self.allowed_types[arg_type](val)
                else:
                    param_dict[key] = val
            elif key == 'options' or (type(val) is str and 'None' in val):
                pass
            elif 'num_iter' in key:
                param_dict['options']['num_iter'] = self.param_dict['num_iter']
            elif key in config.function_defaults['Astra'][self.subfunc_name].iterkeys():
                param_dict['options']['extra_options'][key] = val
            else:
                pass

        # get rid of extra_options if there are none
        if len(param_dict['options']['extra_options']) < 1:
            param_dict['options'].pop('extra_options')

        # if 'astra' in algorithm name, remove it
        if 'astra' in param_dict['options']['method']:
            param_dict['options']['method'] = param_dict['options']['method'].split('_astra')[0]

        return param_dict


class ReadFunctionWidget(FunctionWidget):
    """
    Subclass of FunctionWidget for reader functions. Mostly necessary so that reader can't be removed
    """
    def __init__(self, name, subname, package):
        super(ReadFunctionWidget, self).__init__(name, subname, package, checkable=False,)

    @property
    def sinograms(self):
        return (self.params.child('start_sinogram').value(), self.params.child('end_sinogram').value(),
                self.params.child('step_sinogram').value())

    @property
    def projections(self):
        return (self.params.child('start_projection').value(), self.params.child('end_projection').value(),
                self.params.child('step_projection').value())

    @property
    def chunk(self):
        return self.params.child('sinograms_per_chunk').value()


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

    Attributes
    ----------
    parent : str
        string of the parent folder holding data set
    folder : str
        string of new folder to be written into
    file : str
        file names for reconstruction to write to
    """

    sigPipelineChanged = QtCore.Signal()

    def __init__(self, name, subname, package):
        super(WriteFunctionWidget, self).__init__(name, subname, package)
        # self.params.child('Browse').sigActivated.connect(
        #     lambda: self.params.child('parent folder').setValue( str(QtGui.QFileDialog.getSaveFileName(None,
        #     'Save reconstruction as', self.params.child('parent folder').value())[0])))

        self.parent = self.params.param('parent folder')
        self.folder = self.params.param('folder name')
        self.file = self.params.param('file name')
        self.params.child('Browse').sigActivated.connect(self.setBrowse)

        # connect signals to change full file name whenever smaller names are changed
        self.parent.sigValueChanged.connect(self.pathChanged)
        self.folder.sigValueChanged.connect(self.pathChanged)
        self.file.sigValueChanged.connect(self.pathChanged)

        # set full file name as read only
        self.params.param('fname').setReadonly()

        # self.fname.sigValueChanged.connect(self.fileChanged)
        # self.fname.sigValueChanged.connect(self.folderChanged)
        # self.fname.sigValueChanged.connect(self.parentChanged)

    def setBrowse(self):
        """
        Uses result of browse button in 'parent folder' and 'folder name' fields
        """

        path = str(QtGui.QFileDialog.getSaveFileName(None, 'Save reconstruction as',self.folder.value())[0])

        folder = path.split('/')[-1]
        parent = path.split(folder)[0]

        self.parent.setValue(parent)
        self.folder.setValue(folder)


    def pathChanged(self):
        """
        Changes write name when one of parent/folder/file fields is changed
        """

        self.params.param('fname').setValue(os.path.join(self.parent.value(), self.folder.value(),
                                                         self.file.value()))




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
    sigTestRange(str, object, dict)
    sigPipelineChanged()
        Emitted when the pipeline changes or the reconstruction function is changed
    """

    sigTestRange = QtCore.Signal(str, object, dict)
    sigPipelineChanged = QtCore.Signal()

    center_func_slc = {'Phase Correlation': (0, -1)}  # slice parameters for center functions

    def __init__(self, list_layout, form_layout, function_widgets=None, blank_form=None):
        super(FunctionManager, self).__init__(list_layout, form_layout, feature_widgets=function_widgets,
                                              blank_form=blank_form)
        #TODO: add attribute to keep track of function's order in pipeline
        self.cor_offset = lambda x: x  # dummy
        self.cor_scale = lambda x: x  # dummy
        self.recon_function = None

        # queue for reconstructions
        self.recon_queue = Queue.Queue()


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
            elif 'mbir' in reconpkg.packages and package == reconpkg.packages['mbir']:
                func_widget = TomoCamReconFuncWidget(function, subfunction, package)
            else:
                func_widget = TomoPyReconFunctionWidget(function, subfunction, package)
            self.recon_function = func_widget
            self.sigPipelineChanged.emit()
        elif function == 'Reader':
            func_widget = ReadFunctionWidget(function, subfunction, package)
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

    def saveState(self, datawidget):
        """
        Parameters
        ----------

        datawidget : QWidget
            QWidget (usually in the form of a TomoViewer) that holds data

        Returns
        -------
        run_state : list of four elements representing data necessary for reconstruction
            * lst: list of functools.partial which represent the function pipeline
            * theta: array of 'theta' values which represent the angles at which tomography data was taken
            * center: the center of rotation of the data
            * a string of function names and parameters to be later written into a yaml file
        """

        # extract function pipeline
        lst = []; theta = []
        for function in self.features:
            if not function.enabled or 'Reader' in function.name:
                continue
            fpartial = function.partial
            # set keywords that will be adjusted later by input functions or users
            for arg in inspect.getargspec(function._function)[0]:
                if arg not in fpartial.keywords.iterkeys() or arg in 'center':
                    fpartial.keywords[arg] = '{}'.format(arg)

            # get rid of degenerate keyword arguments
            if 'arr' in fpartial.keywords and 'tomo' in fpartial.keywords:
                fpartial.keywords['tomo'] = fpartial.keywords['arr']
                fpartial.keywords.pop('arr', None)

            # special cases for the 'write' function
            if 'start' in fpartial.keywords:
                fpartial.keywords['start'] = 'start'
            if 'Write' in function.name:
                fpartial.keywords.pop('parent folder', None)
                fpartial.keywords.pop('folder name', None)
                fpartial.keywords.pop('file name', None)

            lst.append((fpartial, function.name))

            if 'Reconstruction' in function.name: # could be bad, depending on if other operations need theta/center
                for param,ipf in function.input_functions.iteritems():
                    if not ipf.enabled:
                        if 'center' in param:
                            center = function.partial.keywords['center']
                        continue
                    # extract center value
                    if 'center' in param:
                        # this portion is taken from old updateFunctionPartial code
                        args = []
                        if ipf.subfunc_name in FunctionManager.center_func_slc:
                            map(args.append, map(datawidget.data.fabimage.__getitem__,
                                                         FunctionManager.center_func_slc[ipf.subfunc_name]))
                        else:
                            args.append(datawidget.getsino())
                        if ipf.subfunc_name == 'Nelder Mead':
                            ipf.partial.keywords['theta'] = function.input_functions['theta'].partial()
                        center = ipf.partial(*args)

                    # extract theta values
                    if 'theta' in param:
                        theta = ipf.partial()


        extract = (config.extract_pipeline_dict(self.features), config.extract_runnable_dict(self.features))
        return [lst, theta, center, extract]

    def loadDataDictionary(self, datawidget, theta, center, slc = None):
        """
        Method to load dictionary of data relevant to a reconstruction

        Parameters
        ----------
        datawidget : QWidget
            QWidget (usually in the form of a TomoViewer) that holds data
        theta : array
            Array of values which represent angles at which tomography data was taken
        center : float
            Center of rotation of data
        slc : Slice
            Slice object to extract relevant portions of sinogram/flats/darks data

        Returns
        -------
        data_dict : dict
            Dictionary of data relevant to reconstruction
        """

        data_dict = OrderedDict()

        if slc is not None and slc[0].start is not None:
            slc_ = (slice(slc[0].start, datawidget.data.shape[0] - 1, slc[0].step) if slc[0].stop is None
                    else slc[0])
            flat_loc = map_loc(slc_, datawidget.data.fabimage.flatindices())
        else:
            flat_loc = datawidget.data.fabimage.flatindices()

        data_dict['tomo'] = datawidget.getsino(slc=slc)
        data_dict['flats'] = datawidget.getflats(slc=slc)
        data_dict['dark'] = datawidget.getdarks(slc=slc)
        data_dict['flat_loc'] = flat_loc
        data_dict['theta'] = theta
        data_dict['center'] = center

        return data_dict


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

        for argname in funcwidget.missing_args: # find a more elegant way to point to the flats and darks
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

        for param, ipf in funcwidget.input_functions.iteritems():
            args = []
            if not ipf.enabled:
                continue
            if param == 'center':
                if ipf.subfunc_name in FunctionManager.center_func_slc:
                    map(args.append, map(datawidget.data.fabimage.__getitem__,
                                         FunctionManager.center_func_slc[ipf.subfunc_name]))
                else:
                    args.append(datawidget.getsino())

                if ipf.subfunc_name == 'Nelder Mead':
                    ipf.partial.keywords['theta'] = funcwidget.input_functions['theta'].partial()
            fpartial.keywords[param] = ipf.partial(*args)

            if stack_dict and param in stack_dict:  # update the stack dict with new kwargs
                stack_dict[param] = fpartial.keywords[param]


        if funcwidget.func_name in ('Padding', 'Downsample', 'Upsample'):
            self.setCenterCorrection(funcwidget.func_name, fpartial.keywords)
        elif 'Reconstruction' in funcwidget.func_name:
            fpartial.keywords['center'] = self.cor_offset(self.cor_scale(fpartial.keywords['center']))
            self.resetCenterCorrection()

        return fpartial


    def updatePartial(self, function, name, data_dict, param_dict):
        """
        Updates the given function partial's keywords - the functools.partial object is the first element of func_tuple

        Parameters
        ----------

        func_tuple : functools.partial
            functools.partial of a function in the processing pipeline
        data_dict : dict
            Dictionary which contains all information relevant to a reconstruction - its elements are loaded into the
            functools.partial's keyword arguments
        param_dict : dict
            Copy of the original keyword arguments of the functools.partial, for reference

        Return
        ------

        function : functools.partial
            Function partial with all keyword arguments loaded
        write : str
            Name of the array function will act on - this allows users to specify which data a function will act on
            in the pipeline
        """

        write = 'tomo'

        for key, val in param_dict.iteritems():
            if val in data_dict.iterkeys():
                if 'arr' in key or 'tomo' in key:
                    write = val
                function.keywords[key] = data_dict[val]

        if name in ('Padding', 'Downsample', 'Upsample'):
            self.setCenterCorrection(name, function.keywords)
        if 'Reconstruction' in name:
            function.keywords['center'] = self.cor_offset(self.cor_scale(function.keywords['center']))
            self.resetCenterCorrection()
        return function, write

    def loadPreviewData(self, datawidget, slc=None, ncore=None, skip_names=['Write', 'Reader'],
                        fixed_func=None, prange=None):
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
        partial_stack : list of partials:
            List with function partials needed to run preview
        self.stack_dict : dict
            Dictionary summarizing functions and parameters representing the pipeline (used for the list of partials)
        data_dict : dict
            Dictionary of data necessary to run a reconstruction preview
        """

        stack_dict = OrderedDict()
        partial_stack = []
        self.lockParams(True)

        func_pipeline, theta, center, yaml_pipe = self.saveState(datawidget)

        # set up dictionary of function keywords
        params_dict = OrderedDict()
        for tuple in func_pipeline:
            params_dict['{}'.format(tuple[1])] = dict(tuple[0].keywords)

        # load data dictionary
        data_dict = self.loadDataDictionary(datawidget, theta, center, slc = slc)

        count = 1
        for func in self.features:
            name = func.name
            func_name = str(count) + ". " + func.func_name
            if not func.enabled:
                continue
            elif func.func_name in skip_names:
                stack_dict[func_name] = {func.subfunc_name: deepcopy(func.exposed_param_dict)}
                count += 1
                continue
            elif fixed_func is not None and func.func_name == fixed_func.func_name:
                func = fixed_func
                for key, val in func.exposed_param_dict.iteritems():
                    if key in 'center':
                        data_dict[key] = val
                    elif key in params_dict[name].iterkeys() and key not in 'center':
                        params_dict[name][key] = val
            stack_dict[func_name] = {func.subfunc_name: deepcopy(func.exposed_param_dict)}
            count += 1

            # load partial_stack
            fpartial = func.partial
            for arg in inspect.getargspec(func._function)[0]:
                if arg not in fpartial.keywords.iterkeys() or arg in 'center':
                    fpartial.keywords[arg] = '{}'.format(arg)
            # get rid of degenerate keyword arguments
            if 'arr' in fpartial.keywords and 'tomo' in fpartial.keywords:
                fpartial.keywords['tomo'] = fpartial.keywords['arr']
                fpartial.keywords.pop('arr', None)

            # if 'ncore' in fpartial.keywords:
            #     fpartial.keywords['ncore'] = ncore
            partial_stack.append((fpartial, name, params_dict[name]))

            for param, ipf in func.input_functions.iteritems():
                if ipf.enabled:
                    if 'Input Functions' not in stack_dict[func_name][func.subfunc_name]:
                        stack_dict[func_name][func.subfunc_name]['Input Functions'] = {}
                    ipf_dict = {param: {ipf.func_name: {ipf.subfunc_name: ipf.exposed_param_dict}}}
                    stack_dict[func_name][func.subfunc_name]['Input Functions'].update(ipf_dict)

                    # update input function keywords in slice preview table
                    if param in stack_dict[func_name][func.subfunc_name]:
                        stack_dict[func_name][func.subfunc_name][param] = data_dict[param]

        self.lockParams(False)
        return partial_stack, stack_dict, data_dict, prange



    def reconGenerator(self, datawidget, run_state, proj, sino, sino_p_chunk, ncore = None):

        """
        Generator for running full reconstruction. Yields messages representing the status of reconstruction
        This is ideally used as a threads.method or the corresponding threads.RunnableIterator.

        Parameters
        ----------
        datawidget : QWidget
            QWidget (usually in the form of a TomoViewer) that holds data
        run_state : list of four elements
            * list of functools.partial which represent the function pipeline
            * array of 'theta' values which represent the angles at which tomography data was taken
            * the center of rotation of the data
            * a string of function names and parameters to be later written into a yaml file
        proj : tuple of int
            Projection range indices (start, end, step)
        sino : tuple of int
            Sinogram range indices (start, end, step)
        sino_p_chunk : int
            Number of sinograms per chunk
        ncore : int
            Number of cores to run functions
        pipeline_dict: dictionary
            Dictionary of parameters referenced during reconstruction

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

        func_pipeline, theta, center, extract = run_state
        yaml_pipe = extract[0]

        # set up dictionary of function keywords
        params_dict = OrderedDict()
        for tuple in func_pipeline:
            params_dict['{}'.format(tuple[1])] = dict(tuple[0].keywords)


        # get save names for pipeline yaml/runnable files
        dir = ""
        for function_tuple in func_pipeline:
            if 'fname' in function_tuple[0].keywords:
                fname = function_tuple[0].keywords['fname']
                for item in fname.split('/')[:-1]:
                    dir += item + '/'
                yml_file = fname + '.yml'
                python_file = fname + '.py'

        # make project directory if it isn't made already
        if not os.path.exists(dir):
            os.makedirs(dir)

        # save function pipeline as runnable
        path = datawidget.path
        runnable = self.extractPipelineRunnable(run_state, params_dict, proj, sino, sino_p_chunk, path, ncore)
        try:
            with open(python_file, 'w') as py:
                py.write(runnable)
        except NameError or IOError:
            yield "Error: pipeline runnable not written - path could not be found"


        # save yaml in reconstruction folder
        for key in yaml_pipe.iterkeys(): # special case for 'center' param
            if 'Recon' in key:
                for subfunc in yaml_pipe[key].iterkeys():
                    if 'Parameters' in yaml_pipe[key][subfunc].iterkeys():
                        yaml_pipe[key][subfunc]['Parameters']['center'] = float(center)


        try:
            with open(yml_file, 'w') as yml:
                yamlmod.ordered_dump(yaml_pipe, yml)
        except NameError or IOError:
            yield "Error: function pipeline yaml not written - path could not be found"

        for i in range(nchunk):

            start, end  = i * sino[2] * sino_p_chunk + sino[0], (i + 1) * sino[2] * sino_p_chunk + sino[0]
            end = end if end < sino[1] else sino[1]

            slc = (slice(*proj), slice(start, end, sino[2]), slice(None, None, None))


            # load data dictionary
            data_dict = self.loadDataDictionary(datawidget, theta, center, slc = slc)
            data_dict['start'] = write_start
            shape = data_dict['tomo'].shape[1]



            for function_tuple in func_pipeline:
                ts = time.time()
                name = function_tuple[1]
                function = function_tuple[0]

                yield 'Running {0} on slices {1} to {2} from a total of {3} slices...'.format(function_tuple[1],
                                                                                              start, end, total_sino)
                function, write = self.updatePartial(function, name, data_dict, params_dict[name])
                data_dict[write] = function()

                yield ' Finished in {:.3f} s\n'.format(time.time() - ts)

            write_start += shape
            del data_dict


        # print final 'finished with recon' message
        yield 'Reconstruction complete. Run time: {:.2f} s'.format(time.time() - start_time)



    def foldSliceStack(self, partial_stack, data_dict):
        """
        Method to run a reconstruction given a list of function partials and the data for functions to act on

        Parameters
        ----------
         partial_stack : list of 3-tuples
            Contains: a list of tuples, each of which have as elements: functools.partial of full keywords to run,
            the name of the associated function, and a copy of the params belonging to that function
        data_dict : dict
            Dictionary containing all data needed to run a reconstruction

        Returns
        -------
        Returns the 'tomo' data in the data_dict, which has been acted on by all functions in the partial_stack
        """
        for tuple in partial_stack:
            function, write = self.updatePartial(tuple[0], tuple[1], data_dict, tuple[2])
            data_dict[write] = function()


        return data_dict['tomo']


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


    def previewFunctionStack(self, datawidget, slc=None, ncore=None, skip_names=['Write'], fixed_func=None):

        """
        Create the function stack and summary dictionary used for running slice previews and 3D previews

        Deprecated : functionality replaced by self.loadPreviewData


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
        return partial_stack, self.stack_dict


    def functionStackGenerator(self, datawidget, pipeline_dict, proj, sino, sino_p_chunk, ncore=None):
        """
        Generator for running full reconstruction. Yields messages representing the status of reconstruction
        This is ideally used as a threads.method or the corresponding threads.RunnableIterator.

        Deprecated : functionality replaced by self.reconGenerator


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
        pipeline_dict: dictionary
            Dictionary of parameters referenced during reconstruction

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
                    fpartial.keywords.pop('parent folder', None)
                    fpartial.keywords.pop('folder name', None)
                    fpartial.keywords.pop('file name', None)
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
            yield "Error: function pipeline yaml not written - path could not be found"

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
        if function.func_name in 'Reader':
            return
        for i in prange:
            function.param_dict[parameter] = i
            # Dynamic FixedFunc "dummed down" FuncWidget class. cool.
            fixed_func = type('FixedFunc', (), {'func_name': function.func_name, 'subfunc_name': function.subfunc_name,
                                                'missing_args': function.missing_args,
                                                'param_dict': function.param_dict,
                                                'exposed_param_dict': function.exposed_param_dict,
                                                'partial': function.partial,
                                                'input_functions': function.input_functions,
                                                '_function': function._function})
            self.sigTestRange.emit('Computing preview for {} parameter {}={}...'.format(function.name, parameter, i),
                                   fixed_func, {'function': function.func_name, parameter: prange})



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
            try:
                func = func.split(". ")[1]
            except IndexError:
                func = func
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

    def extractPipelineRunnable(self, run_state, params, proj, sino, sino_p_chunk, path, ncore=None):
        """
        Saves the function pipeline as a runnable (Python) file.

        Parameters
        ----------
        pipeline : dict
            Dictionary of functions and their necessary parameters to write the function information
        """


        signature = "import time \nimport tomopy \nimport dxchange\nimport h5py\n" \
                    "import numpy as np\nimport numexpr as ne\n\n"

        # set up function pipeline
        runnable_pipe = run_state[3][1]
        func_dict = runnable_pipe['func']
        subfunc_dict = runnable_pipe['subfunc']
        center = run_state[2]

        signature += "def main():\n\n"
        signature += "\t# offset and scale factors in case of padding, upsampling, or downsampling\n"
        signature += "\tcor_offset = 0\n"
        signature += "\tcor_scale = 0\n\n"
        signature += "\tstart_time = time.time()\n\n"

        signature += "\n\tdata = dxchange.read_als_832h5('{}')\n".format(path)
        signature += "\ttomo=data[0]; flats=data[1]; dark=data[2]\n"
        signature += "\tmdata = read_als_832h5_metadata('{}')\n\n".format(path)
        signature += "\t# choose which projections and sinograms go into the reconstruction:\n"
        signature += "\tproj_start = {}; proj_end = {}; proj_step = {}\n".format(proj[0],proj[1],proj[2])
        signature += "\tsino_start = {}; sino_end = {}; sino_step = {}\n".format(sino[0],sino[1],sino[2])
        signature += "\tsino_p_chunk = {2} # chunk size of data during reconstruction\n\n".format(proj, sino, sino_p_chunk)
        signature += "\tproj = (proj_start, proj_end, proj_step)\n"
        signature += "\tsino = (sino_start, sino_end, sino_step)\n\n"
        signature += "\twrite_start = sino[0]\n"
        signature += "\tnchunk = ((sino[1]-sino[0]) // sino[2] - 1) // sino_p_chunk +1\n"
        signature += "\ttotal_sino = (sino[1] - sino[0] - 1) // sino[2] + 1\n"
        signature += "\tif total_sino < sino_p_chunk:\n\t\tsino_p_chunk = total_sino\n\n"

        if 'center' not in subfunc_dict.iterkeys():
            signature += "\tcenter = {}\n".format(center)
        for key, val in subfunc_dict.iteritems():
            signature += "\t{} = {}\n".format(key, val)

        signature += "\n\n\t# MAIN LOOP FOR RECONSTRUCTION\n"
        signature += "\tfor i in range(nchunk):\n"
        signature += "\t\tstart, end = i * sino[2] * sino_p_chunk + sino[0], (i + 1) * sino[2] * sino_p_chunk + sino[0]\n"
        signature += "\t\tend = end if end < sino[1] else sino[1]\n\n"
        signature += "\t\tslc = (slice(*proj), slice(start,end,sino[2]), slice(None, None, None))\n"

        signature += "\t\tdata_dict = loadDataDict(data, mdata, theta, center, slc)\n"
        signature += "\t\tdata_dict['start'] = write_start\n"
        signature += "\t\tshape = data_dict['tomo'].shape[1]\n\n"

        signature += "\t\t# the function pipeline: keywords used in each function are located in the\n"
        signature += "\t\t# 'params' assignment for each function\n\n"
        for func, param_dict in func_dict.iteritems():
            try:
                func = func.split(". ")[1]
            except IndexError:
                pass
            signature += "\t\t# function: {}\n".format(func)
            signature += "\t\tts = time.time()\n"
            signature += "\t\tprint 'Running {0} on slices {1} to {2} from a total of {3} slices'.format("
            signature += "'{}', start, end, total_sino)\n ".format('{}'.format(func))
            signature += "\t\tparams = {}\n".format(param_dict)
            signature += "\t\tkwargs, write, cor_offset, cor_scale = updateKeywords('{}', params, data_dict,".format(func)
            signature += " cor_offset, cor_scale)\n"
            signature += "\t\tdata_dict[write] = {}(**kwargs)\n".format(func)
            signature += "\t\tprint 'Finished in {:.3f} s'.format(time.time()-ts)\n"
            signature += "\t\tprint "" #white space \n\n"
        signature += "\t\twrite_start += shape\n\n"
        signature += "\tprint 'Reconstruction complete. Run time: {:.2f} s'.format(time.time()-start_time)\n"
        signature += "\tprint # white space\n\n"

        # rewrite functions used for processing
        signature += "# helper functions\n\n"
        signature += "def read_als_832h5_metadata(fname):\n"
        signature += "\t with h5py.File(fname, 'r') as f:\n\t\tg=_find_dataset_group(f)\n\t\treturn dict(g.attrs)\n\n"

        signature += "def _find_dataset_group(h5object):\n"
        signature += "\tkeys = h5object.keys()\n \tif len(keys)==1:\n"
        signature += "\t\tif isinstance(h5object[keys[0]], h5py.Group):\n"
        signature += "\t\t\tgroup_keys = h5object[keys[0]].keys()\n"
        signature += "\t\t\tif isinstance(h5object[keys[0]][group_keys[0]], h5py.Dataset):\n"
        signature += "\t\t\t\treturn h5object[keys[0]]\n"
        signature += "\t\t\telse:\n\t\t\t\treturn _find_dataset_group(h5object[keys[0]])\n"
        signature += "\t\telse:\n\t\t\traise Exception('Unable to find dataset group')\n"
        signature += "\telse:\n\t\traise Exception('Unable to find dataset group')\n\n"

        signature += "def flatindices(mdata):\n"
        signature += "\ti0 = int(mdata['i0cycle'])\n\tnproj = int(mdata['nangles'])\n"
        signature += "\tif i0 > 0:\n\t\tindices = list(range(0, nproj, i0))\n"
        signature += "\t\tif indices[-1] != nproj - 1:\n\t\t\tindices.append(nproj - 1)\n"
        signature += "\telif i0 == 0:\n\t\tindices = [0, nproj - 1]\n\treturn indices\n\n"

        signature += "# sets COR correction in case of padding, upsample, or downsample\n"
        signature += "def setCenterCorrection(name, param_dict, cor_offset, cor_scale):\n"
        signature += "\tif 'pad' in name and param_dict['axis'] == 2:\n"
        signature += "\t\tn = param_dict['npad']\n"
        signature += "\t\tcor_offset = n\n"
        signature += "\telif 'downsample' in name and param_dict['axis'] == 2:\n"
        signature += "\t\ts = param_dict['level']\n"
        signature += "\t\tcor_scale = -s\n"
        signature += "\telif 'upsample' in name and param_dict['axis'] == 2:\n"
        signature += "\t\ts = param_dict['level']\n"
        signature += "\t\tcor_scale = -s\n"
        signature += "\treturn cor_offset, cor_scale\n\n"
        signature += "def resetCenterCorrection(cor_offset, cor_scale):\n"
        signature += "\tcor_offset = 0\n\tcor_scale = 0\n"
        signature += "\treturn cor_offset, cor_scale\n\n"

        signature += "# performs COR correction\n"
        signature += "def correctCenter(center, cor_offset, cor_scale):\n"
        signature += "\tif cor_scale<0:\n\t\treturn float(int(center * 2 ** cor_scale)) + cor_offset\n"
        signature += "\telse:\n\t\treturn (center * 2 ** cor_scale) + cor_offset\n\n"


        signature += "def map_loc(slc,loc):\n\tstep = slc.step if slc.step is not None else 1\n"
        signature += "\tind = range(slc.start, slc.stop, step)\n\tloc = np.array(loc)\n\tlow, upp = ind[0], ind[-1]\n"
        signature += "\tbuff = (loc[-1] - loc[0]) / len(loc)\n\tmin_loc = low - buff\n\tmax_loc = upp + buff\n"
        signature += "\tloc = np.intersect1d(loc[loc > min_loc], loc[loc < max_loc])\n\tnew_upp = len(ind)\n"
        signature += "\tloc = (new_upp * (loc - low)) // (upp - low)\n\tif loc[0] < 0:\n\t\tloc[0] = 0\n"
        signature += "\treturn np.ndarray.tolist(loc)\n\n"

        # function for loading data dictionary
        signature += "def loadDataDict(data, mdata, theta,center,slc=None):\n\tdata_dict = {}\n"
        signature += "\tif slc is not None and slc[0].start is not None:\n"
        signature += "\t\tslc_ = slice(slc[0].start,data[0].shape[0],slc[0].step)\n"
        signature += "\t\tflat_loc = map_loc(slc_, flatindices(mdata))\n"
        signature += "\telse:\n\t\tflat_loc = flatindices(mdata)\n\n"
        signature += "\tdata_dict['tomo'] = data[0][slc]\n\tdata_dict['flats'] = data[1][slc]\n"
        signature += "\tdata_dict['dark'] = data[2][slc]\n\tdata_dict['flat_loc'] = flat_loc\n"
        signature += "\tdata_dict['theta'] = theta\n\tdata_dict['center'] = center\n\n"
        signature += "\treturn data_dict\n\n"

        signature += "def updateKeywords(function, param_dict, data_dict, cor_offset, cor_scale):\n"
        signature += "\twrite = 'tomo'\n"
        signature += "\tfor key, val in param_dict.iteritems():\n"
        signature += "\t\tif val in data_dict.iterkeys():\n"
        signature += "\t\t\tif 'arr' in key or 'tomo' in key:\n"
        signature += "\t\t\t\twrite = val\n"
        signature += "\t\t\tparam_dict[key] = data_dict[val]\n"
        signature += "\tfor item in ('pad', 'downsample', 'upsample'):\n"
        signature += "\t\tif item in function:\n "
        signature += "\t\t\tcor_offset, cor_scale = setCenterCorrection(function, param_dict, cor_offset, cor_scale)\n"
        signature += "\tif 'recon' in function:\n"
        signature += "\t\tparam_dict['center'] = correctCenter(param_dict['center'], cor_offset, cor_scale)\n"
        signature += "\t\tcor_offset, cor_scale = resetCenterCorrection(cor_offset, cor_scale)\n"
        signature += "\treturn param_dict, write, cor_offset, cor_scale\n\n\n"

        # write custom functions as functions in python file
        signature += "# the following three functions may be used in the reconstruction pipeline\n"
        signature += "def crop(arr, p11, p12, p21, p22, axis=0):\n"
        signature += "\tslc = []\n"
        signature += "\tpts = [p11, p12, p21, p22]\n"
        signature += "\tfor n in range(len(arr.shape)):\n"
        signature += "\t\tif n == axis:\n"
        signature += "\t\t\tslc.append(slice(None))\n"
        signature += "\t\telse:\n"
        signature += "\t\t\tslc.append(slice(pts.pop(0), -pts.pop(0)))\n"
        signature += "\treturn arr[slc]\n\n"

        signature += "def convert_data(arr, imin=None, imax=None, dtype='uint8', intcast='float32'):\n"
        signature += "\tDTYPE_RANGE = {'uint8': (0, 255), 'uint16': (0, 65535), 'int8': (-128, 127),"
        signature += "'int16': (-32768, 32767),'float32': (-1, 1),'float64': (-1, 1)}\n"
        signature += "\tallowed_dtypes = ('uint8', 'uint16', 'int8', 'int16', 'float32', 'float64')\n"
        signature += "\tif dtype not in allowed_dtypes:\n"
        signature += "\t\traise ValueError('dtype keyword {0} not in allowed keywords {1}'.format(dtype, allowed_dtypes))\n\n"

        signature += "\t# Determine range to cast values\n"
        signature += "\tminset = False\n"
        signature += "\tif imin is None:\n"
        signature += "\t\timin = np.min(arr)\n"
        signature += "\t\tminset = True\n"
        signature += "\tmaxset = False\n"
        signature += " \tif imax is None:\n"
        signature += "\t\timax = np.max(arr)\n"
        signature += "\t\tmaxset = True\n\n"

        signature += "\tnp_cast = getattr(np, str(arr.dtype))\n"
        signature += "\timin, imax = np_cast(imin), np_cast(imax)\n"

        signature += "\t# Determine range of new dtype\n"
        signature += "\tomin, omax = DTYPE_RANGE[dtype]\n"
        signature += "\tomin = 0 if imin >= 0 else omin\n"
        signature += "\tomin, omax = np_cast(omin), np_cast(omax)\n"

        signature += "\tif arr.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,"
        signature += "np.uint32, np.uint64, np.bool_, np.int_, np.intc, np.intp]:\n"
        signature += "\t\tint_cast = getattr(np, str(intcast))\n"
        signature += "\t\tout = np.empty(arr.shape, dtype=int_cast)\n"
        signature += "\t\timin = int_cast(imin)\n"
        signature += "\t\timax = int_cast(imax)\n"
        signature += "\t\tdf = int_cast(imax) - int_cast(imin)\n"
        signature += "\telse:\n"
        signature += "\t\tout = np.empty(arr.shape, dtype=arr.dtype)\n"
        signature += "\t\tdf = imax - imin\n"
        signature += "\tif not minset:\n"
        signature += "\t\tif np.min(arr) < imin:\n"
        signature += "\t\t\tarr = ne.evaluate('where(arr < imin, imin, arr)', out=out)\n"
        signature += "\tif not maxset:\n"
        signature += "\t\tif np.max(arr) > imax:\n"
        signature += "\t\t\tarr = ne.evaluate('where(arr > imax, imax, arr)', out=out)\n"
        signature += "\tne.evaluate('(arr - imin) / df', truediv=True, out=out)\n"
        signature += "\tne.evaluate('out * (omax - omin) + omin', out=out)\n\n"

        signature += "\t# Cast data to specified type\n"
        signature += "\treturn out.astype(np.dtype(dtype), copy=False)\n\n"

        signature += """
def array_operation_add(arr, value=0):
    return ne.evaluate('arr + value')

def array_operation_sub(arr, value=0):
    return ne.evaluate('arr - value', truediv=True)

def array_operation_mult(arr, value=1):
    return ne.evaluate('arr * value')

def array_operation_div(arr, value=1):
    return ne.evaluate('arr / value')

def array_operation_max(arr, value=0):
    return np.maximum(arr, value)"""

        signature += "\n\nif __name__ == '__main__':\n\tmain()"

        return signature



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
