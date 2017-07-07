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
import fabio
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from modpkgs import yamlmod
import numpy as np
import pyqtgraph as pg
from PySide import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import config
import reconpkg
import ui
from xicam.widgets import featurewidgets as fw
from functionmanager import TestRangeDialog


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
        self.params = Parameter.create(name=self.name, children=config.parameters[self.subfunc_name], type='group',
                                       readonly=False)

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

        self.allowed_types = {'str': str, 'int': int, 'float': float, 'bool': bool, 'unicode': unicode}

        self.expand()

    # make it so function widgets never collapse
    def collapse(self):
        pass

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

        if hasattr(self, 'defaults'):
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
        if hasattr(self, 'defaults'):
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

class AnglesFunctionWidget(FunctionWidget):

    @property
    def updated_param_dict(self):

        param_dict = {}
        param_dict['ang1'] = float(self.param_dict['ang1'] + self.param_dict['range'])
        param_dict['ang2'] = float(self.param_dict['ang1'])
        param_dict['nang'] = self.param_dict['nang']

        return param_dict

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
        self.input_functions = {'theta': AnglesFunctionWidget('Projection Angles', 'Projection Angles', closeable=False,
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

        if subname in config.function_defaults.keys():
            self.defaults = config.function_defaults[subname]
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

        if subname in config.function_defaults.keys():
            self.defaults = config.function_defaults[subname]
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

class MaskFunctionWidget(FunctionWidget):
    """
    Subclass of FunctionWidget for functions with masks.
    """

    def __init__(self, name, subname, package, checkable=True, closeable=True, parent=None):
        super(MaskFunctionWidget, self).__init__(name, subname, package, checkable=checkable, closeable=closeable,
                                                 parent=parent)
        self.params.child('Browse mask image').sigActivated.connect(self.setBrowse)
        self.params.child('mask').sigValueChanged.connect(self.showOptions)
        self.params.child('mask path').hide()

    def showOptions(self):
        if self.params.child('mask').value() == 'custom mask':
            self.params.child('mask path').show()
        else:
            self.params.child('mask path').hide()
            if self.params.child('mask').value() == 'StructuredElementL':
                self.params.child('L').show()
            else:
                self.params.child('L').hide()

    def setBrowse(self):
        path = str(QtGui.QFileDialog.getOpenFileName(None, 'Choose mask image')[0])
        try:
            self.mask_image = fabio.open(path).data
            self.params.param('mask path').show()
            self.params.param('mask path').setValue(path)
            self.params.param('mask').setValue('custom mask')
        except IOError:
            pass

    @property
    def updated_param_dict(self):
        param_dict = {}
        if self.params.param('mask').value() == 'custom mask':
            param_dict['mask'] = self.mask_image
        else:
            param_dict['mask'] = self.params.param('mask').value()
        param_dict['image'] = self.param_dict['image']
        param_dict['L'] = self.param_dict['L']
        param_dict['platform'] = self.param_dict['platform']
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
    def sino_chunk(self):
        return self.params.child('sinograms_per_chunk').value()

    @property
    def proj_chunk(self):
        return self.params.child('projections_per_chunk').value()

    @property
    def width(self):
        return (self.params.child('start_width').value(), self.params.child('end_width').value(),
                self.params.child('step_width').value())


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
        # this line causes the 'invalid keyword argument - readonly' error during a reconstructin preview
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


class CORSelectionWidget(QtGui.QWidget):

    cor_detection_funcs = ['Phase Correlation', 'Vo', 'Nelder-Mead']
    sigCORFuncChanged = QtCore.Signal(str, QtGui.QWidget)

    def __init__(self, subname='Phase Correlation', parent=None):
        super(CORSelectionWidget, self).__init__(parent=parent)

        self.layout = QtGui.QVBoxLayout()
        self.function = FunctionWidget(name="Center Detection", subname=subname,
                                package=reconpkg.packages[config.names[subname][1]])
        self.params = pg.parametertree.Parameter.create(name=self.function.name,
                                             children=config.parameters[self.function.subfunc_name], type='group')

        self.param_tree = pg.parametertree.ParameterTree()
        self.param_tree.setMinimumHeight(200)
        self.param_tree.setMinimumWidth(200)
        self.param_tree.setParameters(self.params, showTop=False)
        for key, val in self.function.param_dict.iteritems():
            if key in [p.name() for p in self.params.children()]:
                self.params.child(key).setValue(val)
                self.params.child(key).setDefault(val)

        self.method_box = QtGui.QComboBox()
        self.method_box.currentIndexChanged.connect(self.changeFunction)
        for item in self.cor_detection_funcs:
            self.method_box.addItem(item)
        self.method_box.currentIndexChanged.connect(self.corFuncChanged)

        label = QtGui.QLabel('COR detection function: ')
        method_layout = QtGui.QHBoxLayout()
        method_layout.addWidget(label)
        method_layout.addWidget(self.method_box)

        self.layout.addLayout(method_layout)
        self.layout.addWidget(self.param_tree)
        self.setLayout(self.layout)

    def corFuncChanged(self, index):
        self.sigCORFuncChanged.emit(self.cor_detection_funcs[index], self)

    def changeFunction(self, index):
        subname = self.method_box.itemText(index)
        self.layout.removeWidget(self.param_tree)

        self.function = FunctionWidget(name="Center Detection", subname=subname,
                                package=reconpkg.packages[config.names[subname][1]])
        self.params = pg.parametertree.Parameter.create(name=self.function.name,
                                             children=config.parameters[self.function.subfunc_name], type='group')
        self.param_tree = pg.parametertree.ParameterTree()
        self.param_tree.setMinimumHeight(200)
        self.param_tree.setMinimumWidth(200)
        self.param_tree.setParameters(self.params,showTop = False)
        for key, val in self.function.param_dict.iteritems():
            if key in [p.name() for p in self.params.children()]:
                self.params.child(key).setValue(val)
                self.params.child(key).setDefault(val)

        self.layout.addWidget(self.param_tree)
        self.setLayout(self.layout)