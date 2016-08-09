# -*- coding: utf-8 -*-

from copy import deepcopy
from functools import partial
import inspect
import numpy as np
from collections import OrderedDict
from PySide import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
from xicam import threads
import config
import manager
import reconpkg
import ui
from xicam import msg
import ftrwidgets as fw


class FunctionWidget(fw.FeatureWidget):
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
        self._partial = None

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
        print self.subfunc_name, self.param_dict
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
        if self.previewButton.isChecked() or not self.previewButton.isCheckable():
            return True
        return False

    @enabled.setter
    def enabled(self, val):
        if val and self.previewButton.isCheckable():
            self.previewButton.setChecked(True)
        else:
            self.previewButton.setChecked(False)

    def updateParamsDict(self):
        self.param_dict.update({param.name(): param.value() for param in self.params.children()})
        for p, ipf in self.input_functions.iteritems():
            ipf.updateParamsDict()

    @property
    def partial(self):
        self._partial = partial(self._function, **self.param_dict)
        return self._partial

    @partial.setter
    def partial(self, p):
        self._partial = p

    def addInputFunction(self, parameter, functionwidget):
        self.input_functions[parameter] = functionwidget
        self.addSubFeature(functionwidget)

    @property
    def input_partials(self):
        return None

    @property
    def func_signature(self):
        signature = str(self._function.__name__) + '('
        for arg in self.missing_args:
            signature += '{},'.format(arg)
        for param, value in self.param_dict.iteritems():
            signature += '{0}={1},'.format(param, value) if not isinstance(value, str) else \
                '{0}=\'{1}\','.format(param, value)
        return signature[:-1] + ')'

    def paramChanged(self, param):
        self.param_dict.update({param.name(): param.value()})

    # def getParamDict(self, update=True):
    #     if update:
    #         self.updateParamsDict()
    #     return self.param_dict

    def allReadOnly(self, boolean):
        for param in self.params.children():
            param.setReadonly(boolean)

    def menuRequested(self):
        # Menus when the function widget is right clicked
        pass

    def paramMenuRequested(self, pos):
        # Menus when a parameter in the form is right clicked
        if self.form.currentItem().parent():
            self.parammenu.exec_(self.form.mapToGlobal(pos))

    def testParamTriggered(self):
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
            widget = ui.centerwidget.currentWidget()
            if widget is None: return
            self.updateParamsDict()
            msg.showMessage('Computing previews for {}:{} parameter range...'.format(self.subfunc_name,
                                                                                     param.name()), timeout=0)
            for i in test.selectedRange():
                self.param_dict[param.name()] = i
                manager.pipeline_preview_action(widget, ui.centerwidget.currentWidget().addSlicePreview, update=False,
                                                fixed_funcs={self.subfunc_name: [deepcopy(self.param_dict),
                                                                                  deepcopy(self.partial)]})

class ReconFunctionWidget(FunctionWidget):
    def __init__(self, name, subname, package):

        self.packagename = package.__name__
        self.input_functions = {'theta': FunctionWidget('Projection Angles', 'Projection Angles', closeable=False,
                                                  package=reconpkg.packages['tomopy'], checkable=False)}
        super(ReconFunctionWidget, self).__init__(name, subname, package, input_functions=self.input_functions,
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
        kwargs = deepcopy(self.param_dict)
        if 'cutoff' in kwargs.keys() and 'order' in kwargs.keys():
            kwargs['filter_par'] = list((kwargs.pop('cutoff'), kwargs.pop('order')))
        self._partial = partial(self._function, **kwargs)
        return self._partial

    @property
    def input_partials(self):
        p = []
        if self.center is None or not self.center.previewButton.isChecked():
            p.append((None, None, None))
        else:
            if self.center.subfunc_name == 'Phase Correlation':
                slices = (0, -1)
            else:
                slices = (slice(None, ui.centerwidget.currentWidget().sinogramViewer.currentIndex),)
            if self.center.subfunc_name == 'Nelder-Mead':
                p.append(('center', slices, partial(self.center.partial, theta=self.angles.partial())))
            else:
                p.append(('center', slices, self.center.partial))
        p.append(('theta', None, self.angles.partial))
        return p

    def resetCenter(self):
        self.center = None
        self.input_functions = [self.center, self.angles]

    def addCenterDetectFunction(self, func, subfunc, package=reconpkg.packages['tomopy']):
        self.input_functions['center'] = FunctionWidget(func, subfunc, package=package)
        self.addInputFunction('center', self.input_functions['center'])

    def setCenterParam(self, value):
        self.params.child('center').setValue(value)
        self.params.child('center').setDefault(value)

    def menuRequested(self, pos):
        self.menu.exec_(self.previewButton.mapToGlobal(pos))


class AstraReconFuncWidget(ReconFunctionWidget):
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
        return FunctionWidget.partial(self)
        # self._partial = partial(self._function, **self.param_dict)
        # return self._partial



class TestRangeDialog(QtGui.QDialog):
    """
    Simple QDialgog subclass with three spinBoxes to inter start, end, step for a range to test a particular function
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
    Simple QDialgog subclass with comboBox and lineEdit to choose from a list of available function parameter keywords
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
    Class to manage tomography workflow/pipeline FunctionWidgets
    """

    center_func_slc = {'Phase Correlation': (0, -1)}

    def __init__(self, list_layout, form_layout, function_widgets=None, blank_form=None):
        super(FunctionManager, self).__init__(list_layout, form_layout, feature_widgets=function_widgets,
                                              blank_form=blank_form)
        self.cor_offset = lambda x: x  # dummy
        self.cor_scale = lambda x: x  # dummy
        self.recon_function = None
        self.pipeline_yaml = {}

    # TODO fix this astra check raise error if package not available
    def addFunction(self, function, subfunction, package):
        if function == 'Reconstruction':
            if 'astra' in reconpkg.packages and package == reconpkg.packages['astra']:
                func_widget = AstraReconFuncWidget(function, subfunction, package)
            else:
                func_widget = ReconFunctionWidget(function, subfunction, package)
            self.recon_function = func_widget
        else:
            func_widget = FunctionWidget(function, subfunction, package)
        self.addFeature(func_widget)
        return func_widget  #TODO why do i return this?

    def addInputFunction(self, funcwidget, parameter, function, subfunction, package, **kwargs):
        ipf_widget = FunctionWidget(function, subfunction, package, **kwargs)
        funcwidget.addInputFunction(parameter, ipf_widget)
        return ipf_widget

    def updateParameters(self):
        for function in self.features:
            function.updateParamsDict()

    def lockParams(self, boolean):
        for func in self.features:
            func.allReadOnly(boolean)

    def resetCenterCorrection(self):
        self.cor_offset = lambda x: x  # dummy
        self.cor_scale = lambda x: x  # dummy

    def setCenterCorrection(self, name, param_dict):
        if 'Padding' in name and param_dict['axis'] == 2:
            n = param_dict['npad']
            self.cor_offset = lambda x: x + n
        elif 'Downsample' in name and param_dict['axis'] == 2:
            s = param_dict['level']
            self.cor_scale = lambda x: x / 2 ** s
        elif 'Upsample' in name and param_dict['axis'] == 2:
            s = param_dict['level']
            self.cor_scale = lambda x: x * 2 ** s


    def updateFunctionPartial(self, funcwidget, datawidget, stack_dict, slc):
        fpartial = funcwidget.partial
        for argname in funcwidget.missing_args:
            if argname in 'flats':
                fpartial.keywords[argname] = datawidget.getflats(slc=slc)
            if argname in 'darks':
                fpartial.keywords[argname] = datawidget.getdarks(slc=slc)
        for param, ipf in funcwidget.input_functions.iteritems():
            args = []
            if not ipf.enabled:
                continue
            if param == 'center':  # Need to find a cleaner solution to this
                if ipf.subfunc_name in FunctionManager.center_func_slc:
                    map(args.append, map(datawidget.data.fabimage.__getitem__,
                                         FunctionManager.center_func_slc[ipf.subfunc_name]))
                else:
                    args.append(datawidget.getsino())
                if ipf.subfunc_name == 'Nelder Mead':  # Also needs a cleaner solution
                    ipf.partial.keywords['theta'] = funcwidget.input_functions['theta'].partial()
            fpartial.keywords[param] = ipf.partial(*args)
            stack_dict[param] = fpartial.keywords[param]
        if funcwidget.func_name in ('Padding', 'Downsample', 'Upsample'):
            self.setCenterCorrection(funcwidget.func_name, fpartial.keywords)
        elif 'Reconstruction' in funcwidget.func_name:
            fpartial.keywords['center'] = self.cor_offset(self.cor_scale(fpartial.keywords['center']))
            self.resetCenterCorrection()
        return fpartial

    def previewFunctionStack(self, datawidget, slc=None, ncore=None):
        stack_dict = OrderedDict()
        partial_stack = []
        self.lockParams(True)
        for func in self.features:
            if not func.enabled:
                continue
            elif func.func_name == 'Write':
                stack_dict[func.func_name] = {func.subfunc_name: deepcopy(func.param_dict)}
                continue
            stack_dict[func.func_name] = {func.subfunc_name: deepcopy(func.param_dict)}
            p = self.updateFunctionPartial(func, datawidget, stack_dict[func.func_name][func.subfunc_name], slc)
            # stack_dict[func.func_name][func.subfunc_name].update(p.keywords)
            if 'ncore' in p.keywords:
                p.keywords['ncore'] = ncore
            partial_stack.append(p)
            if func.input_functions:
                ipf_dict = {ipf.subfunc_name: deepcopy(ipf.param_dict) for ipf in func.input_functions.values()
                            if ipf.enabled}
                stack_dict[func.func_name][func.subfunc_name].update(ipf_dict)
        self.lockParams(False)
        return partial_stack, stack_dict

    def foldFunctionStack(self, partial_stack, initializer):
        return reduce(lambda f1, f2: f2(f1), partial_stack, initializer)

