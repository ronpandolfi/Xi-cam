# -*- coding: utf-8 -*-

from copy import deepcopy
from functools import partial
from collections import OrderedDict
import numpy as np
from PySide import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree

import config
import introspect
import manager
import reconpkg
import ui
from xicam import msg
import ftrwidgets as fw
import yamlmod


class FunctionWidget(fw.FeatureWidget):
    def __init__(self, name, subname, package, checkable=True, parent=None):
        self.name = name
        if name != subname:
            self.name += ' (' + subname + ')'
        super(FunctionWidget, self).__init__(self.name, checkable=checkable, parent=parent)

        self.func_name = name
        self.subfunc_name = subname
        self.input_functions = None
        print 'GOT PACKAGE ', package
        self._function = getattr(package, config.names[self.subfunc_name][0])
        self.param_dict = {}
        self._partial = None

        self.params = Parameter.create(name=self.name, children=config.parameters[self.subfunc_name], type='group')

        # Create dictionary with keys and default values that are not shown in the functions form
        self.kwargs_complement = introspect.get_arg_defaults(self._function)
        for key in self.param_dict.keys():
            if key in self.kwargs_complement:
                del self.kwargs_complement[key]

        # Create a list of argument names (this will most generally be the data passed to the function)
        self.args_complement = introspect.get_arg_names(self._function)
        s = set(self.param_dict.keys() + self.kwargs_complement.keys())
        self.args_complement = [i for i in self.args_complement if i not in s]

        self.form = ParameterTree(showHeader=False)
        self.form.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.form.customContextMenuRequested.connect(self.paramMenuRequested)
        self.form.setParameters(self.params, showTop=True)

        self.parammenu = QtGui.QMenu()
        action = QtGui.QAction('Test Parameter Range', self)
        action.triggered.connect(self.testParamTriggered)
        self.parammenu.addAction(action)

        self.previewButton.customContextMenuRequested.connect(self.menuRequested)
        self.menu = QtGui.QMenu()

        self.setDefaults()
        self.updateParamsDict()

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
        for param in self.params.children():
            self.param_dict.update({param.name(): param.value()})
        if self.input_functions is not None:
            for ipf in self.input_functions:
                ipf.updateParamsDict()
        return self.param_dict

    @property
    def partial(self):
        kwargs = dict(self.param_dict, **self.kwargs_complement)
        self._partial = partial(self._function, **kwargs)
        return self._partial

    @partial.setter
    def partial(self, p):
        self._partial = p

    @property
    def input_partials(self):
        return None

    @property
    def func_signature(self):
        signature = str(self._function.__name__) + '('
        for arg in self.args_complement:
            signature += '{},'.format(arg)
        for param, value in self.updateParamsDict.iteritems():
            signature += '{0}={1},'.format(param, value) if not isinstance(value, str) else \
                '{0}=\'{1}\','.format(param, value)
        for param, value in self.kwargs_complement.iteritems():
            signature += '{0}={1},'.format(param, value) if not isinstance(value, str) else \
                '{0}=\'{1}\','.format(param, value)
        return signature[:-1] + ')'

    def paramChanged(self, param):
        self.param_dict.update({param.name(): param.value()})

    def getParamDict(self, update=True):
        if update:
            self.updateParamsDict()
        return self.param_dict

    def setDefaults(self):
        defaults = introspect.get_arg_defaults(self._function)
        for param in self.params.children():
            if param.name() in defaults:
                if isinstance(defaults[param.name()], unicode):
                    defaults[param.name()] = str(defaults[param.name()])
                param.setDefault(defaults[param.name()])
                param.setValue(defaults[param.name()])

    def allReadOnly(self, boolean):
        for param in self.params.children():
            param.setReadonly(boolean)

    def menuRequested(self):
        pass

    def paramMenuRequested(self, pos):
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
        super(ReconFunctionWidget, self).__init__(name, subname, package, checkable=False)

        self.packagename = package.__name__
        self.kwargs_complement['algorithm'] = subname.lower()

        # Input functions
        self.center = None
        self.angles = None

        self.subframe = QtGui.QFrame(self)
        self.subframe.setFrameShape(QtGui.QFrame.StyledPanel)
        self.subframe.setFrameShadow(QtGui.QFrame.Raised)
        self.subframe_layout = QtGui.QVBoxLayout(self.subframe)
        self.subframe_layout.setContentsMargins(5, 5, 5, 5)
        self.subframe_layout.setSpacing(0)
        self.verticalLayout.addWidget(self.subframe)
        self.subframe.hide()

        self.submenu = QtGui.QMenu('Input Function')
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_39.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.submenu.setIcon(icon)
        ui.buildfunctionmenu(self.submenu, config.funcs['Input Functions'][name], self.addInputFunction)
        self.menu.addMenu(self.submenu)

        self.input_functions = [self.center, self.angles]
        self.addInputFunction('Projection Angles', 'Projection Angles', package=reconpkg.packages['tomopy'])

    @property
    def partial(self):
        d = deepcopy(self.param_dict)
        if 'cutoff' in d.keys() and 'order' in d.keys():
            d['filter_par'] = list((d.pop('cutoff'), d.pop('order')))
        kwargs = dict(d, **self.kwargs_complement)
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

    def addInputFunction(self, func, subfunc, package=reconpkg.packages['tomopy']):
        fwidget = self.angles if func == 'Projection Angles' else self.center
        if fwidget is not None and fwidget.subfunc_name != 'Manual':
            if func != 'Projection Angles':
                value = QtGui.QMessageBox.question(self, 'Adding duplicate function',
                                                   '{} input function already in pipeline\n'
                                                   'Do you want to replace it?'.format(func),
                                                   (QtGui.QMessageBox.Yes | QtGui.QMessageBox.No))
                if value is QtGui.QMessageBox.No:
                    return
            try:
                fwidget.deleteLater()
            except AttributeError:
                    pass
        checkable = False if func == 'Projection Angles' else True
        fwidget = FunctionWidget(func, subfunc, package=package, checkable=checkable)
        h = QtGui.QHBoxLayout()
        indent = QtGui.QLabel('  -   ')
        h.addWidget(indent)
        h.addWidget(fwidget)
        fwidget.destroyed.connect(indent.deleteLater)
        self.subframe_layout.addLayout(h)
        if func == 'Projection Angles':
            self.angles = fwidget
        else:
            self.center = fwidget
            self.center.destroyed.connect(self.resetCenter)
        self.input_functions = [self.center, self.angles]
        return fwidget

    def mouseClicked(self):
        super(ReconFunctionWidget, self).mouseClicked()
        self.subframe.show()

    def setCenterParam(self, value):
        self.params.child('center').setValue(value)
        self.params.child('center').setDefault(value)

    def menuRequested(self, pos):
        self.menu.exec_(self.previewButton.mapToGlobal(pos))


class AstraReconFuncWidget(ReconFunctionWidget):
    def __init__(self, name, subname, package):
        super(AstraReconFuncWidget, self).__init__(name, subname, reconpkg.tomopy)
        self.kwargs_complement['algorithm'] = reconpkg.tomopy.astra
        self.kwargs_complement['options'] = {}
        self.kwargs_complement['options']['method'] = subname.replace(' ', '_')
        if 'CUDA' in subname:
            self.kwargs_complement['options']['proj_type'] = 'cuda'
        else:
            self.kwargs_complement['options']['proj_type'] = 'linear'

    @property
    def partial(self):
        d = deepcopy(self.param_dict)
        kwargs = deepcopy(self.kwargs_complement)
        if 'center' in d:
            del d['center']
        kwargs['options'].update(d)
        self._partial = partial(self._function, **kwargs)
        return self._partial



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

    def __init__(self, list_layout, form_layout, function_widgets=None, blank_form=None):
        super(FunctionManager, self).__init__(list_layout, form_layout, feature_widgets=function_widgets,
                                              blank_form=blank_form)

        self.cor_offset = None
        self.cor_scale = lambda x: x  # dummy
        self.recon_function = None
        self.functions = self.features  # rename for readability
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

    @property
    def pipeline_dict(self):
        d = OrderedDict()
        for f in self.functions:
            d[f.func_name] = {f.subfunc_name: {'Parameters': {p.name(): p.value() for p in f.params.children()}}}
            d[f.func_name][f.subfunc_name]['Enabled'] = f.enabled
            if f.func_name == 'Reconstruction':
                d[f.func_name][f.subfunc_name].update({'Package': f.packagename})
            if f.input_functions is not None:
                d[f.func_name][f.subfunc_name]['Input Functions'] = {}
                for ipf in f.input_functions:
                    if ipf is not None:
                        id = {ipf.subfunc_name: {'Parameters': {p.name(): p.value() for p in ipf.params.children()}}}
                        d[f.func_name][f.subfunc_name]['Input Functions'][ipf.func_name] = id
        return d

    def updateParameters(self):
        for function in self.functions:
            function.updateParamsDict()

    def setCenterCorrection(self, name, param_dict):
        global cor_offset, cor_scale
        if 'Padding' in name and param_dict['axis'] == 2:
            n = param_dict['npad']
            cor_offset = lambda x: cor_scale(x) + n
        elif 'Downsample' in name and param_dict['axis'] == 2:
            s = param_dict['level']
            cor_scale = lambda x: x / 2 ** s
        elif 'Upsample' in name and param_dict['axis'] == 2:
            s = param_dict['level']
            cor_scale = lambda x: x * 2 ** s