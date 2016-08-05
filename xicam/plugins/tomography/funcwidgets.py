# -*- coding: utf-8 -*-

from copy import deepcopy
from functools import partial

import numpy as np
from PySide import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree

import configdata
import introspect
import manager
import reconpkg
import ui
from xicam import msg
from xicam.plugins.tomography.features.widgets import FeatureWidget


class FuncWidget(FeatureWidget):
    def __init__(self, function, subfunction, package, checkable=True, parent=None):
        self.name = function
        if function != subfunction:
            self.name += ' (' + subfunction + ')'
        super(FuncWidget, self).__init__(self.name, checkable=checkable, parent=parent)

        self.func_name = function
        self.subfunc_name = subfunction
        self._form = None
        self._partial = None
        self._function = getattr(package, configdata.names[self.subfunc_name][0])
        self.params = Parameter.create(name=self.name, children=configdata.parameters[self.subfunc_name], type='group')
        self.param_dict = {}
        self.input_functions = None
        self.setDefaults()
        self.updateParamsDict()

        # Create dictionary with keys and default values that are not shown in the functions form
        self.kwargs_complement = introspect.get_arg_defaults(self._function)
        for key in self.param_dict.keys():
            if key in self.kwargs_complement:
                del self.kwargs_complement[key]

        # Create a list of argument names (this will most generally be the data passed to the function)
        self.args_complement = introspect.get_arg_names(self._function)
        s = set(self.param_dict.keys() + self.kwargs_complement.keys())
        self.args_complement = [i for i in self.args_complement if i not in s]

        self.parammenu = QtGui.QMenu()
        action = QtGui.QAction('Test Parameter Range', self)
        action.triggered.connect(self.testParamTriggered)
        self.parammenu.addAction(action)

        self.previewButton.customContextMenuRequested.connect(self.menuRequested)
        self.menu = QtGui.QMenu()

    def wireup(self):
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

    @property
    def form(self):
        if self._form is None:
            self._form = ParameterTree(showHeader=False)
            self._form.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self._form.customContextMenuRequested.connect(self.paramMenuRequested)
            self._form.setParameters(self.params, showTop=True)
            self.wireup()
        return self._form

    def updateParamsDict(self):
        for param in self.params.children():
            self.param_dict.update({param.name(): param.value()})
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

class ReconFuncWidget(FuncWidget):
    def __init__(self, function, subfunction, package):
        super(ReconFuncWidget, self).__init__(function, subfunction, package, checkable=False)

        self.packagename = package.__name__
        self.kwargs_complement['algorithm'] = subfunction.lower()

        # Input functions
        self.center = None
        self.angles = None

        self.frame_2 = QtGui.QFrame(self)
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2_layout = QtGui.QVBoxLayout(self.frame_2)
        self.frame_2_layout.setContentsMargins(5, 5, 5, 5)
        self.frame_2_layout.setSpacing(0)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_2.hide()

        self.submenu = QtGui.QMenu('Input Function')
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_39.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.submenu.setIcon(icon)
        ui.buildfunctionmenu(self.submenu, configdata.funcs['Input Functions'][function], self.addInputFunction)
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
        fwidget = FuncWidget(func, subfunc, package=package, checkable=checkable)
        h = QtGui.QHBoxLayout()
        indent = QtGui.QLabel('  -   ')
        h.addWidget(indent)
        h.addWidget(fwidget)
        fwidget.destroyed.connect(indent.deleteLater)
        self.frame_2_layout.addLayout(h)
        if func == 'Projection Angles':
            self.angles = fwidget
        else:
            self.center = fwidget
            self.center.destroyed.connect(self.resetCenter)
        self.input_functions = [self.center, self.angles]
        return fwidget

    def mouseClicked(self):
        super(ReconFuncWidget, self).mouseClicked()
        self.frame_2.show()

    def setCenterParam(self, value):
        self.params.child('center').setValue(value)
        self.params.child('center').setDefault(value)

    def menuRequested(self, pos):
        self.menu.exec_(self.previewButton.mapToGlobal(pos))


class AstraReconFuncWidget(ReconFuncWidget):
    def __init__(self, function, subfunction, package):
        super(AstraReconFuncWidget, self).__init__(function, subfunction, reconpkg.tomopy)
        self.kwargs_complement['algorithm'] = reconpkg.tomopy.astra
        self.kwargs_complement['options'] = {}
        self.kwargs_complement['options']['method'] = subfunction.replace(' ', '_')
        if 'CUDA' in subfunction:
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


        self.setWindowTitle(_translate("Dialog", "Set parameter range", None))
        self.label.setText(_translate("Dialog", "Start", None))
        self.label_2.setText(_translate("Dialog", "End", None))
        self.label_3.setText(_translate("Dialog", "Step", None))

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
        self.setWindowTitle(_translate("Dialog", "Set parameter range", None))

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

