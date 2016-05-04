# -*- coding: utf-8 -*-

from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
import numpy as np
from functools import partial
from time import sleep
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import functionmanager
from collectionsmod import UnsortableOrderedDict
import ui
import functiondata
import introspect


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8

    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


class ROlineEdit(QtGui.QLineEdit):
    def __init__(self, *args, **kwargs):
        super(ROlineEdit, self).__init__(*args, **kwargs)
        self.setReadOnly(True)
        self.setFrame(False)

    def focusOutEvent(self, *args, **kwargs):
        super(ROlineEdit, self).focusOutEvent(*args, **kwargs)
        self.setCursor(QtCore.Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, *args, **kwargs):
        super(ROlineEdit, self).mouseDoubleClickEvent(*args, **kwargs)
        self.setFrame(True)
        self.setFocus()
        self.selectAll()


class featureWidget(QtGui.QWidget):
    def __init__(self, name=''):
        self.name = name

        self._form = None

        super(featureWidget, self).__init__()

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.frame)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.previewButton = QtGui.QPushButton(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.previewButton.sizePolicy().hasHeightForWidth())
        self.previewButton.setSizePolicy(sizePolicy)
        self.previewButton.setStyleSheet("margin:0 0 0 0;")
        self.previewButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/eye.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.previewButton.setIcon(icon)
        self.previewButton.setFlat(True)
        self.previewButton.setCheckable(True)
        self.previewButton.setChecked(True)
        self.previewButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.previewButton)
        self.line = QtGui.QFrame(self.frame)
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_2.addWidget(self.line)
        self.txtName = ROlineEdit(self.frame)
        self.txtName.setObjectName("txtName")
        self.horizontalLayout_2.addWidget(self.txtName)
        self.txtName.setText(name)
        self.line_3 = QtGui.QFrame(self.frame)
        self.line_3.setFrameShape(QtGui.QFrame.VLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout_2.addWidget(self.line_3)
        self.closeButton = QtGui.QPushButton(self.frame)
        self.closeButton.setStyleSheet("margin:0 0 0 0;")
        self.closeButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("gui/close-button.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.closeButton.setIcon(icon1)
        self.closeButton.setFlat(True)
        self.closeButton.setObjectName("pushButton_3")
        self.closeButton.clicked.connect(self.delete)
        self.horizontalLayout_2.addWidget(self.closeButton)
        self.verticalLayout.addWidget(self.frame)

        self.txtName.mousePressEvent = self.mousePressEvent

        self.frame.setFrameShape(QtGui.QFrame.Box)
        self.frame.setCursor(QtCore.Qt.ArrowCursor)

    def delete(self):
        value = QtGui.QMessageBox.question(None, 'Delete this feature?',
                                           'Are you sure you want to delete this function?',
                                           (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
        if value is QtGui.QMessageBox.Yes:
            functionmanager.functions = [feature for feature in functionmanager.functions if feature is not self]
            self.deleteLater()
            ui.showform(ui.blankform)

    def mousePressEvent(self, *args, **kwargs):
        self.showSelf()
        self.setFocus()
        functionmanager.currentindex = functionmanager.functions.index(self)
        super(featureWidget, self).mousePressEvent(*args, **kwargs)

    def showSelf(self):
        ui.showform(self.form)

    # def hideothers(self):
    #     for item in functionmanager.functions:
    #         if hasattr(item, 'frame_2') and item is not self:
    #             item.frame_2.hide()

    # def wireup(self):
    #     if hasattr(self.form, 'txtName'):
    #         self.form.txtName.setText(self.name)
    #         self.form.txtName.textChanged.connect(self.setName)

    def setName(self, name):
        self.name = name
        functionmanager.update()


class FuncWidget(featureWidget):
    def __init__(self, function, subfunction, package):
        self.name = function
        if function != subfunction:
            self.name += ' (' + subfunction + ')'
        super(FuncWidget, self).__init__(self.name)

        self.func_name = function
        self.subfunc_name = subfunction
        self._formpath = 'gui/guiLayer.ui'
        self._form = None
        self._param_dict = None
        self._partial = None
        self.__function = getattr(package, functiondata.names[self.subfunc_name])
        self.params = Parameter.create(name=self.name, children=functiondata.parameters[self.subfunc_name], type='group')

        # Create dictionary with keys and default values that are not shown in the functions form
        self.kwargs_complement = introspect.get_arg_defaults(self.__function)
        for key in self.param_dict.keys():
            if key in self.kwargs_complement:
                del self.kwargs_complement[key]

        # Create a list of argument names (this will most generally be the data passed to the function)
        self.args_complement = introspect.get_arg_names(self.__function)
        s = set(self.param_dict.keys() + self.kwargs_complement.keys())
        self.args_complement = [i for i in self.args_complement if i not in s]

        self.setDefaults()

        self.menu = QtGui.QMenu()
        action = QtGui.QAction('Test Parameter Range', self)
        action.triggered.connect(self.testParamTriggered)
        self.menu.addAction(action)

    def wireup(self):
            for param in self.params.children():
                param.sigValueChanged.connect(self.paramChanged)

    @property
    def form(self):
        if self._form is None:
            self._form = ParameterTree()
            self._form.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self._form.customContextMenuRequested.connect(self.menuActionClicked)
            self._form.setParameters(self.params, showTop=True)
            self.wireup()
        return self._form

    @property
    def param_dict(self):
        if self._param_dict is None:
            self._param_dict = {}
        for param in self.params.children():
            self._param_dict.update({param.name(): param.value()})
        return self._param_dict

    @property
    def partial(self):
        kwargs = dict(self.param_dict, **self.kwargs_complement)
        self._partial = partial(self.__function, **kwargs)
        return self._partial

    @property
    def func_signature(self):
        signature = str(self.__function.__name__) + '('
        for arg in self.args_complement:
            signature += '{},'.format(arg)
        for param, value in self.param_dict.iteritems():
            signature += '{0}={1},'.format(param, value) if not isinstance(value, str) else '{0}=\'{1}\','.format(param, value)
        for param, value in self.kwargs_complement.iteritems():
            signature += '{0}={1},'.format(param, value) if not isinstance(value, str) else '{0}=\'{1}\','.format(param, value)
        return signature[:-1] + ')'

    def paramChanged(self, param):
        self.param_dict.update({param.name(): param.value()})

    def setDefaults(self): #TODO gridrec filter_pars are children of filter_name so are not captured in defaults
        defaults = introspect.get_arg_defaults(self.__function)
        for param in self.params.children():
            if param.name() in defaults:
                if isinstance(defaults[param.name()], unicode):
                    defaults[param.name()] = str(defaults[param.name()])
                param.setDefault(defaults[param.name()])
                param.setValue(defaults[param.name()])

    def run(self, **args):
        for arg in args.keys():
            if arg not in self.args_complement:
                raise ValueError('{} is not an argument to {}'.format(arg, self.__function.__name__))
        if len(args) != len(self.args_complement):
            raise ValueError('{} requires {} more arguments, {} given'.format(self.__function.__name__,
                                                                              len(self.args_complement), len(args)))
        return self.partial(**args)

    def menuActionClicked(self, pos):
        if self.form.currentItem().parent():
            self.menu.exec_(self.form.mapToGlobal(pos))

    def testParamTriggered(self):
        param = self.form.currentItem().param
        if param.type() == 'int' or param.type() == 'float':
            start, end, step = None, None, None
            if 'limits' in param.opts:
                start, end = param.opts['limits']
                step = (end - start)/3 + 1
            elif param.value() is not None:
                start, end, step = param.value()//2, 4*(param.value())//2, param.value()//2
            test = TestRangeDialog(param.type(), (start, end, step))
            if test.exec_():
                for i in np.arange(*test.selectedRange()):
                    param.setValue(i)
                    functionmanager.runpreviewstack()
                    # TODO avoid having to make this to wait
                    # Seems to have a race condition when trying to access data in functionmanager.runpreviewstack()
                    sleep(0.7)
        elif param.type() == 'list':
            test = TestListRangeDialog(param.opts['values'])
            if test.exec_():
                for i in test.selectedOptions():
                    param.setValue(i)
                    functionmanager.runpreviewstack()
                    sleep(0.7)


class ReconFuncWidget(FuncWidget):
    def __init__(self, function, subfunction, package):
        super(ReconFuncWidget, self).__init__(function, subfunction, package)
        self.previewButton.setCheckable(False)
        self.previewButton.setChecked(True)
        self.kwargs_complement['algorithm'] = subfunction.lower()

    def setCenterParam(self, value):
        for param in self.params.children():
            if param.name() == 'center':
                param.setValue(value)


class TestRangeDialog(QtGui.QDialog):
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
            self.spinBox.setMaximum(3 * prange[2])
            self.spinBox_2.setMaximum(3 * prange[2])
            self.spinBox_3.setMaximum(prange[2])

            self.spinBox.setValue(prange[0])
            self.spinBox_2.setValue(prange[1])
            self.spinBox_3.setValue(prange[2])


        self.setWindowTitle(_translate("Dialog", "Set parameter range", None))
        self.label.setText(_translate("Dialog", "Start", None))
        self.label_2.setText(_translate("Dialog", "End", None))
        self.label_3.setText(_translate("Dialog", "Step", None))

    def selectedRange(self):
        # return the end as selected end + step so that the range includes the end
        return self.spinBox.value(), self.spinBox_2.value()  + self.spinBox_3.value(), self.spinBox_3.value()


class TestListRangeDialog(QtGui.QDialog):
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

    def selectedOptions(self):
        return str(self.lineEdit.text()).split(' ')