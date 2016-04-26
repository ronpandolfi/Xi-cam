# -*- coding: utf-8 -*-

from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
from functools import partial
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import numpy as np
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
        self.setReadOnly(True)
        self.setFrame(False)
        self.setCursor(QtCore.Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, *args, **kwargs):
        super(ROlineEdit, self).mouseDoubleClickEvent(*args, **kwargs)
        self.setReadOnly(True)
        self.setFrame(True)
        self.setFocus()
        self.selectAll()


class featureWidget(QtGui.QWidget):
    sigUpdateDisplayUnitCell = QtCore.Signal()

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
        self.pushButton = QtGui.QPushButton(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setStyleSheet("margin:0 0 0 0;")
        self.pushButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/eye.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon)
        self.pushButton.setFlat(True)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
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
        self.pushButton_3 = QtGui.QPushButton(self.frame)
        self.pushButton_3.setStyleSheet("margin:0 0 0 0;")
        self.pushButton_3.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("gui/close-button.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_3.setIcon(icon1)
        self.pushButton_3.setFlat(True)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.delete)
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        self.verticalLayout.addWidget(self.frame)

        self.txtName.mousePressEvent = self.mousePressEvent

        self.frame.setFrameShape(QtGui.QFrame.Box)
        self.frame.setCursor(QtCore.Qt.ArrowCursor)

    @property
    def form(self):
        if self._form is None:
            self._form = loadform(self._formpath)
            self.wireup()
        return self._form

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
        self.hideothers()
        self.setFocus()
        functionmanager.currentfunction = functionmanager.functions.index(self)
        super(featureWidget, self).mousePressEvent(*args, **kwargs)

    def showSelf(self):
        ui.showform(self.form)

    def hideothers(self):
        for item in functionmanager.functions:
            if hasattr(item, 'frame_2') and item is not self:
                item.frame_2.hide()

    def wireup(self):
        if hasattr(self.form, 'txtName'):
            self.form.txtName.setText(self.name)
            self.form.txtName.textChanged.connect(self.setName)


    def setName(self, name):
        self.name = name
        functionmanager.update()


    def toDict(self):
        return {}


class form(QtGui.QWidget):
    def __init__(self, name):
        super(form, self).__init__()
        self._form = None
        self.wireup()
        self.form

    @property
    def form(self):
        if self._form is None:
            self._form = loadform(self._formpath)
        return self._form

    def wireup(self):
        pass


class func(featureWidget):
    def __init__(self, function, subfunction, package):
        self.name = function
        if function != subfunction:
            self.name += ' (' + subfunction + ')'
        super(func, self).__init__(self.name)

        self.func_name = function
        self.subfunc_name = subfunction
        self._formpath = 'gui/guiLayer.ui'
        self._form = None
        self._param_dict = None
        self._partial = None
        self.__function = getattr(package, functiondata.names[self.subfunc_name])
        self.params = Parameter.create(name=self.name, children=functiondata.parameters[self.subfunc_name], type='group')

        self.kwargs_complement = introspect.get_arg_defaults(self.__function)
        if function == 'Reconstruction':
            self.kwargs_complement['algorithm'] = subfunction.lower()
        for key in self.param_dict.keys():
            if key in self.kwargs_complement:
                del self.kwargs_complement[key]
        self.args_complement = introspect.get_arg_names(self.__function)
        s = set(self.param_dict.keys() + self.kwargs_complement.keys())
        self.args_complement = [i for i in self.args_complement if i not in s]

        self.setDefaults()

        self.menu = QtGui.QMenu()
        action = QtGui.QAction('Test Parameter Range', self)
        action.triggered.connect(self.testParamRequested)
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
        print self.func_signature
        if self.form.currentItem().parent():
            self.menu.exec_(self.form.mapToGlobal(pos))

    def testParamRequested(self):
        print self.form.currentItem()


def loadform(path):
    guiloader = QUiLoader()
    f = QtCore.QFile(path)
    f.open(QtCore.QFile.ReadOnly)
    form = guiloader.load(f)
    f.close()
    return form


class hideableGroupParameterItem(pTypes.GroupParameterItem):
    def optsChanged(self, param, opts):
        super(hideableGroupParameterItem, self).optsChanged(param, opts)
        if 'visible' in opts:
            self.setHidden(not opts['visible'])


class TestRangeDialog(QtGui.QDialog):

    def __init__(self, dtype, range=None, values=None, **opts):
        super(TestRangeDialog, self).__init__(**opts)
        l = QtGui.QHBoxLayout(self)
        self.button_box