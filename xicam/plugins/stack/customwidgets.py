# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'vector.ui'
#
# Created: Sat Oct 17 11:23:24 2015
# by: PyQt4 UI code generator 4.11.3
#
# WARNING! All changes made in this file will be lost!

import json
from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import numpy as np
from pyFAI import detectors
import functionmanager
from collectionsmod import UnsortableOrderedDict
import ui
import functiondata


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
        self.setReadOnly(False)
        self.setFrame(True)
        self.setFocus()
        self.selectAll()


# Form implementation generated from reading ui file 'layer.ui'
#
# Created: Thu Oct 22 09:52:22 2015
# by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!


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
        # self.pushButton_2 = QtGui.QPushButton(self.frame)
        # sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        # self.pushButton_2.setSizePolicy(sizePolicy)
        # self.pushButton_2.setStyleSheet("margin:0 0 0 0;")
        # self.pushButton_2.setFlat(True)
        # self.pushButton_2.setObjectName("pushButton_2")
        # self.horizontalLayout_2.addWidget(self.pushButton_2)
        # self.line_2 = QtGui.QFrame(self.frame)
        # self.line_2.setFrameShape(QtGui.QFrame.VLine)
        # self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        # self.line_2.setObjectName("line_2")
        # self.horizontalLayout_2.addWidget(self.line_2)
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

        # self.pushButton_2.setText("O")

        self.frame.setFrameShape(QtGui.QFrame.Box)
        self.frame.setCursor(QtCore.Qt.ArrowCursor)



    def delete(self):
        value = QtGui.QMessageBox.question(None, 'Delete this feature?',
                                           'Are you sure you want to delete this feature?',
                                           (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
        if value is QtGui.QMessageBox.Yes:
            functionmanager.functions = [feature for feature in functionmanager.functions if feature is not self]
            self.deleteLater()
            ui.showForm(ui.blankForm)

    def mousePressEvent(self, *args, **kwargs):
        functionmanager.currentfunction = functionmanager.functions.index(self)
        print functionmanager.currentfunction
        self.showSelf()
        self.hideothers()
        self.setFocus()
        super(featureWidget, self).mousePressEvent(*args, **kwargs)

    def showSelf(self):
        ui.showForm(self.form)


    def hideothers(self):
        for item in functionmanager.functions:
            if hasattr(item, 'frame_2') and item is not self:
                item.frame_2.hide()

    @property
    def form(self):
        if self._form is None:
            self._form = loadform(self._formpath)
            self.wireup()
        return self._form

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
    def __init__(self, function, subfunction):
        self._formpath = 'gui/guiLayer.ui'
        self.name = function
        if function != subfunction:
            self.name += ' (' + subfunction + ')'
        super(func, self).__init__(self.name)
        self.function = function
        self.subfunction = subfunction
        self.params = Parameter.create(name=self.name, children=functiondata.parameters[self.subfunction], type='group')

    @property
    def form(self):
        if self._form is None:
            self.parameterTree = ParameterTree()
            self.parameterTree.setParameters(self.params, showTop=True)
            self._form = self.parameterTree
            self.wireup()
        return self._form

    def wireup(self):
        pass



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


class StepParameter(pTypes.GroupParameter):
    itemClass = hideableGroupParameterItem

    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)

        self.Min = pTypes.SimpleParameter(name='Minimum', type='float', value=0)
        self.Max = pTypes.SimpleParameter(name='Maximum', type='float', value=0)
        self.Step = pTypes.SimpleParameter(name='Step', type='float', value=0)

        self.addChildren([self.Min, self.Max, self.Step])

    def toDict(self):
        d = dict()
        d['min'] = self.Min.value()
        d['max'] = self.Max.value()
        d['step'] = self.Step.value()

        return d


class MinMaxParameter(pTypes.GroupParameter):
    itemClass = hideableGroupParameterItem

    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)

        self.Min = pTypes.SimpleParameter(name='Minimum', type='float', value=0)
        self.Max = pTypes.SimpleParameter(name='Maximum', type='float', value=0)

        self.addChildren([self.Min, self.Max])

    def value(self):
        return [self.Min.value(), self.Max.value()]


    def toDict(self):
        d = UnsortableOrderedDict()
        d['min'] = self.Min.value()
        d['max'] = self.Max.value()

        return d
