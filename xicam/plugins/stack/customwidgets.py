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


class vector(QtGui.QWidget):
    sigValueChanged = QtCore.Signal()
    sigChanged = sigValueChanged


    def __init__(self):
        super(vector, self).__init__()

        self.horizontalLayout = QtGui.QHBoxLayout(self)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.UnitCellVec1LeftParenthesis3D = QtGui.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.UnitCellVec1LeftParenthesis3D.setFont(font)
        self.UnitCellVec1LeftParenthesis3D.setObjectName(_fromUtf8("UnitCellVec1LeftParenthesis3D"))
        self.horizontalLayout.addWidget(self.UnitCellVec1LeftParenthesis3D)
        self.value1 = QtGui.QDoubleSpinBox(self)
        self.value1.setDecimals(1)
        self.value1.setMinimum(-1000.0)
        self.value1.setMaximum(1000.0)
        self.value1.setSingleStep(0.5)
        self.value1.setProperty("value", 0.0)
        self.value1.setObjectName(_fromUtf8("value1"))
        self.horizontalLayout.addWidget(self.value1)
        self.UnitCellVec1Comma1 = QtGui.QLabel(self)
        self.UnitCellVec1Comma1.setObjectName(_fromUtf8("UnitCellVec1Comma1"))
        self.horizontalLayout.addWidget(self.UnitCellVec1Comma1)
        self.value2 = QtGui.QDoubleSpinBox(self)
        self.value2.setDecimals(1)
        self.value2.setMinimum(-1000.0)
        self.value2.setMaximum(1000.0)
        self.value2.setSingleStep(0.5)
        self.value2.setObjectName(_fromUtf8("value2"))
        self.horizontalLayout.addWidget(self.value2)
        self.UnitCellVec1Comma2 = QtGui.QLabel(self)
        self.UnitCellVec1Comma2.setObjectName(_fromUtf8("UnitCellVec1Comma2"))
        self.horizontalLayout.addWidget(self.UnitCellVec1Comma2)
        self.value3 = QtGui.QDoubleSpinBox(self)
        self.value3.setDecimals(1)
        self.value3.setMinimum(-1000.0)
        self.value3.setMaximum(1000.0)
        self.value3.setSingleStep(0.5)
        self.value3.setObjectName(_fromUtf8("value3"))
        self.horizontalLayout.addWidget(self.value3)
        self.UnitCellVec1RightParenthesis3D = QtGui.QLabel(self)
        self.UnitCellVec1RightParenthesis3D.setFont(font)
        self.UnitCellVec1RightParenthesis3D.setObjectName(_fromUtf8("UnitCellVec1RightParenthesis3D"))
        self.horizontalLayout.addWidget(self.UnitCellVec1RightParenthesis3D)

        self.UnitCellVec1LeftParenthesis3D.setText("(")
        self.UnitCellVec1Comma1.setText(",")
        self.UnitCellVec1Comma2.setText(",")
        self.UnitCellVec1RightParenthesis3D.setText(")")

        self.value1.valueChanged.connect(self.sigValueChanged)
        self.value2.valueChanged.connect(self.sigValueChanged)
        self.value3.valueChanged.connect(self.sigValueChanged)

    def value(self):
        return self.value1.value(), self.value2.value(), self.value3.value()


    def setValue(self, v):
        self.value1.setValue(v[0])
        self.value2.setValue(v[1])
        self.value3.setValue(v[2])


    def setEnabled(self, enabled):
        self.value1.setEnabled(enabled)
        self.value2.setEnabled(enabled)
        self.value3.setEnabled(enabled)


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
            functionmanager.features = [feature for feature in functionmanager.features if feature is not self]
            self.deleteLater()
            ui.showForm(ui.blankForm)

    def mousePressEvent(self, *args, **kwargs):
        self.showSelf()
        self.hideothers()
        self.setFocus()
        super(featureWidget, self).mousePressEvent(*args, **kwargs)

    def showSelf(self):
        ui.showForm(self.form)


    def hideothers(self):
        for item in functionmanager.features:
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
    def __init__(self, function,subfunction):
        self._formpath = 'gui/guiLayer.ui'
        name = function
        print function,subfunction
        if function != subfunction:
            name += ' (' + subfunction + ')'
        super(func, self).__init__(name)
        self.function = function
        self.subfunction = subfunction
        self.params = Parameter.create(name=name, children=functiondata.parameters[self.subfunction], type='group')

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




class VectorParameterItem(pTypes.WidgetParameterItem):
    def makeWidget(self):
        w = vector()
        opts = self.param.opts
        value = opts.get('value', None)
        if value is not None:
            w.setValue(value)

        self.value = w.value
        self.setValue = w.setValue

        return w

    def valueChanged(self, *args, **kwargs):
        super(VectorParameterItem, self).valueChanged(*args, **kwargs)


class VectorParameter(Parameter):
    itemClass = VectorParameterItem

    def __init__(self, *args, **kwargs):
        super(VectorParameter, self).__init__(*args, **kwargs)


    def defaultValue(self):
        return (0, 0, 0)


registerParameterType('Vector', VectorParameter, override=True)


class ScalableGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"
        pTypes.GroupParameter.__init__(self, **opts)

    def addNew(self):
        self.addChild(dict(name="Point %d" % (len(self.childs) + 1), type='Vector', removable=True, renamable=True))

    def toArray(self):
        return [list(child.value()) for child in self.children()]

    def toDict(self):
        d = dict()
        for child in self.children():
            d[child.opts['name']] = list(child.value())
        return d


class hideableGroupParameterItem(pTypes.GroupParameterItem):
    def optsChanged(self, param, opts):
        super(hideableGroupParameterItem, self).optsChanged(param, opts)
        if 'visible' in opts:
            self.setHidden(not opts['visible'])


class DistParameter(pTypes.GroupParameter):
    itemClass = hideableGroupParameterItem

    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)

        self.DistributionChoice = pTypes.ListParameter(name='Distribution', type='list', value=0,
                                                       values=['Single value', 'Uniform', 'Random', 'Gaussian'])
        self.Value = pTypes.SimpleParameter(name=opts['name'], type='float', value=0)
        self.Min = pTypes.SimpleParameter(name='Minimum', type='float', value=0)
        self.Max = pTypes.SimpleParameter(name='Maximum', type='float', value=0)
        self.Mean = pTypes.SimpleParameter(name='Mean', type='float', value=0)
        self.Variance = pTypes.SimpleParameter(name='Variance', type='float', value=0)
        self.N = pTypes.SimpleParameter(name='Number of samples', type='int', value=0)

        self.Min.hide()
        self.Max.hide()
        self.Mean.hide()
        self.Variance.hide()
        self.N.hide()

        self.DistributionChoice.sigValueChanged.connect(self.distributionChanged)

        self.addChildren([self.DistributionChoice, self.Value, self.Min, self.Max, self.Variance, self.N])


    def distributionChanged(self, _, choice):
        # print choice
        if choice == 'Uniform':
            self.Value.hide()
            self.Min.show()
            self.Max.show()
            self.Mean.hide()
            self.Variance.hide()
            self.N.show()
        elif choice == 'Single value':
            self.Value.show()
            self.Min.hide()
            self.Max.hide()
            self.Mean.hide()
            self.Variance.hide()
            self.N.hide()
        elif choice == 'Random':
            self.Value.hide()
            self.Min.show()
            self.Max.show()
            self.Mean.hide()
            self.Variance.hide()
            self.N.show()
        elif choice == 'Gaussian':
            self.Value.hide()
            self.Min.show()
            self.Max.show()
            self.Mean.show()
            self.Variance.show()
            self.N.show()

    def toDict(self):
        d = UnsortableOrderedDict()
        choice = self.DistributionChoice.value()
        d['type'] = self.opts['higKey'].lower()
        if choice == 'Uniform':
            d['min'] = self.Min.value()
            d['max'] = self.Max.value()
            d['N'] = self.N.value()
            d['stat'] = 'uniform'
        elif choice == 'Single value':
            d['min'] = self.Value.value()
            d['stat'] = 'single'
        elif choice == 'Random':
            d['min'] = self.Min.value()
            d['max'] = self.Max.value()
            d['N'] = self.N.value()
            d['stat'] = 'random'
        elif choice == 'Gaussian':
            d['min'] = self.Min.value()
            d['max'] = self.Max.value()
            d['N'] = self.N.value()
            d['stddev'] = np.sqrt(self.Variance.value())
            d['mean'] = self.Mean.value()
            d['stat'] = 'gaussian'
        return d


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