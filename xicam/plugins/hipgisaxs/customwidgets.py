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
import hig
import featuremanager
import ui
import display
from modpkgs.collectionsmod import UnsortableOrderedDict

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

        self.makeValues()

        self.horizontalLayout = QtGui.QHBoxLayout(self)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.UnitCellVec1LeftParenthesis3D = QtGui.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.UnitCellVec1LeftParenthesis3D.setFont(font)
        self.UnitCellVec1LeftParenthesis3D.setObjectName(_fromUtf8("UnitCellVec1LeftParenthesis3D"))
        self.horizontalLayout.addWidget(self.UnitCellVec1LeftParenthesis3D)
        self.value1.setMinimum(-1000)
        self.value1.setMaximum(1000)
        self.value1.setSingleStep(1)
        self.value1.setProperty("value", 0.0)
        self.value1.setObjectName(_fromUtf8("value1"))
        self.horizontalLayout.addWidget(self.value1)
        self.UnitCellVec1Comma1 = QtGui.QLabel(self)
        self.UnitCellVec1Comma1.setObjectName(_fromUtf8("UnitCellVec1Comma1"))
        self.horizontalLayout.addWidget(self.UnitCellVec1Comma1)
        self.value2.setMinimum(-1000)
        self.value2.setMaximum(1000)
        self.value2.setSingleStep(1)
        self.value2.setObjectName(_fromUtf8("value2"))
        self.horizontalLayout.addWidget(self.value2)
        self.UnitCellVec1Comma2 = QtGui.QLabel(self)
        self.UnitCellVec1Comma2.setObjectName(_fromUtf8("UnitCellVec1Comma2"))
        self.horizontalLayout.addWidget(self.UnitCellVec1Comma2)
        self.value3.setMinimum(-1000)
        self.value3.setMaximum(1000)
        self.value3.setSingleStep(1)
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

    def makeValues(self):
        self.value1 = QtGui.QDoubleSpinBox()
        self.value2 = QtGui.QDoubleSpinBox()
        self.value3 = QtGui.QDoubleSpinBox()
        for v in [self.value1,self.value2,self.value3]:
            v.setDecimals(3)

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

class intvector(vector):
    sigValueChanged = QtCore.Signal()
    sigChanged = sigValueChanged

    def makeValues(self):
        self.value1 = QtGui.QSpinBox()
        self.value2 = QtGui.QSpinBox()
        self.value3 = QtGui.QSpinBox()


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
        self.frame.setObjectName("frame")
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
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_47.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
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
        icon1.addPixmap(QtGui.QPixmap("xicam/gui/icons_46.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
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

        _ = self.form  # force cache form

    def delete(self):
        value = QtGui.QMessageBox.question(None, 'Delete this feature?',
                                           'Are you sure you want to delete this feature?',
                                           (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
        if value is QtGui.QMessageBox.Yes:
            featuremanager.features.remove(self)
            ui.showForm(ui.blankForm)
            self.deleteLater()

        display.redraw()

    def mousePressEvent(self, *args, **kwargs):
        self.showSelf()
        self.hideothers()
        self.setFocus()
        super(featureWidget, self).mousePressEvent(*args, **kwargs)

    def showSelf(self):
        ui.showForm(self.form)

    def hideothers(self):
        for item in featuremanager.features:
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
        featuremanager.update()

    def toDict(self):
        return {}


class substrate(featureWidget):
    def __init__(self, name='Substrate'):
        self._formpath = 'xicam/gui/guiSubstrate.ui'
        super(substrate, self).__init__(name)

    @property
    def form(self):
        if self._form is None:
            self.Material = pTypes.ListParameter(type='list', name='Material',
                                                 values=['Custom...', 'Si (10 keV)', 'Polystyrene (10 keV)',
                                                         'PMMA (10 keV)', 'Air', 'Vacuum'], value=0)
            self.delta = pTypes.SimpleParameter(type='float', name='delta', value=4.60093997E-06, step=.01)
            self.beta = pTypes.SimpleParameter(type='float', name='beta', value=3.91313968E-08, step=.01)
            params = [self.Material, self.delta, self.beta]

            self.parameter = Parameter.create(name='params', type='group', children=params)

            self.parameterTree = ParameterTree()
            self.parameterTree.setParameters(self.parameter, showTop=False)
            self._form = self.parameterTree
            self.wireup()
        return self._form

    def wireup(self):

        self.Material.sigValueChanged.connect(self.materialChanged)
        self.setConnected(True)

    def setConnected(self, connect):
        for item in [self.delta, self.beta]:
            if connect:
                item.sigValueChanged.connect(self.settoCustom)
            else:
                item.sigValueChanged.disconnect(self.settoCustom)

    def settoCustom(self):
        self.Material.setValue(0)

    def materialChanged(self, _, material):
        # print material
        self.setConnected(False)
        if material == 'PMMA (10 keV)':
            self.delta.setValue(0.5)
            self.beta.setValue(0.5)
        if material == 'Si (10 keV)':
            self.delta.setValue(4.60093997E-06)
            self.beta.setValue(3.91313968E-08)

        self.setConnected(True)

    def toDict(self):
        return UnsortableOrderedDict([('key', 'substr'),
                                      ('name', self.name),
                                      ('order', -1),
                                      ('material', self.Material.value()),
                                      ('refindex', UnsortableOrderedDict([('delta', self.delta.value()),
                                                                          ('beta', self.beta.value())]))])


class layer(featureWidget):
    def __init__(self, name=None):
        self._formpath = 'xicam/gui/guiLayer.ui'
        if name is None:
            name = 'Layer ' + str(featuremanager.layercount() + 1)
        super(layer, self).__init__(name)

    @property
    def form(self):
        if self._form is None:
            self.Thickness = pTypes.SimpleParameter(type='float', name='Thickness', value=.0000001, step=1e-9,
                                                    suffix='m', siPrefix=True)
            self.Material = pTypes.ListParameter(type='list', name='Material',
                                                 values=['Custom...', 'Si (10 keV)', 'Polystyrene (10 keV)',
                                                         'PMMA (10 keV)', 'Air', 'Vacuum'], value=0)
            self.delta = pTypes.SimpleParameter(type='float', name='delta', value=4.60093997E-06, step=.01)
            self.beta = pTypes.SimpleParameter(type='float', name='beta', value=3.91313968E-08, step=.01)
            self.TransVec = VectorParameter(name='Translation')
            params = [self.Thickness, self.Material, self.delta, self.beta, self.TransVec]

            self.parameter = Parameter.create(name='params', type='group', children=params)

            self.parameterTree = ParameterTree()
            self.parameterTree.setParameters(self.parameter, showTop=False)
            self._form = self.parameterTree
            self.wireup()
        return self._form

    def wireup(self):

        self.Material.sigValueChanged.connect(self.materialChanged)
        self.setConnected(True)

    def setConnected(self, connect):
        for item in [self.delta, self.beta]:
            if connect:
                item.sigValueChanged.connect(self.settoCustom)
            else:
                item.sigValueChanged.disconnect(self.settoCustom)

    def settoCustom(self):
        self.Material.setValue(0)

    def materialChanged(self, _, material):
        # print material
        self.setConnected(False)
        if material == 'PMMA (10 keV)':
            self.delta.setValue(0.5)
            self.beta.setValue(0.5)

        self.setConnected(True)

    def toDict(self):
        return UnsortableOrderedDict([('key', self.name),
                                      ('order', featuremanager.features.index(self)),
                                      ('material', self.Material.value()),
                                      ('thickness', self.Thickness.value()),
                                      ('transvec', list(self.TransVec.value())),
                                      ('refindex', UnsortableOrderedDict([('delta', self.delta.value()),
                                                                          ('beta', self.beta.value())]))])


class particle(featureWidget):
    def __init__(self, name=None):
        if name is None:
            name = 'Particle ' + str(featuremanager.particlecount() + 1)
        super(particle, self).__init__(name)

        self.frame_2 = QtGui.QFrame(self)
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout = QtGui.QGridLayout(self.frame_2)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.line_4 = QtGui.QFrame(self.frame_2)
        self.line_4.setFrameShape(QtGui.QFrame.HLine)
        self.line_4.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout.addWidget(self.line_4, 1, 2, 1, 1)
        self.pushButton_4 = QtGui.QPushButton(self.frame_2)
        self.pushButton_4.setFlat(True)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.showStructure)
        self.gridLayout.addWidget(self.pushButton_4, 0, 2, 1, 1)
        self.pushButton_5 = QtGui.QPushButton(self.frame_2)
        self.pushButton_5.setFlat(True)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.showEnsemble)
        self.gridLayout.addWidget(self.pushButton_5, 2, 2, 1, 1)
        spacerItem = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 0, 1, 1)
        self.line_6 = QtGui.QFrame(self.frame_2)
        self.line_6.setFrameShape(QtGui.QFrame.VLine)
        self.line_6.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.gridLayout.addWidget(self.line_6, 0, 1, 1, 1)
        self.line_7 = QtGui.QFrame(self.frame_2)
        self.line_7.setFrameShape(QtGui.QFrame.VLine)
        self.line_7.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.gridLayout.addWidget(self.line_7, 2, 1, 1, 1)

        self.verticalLayout.addWidget(self.frame_2)
        self.pushButton_4.setText("Structure")
        self.pushButton_5.setText("Ensemble")
        self.frame_2.hide()

        self.frame_2.setFrameShape(QtGui.QFrame.Box)

        self.ensemble = ensemble(self)
        self.structure = structure(self)

    @property
    def form(self):
        if self._form is None:
            # PARTICLE STYLE
            self.Type = pTypes.ListParameter(name='Particle type', values=['Sphere', 'Box', 'Cylinder', 'Pyramid'],
                                             value=0)
            self.Type.sigValueChanged.connect(self.typeChanged)

            # SPHERE
            self.Radius = DistParameter(name='Radius', higKey='radius', value=10)

            # BOX
            self.Length = DistParameter(name='Length (x)', higKey='xsize', value=10)
            self.Width = DistParameter(name='Width (y)', higKey='ysize', value=10)
            self.Height = DistParameter(name='Height (z)', higKey='height', value=10)

            # CYLINDER
            # (redundant)

            # PYRAMID
            self.BaseAngle = DistParameter(name='Angle with base (deg)', higKey='baseangle', value=15)

            # ROTATIONS

            self.XRotation = pTypes.SimpleParameter(name='X Rotation', value=0, type='float')
            self.YRotation = pTypes.SimpleParameter(name='Y Rotation', value=0, type='float')
            self.ZRotation = pTypes.SimpleParameter(name='Z Rotation', value=0, type='float')

            # Refractive index
            self.delta = pTypes.SimpleParameter(type='float', name='delta', value=4.60093997E-06, step=.01)
            self.beta = pTypes.SimpleParameter(type='float', name='beta', value=3.91313968E-08, step=.01)
            self.RefractiveIndex = pTypes.GroupParameter(name='Refractive Index', children=[self.delta, self.beta])

            self.hideAll()
            self.Radius.show()

            params = [self.Type, self.Radius, self.Length, self.Width, self.Height, self.BaseAngle,
                      self.XRotation, self.YRotation, self.ZRotation, self.RefractiveIndex]
            self.parameter = Parameter.create(name='params', type='group', children=params)

            self.parameterTree = ParameterTree()
            self.parameterTree.setParameters(self.parameter, showTop=False)

            self.parameter.sigTreeStateChanged.connect(display.redraw)

            self._form = self.parameterTree
        return self._form

    def hideAll(self):
        self.Radius.hide()
        self.Length.hide()
        self.Width.hide()
        self.Height.hide()
        self.BaseAngle.hide()

    def typeChanged(self, _, choice):
        self.hideAll()
        for param in self.relevantParams():
            param.show()

    def wireup(self):
        super(particle, self).wireup()
        self.form
        self.setConnected(True)

        self.LatticeChoice.sigValueChanged.connect(self.changeUnitCellType)

    def showStructure(self):
        ui.showForm(self.structure.form)

    def showEnsemble(self):
        ui.showForm(self.ensemble.form)

    def mousePressEvent(self, *args, **kwargs):
        super(particle, self).mousePressEvent(*args, **kwargs)

        self.frame_2.show()

    def relevantParams(self):
        choice = self.Type.value()
        if choice == 'Sphere':
            return [self.Radius]
        elif choice == 'Box':
            return [self.Length, self.Width, self.Height]
        elif choice == 'Cylinder':
            return [self.Radius, self.Height]
        elif choice == 'Pyramid':
            return [self.Length, self.Width, self.Height, self.BaseAngle]

    def toDict(self):

        return UnsortableOrderedDict([('name', self.Type.value().lower()),
                                      ('key', self.name),
                                      ('params', [param.toDict() for param in self.relevantParams()]),
                                      ('xrot',self.XRotation.value()),
                                      ('yrot', self.YRotation.value()),
                                      ('zrot', self.ZRotation.value()),
                                      ('refindex', {'delta': self.delta.value(), 'beta': self.beta.value()})])


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


class ensemble(form):
    def __init__(self, parent, name='Ensemble'):
        self.parent = parent
        self._formpath = 'xicam/gui/guiEnsemble.ui'
        super(ensemble, self).__init__(name)


        # def wireup(self):
        # super(ensemble, self).wireup()
        # self.form.addGrainButton.clicked.connect(self.addGrain)

        # def addGrain(self):
        # self.addChild(grain())
        #     featuresmodel.layoutChanged.emit()

    @property
    def form(self):
        if self._form is None:
            # self.Axis = pTypes.ListParameter(name='Axis', values=['None', 'x', 'y', 'z'], value=0)
            self.addRotationAction = pTypes.ActionParameter(name='Add Rotation')
            self.addRotationAction.sigActivated.connect(self.addRotation)

            params = [self.addRotationAction]

            self.parameter = Parameter.create(name='params', type='group', children=params)

            self.parameterTree = ParameterTree()
            self.parameterTree.setParameters(self.parameter, showTop=False)
            self._form = self.parameterTree
        return self._form

    def addRotation(self):
        n = len(self.parameter.children())
        self.parameter.addChild(RotationParameter(name='Rotation ' + str(n)))

    def changeAxis(self, _, choice):
        # print choice
        if choice == 'None':
            self.Rotation.hide()
        else:
            self.Rotation.show()

    def toDict(self):
        # check that all the distributions are the same
        allsame=True
        if len(self.parameter.children())>1:
            disttype=self.parameter.children()[1].AngleDist.DistributionChoice.value()
            for rot in [rot for rot in self.parameter.children() if type(rot) is RotationParameter]:
                if not rot.AngleDist.DistributionChoice.value() == disttype:
                    allsame = False
        else:
            return dict()

        if not allsame:
            from xicam import dialogs
            dialogs.infodialog("HipGISAXS currently supports only 1 type of orientation distribution at a time. Please edit the ensemble.","Too many distributions")
            raise NotImplementedError

        stat = {'Uniform':'range','Random':'random','Bragg':'bragg','Gaussian':'gaussian'}

        rots = [('rot' + str(self.parameter.children().index(rot)), rot.toDict()) for rot in self.parameter.children()
                if type(rot) is RotationParameter]
        rots = [('stat',stat[disttype])]+rots
        return UnsortableOrderedDict(rots)


class structure(form):
    sigUpdateDisplayUnitCell = QtCore.Signal()

    def __init__(self, parent, name='Structure'):
        self.parent = parent
        self._formpath = 'xicam/gui/guiStructure.ui'
        self.vector1 = vector()
        self.vector2 = vector()
        self.vector3 = vector()
        super(structure, self).__init__(name)

        # Testing

        # {'name': 'Basic parameter data types', 'type': 'group', 'children': [
        # {'name': 'Integer', 'type': 'int', 'value': 10},
        #         {'name': 'Float', 'type': 'float', 'value': 10.5, 'step': 0.1},
        #         {'name': 'String', 'type': 'str', 'value': "hi"},
        #         {'name': 'List', 'type': 'list', 'values': [1,2,3], 'value': 2},
        #         {'name': 'Named List', 'type': 'list', 'values': {"one": 1, "two": "twosies", "three": [3,3,3]}, 'value': 2},
        #         {'name': 'Boolean', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
        #         {'name': 'Color', 'type': 'color', 'value': "FF0", 'tip': "This is a color button"},
        #         {'name': 'Gradient', 'type': 'colormap'},
        #         {'name': 'Subgroup', 'type': 'group', 'children': [
        #             {'name': 'Sub-param 1', 'type': 'int', 'value': 10},
        #             {'name': 'Sub-param 2', 'type': 'float', 'value': 1.2e6},
        #             ]},
        #         {'name': 'Text Parameter', 'type': 'text', 'value': 'Some text...'},
        #         {'name': 'Action Parameter', 'type': 'action'},
        #         ]},
        #     {'name': 'Numerical Parameter Options', 'type': 'group', 'children': [
        #         {'name': 'Units + SI prefix', 'type': 'float', 'value': 1.2e-6, 'step': 1e-6, 'siPrefix': True, 'suffix': 'V'},
        #         {'name': 'Limits (min=7;max=15)', 'type': 'int', 'value': 11, 'limits': (7, 15), 'default': -6},
        #         {'name': 'DEC stepping', 'type': 'float', 'value': 1.2e6, 'dec': True, 'step': 1, 'siPrefix': True, 'suffix': 'Hz'},
        #
        #         ]},
        #     {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
        #         {'name': 'Save State', 'type': 'action'},
        #         {'name': 'Restore State', 'type': 'action', 'children': [
        #             {'name': 'Add missing items', 'type': 'bool', 'value': True},
        #             {'name': 'Remove extra items', 'type': 'bool', 'value': True},
        #             ]},
        #         ]},
        #     {'name': 'Extra Parameter Options', 'type': 'group', 'children': [
        #         {'name': 'Read-only', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'readonly': True},
        #         {'name': 'Renamable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'renamable': True},
        #         {'name': 'Removable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'removable': True},
        #         ]},

    @property
    def form(self):
        if self._form is None:
            self.dimensions = pTypes.ListParameter(name='Dimensions', type='list', value='3D',
                                                   values=['1D', '2D', '3D'])

            # 1-D/2D type
            self.non3DType = pTypes.ListParameter(name='Type', type='list', values=['Paracrystal', 'Percus Yevick'])
            self.non3DType.hide()
            self.domainsize = pTypes.SimpleParameter(name='Domain Size', type='float', value=100)
            self.domainsize.hide()
            self.non3Dlattice = pTypes.ListParameter(name='Planar Lattice', type='list', values=['Hexagonal','Cubic'])
            self.non3Dlattice.hide()

            # 1-D
            self.xmean = pTypes.SimpleParameter(name='Mean', type='float', value=100)
            self.xstd = pTypes.SimpleParameter(name='Std', type='float', value=1)
            self.xspacing = hideableGroupParameter(name='X Spacing', children=[self.xmean, self.xstd])
            self.xspacing.hide()

            # 2-D
            self.ymean = pTypes.SimpleParameter(name='Mean', type='float', value=100)
            self.ystd = pTypes.SimpleParameter(name='Std', type='float', value=1)
            self.yspacing = hideableGroupParameter(name='Y Spacing', children=[self.ymean, self.ystd])
            self.yspacing.hide()

            # 3-D
            self.LatticeA = VectorParameter(name='A', value=(30, 0, 0))
            self.LatticeB = VectorParameter(name='B', value=(0, 30, 0))
            self.LatticeC = VectorParameter(name='C', value=(0, 0, 30))
            self.LatticeChoice = pTypes.ListParameter(name='Type', type='list',
                                                      values=["Custom...", "Simple Cubic", "Body Centered Cubic",
                                                              'Face Centered Cubic', 'Hexagonal'])
            self.Repetition = IntVectorParameter(name='Repetition', value=(2, 2, 2), type=int,step=1)
            self.Scaling = pTypes.SimpleParameter(name='Scaling', value=1, type='float')
            self.Lattice = hideableGroupParameter(name='Lattice', children=[self.LatticeChoice,
                                                                            self.LatticeA,
                                                                            self.LatticeB,
                                                                            self.LatticeC])

            self.Basis = ScalableGroup(name='Basis', children=[VectorParameter(name='Point 1')])
            self.Position = VectorParameter(name='Position')
            self.iratio = pTypes.SimpleParameter(name='I Ratio', value=1, type='float')

            params = [self.dimensions,
                      self.non3DType,
                      self.non3Dlattice,
                      self.xspacing,
                      self.yspacing,
                      self.domainsize,
                      self.Lattice,
                      self.Repetition,
                      self.Scaling,
                      self.Basis,
                      self.Position,
                      self.iratio
                      ]

            self.parameter = Parameter.create(name='params', type='group', children=params)

            self.parameterTree = ParameterTree()
            self.parameterTree.setParameters(self.parameter, showTop=False)
            self._form = self.parameterTree

            self. parameter.sigTreeStateChanged.connect(display.redraw)
        return self._form

    def wireup(self):
        super(structure, self).wireup()
        self.form
        self.setConnected(True)

        self.LatticeChoice.sigValueChanged.connect(self.changeUnitCellType)
        self.dimensions.sigValueChanged.connect(self.changeDimensions)

    def setConnected(self, connect):

        for item in [self.LatticeA, self.LatticeB, self.LatticeC]:
            if connect:
                #item.sigValueChanged.connect(self.showUnitCell)
                item.sigValueChanged.connect(self.switchtoCustom)
            else:
                #item.sigValueChanged.disconnect(self.showUnitCell)
                item.sigValueChanged.disconnect(self.switchtoCustom)

    # def showUnitCell(self):
    #     display.redraw()


    def switchtoCustom(self):
        self.LatticeChoice.setValue(0)

    def changeDimensions(self, _, choice):
        self.resetDimensions()

        if choice == '1D':
            self.xspacing.show()
            self.non3DType.show()
            self.domainsize.show()
            self.non3Dlattice.show()
        elif choice == '2D':
            self.xspacing.show()
            self.yspacing.show()
            self.non3DType.show()
            self.domainsize.show()
            self.non3Dlattice.show()
        elif choice == '3D':
            self.Lattice.show()
            self.Basis.show()
            self.Scaling.show()
            self.Repetition.show()

        #self.showUnitCell()

    def resetDimensions(self):
        self.domainsize.hide()
        self.non3DType.hide()
        self.xspacing.hide()
        self.yspacing.hide()
        self.Lattice.hide()
        self.Basis.hide()
        self.Scaling.hide()
        self.Repetition.hide()
        self.non3Dlattice.show()

    def changeUnitCellType(self, _, choice):

        self.setConnected(False)

        if choice == 'Simple Cubic':
            self.LatticeA.setValue((30, 0, 0))
            self.LatticeB.setValue((0, 30, 0))
            self.LatticeC.setValue((0, 0, 30))

        elif choice == 'Body Centered Cubic':
            self.LatticeA.setValue((-30, 30, 15))
            self.LatticeB.setValue((30, 30, 30))
            self.LatticeC.setValue((-30, -30, 30))

        elif choice == 'Face Centered Cubic':
            self.LatticeA.setValue((30, 30, 0))
            self.LatticeB.setValue((0, 30, 30))
            self.LatticeC.setValue((30, 0, 30))

        elif choice == 'Hexagonal Planar':
            self.LatticeA.setValue((10*3, 0, 0))
            self.LatticeB.setValue((5*3, 3 ** .5 * .5*10.*3, 0))
            self.LatticeC.setValue((0, 0, 10*3))

        # TODO: Add HCP

        #self.showUnitCell()

        self.setConnected(True)

    def toStructureDict(self):

        if len(self.parent.ensemble.toDict()):
            ensemble = [('maxgrains', [360, 1, 1])]
            ensemble.append(('orientations', self.parent.ensemble.toDict()))
        else:
            ensemble = [('maxgrains', [1, 1, 1])]

        return UnsortableOrderedDict([('key', 'st' + self.parent.name),
                                      ('iratio', self.iratio.value()),
                                      ('transvec', list(self.Position.value())),
                                      ('grain', self.toGrainDict()),
                                      ('ensemble', UnsortableOrderedDict(ensemble))])

    def toUnitCellDict(self):
        return UnsortableOrderedDict([('key', 'u' + self.parent.name),
                                      ('elements', [UnsortableOrderedDict([('shape_key', self.parent.name),
                                                                           ('locations', self.Basis.toArray())])])])

    def toGrainDict(self):
        z = featuremanager.features.index(self.parent)-1
        # get next lower layer
        while type(featuremanager.features[z]) is not layer and z>=0:
            z-=1
        if type(featuremanager.features[z]) is layer:
            layer_key=featuremanager.features[z].name
        else:
            layer_key=''
        grain = [('unitcell_key', 'u' + self.parent.name),
                 ('layer_key', layer_key)]  # IMPORTANT! How should this be determined?...
        if self.dimensions.value() == '1D':
            grain.extend([('type', self.non3DType.value().lower()),
                          ('lattice',self.non3Dlattice.value().lower()),
                          ('xspacing', UnsortableOrderedDict([('mean',self.xmean.value()),
                                                              ('std',self.xstd.value())])),
                          ('domain', self.domainsize.value())])  # Where does hex/cubic go?...
        elif self.dimensions.value() == '2D':
            grain.extend([('type', self.non3DType.value().lower()),
                          ('lattice',self.non3Dlattice.value().lower()),
                          ('xspacing', UnsortableOrderedDict([('mean',self.xmean.value()),
                                                              ('std',self.xstd.value())])),
                          ('yspacing', UnsortableOrderedDict([('mean',self.ymean.value()),
                                                              ('std',self.ystd.value())])),
                          ('domain', self.domainsize.value())])
        else:
            grain.extend([('lattice', UnsortableOrderedDict([('a', list(self.LatticeA.value())),
                                                             ('b', list(self.LatticeB.value())),
                                                             ('c', list(self.LatticeC.value())),
                                                             # ('basis', self.Basis.toDict()),
                                                             ])),
                          ('scaling', self.Scaling.value()),
                          ('repetition', list(self.Repetition.value()))
                          ])
        return UnsortableOrderedDict(grain)


def loadform(path):
    guiloader = QUiLoader()
    f = QtCore.QFile(path)
    f.open(QtCore.QFile.ReadOnly)
    form = guiloader.load(f)
    f.close()
    return form


class scattering(form):
    def __init__(self, name='Scattering'):
        super(scattering, self).__init__(name)

    @property
    def form(self):
        if self._form is None:
            self.Experiment = pTypes.ListParameter(name='Experiment', values=['GISAXS', 'SAXS'], value=0)
            self.Energy = pTypes.SimpleParameter(name='Photon Energy', type='int', value=10000, suffix='eV',
                                                 siPrefix=True, step=100)
            self.Incidence = StepParameter(name='Incidence Angle(s)', min=.1, max=0, step=0)

            # self.Rotation = StepParameter(name='In-plane Rotation')
            # self.Tilt = StepParameter(name='Tilt', suffix='deg')

            self.IncidenceAngleVisual = pTypes.SimpleParameter(name='Incidence angle', value=0.120, suffix=' deg',
                                                               type='float')
            # self.TiltAngleVisual = pTypes.SimpleParameter(name='Tilt Angle', value=0, suffix=' deg', type='float')
            self.Visualization = pTypes.GroupParameter(name='Beam Visualization (display only)',
                                                       children=[self.IncidenceAngleVisual, self.TiltAngleVisual])

            params = [self.Experiment, self.Energy, self.Incidence, self.Visualization]

            self.parameter = Parameter.create(name='params', type='group', children=params)

            self.parameterTree = ParameterTree()
            self.parameterTree.setParameters(self.parameter, showTop=False)
            self._form = self.parameterTree
        return self._form

    def toDict(self):
        return UnsortableOrderedDict([('pathprefix', '.'),
                                      ('runname', 'BLAH'),
                                      ('expt', self.Experiment.value().lower()),
                                      ('alphai', self.Incidence.toDict()),
                                      ('photon', UnsortableOrderedDict([('value', self.Energy.value()),
                                                                        ('unit', 'ev')])),
                                      ('detector', UnsortableOrderedDict([('pixelsize', 0.172),
                                                                          ('sdd', 4128.62),
                                                                          ('directbeam', [489.916, 843.076])]))])


class detector(form):
    def __init__(self, name='Computation'):
        super(detector, self).__init__(name)

    @property
    def form(self):
        if self._form is None:
            detectornames = ['Custom...'] + [detector().name for detector in detectors.ALL_DETECTORS.values()]
            self.DetectorChoice = pTypes.ListParameter(name='Detector Model', values=detectornames, value='Custom...')
            self.DetectorChoice.sigValueChanged.connect(self.changeDetector)

            self.Width = pTypes.SimpleParameter(name='Width', type='int', suffix=' px', value=300)
            self.Height = pTypes.SimpleParameter(name='Height', type='int', suffix=' px', value=300)
            self.DetectorResolution = pTypes.GroupParameter(name='Detector Resolution',
                                                            children=[self.Width, self.Height])
            self.setConnected(True)

            self.Qparallel = MinMaxParameter(name='Q Parallel', min=-2, max=2)
            self.Qz = MinMaxParameter(name='Q Z', min=0, max=4)
            self.Qrange = pTypes.GroupParameter(name='Q Range', children=[self.Qparallel, self.Qz])

            self.smearing = pTypes.SimpleParameter(name='Smearing', value=0, type='float')

            self.Experiment = pTypes.ListParameter(name='Experiment', values=['GISAXS', 'SAXS'], value=0)
            self.Energy = pTypes.SimpleParameter(name='Photon Energy', type='int', value=10000, suffix='eV',
                                                 siPrefix=True, step=100)
            self.Incidence = StepParameter(name='Incidence Angle(s)', min=.1, max=0, step=0)

            params = [self.DetectorResolution, self.Qrange, self.smearing, self.Experiment, self.Incidence]

            self.parameter = Parameter.create(name='params', type='group', children=params)

            self.parameterTree = ParameterTree()
            self.parameterTree.setParameters(self.parameter, showTop=False)
            self._form = self.parameterTree
        return self._form

    def setConnected(self, connect):
        if connect:
            self.Width.sigValueChanged.connect(self.shapeChanged)
            self.Height.sigValueChanged.connect(self.shapeChanged)
        else:
            self.Width.sigValueChanged.disconnect(self.shapeChanged)
            self.Height.sigValueChanged.disconnect(self.shapeChanged)

    def changeDetector(self, _, choice):
        if not choice == 'Custom...':
            self.setConnected(False)
            detectornames = [detector().name for detector in detectors.ALL_DETECTORS.values()]
            detectorindex = detectornames.index(choice)
            detector = detectors.ALL_DETECTORS.values()[detectorindex]()
            self.Width.setValue(detector.MAX_SHAPE[1])
            self.Height.setValue(detector.MAX_SHAPE[0])
            self.setConnected(True)

    def shapeChanged(self):
        self.DetectorChoice.setValue('Custom...')

    def toDict(self):
        return UnsortableOrderedDict([('path', '.'),
                                      ('runname', 'BLAH'),
                                      ('expt', self.Experiment.value().lower()),
                                      ('alphai', self.Incidence.toDict()),
                                      ('photon', UnsortableOrderedDict([('energy', self.Energy.value()),
                                                                        ('unit', 'ev')])),
                                      ('smearing', 0),
                                      ('detector', self.DetectorChoice.value()),
                                      ('output', UnsortableOrderedDict([('type', 'qspace'),
                                                                        ('minpoint', [self.Qparallel.value()[0],
                                                                                      self.Qz.value()[0]]),
                                                                        ('maxpoint', [self.Qparallel.value()[1],
                                                                                      self.Qz.value()[1]])])),
                                      ('resolution', [self.Width.value(), self.Height.value()])
                                      ])


class VectorParameterItem(pTypes.WidgetParameterItem):
    def makeWidget(self, cls = vector):
        w = cls()
        opts = self.param.opts
        value = opts.get('value', None)
        if value is not None:
            w.setValue(value)

        self.value = w.value
        self.setValue = w.setValue

        return w

    def valueChanged(self, *args, **kwargs):
        super(VectorParameterItem, self).valueChanged(*args, **kwargs)

class IntVectorParameterItem(VectorParameterItem):
    def makeWidget(self):
        return super(IntVectorParameterItem, self).makeWidget(cls=intvector)


class VectorParameter(Parameter):
    itemClass = VectorParameterItem

    def __init__(self, *args, **kwargs):
        super(VectorParameter, self).__init__(*args, **kwargs)

    def defaultValue(self):
        return (0, 0, 0)

class IntVectorParameter(VectorParameter):
    itemClass = IntVectorParameterItem

registerParameterType('Vector', VectorParameter, override=True)


class hideableGroupParameterItem(pTypes.GroupParameterItem):
    def optsChanged(self, param, opts):
        super(hideableGroupParameterItem, self).optsChanged(param, opts)
        if 'visible' in opts:
            self.setHidden(not opts['visible'])


class hideableGroupParameter(pTypes.GroupParameter):
    itemClass = hideableGroupParameterItem


class ScalableGroup(hideableGroupParameter):
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


class RotationDistParameter(pTypes.GroupParameter):
    itemClass = hideableGroupParameterItem

    def __init__(self, value=None, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)

        self.DistributionChoice = pTypes.ListParameter(name='Distribution', type='list', value=0,
                                                       values=['Uniform', 'Random', 'Gaussian', 'Bragg'])
        self.Min = pTypes.SimpleParameter(name='Minimum', type='float', value=0)
        self.Max = pTypes.SimpleParameter(name='Maximum', type='float', value=0)
        self.Mean = pTypes.SimpleParameter(name='Mean', type='float', value=0)
        self.Variance = pTypes.SimpleParameter(name='Variance', type='float', value=0)

        self.Mean.hide()
        self.Variance.hide()

        self.DistributionChoice.sigValueChanged.connect(self.distributionChanged)

        self.addChildren([self.DistributionChoice, self.Min, self.Max, self.Mean, self.Variance])

    def distributionChanged(self, _, choice):
        # print choice
        if choice == 'Uniform':
            self.Min.show()
            self.Max.show()
            self.Mean.hide()
            self.Variance.hide()
        elif choice == 'Random':
            self.Min.show()
            self.Max.show()
            self.Mean.hide()
            self.Variance.hide()
        elif choice == 'Gaussian':
            self.Min.show()
            self.Max.show()
            self.Mean.show()
            self.Variance.show()
        elif choice == 'Bragg':
            self.Min.hide()
            self.Max.hide()
            self.Mean.hide()
            self.Variance.hide()

    def toDict(self):
        d = UnsortableOrderedDict()
        choice = self.DistributionChoice.value()

        if choice == 'Uniform':
            d['min'] = self.Min.value()
            d['max'] = self.Max.value()
            d['stat'] = 'uniform'
        elif choice == 'Random':
            d['min'] = self.Min.value()
            d['max'] = self.Max.value()
            d['stat'] = 'random'
        elif choice == 'Gaussian':
            d['min'] = self.Min.value()
            d['max'] = self.Max.value()
            d['std'] = float(np.sqrt(self.Variance.value()))
            d['mean'] = self.Mean.value()
            d['stat'] = 'gaussian'
        return d


class DistParameter(pTypes.GroupParameter):
    itemClass = hideableGroupParameterItem

    def __init__(self, value=None, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)

        self.DistributionChoice = pTypes.ListParameter(name='Distribution', type='list', value=0,
                                                       values=['Single value', 'Uniform', 'Random', 'Gaussian'])
        self.Value = pTypes.SimpleParameter(name=opts['name'], type='float', value=value)
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

        self.addChildren([self.DistributionChoice, self.Value, self.Min, self.Max, self.Mean, self.Variance, self.N])

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

        if 'higKey' in self.opts:
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
            d['std'] = np.sqrt(self.Variance.value())
            d['mean'] = self.Mean.value()
            d['stat'] = 'gaussian'
        return d


class RotationParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        super(RotationParameter, self).__init__(**opts)

        self.AxisChoice = pTypes.ListParameter(name='Axis', type='list', value=0,
                                               values=['X', 'Y', 'Z'])
        self.AngleDist = RotationDistParameter(name='Angles')
        self.delete = pTypes.ActionParameter(name='Remove Rotation')
        self.delete.sigActivated.connect(self.remove)  # Will this work?

        self.addChildren([self.AxisChoice, self.AngleDist, self.delete])

    def toDict(self):
        d = self.AngleDist.toDict()
        d['axis'] = self.AxisChoice.value().lower()
        return d


class StepParameter(pTypes.GroupParameter):
    itemClass = hideableGroupParameterItem

    def __init__(self, min=None, max=None, step=None, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)

        self.Min = pTypes.SimpleParameter(name='Minimum', type='float', value=min)
        self.Max = pTypes.SimpleParameter(name='Maximum', type='float', value=max)
        self.Step = pTypes.SimpleParameter(name='Step', type='float', value=step)

        self.addChildren([self.Min, self.Max, self.Step])

    def toDict(self):
        d = dict()
        d['min'] = self.Min.value()
        d['max'] = self.Max.value()
        d['step'] = self.Step.value()

        return d


class MinMaxParameter(pTypes.GroupParameter):
    itemClass = hideableGroupParameterItem

    def __init__(self, min=None, max=None, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)

        self.Min = pTypes.SimpleParameter(name='Minimum', type='float', value=min)
        self.Max = pTypes.SimpleParameter(name='Maximum', type='float', value=max)

        self.addChildren([self.Min, self.Max])

    def value(self):
        return [self.Min.value(), self.Max.value()]

    def toDict(self):
        d = UnsortableOrderedDict()
        d['min'] = self.Min.value()
        d['max'] = self.Max.value()

        return d
