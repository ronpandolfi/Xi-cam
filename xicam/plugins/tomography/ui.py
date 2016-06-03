import os
from functools import partial
import numpy as np
from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
from psutil import cpu_count
import pyqtgraph as pg
from pyqtgraph import parametertree as pt
import toolbar as ttoolbar
import fdata
import fmanager

blankform = None
propertytable = None
configparams = None
paramformstack = None
functionwidget = None
centerwidget = None


def loadUi():
    global blankform, propertytable, configparams, functionwidget, paramformstack, centerwidget

    toolbar = ttoolbar.tomotoolbar()

    centerwidget = QtGui.QTabWidget()

    centerwidget.setDocumentMode(True)
    centerwidget.setTabsClosable(True)

    bottomwidget = None

    # Load the gui from file
    functionwidget = QUiLoader().load('gui/tomographyleft.ui')

    functionwidget.clearButton.clicked.connect(fmanager.clear_action)
    functionwidget.moveUpButton.clicked.connect(
        lambda: fmanager.swap_functions(fmanager.currentindex,
                                        fmanager.currentindex - 1))
    functionwidget.moveDownButton.clicked.connect(
        lambda: fmanager.swap_functions(fmanager.currentindex,
                                        fmanager.currentindex + 1))

    addfunctionmenu = QtGui.QMenu()
    buildfunctionmenu(addfunctionmenu, fdata.funcs['Functions'], fmanager.add_action)

    functionwidget.addFunctionButton.setMenu(addfunctionmenu)
    functionwidget.addFunctionButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
    functionwidget.addFunctionButton.setArrowType(QtCore.Qt.NoArrow)

    filefuncmenu = QtGui.QMenu()
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap("gui/open_32.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    openaction = QtGui.QAction(icon, 'Open', filefuncmenu,)
    openaction.triggered.connect(fmanager.open_pipeline_file)
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap("gui/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    saveaction = QtGui.QAction(icon, 'Save', filefuncmenu)
    saveaction.triggered.connect(lambda :fmanager.save_function_pipeline(fmanager.create_pipeline_dict()))
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap("gui/refresh.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    refreshaction = QtGui.QAction(icon, 'Refresh', filefuncmenu)
    refreshaction.triggered.connect(lambda: fmanager.load_function_pipeline(
                                                           'yaml/tomography/default_pipeline.yml'))
    filefuncmenu.addActions([openaction, saveaction, refreshaction])

    functionwidget.fileButton.setMenu(filefuncmenu)
    functionwidget.fileButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
    functionwidget.fileButton.setArrowType(QtCore.Qt.NoArrow)

    leftwidget = QtGui.QWidget()

    l = QtGui.QVBoxLayout()
    l.setContentsMargins(0, 0, 0, 0)

    paramtree = pt.ParameterTree()
    paramformstack = QtGui.QStackedWidget()
    paramformstack.addWidget(paramtree)
    paramformstack.setFixedHeight(160)
    l.addWidget(paramformstack)
    l.addWidget(functionwidget)

    leftwidget.setLayout(l)

    rightwidget = QtGui.QSplitter(QtCore.Qt.Vertical)

    configtree = pt.ParameterTree()
    configtree.setMinimumHeight(230)

    params = [{'name': 'Start Sinogram', 'type': 'int', 'value': 0, 'default': 0, },
              {'name': 'End Sinogram', 'type': 'int'},
              {'name': 'Step Sinogram', 'type': 'int', 'value': 1, 'default': 1},
              {'name': 'Start Projection', 'type': 'int', 'value': 0, 'default': 0},
              {'name': 'End Projection', 'type': 'int'},
              {'name': 'Step Projection', 'type': 'int', 'value': 1, 'default': 1},
              # {'name': 'Ouput Format', 'type': 'list', 'values': ['TIFF (.tiff)'], 'default': 'TIFF (.tiff)'},
              # {'name': 'Output Name', 'type': 'str'},
              # {'name': 'Browse', 'type': 'action'},
              {'name': 'Cores', 'type': 'int', 'value': cpu_count(), 'default': cpu_count(), 'limits':[1, cpu_count()]},
              {'name': 'Sinogram Chunks', 'type': 'int', 'value': 1},
              {'name': 'Sinograms/Chunk', 'type': 'int', 'value': 0}]

    configparams = pt.Parameter.create(name='Configuration', type='group', children=params)
    configtree.setParameters(configparams, showTop=False)
    # configparams.param('Browse').sigActivated.connect(
    #     lambda: configparams.param('Output Name').setValue(
    #         str(QtGui.QFileDialog.getSaveFileName(None, 'Save reconstruction as',
    #                                               configparams.param('Output Name').value())[0])))

    sinostart = configparams.param('Start Sinogram')
    sinoend = configparams.param('End Sinogram')
    sinostep = configparams.param('Step Sinogram')
    nsino = lambda: (sinoend.value() - sinostart.value() + 1) // sinostep.value()
    chunks = configparams.param('Sinogram Chunks')
    sinos = configparams.param('Sinograms/Chunk')
    chunkschanged = lambda: sinos.setValue(np.round(nsino() / chunks.value()), blockSignal=sinoschanged)
    sinoschanged = lambda: chunks.setValue((nsino() - 1) // sinos.value() + 1, blockSignal=chunkschanged)
    chunks.sigValueChanged.connect(chunkschanged)
    sinos.sigValueChanged.connect(sinoschanged)
    sinostart.sigValueChanged.connect(chunkschanged)
    sinoend.sigValueChanged.connect(chunkschanged)
    sinostep.sigValueChanged.connect(chunkschanged)
    chunks.setValue(1)

    rightwidget.addWidget(configtree)

    propertytable = pg.TableWidget() #QtGui.QTableView()
    propertytable.verticalHeader().hide()
    propertytable.horizontalHeader().setStretchLastSection(True)
    propertytable.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)

    rightwidget.addWidget(propertytable)
    propertytable.hide()


    blankform = QtGui.QLabel('Select a function from\n below to set parameters...')
    blankform.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
    blankform.setAlignment(QtCore.Qt.AlignCenter)
    showform(blankform)

    return leftwidget, centerwidget, rightwidget, bottomwidget, toolbar


def showform(widget):
    paramformstack.addWidget(widget)
    paramformstack.setCurrentWidget(widget)


def buildfunctionmenu(menu, fdata, actionslot):
    for func,subfuncs in fdata.iteritems():
        if len(subfuncs) > 1 or func != subfuncs[0]:
            funcmenu = QtGui.QMenu(func)
            menu.addMenu(funcmenu)
            for subfunc in subfuncs:
                if isinstance(subfuncs, dict) and len(subfuncs[subfunc]) > 0:
                    optsmenu = QtGui.QMenu(subfunc)
                    funcmenu.addMenu(optsmenu)
                    for opt in subfuncs[subfunc]:
                        funcaction = QtGui.QAction(opt, funcmenu)
                        funcaction.triggered.connect(partial(actionslot, func, opt))
                        optsmenu.addAction(funcaction)
                else:
                    funcaction = QtGui.QAction(subfunc, funcmenu)
                    funcaction.triggered.connect(partial(actionslot, func, subfunc))
                    funcmenu.addAction(funcaction)
        elif len(subfuncs) == 1:
            funcaction = QtGui.QAction(func, menu)
            funcaction.triggered.connect(partial(actionslot, func, func))
            menu.addAction(funcaction)


def setconfigparams(sino, proj):
    configparams.child('End Sinogram').setValue(sino)
    configparams.child('End Sinogram').setLimits([0, sino])
    configparams.child('Start Sinogram').setLimits([0, sino])
    configparams.child('Step Sinogram').setLimits([0, sino])
    configparams.child('End Projection').setValue(proj)
    configparams.child('End Projection').setLimits([0, proj])
    configparams.child('Start Projection').setLimits([0, proj])
    configparams.child('Step Projection').setLimits([0, proj])
    # configparams.child('Output Name').setValue(outname)

