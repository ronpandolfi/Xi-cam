import os
from functools import partial
from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
from psutil import cpu_count
import pyqtgraph as pg
from pyqtgraph import parametertree as pt
import toolbar as ttoolbar
import config
import manager
import widgets

blankform = None
propertytable = None
configparams = None
paramformstack = None
functionwidget = None
centerwidget = None
bottomwidget = None


def loadUi():
    global blankform, propertytable, configparams, functionwidget, paramformstack, centerwidget, bottomwidget

    toolbar = ttoolbar.tomotoolbar()

    centerwidget = QtGui.QTabWidget()

    centerwidget.setDocumentMode(True)
    centerwidget.setTabsClosable(True)

    bottomwidget = widgets.RunViewer()

    # Load the gui from file
    functionwidget = QUiLoader().load('gui/tomographyleft.ui')

    # Add some tool tips
    functionwidget.addFunctionButton.setToolTip('Add function to pipeline')
    functionwidget.clearButton.setToolTip('Clear pipeline')
    functionwidget.fileButton.setToolTip('Save/Load pipeline')
    functionwidget.moveDownButton.setToolTip('Move selected function down')
    functionwidget.moveUpButton.setToolTip('Move selected function up')

    functionwidget.clearButton.clicked.connect(manager.clear_action)
    functionwidget.moveUpButton.clicked.connect(
        lambda: manager.swap_functions(manager.currentindex,
                                       manager.currentindex - 1))
    functionwidget.moveDownButton.clicked.connect(
        lambda: manager.swap_functions(manager.currentindex,
                                       manager.currentindex + 1))

    addfunctionmenu = QtGui.QMenu()
    buildfunctionmenu(addfunctionmenu, config.funcs['Functions'], manager.add_action)

    functionwidget.addFunctionButton.setMenu(addfunctionmenu)
    functionwidget.addFunctionButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
    functionwidget.addFunctionButton.setArrowType(QtCore.Qt.NoArrow)

    filefuncmenu = QtGui.QMenu()
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap("gui/icons_55.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    openaction = QtGui.QAction(icon, 'Open', filefuncmenu,)
    openaction.triggered.connect(manager.open_pipeline_file)
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap("gui/icons_59.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    saveaction = QtGui.QAction(icon, 'Save', filefuncmenu)
    saveaction.triggered.connect(lambda :manager.save_function_pipeline(manager.create_pipeline_dict()))
    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap("gui/icons_56.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    refreshaction = QtGui.QAction(icon, 'Refresh', filefuncmenu)
    refreshaction.triggered.connect(lambda: manager.load_function_pipeline('yaml/tomography/default_pipeline.yml'))
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
    l.addWidget(paramformstack)
    l.addWidget(functionwidget)

    leftwidget.setLayout(l)
    icon = QtGui.QIcon(QtGui.QPixmap("gui/icons_49.png"))
    leftmodes = [(leftwidget, icon)]

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
              {'name': 'Sinograms/Chunk', 'type': 'int', 'value': 20*cpu_count()},
              {'name': 'CPU Cores', 'type': 'int', 'value': cpu_count(), 'default': cpu_count(),
               'limits':[1, cpu_count()]}]

    configparams = pt.Parameter.create(name='Configuration', type='group', children=params)
    configtree.setParameters(configparams, showTop=False)
    # configparams.param('Browse').sigActivated.connect(
    #     lambda: configparams.param('Output Name').setValue(
    #         str(QtGui.QFileDialog.getSaveFileName(None, 'Save reconstruction as',
    #                                               configparams.param('Output Name').value())[0])))
    rightwidget.addWidget(configtree)

    propertytable = pg.TableWidget() #QtGui.QTableView()
    propertytable.verticalHeader().hide()
    propertytable.horizontalHeader().setStretchLastSection(True)
    propertytable.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)

    rightwidget.addWidget(propertytable)
    propertytable.hide()
    rightmodes = [(rightwidget, QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.File))]

    blankform = QtGui.QLabel('Select a function from\n below to set parameters...')
    blankform.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
    blankform.setAlignment(QtCore.Qt.AlignCenter)
    showform(blankform)

    return leftmodes, centerwidget, rightwidget, bottomwidget, toolbar


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

