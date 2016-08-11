from functools import partial
from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
from psutil import cpu_count
import pyqtgraph as pg
from pyqtgraph import parametertree as pt
import reconpkg
import toolbar
import widgets

property_table = None
config_params = None
paramformstack = None
functionwidget = None
centerwidget = None
bottomwidget = None


class UIform(object):
    def setupUi(self):

        self.toolbar = toolbar.tomotoolbar()
        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.bottomwidget = widgets.RunViewer()
        self.functionwidget = QUiLoader().load('gui/tomographyleft.ui')
        self.functionwidget.functionsList.setAlignment(QtCore.Qt.AlignBottom)

        self.functionwidget.addFunctionButton.setToolTip('Add function to pipeline')
        self.functionwidget.clearButton.setToolTip('Clear pipeline')
        self.functionwidget.fileButton.setToolTip('Save/Load pipeline')
        self.functionwidget.moveDownButton.setToolTip('Move selected function down')
        self.functionwidget.moveUpButton.setToolTip('Move selected function up')

        self.addfunctionmenu = QtGui.QMenu()
        self.functionwidget.addFunctionButton.setMenu(self.addfunctionmenu)
        self.functionwidget.addFunctionButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.functionwidget.addFunctionButton.setArrowType(QtCore.Qt.NoArrow)

        filefuncmenu = QtGui.QMenu()
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_55.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.openaction = QtGui.QAction(icon, 'Open', filefuncmenu,)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_59.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.saveaction = QtGui.QAction(icon, 'Save', filefuncmenu)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_56.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.refreshaction = QtGui.QAction(icon, 'Reset', filefuncmenu)
        filefuncmenu.addActions([self.openaction, self.saveaction, self.refreshaction])

        self.functionwidget.fileButton.setMenu(filefuncmenu)
        self.functionwidget.fileButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.functionwidget.fileButton.setArrowType(QtCore.Qt.NoArrow)

        leftwidget = QtGui.QSplitter(QtCore.Qt.Vertical)

        paramtree = pt.ParameterTree()
        self.param_form = QtGui.QStackedWidget()
        self.param_form.addWidget(paramtree)
        leftwidget.addWidget(self.param_form)
        leftwidget.addWidget(self.functionwidget)

        icon = QtGui.QIcon(QtGui.QPixmap("gui/icons_49.png"))
        self.leftmodes = [(leftwidget, icon)]

        rightwidget = QtGui.QSplitter(QtCore.Qt.Vertical)

        configtree = pt.ParameterTree()
        configtree.setMinimumHeight(230)

        params = [{'name': 'Start Sinogram', 'type': 'int', 'value': 0, 'default': 0, },
                  {'name': 'End Sinogram', 'type': 'int'},
                  {'name': 'Step Sinogram', 'type': 'int', 'value': 1, 'default': 1},
                  {'name': 'Start Projection', 'type': 'int', 'value': 0, 'default': 0},
                  {'name': 'End Projection', 'type': 'int'},
                  {'name': 'Step Projection', 'type': 'int', 'value': 1, 'default': 1},
                  # {'name': 'Browse', 'type': 'action'},
                  {'name': 'Sinograms/Chunk', 'type': 'int', 'value': 20*cpu_count()},
                  {'name': 'CPU Cores', 'type': 'int', 'value': cpu_count(), 'default': cpu_count(),
                   'limits':[1, cpu_count()]}]

        self.config_params = pt.Parameter.create(name='Configuration', type='group', children=params)
        configtree.setParameters(self.config_params, showTop=False)
        # self.config_params.param('Browse').sigActivated.connect(
        #     lambda: self.config_params.param('Output Name').setValue(
        #         str(QtGui.QFileDialog.getSaveFileName(None, 'Save reconstruction as',
        #                                               self.config_params.param('Output Name').value())[0])))
        rightwidget.addWidget(configtree)

        self.property_table = pg.TableWidget()
        self.property_table.verticalHeader().hide()
        self.property_table.horizontalHeader().setStretchLastSection(True)
        self.property_table.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)

        rightwidget.addWidget(self.property_table)
        self.property_table.hide()
        self.rightmodes = [(rightwidget, QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.File))]

    def connectTriggers(self, open, save, reset, moveup, movedown, clear):
        self.openaction.triggered.connect(open)
        self.saveaction.triggered.connect(save)
        self.refreshaction.triggered.connect(reset)
        self.functionwidget.moveDownButton.clicked.connect(moveup)
        self.functionwidget.moveUpButton.clicked.connect(movedown)
        self.functionwidget.clearButton.clicked.connect(clear)

    def setConfigParams(self, proj, sino):
        print 'Setting config params to ', proj, sino
        self.config_params.child('End Sinogram').setLimits([0, sino])
        self.config_params.child('Start Sinogram').setLimits([0, sino])
        self.config_params.child('Step Sinogram').setLimits([0, sino + 1])
        self.config_params.child('End Sinogram').setValue(sino)
        self.config_params.child('End Sinogram').setDefault(sino)
        self.config_params.child('End Projection').setLimits([0, proj])
        self.config_params.child('Start Projection').setLimits([0, proj])
        self.config_params.child('Step Projection').setLimits([0, proj + 1])
        self.config_params.child('End Projection').setValue(proj)
        self.config_params.child('End Projection').setDefault(proj)


def build_function_menu(menu, functree, functiondata, actionslot):
    for func, subfuncs in functree.iteritems():
        if len(subfuncs) > 1 or func != subfuncs[0]:
            funcmenu = QtGui.QMenu(func)
            menu.addMenu(funcmenu)
            for subfunc in subfuncs:
                if isinstance(subfuncs, dict) and len(subfuncs[subfunc]) > 0:
                    optsmenu = QtGui.QMenu(subfunc)
                    funcmenu.addMenu(optsmenu)
                    for opt in subfuncs[subfunc]:
                        funcaction = QtGui.QAction(opt, funcmenu)
                        funcaction.triggered.connect(partial(actionslot, func, opt,
                                                             reconpkg.packages[functiondata[opt][1]]))
                        optsmenu.addAction(funcaction)
                else:
                    funcaction = QtGui.QAction(subfunc, funcmenu)
                    funcaction.triggered.connect(partial(actionslot, func, subfunc,
                                                         reconpkg.packages[functiondata[subfunc][1]]))
                    funcmenu.addAction(funcaction)
        elif len(subfuncs) == 1:
            funcaction = QtGui.QAction(func, menu)
            funcaction.triggered.connect(partial(actionslot, func, func, reconpkg.packages[functiondata[func][1]]))
            menu.addAction(funcaction)