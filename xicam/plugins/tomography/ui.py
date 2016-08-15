

__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


from functools import partial
from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
from psutil import cpu_count
import pyqtgraph as pg
from pyqtgraph import parametertree as pt
import reconpkg
import viewers


class UIform(object):
    def setupUi(self):

        self.toolbar = Toolbar()
        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.bottomwidget = viewers.RunConsole()
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
                  {'name': 'Sinograms/Chunk', 'type': 'int', 'value': 20*cpu_count()},
                  {'name': 'CPU Cores', 'type': 'int', 'value': cpu_count(), 'default': cpu_count(),
                   'limits':[1, cpu_count()]}]

        self.config_params = pt.Parameter.create(name='Configuration', type='group', children=params)
        configtree.setParameters(self.config_params, showTop=False)

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

    def setProjParams(self, end, start=0):
        self.config_params.child('End Projection').setLimits([0, end])
        self.config_params.child('Start Projection').setLimits([0, end])
        self.config_params.child('Step Projection').setLimits([0, end + 1])
        self.config_params.child('End Projection').setValue(end)
        self.config_params.child('End Projection').setDefault(end)
        self.config_params.child('Start Projection').setValue(start)

    def setSinoParams(self, end, start=0):
        self.config_params.child('End Sinogram').setLimits([0, end])
        self.config_params.child('Start Sinogram').setLimits([0, end])
        self.config_params.child('Step Sinogram').setLimits([0, end + 1])
        self.config_params.child('End Sinogram').setValue(end)
        self.config_params.child('End Sinogram').setDefault(end)
        self.config_params.child('Start Sinogram').setValue(start)



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
                        try:
                            funcaction.triggered.connect(partial(actionslot, func, opt,
                                                                 reconpkg.packages[functiondata[opt][1]]))
                            optsmenu.addAction(funcaction)
                        except KeyError:
                            pass
                else:
                    funcaction = QtGui.QAction(subfunc, funcmenu)
                    try:
                        funcaction.triggered.connect(partial(actionslot, func, subfunc,
                                                             reconpkg.packages[functiondata[subfunc][1]]))
                        funcmenu.addAction(funcaction)
                    except KeyError:
                        pass
        elif len(subfuncs) == 1:
            try:
                funcaction = QtGui.QAction(func, menu)
                funcaction.triggered.connect(partial(actionslot, func, func, reconpkg.packages[functiondata[func][1]]))
                menu.addAction(funcaction)
            except KeyError:
                pass


class Toolbar(QtGui.QToolBar):
    """
    QToolbar subclass used in Tomography plugin
    """

    def __init__(self):
        super(Toolbar, self).__init__()

        self.actionRun_SlicePreview = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_50.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_SlicePreview.setIcon(icon)
        self.actionRun_SlicePreview.setToolTip('Slice preview')

        self.actionRun_3DPreview = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_42.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_3DPreview.setIcon(icon)
        self.actionRun_3DPreview.setToolTip('3D preview')

        self.actionRun_FullRecon = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_34.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_FullRecon.setIcon(icon)
        self.actionRun_FullRecon.setToolTip('Full reconstruction')

        self.actionPolyMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_05.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPolyMask.setIcon(icon)
        self.actionPolyMask.setText("Polygon mask")

        self.actionCircMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_05.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionCircMask.setIcon(icon)
        self.actionCircMask.setText("Circular mask")

        self.actionRectMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_05.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRectMask.setIcon(icon)
        self.actionRectMask.setText('Rectangular mask')

        self.actionMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_03.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionMask.setIcon(icon)

        maskmenu = QtGui.QMenu(self)
        maskmenu.addAction(self.actionRectMask)
        maskmenu.addAction(self.actionCircMask)
        maskmenu.addAction(self.actionPolyMask)
        toolbuttonMasking = QtGui.QToolButton(self)
        toolbuttonMasking.setDefaultAction(self.actionMask)
        toolbuttonMasking.setMenu(maskmenu)
        toolbuttonMasking.setPopupMode(QtGui.QToolButton.InstantPopup)
        toolbuttonMaskingAction = QtGui.QWidgetAction(self)
        toolbuttonMaskingAction.setDefaultWidget(toolbuttonMasking)


        self.actionCenter = QtGui.QWidgetAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_28.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionCenter.setIcon(icon)
        self.actionCenter.setToolTip('Overlay center of rotation detection')
        self.toolbuttonCenter = QtGui.QToolButton(parent=self)
        self.toolbuttonCenter.setPopupMode(QtGui.QToolButton.InstantPopup)
        self.actionCenter.setDefaultWidget(self.toolbuttonCenter)
        self.actionCenter.setCheckable(True)
        self.toolbuttonCenter.setDefaultAction(self.actionCenter)

        self.actionROI = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_60.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionROI.setIcon(icon)
        self.actionROI.setToolTip('Select region of interest')
        self.actionROI.setCheckable(True)

        self.setIconSize(QtCore.QSize(32, 32))

        self.addAction(self.actionRun_FullRecon)
        self.addAction(self.actionRun_SlicePreview)
        self.addAction(self.actionRun_3DPreview)
        self.addAction(self.actionCenter)
        self.addAction(self.actionROI)
        self.addAction(toolbuttonMaskingAction)


    def connecttriggers(self, slicepreview, preview3D, fullrecon, center, roiselection):
        """
        Connect toolbar action signals to give slots

        Parameters
        ----------
        slicepreview : QtCore.Slot
            Slot to connect actionRun_SlicePreview
        preview3D : QtCore.Slot
            Slot to connect actionRun_3DPreview
        fullrecon : QtCore.Slot
            Slot to connect actionRun_FullRecon
        center : QtCore.Slot
            Slot to connect actionCenter

        """
        self.actionRun_SlicePreview.triggered.connect(slicepreview)
        self.actionRun_3DPreview.triggered.connect(preview3D)
        self.actionRun_FullRecon.triggered.connect(fullrecon)
        self.actionCenter.toggled.connect(center)
        self.actionROI.toggled.connect(roiselection)