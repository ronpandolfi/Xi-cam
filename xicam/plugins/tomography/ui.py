

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
from collections import deque
import xicam.widgets.featurewidgets as fw
from xicam.widgets.customwidgets import DataTreeWidget
import reconpkg
import os
import viewers


class UIform(object):
    """
    Class for tomography plugin ui setup.

    Attributes
    ----------

    toolbar : QtGui.QToolBar
        Toolbar shown in plugin
    leftmodes : list of tuples
        leftmodes list for standard base plugin initialization [widget, tab icon]
    righmodes : list of tuples
        rightmodes list for standard base plugin initialization [widget, tab icon]
    param_form : QtGui.QStackedWidget
        Container for ParameterTree's used to display function parameters
    functionWidget : QtGui.QWidget
        Workflow pipeline GUI in plugin's leftmodes
    property_table : pyqtgraph.TableWidget
        TableWidget to display dataset metadata on right widget
    config_params : pyqtgraph.Parameter
        Parameter holding the run configuration parameters (ie sino start, sino end, sino step, number of cores)

    Methods
    -------
    connectTriggers
        Connect leftwidget (function mangement buttons) triggers to corresponding slots
    setConfigParams
        Sets configuration parameters in pg.Parameter inside rightwidget
    """

    def setupUi(self):
        """Set up the UI for tomography plugin"""

        self.toolbar = Toolbar()
        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.bottomwidget = viewers.RunConsole()
        self.functionwidget = QUiLoader().load('xicam/gui/tomographyleft.ui')
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
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_55.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.openaction = QtGui.QAction(icon, 'Open', filefuncmenu)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_59.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.saveaction = QtGui.QAction(icon, 'Save', filefuncmenu)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_56.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.refreshaction = QtGui.QAction(icon, 'Reset', filefuncmenu)
        filefuncmenu.addActions([self.openaction, self.saveaction, self.refreshaction])

        self.functionwidget.fileButton.setMenu(filefuncmenu)
        self.functionwidget.fileButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.functionwidget.fileButton.setArrowType(QtCore.Qt.NoArrow)

        leftwidget = QtGui.QSplitter(QtCore.Qt.Vertical)

        paramtree = pt.ParameterTree()
        self.param_form = QtGui.QStackedWidget()
        self.param_form.addWidget(paramtree)
        self.property_table = pg.TableWidget()
        self.property_table.verticalHeader().hide()
        self.property_table.horizontalHeader().setStretchLastSection(True)
        self.property_table.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        leftwidget.addWidget(self.param_form)
        leftwidget.addWidget(self.functionwidget)

        self.queue = ReconManager()


        icon_functions = QtGui.QIcon(QtGui.QPixmap("xicam/gui/icons_49.png"))
        icon_properties = QtGui.QIcon(QtGui.QPixmap("xicam/gui/icons_61.png")) #metadata icon
        icon_queue = QtGui.QIcon(QtGui.QPixmap("xicam/gui/icons_63.png"))
        self.leftmodes = [(leftwidget, icon_functions),(self.queue, icon_queue),(self.property_table, icon_properties)]

        # rightwidget = QtGui.QSplitter(QtCore.Qt.Vertical)
        #
        # configtree = pt.ParameterTree()
        # configtree.setMinimumHeight(230)
        #
        # params = [{'name': 'Start Sinogram', 'type': 'int', 'value': 0, 'default': 0, },
        #           {'name': 'End Sinogram', 'type': 'int'},
        #           {'name': 'Step Sinogram', 'type': 'int', 'value': 1, 'default': 1},
        #           {'name': 'Start Projection', 'type': 'int', 'value': 0, 'default': 0},
        #           {'name': 'End Projection', 'type': 'int'},
        #           {'name': 'Step Projection', 'type': 'int', 'value': 1, 'default': 1},
        #           {'name': 'Sinograms/Chunk', 'type': 'int', 'value': 5*cpu_count()},
        #           {'name': 'CPU Cores', 'type': 'int', 'value': cpu_count(), 'default': cpu_count(),
        #            'limits':[1, cpu_count()]}]
        #
        # self.config_params = pt.Parameter.create(name='Configuration', type='group', children=params)
        # configtree.setParameters(self.config_params, showTop=False)
        # rightwidget.addWidget(configtree)



    def connectTriggers(self, open, save, reset, moveup, movedown, clear):
        """
        Connect leftwidget (function mangement buttons) triggers to corresponding slots

        Parameters
        ----------
        open : QtCore.Slot
            Slot to handle signal from open button
        save QtCore.Slot
            Slot to handle signal from save button
        reset QtCore.Slot
            Slot to handle signal from reset button
        moveup QtCore.Slot
            Slot to handle signal to move a function widget upwards
        movedown QtCore.Slot
            Slot to handle signal to move a function widget downwards
        clear QtCore.Slot
            Slot to handle signal from clear button
        """

        self.openaction.triggered.connect(open)
        self.saveaction.triggered.connect(save)
        self.refreshaction.triggered.connect(reset)
        self.functionwidget.moveDownButton.clicked.connect(moveup)
        self.functionwidget.moveUpButton.clicked.connect(movedown)
        self.functionwidget.clearButton.clicked.connect(clear)
    #
    # def setConfigParams(self, proj, sino):
    #     self.config_params.child('End Sinogram').setLimits([0, sino])
    #     self.config_params.child('Start Sinogram').setLimits([0, sino])
    #     self.config_params.child('Step Sinogram').setLimits([0, sino + 1])
    #     self.config_params.child('End Sinogram').setValue(sino)
    #     self.config_params.child('End Sinogram').setDefault(sino)
    #     self.config_params.child('End Projection').setLimits([0, proj])
    #     self.config_params.child('Start Projection').setLimits([0, proj])
    #     self.config_params.child('Step Projection').setLimits([0, proj + 1])
    #     self.config_params.child('End Projection').setValue(proj)
    #     self.config_params.child('End Projection').setDefault(proj)


def build_function_menu(menu, functree, functiondata, actionslot):
    """
    Builds the function menu's and submenu's anc connects them to the corresponding slot to add them to the workflow
    pipeline

    Parameters
    ----------
    menu : QtGui.QMenu
        Menu object to populate with submenu's and actions
    functree : dict
        Dictionary specifying the depth levels of functions. See functions.yml entry "Functions"
    functiondata : dict
        Dictionary with function information. See function_names.yml
    actionslot : QtCore.Slot
        slot where the function action triggered signal shoud be connected
    """

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

    Attributes
    ----------
    actionRun_SlicePreview : QtGui.QAction
    actionRun_3DPreview : QtGui.QAction
    actionRun_FullRecon : QtGui.QAction
    actionCenter : QtGui.QAction
    actionROI : QtGui.QAction
    actionPolyMask : QtGui.QAction
    actionCircMask : QtGui.QAction
    actionRectMask : QtGui.QAction
    actionMask : QtGui.QAction

    Methods
    -------
    connecttriggers
        Connect toolbar action signals to give slots
    """

    def __init__(self):
        super(Toolbar, self).__init__()

        self.actionRun_SlicePreview = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_50.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_SlicePreview.setIcon(icon)
        self.actionRun_SlicePreview.setToolTip('Slice preview')

        self.actionRun_MultiSlicePreview = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_62.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_MultiSlicePreview.setIcon(icon)
        self.actionRun_MultiSlicePreview.setToolTip('Multi-slice preview')

        self.actionRun_3DPreview = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_42.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_3DPreview.setIcon(icon)
        self.actionRun_3DPreview.setToolTip('3D preview')

        self.actionRun_FullRecon = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_34.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_FullRecon.setIcon(icon)
        self.actionRun_FullRecon.setToolTip('Full reconstruction')

        self.actionMBIR = QtGui.QWidgetAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_06.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionMBIR.setIcon(icon)
        self.actionMBIR.setToolTip('MBIR reconstruction')
        self.toolbuttonMBIR = QtGui.QToolButton(parent=self)
        self.toolbuttonMBIR.setPopupMode(QtGui.QToolButton.InstantPopup)
        self.actionMBIR.setDefaultWidget(self.toolbuttonMBIR)
        self.actionMBIR.setCheckable(True)
        self.toolbuttonMBIR.setDefaultAction(self.actionMBIR)

        self.actionCenter = QtGui.QWidgetAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_28.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionCenter.setIcon(icon)
        self.actionCenter.setToolTip('Center of rotation detection')
        self.toolbuttonCenter = QtGui.QToolButton(parent=self)
        self.toolbuttonCenter.setPopupMode(QtGui.QToolButton.InstantPopup)
        self.actionCenter.setDefaultWidget(self.toolbuttonCenter)
        self.actionCenter.setCheckable(True)
        self.toolbuttonCenter.setDefaultAction(self.actionCenter)

        self.actionPolyMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_05.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPolyMask.setIcon(icon)
        self.actionPolyMask.setText("Polygon mask")

        self.actionCircMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_05.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionCircMask.setIcon(icon)
        self.actionCircMask.setText("Circular mask")

        self.actionRectMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_05.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRectMask.setIcon(icon)
        self.actionRectMask.setText('Rectangular mask')

        self.actionMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_03.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
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

        # TODO working on ROI Selection TOOL
        self.actionROI = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_60.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionROI.setIcon(icon)
        self.actionROI.setToolTip('Select region of interest')

        self.setIconSize(QtCore.QSize(32, 32))

        self.addAction(self.actionRun_FullRecon)
        self.addAction(self.actionRun_SlicePreview)
        self.addAction(self.actionRun_MultiSlicePreview)
        self.addAction(self.actionRun_3DPreview)
        self.addAction(self.actionMBIR)
        self.addAction(self.actionCenter)
        self.addAction(self.actionROI)
        # self.addAction(toolbuttonMaskingAction)


    def connectTriggers(self, slicepreview, multislicepreview, preview3D, fullrecon, center, roiselection, mbir):

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
        self.actionRun_MultiSlicePreview.triggered.connect(multislicepreview)
        self.actionRun_3DPreview.triggered.connect(preview3D)
        self.actionRun_FullRecon.triggered.connect(fullrecon)
        self.actionCenter.toggled.connect(center)
        self.actionMBIR.toggled.connect(mbir)
        self.actionROI.triggered.connect(roiselection)

class ReconManager(QtGui.QSplitter):
    """
    Widget to show reconstruction queue on leftwidget tab

    Attributes
    ----------
    queue_form: QtTui.QStackedWidget
        widget to show dictionary representation of pipeline for each reconstruction in queue
    queue: featurewidgets.FeatureManager
        widget to hold featurewidgets representing reconstruction jobs

    Signals
    -------
    signReconDeleted(int)
        Emitted when user deletes recon job sitting on queue. Emits index of job.
    sigReconSwapped(int, int)
        Emitted when user tries to change order of jobs on queue. Emits indices of the two to be switched
    """

    sigReconDeleted = QtCore.Signal(int)
    sigReconSwapped = QtCore.Signal(int, int)

    def __init__(self, *args, **kwargs):

        super(ReconManager, self).__init__(*args, **kwargs)
        queue_ui = QUiLoader().load('xicam/gui/tomographyqueue.ui')
        self.queue_form = QtGui.QStackedWidget()
        queue_ui.functionsList.setAlignment(QtCore.Qt.AlignBottom)
        queue_ui.moveDownButton.setToolTip('Move selected job down in queue')
        queue_ui.moveUpButton.setToolTip('Move selected job up in queue')
        self.queue = fw.FeatureManager(queue_ui.functionsList, self.queue_form, blank_form='Click items below to see reconstruction jobs on queue.')
        queue_ui.moveDownButton.clicked.connect(self.moveDown)
        queue_ui.moveUpButton.clicked.connect(self.moveUp)

        self.setOrientation(QtCore.Qt.Vertical)
        self.addWidget(self.queue_form)
        self.addWidget(queue_ui)


    def addRecon(self, args):
        """
        Parameters
        ----------
        args: 7-tuple
            Tuple with args for a reconstruction. They are:
            1). datawidget: Datawidget containing data
            2). tuple: function pipeline, and reconstruction params (like theta, COR)
            3). type: projections to reconstruct
            4). tuple: sinograms to reconstruct
            5). int: sinograms per reconstruction iteration
            6). tuple: width dimensions for reconstruction
            7). int: cpus available
        """

        name = os.path.basename(args[0].path)
        widget = fw.FeatureWidget(name, checkable=False)
        widget.previewButton.hide()
        widget.line.hide()
        widget.sigDelete.connect(self.reconDeleted)

        form = DataTreeWidget()
        data = args[1][3][0]
        form.setData(data, hideRoot=True)
        form.header().hide()
        form.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        form.setSelectionBehavior(QtGui.QAbstractItemView.SelectItems)

        widget.form = form

        self.queue_form.addWidget(form)
        self.queue.addFeature(widget)


    def removeRecon(self, idx):
        """
        Function to remove job from queue

        Parameters
        ----------
        idx: int
            index of job to be removed

        """

        feature = self.queue.features[idx]
        self.queue.removeFeature(feature)

    def reconDeleted(self, funcwidget):
        """
        Slot to connect to user deleting job on queue

        Parameters
        ----------
        funcwidget: featurewidgets.FunctionWidget
            functionwidget to be deleted

        Signals
        -------
        sigReconDeleted(idx)
            Emits index of job to be deleted
        """

        idx = self.queue.features.index(funcwidget)
        self.sigReconDeleted.emit(idx)

    def moveUp(self):
        """
        Slot to connect to user swapping recon job and job above it
        """

        idx1 = self.queue.features.index(self.queue.selectedFeature)
        idx2 = self.queue.features.index(self.queue.nextFeature)

        if idx1 == 0 or idx2 == 0:
            return
        else:
            self.queue.swapFeatures(self.queue.selectedFeature, self.queue.nextFeature)
            self.sigReconSwapped.emit(idx1, idx2)

    def moveDown(self):
        """
        Slot to connect to user swapping recon job and job below it
        """
        idx1 = self.queue.features.index(self.queue.selectedFeature)
        idx2 = self.queue.features.index(self.queue.previousFeature)

        if idx1 == 0 or idx2 == 0:
            return
        else:
            self.queue.swapFeatures(self.queue.selectedFeature, self.queue.previousFeature)
            self.sigReconSwapped.emit(idx1, idx2)










