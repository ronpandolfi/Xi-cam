from xicam.plugins import base
from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
from xicam.plugins.tomography.viewers import RunConsole
from pipeline import msg
import f3d_viewers
import os
import xicam.widgets.featurewidgets as fw
import pyqtgraph as pg


class plugin(base.plugin):

    name = "F3D"

    def __init__(self, placeholders, *args, **kwargs):


        # self.centerwidget.currentChanged.connect(self.currentChanged)
        # self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)
        #
        # # DRAG-DROP
        # self.centerwidget.setAcceptDrops(True)
        # self.centerwidget.dragEnterEvent = self.dragEnterEvent
        # self.centerwidget.dropEvent = self.dropEvent

        self.toolbar = Toolbar()
        self.toolbar.connectTriggers(self.run)

        self.functionwidget = QUiLoader().load('xicam/gui/tomographyleft.ui')
        self.functionwidget.functionsList.setAlignment(QtCore.Qt.AlignBottom)

        self.functionwidget.addFunctionButton.setToolTip('Add function to pipeline')
        self.functionwidget.clearButton.setToolTip('Clear pipeline')
        self.functionwidget.fileButton.setToolTip('Save/Load pipeline')
        self.functionwidget.moveDownButton.setToolTip('Move selected function down')
        self.functionwidget.moveUpButton.setToolTip('Move selected function up')

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
        paramtree = pg.parametertree.ParameterTree()
        self.param_form = QtGui.QStackedWidget()
        self.param_form.addWidget(paramtree)
        self.property_table = pg.TableWidget()
        self.property_table.verticalHeader().hide()
        self.property_table.horizontalHeader().setStretchLastSection(True)
        self.property_table.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        leftwidget.addWidget(self.param_form)
        leftwidget.addWidget(self.functionwidget)

        self.log = RunConsole()

        icon_functions = QtGui.QIcon(QtGui.QPixmap("xicam/gui/icons_49.png"))
        icon_log = QtGui.QIcon(QtGui.QPixmap("xicam/gui/icons_64.png"))

        self.leftmodes = [(leftwidget, icon_functions), (self.log, icon_log)]

        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)
        self.rightwidget = None

        self.manager = FilterManager(self.functionwidget.functionsList, self.param_form,
                                       blank_form='Select a filter from\n below to set parameters...')

        super(plugin, self).__init__(placeholders, *args, **kwargs)

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if op_sys == 'Darwin':
                fname = str(NSURL.URLWithString_(str(url.toString())).filePathURL().path())
            else:
                fname = str(url.toLocalFile())
            if os.path.isfile(fname):
                self.openfiles([fname])
            e.accept()

    def dragEnterEvent(self, e):
        e.accept()

    def openfiles(self, paths):
        """
        Override openfiles method in base plugin. Used to open a tomography dataset from the recognized file formats
        and instantiate a viewer.TomoViewer tab. This function takes quite a bit, consider running this in a background
        thread

        Parameters
        ----------
        paths : str/list
            Path to file. Currently only one file is supported. Multiple paths (ie stack of tiffs should be easy to
            implement using the formats.StackImage class.

        """

        msg.showMessage('Loading file...', timeout=10)
        self.activate()
        if type(paths) is list:
            paths = paths[0]

        widget = f3d_viewers.F3DViewer(files=paths)
        widget.sigSetDefaults.connect(self.manager.setPipelineFromDict)
        self.centerwidget.addTab(widget, os.path.basename(paths))
        self.centerwidget.setCurrentWidget(widget)

    def opendirectory(self, file, operation=None):
        msg.showMessage('Loading directory...', timeout=10)
        self.activate()
        if type(file) is list:
            file = file[0]

        files = [os.path.join(file, path) for path in os.listdir(file) if path.endswith('.tif') or path.endswith('.tiff')]
        widget = f3d_viewers.F3DViewer(files=files)
        self.centerwidget.addTab(widget, os.path.basename(file))
        self.centerwidget.setCurrentWidget(widget)

    def currentWidget(self):
        """
        Return the current widget (viewer.TomoViewer) from the centerwidgets tabs
        """

        try:
            return self.centerwidget.currentWidget()
        except AttributeError:
            return None

    def currentChanged(self, index):
        """
        Slot to recieve centerwidgets currentchanged signal when a new tab is selected
        """

        try:
            self.setPipelineValues()
            self.manager.updateParameters()
            self.toolbar.actionCenter.setChecked(False)
        except (AttributeError, RuntimeError) as e:
            msg.logMessage(e.message, level=msg.ERROR)

    def reconnectTabs(self):
        """
        Reconnect TomoViewers when the pipeline is reset
        """
        for idx in range(self.centerwidget.count()):
            self.centerwidget.widget(idx).wireupCenterSelection(self.manager.recon_function)
            self.centerwidget.widget(idx).sigSetDefaults.connect(self.manager.setPipelineFromDict)

    def tabCloseRequested(self, index):
        """
        Slot to receive signal when a tab is closed. Simply resets configuration parameters and clears metadata table

        Parameters
        ----------
        index : int
            Index of tab that is being closed.
        """

        self.centerwidget.widget(index).deleteLater()

    def run(self):
        pass

class Toolbar(QtGui.QToolBar):

    def __init__(self):
        super(Toolbar, self).__init__()

        self.actionRun = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_34.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun.setIcon(icon)
        self.actionRun.setToolTip('Full reconstruction')

        self.setIconSize(QtCore.QSize(32, 32))

        self.addAction(self.actionRun)


    def connectTriggers(self, run):

        self.actionRun.triggered.connect(run)

class FilterManager(fw.FeatureManager):
    def __init__(self, list_layout, form_layout, function_widgets=None, blank_form=None):

        super(FilterManager, self).__init__(list_layout, form_layout, feature_widgets=function_widgets,
                                              blank_form=blank_form)
