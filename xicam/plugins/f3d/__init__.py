from xicam.plugins import base
from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
from xicam.plugins.tomography.viewers import RunConsole
from pipeline.loader import StackImage
from pipeline import msg
from functools import partial
import f3d_viewers
import os
import pyqtgraph as pg
import filtermanager as fm
import importer


class plugin(base.plugin):

    name = "F3D"

    sigFilterAdded = QtCore.Signal(dict)


    def __init__(self, placeholders, *args, **kwargs):

        self.toolbar = Toolbar()
        self.toolbar.connectTriggers(self.run)
        # self.build_toolbutton_menu(self.toolbar.addMaskMenu, 'Open file for mask', self.openMaskFile)
        # self.build_toolbutton_menu(self.toolbar.addMaskMenu, 'Open directory for mask', self.openMaskFolder)


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
        # self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent

        self.rightwidget = F3DOptionsWidget()


        self.manager = fm.FilterManager(self.functionwidget.functionsList, self.param_form,
                                       blank_form='Select a filter from\n below to set parameters...')

        self.functionwidget.fileButton.setMenu(filefuncmenu)
        self.functionwidget.fileButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.functionwidget.fileButton.setArrowType(QtCore.Qt.NoArrow)

        self.addfunctionmenu = QtGui.QMenu()
        self.functionwidget.addFunctionButton.setMenu(self.addfunctionmenu)
        self.functionwidget.addFunctionButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.functionwidget.addFunctionButton.setArrowType(QtCore.Qt.NoArrow)
        self.openaction.triggered.connect(self.loadPipeline)
        self.saveaction.triggered.connect(self.savePipeline)
        self.functionwidget.moveDownButton.clicked.connect(
            lambda: self.manager.swapFeatures(self.manager.selectedFeature,self.manager.previousFeature))
        self.functionwidget.moveUpButton.clicked.connect(
            lambda: self.manager.swapFeatures(self.manager.selectedFeature, self.manager.nextFeature))
        self.functionwidget.clearButton.clicked.connect(self.clearPipeline)




        super(plugin, self).__init__(placeholders, *args, **kwargs)

        self.filter_images = {}

        self.sigFilterAdded.connect(self.manager.updateFilterMasks)
        self.build_function_menu(self.addfunctionmenu, importer.filters, self.manager.addFilter)


    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if op_sys == 'Darwin':
                fname = str(NSURL.URLWithString_(str(url.toString())).filePathURL().path())
            else:
                fname = str(url.toLocalFile())
            if os.path.isfile(fname):
                self.openfiles([fname])
            if os.path.isdir(fname):
                self.opendirectory([fname])
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

        # check if file is already in filter_images, and load if it is not
        if not paths in self.filter_images.iterkeys():
            self.filter_images[paths] = widget
            self.sigFilterAdded.emit(self.filter_images)

        self.centerwidget.addTab(widget, os.path.basename(paths))
        self.centerwidget.setCurrentWidget(widget)
        msg.showMessage('Done.', timeout=10)

    def opendirectory(self, file, operation=None):
        msg.showMessage('Loading directory...', timeout=10)
        self.activate()
        if type(file) is list:
            file = file[0]

        files = [os.path.join(file, path) for path in os.listdir(file) if path.endswith('.tif') or path.endswith('.tiff')]
        widget = f3d_viewers.F3DViewer(files=files)

        # check if file is already in filter_images, and load if it is not
        if not file in self.filter_images.iterkeys():
            self.filter_images[file] = widget
            self.sigFilterAdded.emit(self.filter_images)

        self.centerwidget.addTab(widget, os.path.basename(file))
        self.centerwidget.setCurrentWidget(widget)
        msg.showMessage('Done.', timeout=10)


    ## TODO: have separate readers for masks? maybe just have all open images be available as masks

    # def openMaskFile(self):
    #
    #     mask_path = QtGui.QFileDialog().getOpenFileName(caption="Select file to open as mask: ")
    #
    #     if not mask_path[0]:
    #         return
    #     try:
    #         mask = StackImage(mask_path[0]).fabimage.rawdata
    #     except AttributeError:
    #         self.log.log2local("Could not open file \'{}\'".format(mask_path[0]))
    #
    #     self.filter_images[mask_path[0]] = mask
    #     self.log.log2local('Successfully loaded \'{}\' as mask'.format(os.path.basename(mask_path[0])))
    #     self.leftwidget.setCurrentWidget(self.log)
    #     self.sigFilterAdded.emit(self.filter_images)
    #
    #
    #
    # def openMaskFolder(self):
    #     mask_path = QtGui.QFileDialog().getExistingDirectory(caption=
    #                                                 "Select directory to search for mask images: ")
    #
    #     if not mask_path:
    #         return
    #     try:
    #         files = [os.path.join(mask_path, path) for path in os.listdir(mask_path) if
    #                  path.endswith('.tif') or path.endswith('.tiff')]
    #         mask = StackImage(files).fabimage.rawdata
    #     except AttributeError:
    #         self.log.log2local("Could not open directory \'{}\'".format(mask_path))
    #
    #     self.filter_images[mask_path] = mask
    #     self.log.log2local('Successfully loaded images in \'{}\' as mask'.format(os.path.basename(mask_path)))
    #     self.leftwidget.setCurrentWidget(self.log)
    #     self.sigFilterAdded.emit(self.filter_images)

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

    def build_function_menu(self, menu, filter_data, actionslot):
        """
        Builds the filter menu and connects it to the corresponding slot to add them to the workflow pipeline

        Parameters
        ----------
        menu : QtGui.QMenu
            Menu object to populate with filter names
        functiondata : dict
            Dictionary with function information. See importer.filters.yml
        actionslot : QtCore.Slot
            slot where the function action triggered signal should be connected
        """

        for func, options in filter_data.iteritems():
                try:
                    funcaction = QtGui.QAction(func, menu)
                    funcaction.triggered.connect(partial(actionslot, func))
                    menu.addAction(funcaction)
                except KeyError:
                    pass

    def build_toolbutton_menu(self, menu, heading, actionslot):

        action = QtGui.QAction(heading, menu)
        action.triggered.connect(actionslot)
        menu.addAction(action)

    def loadPipeline(self):
        pass

    def savePipeline(self):
        pass

    def clearPipeline(self):
        pass

class Toolbar(QtGui.QToolBar):

    def __init__(self):
        super(Toolbar, self).__init__()

        self.actionRun = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_34.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun.setIcon(icon)
        self.actionRun.setToolTip('Run pipeline')

        # self.actionAddMask = QtGui.QToolButton(self)
        # self.addMaskMenu = QtGui.QMenu()
        # self.actionAddMask.setMenu(self.addMaskMenu)
        # self.actionAddMask.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
        # self.actionAddMask.setArrowType(QtCore.Qt.NoArrow)
        # icon = QtGui.QIcon()
        # icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_08.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # self.actionAddMask.setIcon(icon)
        # self.actionAddMask.setToolTip('Add additional mask from disk')

        self.setIconSize(QtCore.QSize(32, 32))

        self.addAction(self.actionRun)
        # self.addWidget(self.actionAddMask)

    def connectTriggers(self, run):

        self.actionRun.triggered.connect(run)

class F3DOptionsWidget(QtGui.QWidget):
    """
    bottomwidget for f3d plugin
    """

    def __init__(self, parent=None):
        super(F3DOptionsWidget, self).__init__(parent=parent)
        layout = QtGui.QVBoxLayout()
        options = QtGui.QLabel('Devices Options')
        self.devices = {}



        layout.addWidget(options)
        layout.addSpacing(10)

        counter = 0
        for device, cores in self.readAvailableDevices().iteritems():
            self.devices[device] = DeviceWidget(device, counter, cores)
            if counter == 0:
                self.devices[device].checkbox.setChecked(True)
            layout.addWidget(self.devices[device])
            counter += 1

        layout.addSpacing(30)

        # widget to hold virtual stack options
        l = QtGui.QVBoxLayout()
        h = QtGui.QHBoxLayout()
        self.virtual_stack = QtGui.QCheckBox()
        self.virtual_stack.stateChanged.connect(self.findDirectory)
        self.output = QtGui.QLineEdit(' ')
        self.output.setReadOnly(True)
        h.addWidget(self.virtual_stack)
        h.addWidget(QtGui.QLabel('Use Virtual Stack'))
        l.addLayout(h)
        l.addWidget(self.output)
        layout.addLayout(l)
        layout.addSpacing(10)

        self.intermediate_steps = QtGui.QCheckBox()
        h_layout = QtGui.QHBoxLayout()
        h_layout.addWidget(self.intermediate_steps)
        h_layout.addWidget(QtGui.QLabel('Show Intermediate Steps '))
        layout.addLayout(h_layout)
        layout.addStretch(50)

        self.setLayout(layout)

    @property
    def use_virtual(self):
        return self.virtual_stack.checkState()

    @property
    def use_intermediate(self):
        return self.intermediate_steps.checkState()

    def findDirectory(self, bool):

        if bool:
            path = QtGui.QFileDialog().getExistingDirectory(caption=
                                                    "Choose output directory: ")
            if path: self.output.setText(path)


    def readAvailableDevices(self):
        """
        Somehow read and return list of all gpus usable for processing
        """

        return {'GeForce GTX TITAN': 2160, 'Something else': 4000}

class DeviceWidget(QtGui.QWidget):

    """
    Widget to hold checkbox, name, and spinbox for cores (?) used in processing
    """

    def __init__(self, name, number, cores):
        super(DeviceWidget, self).__init__(parent=None)


        top_layout = QtGui.QHBoxLayout()
        bottom_layout = QtGui.QVBoxLayout()
        layout = QtGui.QVBoxLayout()

        self.enabled = False
        self.name = name
        self.cores = cores

        self.checkbox = QtGui.QCheckBox()
        self.label = QtGui.QLabel('Device {} ({})'.format(str(number), self.name))
        self.corebox = QtGui.QSpinBox()
        self.corebox.setMinimum(1)
        self.corebox.setMaximum(self.cores)
        self.corebox.setValue(self.cores)
        self.checkbox.stateChanged.connect(self.checkbox_changed)
        self.corebox.valueChanged.connect(self.corebox_changed)

        top_layout.addWidget(self.checkbox)
        top_layout.addWidget(self.label)
        bottom_layout.addWidget(self.corebox)
        layout.addLayout(top_layout)
        layout.addLayout(bottom_layout)

        self.setLayout(layout)

    def checkbox_changed(self, enabled):
        self.enabled = enabled

    def corebox_changed(self, val):
        self.cores = val

