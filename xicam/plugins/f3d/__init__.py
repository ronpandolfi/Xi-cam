from xicam.plugins import base

class plugin(base.plugin):

    name = "F3D"

    def __init__(self, placeholders, *args, **kwargs):
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent

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

        widget = TomoViewer(paths=paths)
        widget.sigSetDefaults.connect(self.manager.setPipelineFromDict)
        widget.wireupCenterSelection(self.manager.recon_function)
        self.centerwidget.addTab(widget, os.path.basename(paths))
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

        self.ui.setConfigParams(0, 0)
        self.ui.property_table.clear()
        self.ui.property_table.hide()
        self.centerwidget.widget(index).deleteLater()