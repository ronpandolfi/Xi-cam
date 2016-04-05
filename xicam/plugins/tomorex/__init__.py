#! /usr/bin/env python

__author__ = "Ronald J Pandolfi"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"

import os

from PySide import QtCore, QtGui
from xicam.plugins import base

import treader, datatree, toolbars, reconwizard
from xicam.plugins.tomorex import metadata, treader, reconwizard


class plugin(base.plugin):
    """
    Class for dataset viewing
    """

    name = 'TomoRex'
    allowed_exts = ('h5', 'tiff', 'tif', '.TIFF', 'npy')
    sigTomoTabChanged = QtCore.Signal(int)
    sigTomoTabClosed = QtCore.Signal()

    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.closedataset)
        self.centerwidget.setStyleSheet('background-color:#212121;')
        self.centerwidget.currentChanged.connect(self.currentTabChanged)

        self.rightwidget = datatree.MetadataTree(metadata.BL832_PARAMS)
        self.toolbar = toolbars.TomoToolbar(self)
        self.toolbar.connect_triggers(self.reconWizard)

        super(plugin, self).__init__(*args, **kwargs)

    def opendataset(self, path, datatype='raw'):
        file_name = os.path.split(path)[1]
        file_name, ext = file_name.split('.')

        if ext is not None and ext not in self.allowed_exts:
            raise IOError('%s file format not supported' % ext)

        widget = TomoTab(file_name, path, datatype)
        self.centerwidget.addTab(widget, file_name)
        self.centerwidget.setCurrentWidget(widget)
        #self.centerwidget.currentWidget().show()
        # index = self.getCurrentIndex()

    def currentTabChanged(self):
        self.sigTomoTabChanged.emit(self.getCurrentIndex())

    def closedataset(self, index):
        widget = self.getCurrentWidget()
        widget.deleteLater()
        self.centerwidget.removeTab(index)

    def getCurrentWidget(self):
        if self.centerwidget.currentWidget() is None:
            return None
        return self.centerwidget.currentWidget()

    def getCurrentIndex(self):
        return self.centerwidget.indexOf(self.getCurrentWidget())

    def preview(self):
        index = self.getCurrentIndex()

    def reconWizard(self):
        wizard = reconwizard.Wizard(fileexplorer=self.fileexplorer)
        #wizard.sigExternalJob.connect(plugins.plugins['Job Monitor'].instance.addPopenJob)
        wizard.exec_()


class TomoTab(QtGui.QWidget):
    """
    Tabs in tomoviewer. These are placeholders for an ImageView or other visualization and hold information of the
    dataset
    """

    def __init__(self, name, path, datatype, viswidget=None):
        super(TomoTab, self).__init__()

        self.name = name
        self.path = path
        self.datatype = datatype
        self.metadata = self.readMetadata()
        self.viswidget = viswidget
        self.layout = QtGui.QStackedLayout(self)

    def readMetadata(self):
        metadata = None
        try:
            metadata = treader.read_als_832h5_metadata(self.path)
        except IOError:
            pass

        return metadata

    def addVisWidget(self):
        self.layout.addWidget(self.viswidget)

    def removeVisWidget(self):
        self.layout.removeWidget(self.viswidget)