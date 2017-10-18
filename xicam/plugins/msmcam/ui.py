__author__ = "Dinesh Kumar"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque",
               "Holden Parks", "Alexander Hexemer"]
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Alpha"


from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
from pyqtgraph import parametertree as paramtree


class UI(object):
    def __init__(self):

        super(UI, self).__init__()
        self.toolbar = Toolbar()
        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.closeTab)

    def closeTab(self, index):
        self.centerwidget.removeTab(index)
        

class Toolbar(QtGui.QToolBar):
    """
    QToolbar subclass used in Tomography plugin

    Attributes
    ----------
    actionMask : QtGui.QAction
    actionFilter : QtGui.QAction
    actionSegment : QtGui.QAction

    Methods
    -------
    connecttriggers
        Connect toolbar action signals to given slots
    """

    def __init__(self):
        super(Toolbar, self).__init__()

        self.actionMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_50.png"), QtGui.QIcon.Normal, 
QtGui.QIcon.On)
        self.actionMask.setIcon(icon)
        self.actionMask.setToolTip('Load Mask')
	self.addAction(self.actionMask)

        self.actionFilter = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_62.png"), QtGui.QIcon.Normal, 
QtGui.QIcon.On)
        self.actionFilter.setIcon(icon)
        self.actionFilter.setToolTip('Select Filter(s)')
        self.addAction(self.actionFilter)

        self.actionSegment = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_42.png"), QtGui.QIcon.Normal, 
QtGui.QIcon.On)
        self.actionSegment.setIcon(icon)
        self.actionSegment.setToolTip('Seg. Methods')
	self.addAction(self.actionSegment)

    def connectTriggers(self, loadmask, applyfilter, runsegmentation):
        """
        Connect toolbar action signals to given slots

        Parameters
        ----------
        loadmask : QtCore.Slot
           Slot to connect actionMask 
        selectfilters : QtCore.Slot
           Slot to connect actionFilter
        selectalgos : QtCore.Slot
           Slot to connect actionSegment
        """
        self.actionMask.triggered.connect(loadmask)
        self.actionFilter.triggered.connect(applyfilter)
        self.actionSegment.triggered.connect(runsegmentation)
