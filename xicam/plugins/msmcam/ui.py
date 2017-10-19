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
from pyqtgraph.parametertree import Parameter, ParameterTree


class UI(object):
    def __init__(self):

        super(UI, self).__init__()
        self.toolbar = Toolbar()
        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.closeTab)
        self.rightwidget = self.setParams()

    def closeTab(self, index):
        self.centerwidget.removeTab(index)

    def setParams(self, mode=None):
        """
        TODO: create a separate class 
        """
        # general parameters
        p = [ 
                { 'name': 'Downsample Scale', 'type': 'float', 'value': 1, 'default': 1 },
                { 'name': 'Filter', 'type': 'group' },
                { 'name': 'Segmentation', 'type': 'group' }
                ]
        self.params = Parameter.create(name='Parameters', type='group', children=p)

        # median filter
        p1 = Parameter.create(name='Median', type='bool', value=True)
        p1.addChild(Parameter.create(name='Mask Size', type='int', value=5))
        self.params.child('Filter').addChild(p1)
        #bilateral fileter
        p1 = Parameter.create(name='Bilateral', type='bool', value=False)
        p1.addChild(Parameter.create(name='Sigma Spatial', type='int', value=5))
        p1.addChild(Parameter.create(name='Sigma Color', type='float', value=0.05))
        self.params.child('Filter').addChild(p1)

        # segmentation
        # k-means
        p1 = Parameter.create(name='k-means', type='bool', value=True)
        p1.addChild(Parameter.create(name='Clusters', type='int', value=3))
        self.params.child('Segmentation').addChild(p1)
        # SRM
        p1 = Parameter.create(name='SRM', type='bool', value=True)
        p1.addChild(Parameter.create(name='Clusters', type='int', value=2))
        self.params.child('Segmentation').addChild(p1)

        param_tree = ParameterTree()
        param_tree.setParameters(self.params, showTop=False)
        return param_tree
        
        

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
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_08.png"), QtGui.QIcon.Normal, 
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
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_34.png"), QtGui.QIcon.Normal, 
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
