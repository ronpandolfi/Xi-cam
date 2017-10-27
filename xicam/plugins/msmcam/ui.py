
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
from .viewer import CenterWidget
from pyqtgraph.parametertree import Parameter, ParameterTree


class UI(object):
    def __init__(self):

        super(UI, self).__init__()
        self.toolbar = Toolbar()
        self.centerwidget = CenterWidget()
        self.rightwidget = self.setParams()

    def setParams(self, mode=None):
        """
        TODO: create a separate class 
        """
        # general parameters
        p = [ 
                { 'name': 'Downsample Scale', 'type': 'float', 'value': 1, 'default': 1 },
                { 'name': 'No. of Slices', 'type': 'int', 'value': 10, 'default': 10 },
                { 'name': 'Fiber Data', 'type': 'bool', 'value': False, 'default': False },
                { 'name': 'In Memory', 'type': 'bool', 'value' : True, 'default': True },
                { 'name': 'Filter', 'type': 'group', 'expanded': False },
                { 'name': 'Segmentation', 'type': 'group', 'expanded': False }
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
        p1 = Parameter.create(name='Invert', type='bool', value=False, default=False)
        self.params.child('Segmentation').addChild(p1)
        p1 = Parameter.create(name='Multiphase', type='bool', value=False, default=False)
        self.params.child('Segmentation').addChild(p1)

        # k-means
        p1 = Parameter.create(name='k-means', type='bool', value=True)
        p1.addChild(Parameter.create(name='Clusters', type='int', value=3))
        self.params.child('Segmentation').addChild(p1)
        # SRM
        p1 = Parameter.create(name='SRM', type='bool', value=False)
        self.params.child('Segmentation').addChild(p1)

        # RMRF
        p1 = Parameter.create(name='PMRF', type='bool', value=False, readonly=True)
        p1.addChild(Parameter.create(name='Clusters', type='int', value=2, readonly=True))
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

        # Masking action
        self.actionMask = self.newAction(icon_path='xicam/gui/icons_08.png',
            tooltip='Mask')
        self.addAction(self.actionMask)

        # ROI
        self.actionROI = self.newAction(icon_path='xicam/gui/icons_roi.png',
            tooltip='ROI', enabled=True)
        self.addAction(self.actionROI)

        # Filter action
        self.actionFilter = self.newAction(icon_path='xicam/gui/icons_62.png', 
            tooltip='Filter')
        self.addAction(self.actionFilter)

        # Segmentation action
        self.actionSegment = self.newAction(icon_path='xicam/gui/icons_34.png', 
            tooltip='Run Segmentation')
        self.addAction(self.actionSegment)

        # export config action
        self.actionSaveCfg = self.newAction(icon_path="xicam/gui/write_inp.png", 
            tooltip='Export Config')
        self.addAction(self.actionSaveCfg)

        # select and display segmentation results
        self.actionShowKmeans = self.newAction(icon_path='xicam/gui/icons_kmeans.png',
            tooltip='k-means')
        self.actionShowSRM = self.newAction(icon_path='xicam/gui/icons_srm.png',
            tooltip='SRM')
        self.actionShowPRMF = self.newAction(icon_path='xicam/gui/icons_pmrf.png',
            tooltip='PRMF')
        menu = QtGui.QMenu()
        menu.addAction(self.actionShowKmeans)
        menu.addAction(self.actionShowSRM)
        menu.addAction(self.actionShowPRMF)
        btnShowSegmentation = QtGui.QToolButton()
        btnShowSegmentation.setDefaultAction(self.actionShowKmeans)
        btnShowSegmentation.setMenu(menu)
        btnShowSegmentationAction = QtGui.QWidgetAction(self)
        btnShowSegmentationAction.setDefaultWidget(btnShowSegmentation)
        self.addAction(btnShowSegmentationAction)


    @staticmethod
    def newAction(icon_path=None, tooltip=None, enabled=False):
        action = QtGui.QAction(None)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal, QtGui.QIcon.On)
        action.setIcon(icon)
        action.setToolTip(tooltip)
        action.setEnabled(enabled)
        return action
