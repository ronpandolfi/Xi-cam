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
from pyqtgraph import ImageView

class CenterWidget(QtGui.QTabWidget):
    """

    """
    def __init__(self, *args, **kwargs):
        
        super(CenterWidget, self).__init__(*args, **kwargs)
        self.setTabPosition(QtGui.QTabWidget.South)
        self.setTabShape(QtGui.QTabWidget.Rounded)
        self.setTabsClosable(False)
        self.setMovable(True)


        self.tab = { 'image': Viewer(), 'filtered': Viewer(), 'segmented': Viewer() }
        self.addTab(self.tab['image'], 'Image')
        self.addTab(self.tab['filtered'], 'Filtered')
        self.addTab(self.tab['segmented'], 'Segmented')

class Viewer(ImageView):
    """
    """
    def __init__(self, *args, **kwargs):
        super(Viewer, self).__init__(*args, **kwargs)
        self.ui.roiBtn.setParent(None)
        self.ui.menuBtn.setParent(None)
