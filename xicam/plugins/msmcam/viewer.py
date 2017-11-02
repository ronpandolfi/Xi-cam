from __future__ import division
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
from pyqtgraph import RectROI

from collections import OrderedDict

class CenterWidget(QtGui.QTabWidget):
    """
    Sets up Main panel to display Images.
    """
    def __init__(self, *args, **kwargs):
        
        super(CenterWidget, self).__init__(*args, **kwargs)
        self.setTabPosition(QtGui.QTabWidget.South)
        self.setTabShape(QtGui.QTabWidget.Rounded)
        self.setTabsClosable(False)
        self.setMovable(False)
        self.roi = None
        self.ROISET = False

        self.tab = OrderedDict({ 'image': Viewer(), 'filtered': Viewer(), 'segmented': Viewer() })
        self.addTab(self.tab['image'], 'Image')
        self.addTab(self.tab['filtered'], 'Filtered')
        self.addTab(self.tab['segmented'], 'Segmented')

    def setROI(self):
        img = self.currentWidget()
        if self.ROISET is False:
            w, h = img.size().toTuple()
            pos = [w//2, h//2]
            s = max(w//10, 30)
            self.roi = RectROI(pos, [s, s], pen=(0, 9), centered=True, sideScalers=True)
            img.addItem(self.roi)
            self.ROISET = True
        else:
            self.ROISET = False
            img.removeItem(self.roi)
            self.roi.deleteLater()

class Viewer(ImageView):
    """
    Subclass pyqtgraph.ImageView to remove ROI and Menu widgets
    """
    def __init__(self, *args, **kwargs):
        super(Viewer, self).__init__(*args, **kwargs)
        self.ui.roiBtn.setParent(None)
        self.ui.menuBtn.setParent(None)

