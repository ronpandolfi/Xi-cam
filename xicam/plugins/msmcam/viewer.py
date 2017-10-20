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

class Viewer(QtGui.QWidget):
    """
    

    """
    def __init__(self, path= None, data = None, *args, **kwargs):
        if path is None and data is None:
            raise ValueError('Either path or data must be provided')

        super(Viewer, self).__init__(*args, **kwargs)
        
        viewer = QtGui.QTabWidget()
        viewer.TabPosition = QtGui.QTabWidget.South
        viewer.TabShape = QtGui.QTabWidget.Rounded
        
        if data is None:
            data = self.loaddata(path)
        
        imv = ImageView(viewer)
        imv.setImage(data) 
        
