import pyqtgraph as pg
from PySide.QtGui import *
from PySide.QtCore import *

import numpy as np

class WaferView(pg.ImageView):
    sigPlot = Signal(object) # emits 2-d cake array
    def __init__(self):
        super(WaferView, self).__init__()

    def mousePressEvent(self,event):
        print event.pos()
        #get cake data from file
        #...
        #emit cake data
        cake = np.zeros((10,10))
        self.sigPlot.emit(cake)


class LocalView(QTabWidget):
    def __init__(self):
        super(LocalView, self).__init__()

        self.view1D = pg.PlotWidget()
        self.view2D = pg.ImageView()

        self.addTab(self.view1D,'1D')
        self.addTab(self.view2D,'Cake')

        self.setTabPosition(self.West)
        self.setTabShape(self.Triangular)

    @Slot(object)
    def plot(self,cake):
        #display cake and 1D in views
        pass