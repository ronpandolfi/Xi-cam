import pyqtgraph as pg

import integration
import center_approx
from PySide.QtGui import QVBoxLayout
from PySide.QtGui import QSplitter
from PySide.QtGui import QWidget
from PySide.QtCore import Qt


class imageTab(QWidget):
    def __init__(self, imgdata, experiment):
        super(imageTab, self).__init__()

        self.imgdata = imgdata
        self.experiment = experiment

        self.imgview = pg.ImageView(self)
        self.imgview.setImage(imgdata)
        self.imgview.autoRange()

        menu = self.imgview.imageItem.ctrlMenu

        self.viewbox = self.imgview.getView()

        self.Layout = QVBoxLayout()
        self.Splitter = QSplitter(Qt.Vertical)
        self.Layout.addWidget(self.Splitter)
        self.Splitter.addWidget(self.imgview)
        self.setLayout(self.Layout)

        ##
        self.findcenter()
        self.radialintegrate()
        ##

    def findcenter(self):
        [x, y] = center_approx.center_approx(self.imgdata)
        print(x, y)
        center = pg.ScatterPlotItem([y], [x], pen=None, symbol='o')
        self.viewbox.addItem(center)
        self.experiment.setValue('Center X', x)
        self.experiment.setValue('Center Y', y)

    def radialintegrate(self):
        q, radialprofile = integration.radialintegrate(self.imgdata, self.experiment)
        self.integration = pg.PlotWidget()
        self.integration.plot(q, radialprofile)
        self.Splitter.addWidget(self.integration)

    def polymask(self):
        ROI = pg.PolyLineROI([[80, 60], [90, 30], [60, 40]], pen=(6, 9), closed=True)


class polymask(pg.ROI):
    pass

