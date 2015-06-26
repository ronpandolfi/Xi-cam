import pyqtgraph as pg
from PySide import QtGui
import viewer
import numpy as np
import pipeline
import os


class timelinetabtracker(QtGui.QWidget):
    """
    A light-weight version of the timeline tab that is retained when the full tab is disposed. Used to generate tab.
    """
    def __init__(self, paths, experiment, parent):
        super(timelinetabtracker, self).__init__()

        self.paths = paths
        self.experiment = experiment
        self.parent = parent
        self.tab = None

        parent.listmodel.widgetchanged()

        self.isloaded = False

    def load(self):
        if not self.isloaded:
            self.layout = QtGui.QHBoxLayout(self)
            simg = pipeline.loader.imageseries(self.paths, self.experiment)
            self.tab = timelinetab(simg, self.parent)
            self.layout.addWidget(self.tab)

            # print('Successful load! :P')
            self.isloaded = True


    def unload(self):
        if self.isloaded:
            self.layout.parent = None
            self.layout.deleteLater()
            # self.tab = None
            print('Successful unload!')
            self.isloaded = False


class timelinetab(viewer.imageTab):
    def __init__(self, simg, parentwindow):
        self.variation = dict()
        self.simg = simg
        dimg = simg.first()

        super(timelinetab, self).__init__(dimg, parentwindow)

        # self.paths = dict(zip(range(len(paths)), sorted(paths)))
        self.parentwindow = parentwindow
        self.setvariationmode(0)
        self.gotomax()
        self.previousPos = int(round(self.parentwindow.timeruler.value()))

    def reduce(self):
        pass
        # self.skipframes = (self.variationy[0:-1] / self.variationy[1:]) > 0.1

    def appendimage(self, d, paths):
        paths = [os.path.join(d, path) for path in paths]
        self.simg.appendimages(paths)

        self.plotvariation()

    def setvariationmode(self, index):
        self.operationindex = index
        self.simg.scan(self.operationindex)
        self.plotvariation()

    def plotvariation(self):
        if len(self.simg.variation) == 0:
            return None

        # TODO: plot variation with indices, and skipped frames; skip None's

        variation = np.array(self.simg.variation.items())
        variation = variation[variation[:, 0].argsort()]
        self.parentwindow.timeline.clear()
        self.parentwindow.timeline.enableAutoScale()
        self.parentwindow.timeruler = pg.InfiniteLine(pen=pg.mkPen('#FFA500', width=3), movable=True)
        self.parentwindow.timeline.addItem(self.parentwindow.timeruler)
        self.parentwindow.timeruler.setBounds([0, max(variation[:, 0])])
        self.parentwindow.timeruler.sigPositionChanged.connect(self.timerulermoved)
        self.parentwindow.timeline.plot(variation[:, 0], variation[:, 1])
        # self.parentwindow.timearrow = pg.ArrowItem(angle=-90, tipAngle=30, baseAngle=20,headLen=15,tailLen=None,brush=None,pen=pg.mkPen('#FFA500',width=3))
        #self.parentwindow.timeline.addItem(self.parentwindow.timearrow)
        #self.parentwindow.timearrow.setPos(0,self.variation[0])

    def timerulermoved(self):
        pos = int(round(self.parentwindow.timeruler.value()))


        # snap to int
        self.parentwindow.timeruler.blockSignals(True)
        self.parentwindow.timeruler.setValue(pos)
        self.parentwindow.timeruler.blockSignals(False)
        #self.parentwindow.timearrow.setPos(pos,self.variation[pos])

        if pos != self.previousPos:
            print pos
            self.redrawframe()
        self.previousPos = pos

    def redrawframe(self):
        key = self.parentwindow.timeruler.value() + 1
        self.dimg = self.simg.getDiffImage(key)
        self.redrawimage()

    def gotomax(self):
        pass
        #self.parentwindow.timeruler.setValue(np.argmax(self.variationy))