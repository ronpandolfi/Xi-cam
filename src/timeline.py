import pyqtgraph as pg
import fabio
from PySide import QtGui
import viewer
import integration
import numpy as np


class timelinetabtracker(QtGui.QWidget):
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
            self.tab = timelinetab(self.paths, self.experiment, self.parent)
            self.layout.addWidget(self.tab)

            print('Successful load! :P')
            self.isloaded = True


    def unload(self):
        if self.isloaded:
            self.layout.parent = None
            self.layout.deleteLater()
            # self.tab = None
            print('Successful unload!')
            self.isloaded = False


class timelinetab(viewer.imageTab):
    def __init__(self, paths, experiment, parentwindow):
        imgdata = fabio.open(paths[0]).data
        super(timelinetab, self).__init__(imgdata, experiment, parentwindow)

        self.paths = paths
        self.experiment = experiment
        self.parentwindow = parentwindow
        self.variation = np.zeros(self.paths.__len__() - 1)
        self.scan()
        self.plotvariation()
        self.gotomax()


    def scan(self):
        operation = lambda curr, prev: np.sum(np.square(curr.prev))
        operation2 = lambda curr, prev: np.sum(np.abs(curr - prev))
        # print(':P')
        #print self.paths

        #get the first frame's profile
        prev = fabio.open(self.paths[0]).data
        for i in range(self.paths.__len__() - 1):
            #print i, self.paths[i]
            curr = fabio.open(self.paths[i + 1]).data
            if curr is None:
                self.variation[i] = None
                continue

            #print curr, prev,'\n'

            self.variation[i] = operation2(curr, prev)
            prev = curr
        print self.variation


    def plotvariation(self):
        self.parentwindow.timeline.clear()
        self.parentwindow.timeruler = pg.InfiniteLine(pen=pg.mkPen('#FFA500', width=3), movable=True)
        self.parentwindow.timeline.addItem(self.parentwindow.timeruler)
        self.parentwindow.timeruler.setBounds([0, self.variation.__len__() - 1])
        self.parentwindow.timeruler.sigPositionChanged.connect(self.timerulermoved)
        self.parentwindow.timeline.plot(self.variation)
        # self.parentwindow.timearrow = pg.ArrowItem(angle=-90, tipAngle=30, baseAngle=20,headLen=15,tailLen=None,brush=None,pen=pg.mkPen('#FFA500',width=3))
        #self.parentwindow.timeline.addItem(self.parentwindow.timearrow)
        #self.parentwindow.timearrow.setPos(0,self.variation[0])

    def timerulermoved(self):
        pos = int(round(self.parentwindow.timeruler.value()))
        self.parentwindow.timeruler.setValue(pos)
        #self.parentwindow.timearrow.setPos(pos,self.variation[pos])
        self.redrawframe()

    def redrawframe(self):
        path = self.paths[self.parentwindow.timeruler.value() + 1]
        self.imgdata = np.rot90(fabio.open(path).data, 2)
        self.redrawimage()

    def gotomax(self):
        self.parentwindow.timeruler.setValue(np.argmax(self.variation))