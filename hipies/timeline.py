import pyqtgraph as pg
# import loader
from PySide import QtGui
import viewer
import numpy as np
import pipeline



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
    def __init__(self, paths, experiment, parentwindow):

        imgdata, paras = pipeline.loader.loadpath(paths[0])
        super(timelinetab, self).__init__(imgdata, experiment, parentwindow)

        self.paths = paths
        self.experiment = experiment
        self.parentwindow = parentwindow
        self.setvariationmode(0)
        self.gotomax()

    def reduce(self):
        self.skipframes = (self.variation[0:-1] / self.variation[1:]) > 0.1


    def scan(self):
        self.variation = np.zeros(self.paths.__len__() - 2)
        # operations = [lambda c, p, n: np.sum(np.square(c - p) / p),  # Chi squared
        #              lambda c, p, n: np.sum(np.abs(c - p)),  # Absolute difference
        #              lambda c, p, n: np.sum(np.abs(c - p) / p),  # Norm. absolute difference
        #              lambda c, p, n: np.sum(c),  # Sum intensity
        #              lambda c, p, n: np.sum(np.abs(n-c) / c)-np.sum(np.abs(c-p) / c)] # Norm. abs. diff. derivative
        # print(':P')
        #print self.paths

        #get the first frame's profile
        prev, _ = pipeline.loader.loadpath(self.paths[0])
        curr, _ = pipeline.loader.loadpath(self.paths[1])
        for i in range(self.paths.__len__() - 2):
            #print i, self.paths[i]
            nxt, _ = pipeline.loader.loadpath(self.paths[i + 2])
            if curr is None:
                self.variation[i] = None
                prev = curr.copy()
                curr = nxt.copy()
                continue

            #print curr, prev,'\n'
            with np.errstate(divide='ignore', invalid='ignore'):
                self.variation[i] = pipeline.variation.variation(self.operationindex, prev, curr,
                                                                 nxt)  #operations[self.operationindex](curr, prev)
            prev = curr.copy()
            curr = nxt.copy()
            # print self.variation

    def setvariationmode(self, index):
        self.operationindex = index
        self.scan()
        self.plotvariation()


    def plotvariation(self):
        self.parentwindow.timeline.clear()
        self.parentwindow.timeline.enableAutoScale()
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
        data, paras = pipeline.loader.loadpath(path)
        self.imgdata = np.rot90(data, 3)
        self.redrawimage()

    def gotomax(self):
        self.parentwindow.timeruler.setValue(np.argmax(self.variation))