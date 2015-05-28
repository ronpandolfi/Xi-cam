import pyqtgraph as pg
# import loader
from PySide import QtGui
import viewer
import numpy as np
import pipeline
import os
from operator import itemgetter


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
        self.variation = dict()
        if len(paths) > 0:
            imgdata, paras = pipeline.loader.loadpath(paths[0])
        else:
            imgdata = None
        super(timelinetab, self).__init__(imgdata, experiment, parentwindow)

        self.paths = dict(zip(range(len(paths)), sorted(paths)))
        self.experiment = experiment
        self.parentwindow = parentwindow
        self.setvariationmode(0)
        self.gotomax()

    def reduce(self):
        pass
        # self.skipframes = (self.variationy[0:-1] / self.variationy[1:]) > 0.1


    def scan(self):
        if len(self.paths) < 3:
            return None

        self.variation = dict()
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
            nxt, _ = pipeline.loader.loadpath(self.paths[i + 2])
            #print i, self.paths[i]
            if os.path.splitext(self.paths[i + 1])[1] == '.nxs':
                variationy = pipeline.loader.readvariation(self.paths[i])


            else:

                if curr is None:
                    prev = curr.copy()
                    curr = nxt.copy()
                    continue

                # print curr, prev,'\n'
                with np.errstate(divide='ignore', invalid='ignore'):
                    variationy = pipeline.variation.variation(self.operationindex, prev, curr,
                                                              nxt)  #operations[self.operationindex](curr, prev)

            variationx = int(os.path.splitext(os.path.basename(self.paths[i + 1]).split('_')[-1])[0])
            self.variation[variationx] = variationy
            prev = curr.copy()
            curr = nxt.copy()
            # print self.variation

    def appendimage(self, d, paths):
        for path in paths:
            path = os.path.join(d, path)
            variation = pipeline.loader.readvariation(path)
            # print ('var:',variation)
            #print(path)
            try:
                frame = int(os.path.splitext(os.path.basename(path).split('_')[-1])[0])
            except ValueError:
                print 'Path has no frame number:', path
                continue
            self.variation[frame] = variation

            self.paths[frame] = path

        self.plotvariation()
        #self.parentwindow.app.processEvents()


    def setvariationmode(self, index):
        self.operationindex = index
        self.scan()
        self.plotvariation()


    def plotvariation(self):
        if len(self.variation) == 0:
            return None

        # TODO: plot variation with indices, and skipped frames; skip None's
        # print self.variation
        #self.variation = sorted(self.variation, key=itemgetter(0))
        #print self.variation
        variation = np.array(self.variation.items())
        #print variation.shape
        variation = variation[variation[:, 0].argsort()]
        #print variation.shape
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
        self.parentwindow.timeruler.setValue(pos)
        #self.parentwindow.timearrow.setPos(pos,self.variation[pos])
        self.redrawframe()

    def redrawframe(self):
        path = self.paths[self.parentwindow.timeruler.value() + 1]
        data, paras = pipeline.loader.loadpath(path)
        self.imgdata = np.rot90(data, 3)
        self.redrawimage()

    def gotomax(self):
        pass
        #self.parentwindow.timeruler.setValue(np.argmax(self.variationy))