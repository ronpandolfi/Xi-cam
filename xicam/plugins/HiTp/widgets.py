"""
Created on Apr 2017

@author: Ron Pandolfi
"""

import pyqtgraph as pg
from PySide.QtGui import *
from PySide.QtCore import *
import pandas as pd
import weakref
import numpy as np
from pyqtgraph import functions as fn
import os.path
import scipy
from pypif import pif
from pypif.obj import *

class WaferView(QWidget):
    sigPlot = Signal(object,object,object)  # emits 2-d cake array, 1-D Q, 1-D chi

    csvkeys = {'crystallinity':'Imax/Iave','peaks':'num_of_peaks', 'texture':'texture_sum', 'SNR':'SNR', 'NND':'neighbor_distances', 'Imax':'Imax'}

    def __init__(self):
        super(WaferView, self).__init__()

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.WaferPlot = pg.PlotWidget()
        self.layout.addWidget(self.WaferPlot)
        self.HLUT = ScatterHistogramLUTWidget()
        self.layout.addWidget(self.HLUT)
        ## wrap functions from histogram
        for fn in ['setHistogramRange', 'autoHistogramRange', 'getLookupTable', 'getLevels']:
            setattr(self, fn, getattr(self.HLUT, fn))

        # self.imageItem = weakref.ref(img)
        # img.sigImageChanged.connect(self.imageChanged)
        # img.setLookupTable(self.getLookupTable)  ## send function pointer, not the result
        # #self.gradientChanged()
        # self.regionChanged()
        # self.imageChanged(autoLevel=True)

        self.waferplotitem = LUTScatterPlotItem(size=10, pen=pg.mkPen('w'), cmap='viridis')
        self.WaferPlot.addItem(self.waferplotitem)
        self.HLUT.setScatterItem(self.waferplotitem)
        ## Make all plots clickable
        # self.lastClicked = None

        # def clicked(plot, points):
        #     p = self.lastClicked
        #     if p: p.setPen(p.brush().color(), width=0)
        #     p = points[0]
        #     p.setPen('w', width=5)
        #     self.lastClicked = p

        # self.waferplotitem.sigClicked.connect(clicked)

        self.WaferPlot.scene().sigMouseClicked.connect(self.mouseClicked)
        self.waferplotitem.sigPlotReduced.connect(self.itemClicked)

        self.lastcsv = None
        self.mode = 'SNR'

    def itemClicked(self, x, y):
        d = pd.read_csv(self.lastcsv)
        clickeditem = d.loc[(d['p_x'] == x) & (d['p_y'] == y)]
        path = clickeditem.path.values[-1]
        path = os.path.dirname(path) + '/Processed/' + os.path.splitext(os.path.basename(path))[0] + '_Qchi.mat'
        mat = scipy.io.loadmat(path)
        cake=mat['cake']
        Q=mat['Q']
        chi=mat['chi']

        self.sigPlot.emit(cake,Q,chi)

        # print('clicked:',x,y)


    def mouseClicked(self, event):
        '''

        Parameters
        ----------
        event : QMouseEvent

        '''
        print(event.pos())
        # get cake data from file
        # ...
        # emit cake data
        # cake = np.zeros((10, 10))
        # self.sigPlot.emit(cake)
        event.accept()

    @Slot(int)
    def setMode(self, modeindex):
        self.mode = self.csvkeys.keys()[modeindex]
        self.redrawfromCSV()

    @Slot(str)
    def redrawfromCSV(self, csv=None):
        '''

        Parameters
        ----------
        csv : str
            filepath reference to CSV file to be displayed
        '''
        if not csv: csv = self.lastcsv
        self.lastcsv = csv
        # print csv
        # print 'loading csv into dataframe'
        p = pd.read_csv(csv)
        # print list(p)
        x = np.nan_to_num(p['p_x'])
        y = np.nan_to_num(p['p_y'])
        z = np.nan_to_num(p[self.csvkeys[self.mode]])
        # print x, y
        d = (x + y).argsort()
        x, y, z = (x[d],
                   y[d],
                   z[d])
        # zmin = min(z)
        # zrange = np.ptp(z)
        # z = np.nan_to_num((z-zmin)/zrange *256)

        # viridisposs = [0.0,0.25,0.5,0.75,1.0]
        # viridiscolors = [(68, 1, 84, 255),(58, 82, 139, 255),(32, 144, 140, 255),(94, 201, 97, 255),(253, 231, 36, 255)]
        # viridismap = pg.ColorMap(viridisposs,viridiscolors,mode=pg.ColorMap.RGB)


        points = [{'pos': (x[i], y[i]),
                   'data': z[i],
                   'size': 30,
                   # 'brush': viridismap.map(z[i]),
                   # 'pen':pg.mkPen(width=0,color=viridismap.map(z[i]))
                   } for i in range(len(z))]

        self.waferplotitem.setPoints(points)
        self.WaferPlot.autoRange()

    def export(self):
        data = pd.read_csv(self.lastcsv)

        # build package
        # for row in data.iterrows():



        return payload


class LUTScatterPlotItem(pg.ScatterPlotItem):
    sigPlotReduced = Signal(float,float)

    def __init__(self, *args, **kwargs):
        super(LUTScatterPlotItem, self).__init__(*args, **kwargs)
        self.levels = [0, 1]
        self.lastClicked=None

    def mouseClickEvent(self,ev):
        super(LUTScatterPlotItem, self).mouseClickEvent(ev)
        if ev.isAccepted():
            p = self.lastClicked
            if p: p.setPen(p.brush().color(), width=0)
            p = self.ptsClicked[0]
            p.setPen('w', width=5)
            self.lastClicked = p
            self.sigPlotReduced.emit(p._data[0],p._data[1])

    def setColorMap(self, cmap):
        self.cmap = cmap

    def addPoints(self, *args, **kargs):
        """
                Add new points to the scatter plot.
                Arguments are the same as setData()
                """

        self.lastargs = args

        ## deal with non-keyword arguments
        if len(args) == 1:
            kargs['spots'] = args[0]
        elif len(args) == 2:
            kargs['x'] = args[0]
            kargs['y'] = args[1]
        elif len(args) > 2:
            raise Exception('Only accepts up to two non-keyword arguments.')

        ## convert 'pos' argument to 'x' and 'y'
        if 'pos' in kargs:
            pos = kargs['pos']
            if isinstance(pos, np.ndarray):
                kargs['x'] = pos[:, 0]
                kargs['y'] = pos[:, 1]
            else:
                x = []
                y = []
                for p in pos:
                    if isinstance(p, QPointF):
                        x.append(p.x())
                        y.append(p.y())
                    else:
                        x.append(p[0])
                        y.append(p[1])
                kargs['x'] = x
                kargs['y'] = y

        ## determine how many spots we have
        if 'spots' in kargs:
            numPts = len(kargs['spots'])
        elif 'y' in kargs and kargs['y'] is not None:
            numPts = len(kargs['y'])
        else:
            kargs['x'] = []
            kargs['y'] = []
            numPts = 0

        ## Extend record array
        oldData = self.data
        self.data = np.empty(len(oldData) + numPts, dtype=self.data.dtype)
        ## note that np.empty initializes object fields to None and string fields to ''

        self.data[:len(oldData)] = oldData
        # for i in range(len(oldData)):
        # oldData[i]['item']._data = self.data[i]  ## Make sure items have proper reference to new array

        newData = self.data[len(oldData):]
        newData['size'] = -1  ## indicates to use default size

        if 'spots' in kargs:
            spots = kargs['spots']
            for i in range(len(spots)):
                spot = spots[i]

                for k in spot:
                    if k == 'pos':
                        pos = spot[k]
                        if isinstance(pos, QPointF):
                            x, y = pos.x(), pos.y()
                        else:
                            x, y = pos[0], pos[1]
                        newData[i]['x'] = x
                        newData[i]['y'] = y
                    elif k == 'pen':
                        newData[i][k] = fn.mkPen(spot[k])
                    elif k == 'brush':
                        newData[i][k] = fn.mkBrush(spot[k])
                    elif k in ['x', 'y', 'size', 'symbol', 'brush', 'data']:
                        newData[i][k] = spot[k]
                    else:
                        pass
                        # raise Exception("Unknown spot parameter: %s" % k)
        elif 'y' in kargs:
            newData['x'] = kargs['x']
            newData['y'] = kargs['y']

        if 'pxMode' in kargs:
            self.setPxMode(kargs['pxMode'])
        if 'antialias' in kargs:
            self.opts['antialias'] = kargs['antialias']

        ## Set any extra parameters provided in keyword arguments
        for k in ['pen', 'brush', 'symbol', 'size']:
            if k in kargs:
                setMethod = getattr(self, 'set' + k[0].upper() + k[1:])
                setMethod(kargs[k], update=False, dataSet=newData, mask=kargs.get('mask', None))

        if 'data' in kargs:
            self.setPointData(kargs['data'], dataSet=newData)

        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.bounds = [None, None]
        newData = self.colorSpots(newData)
        self.invalidate()
        self.updateSpots(newData)
        self.sigPlotChanged.emit(self)

    def colorSpots(self, newData):
        if hasattr(self, 'cmap'):
            colormap = self.cmap()()
            l = self.levels
            for p in newData:
                c = colormap.map((p['data'] - l[0]) / (l[1] - l[0]))
                p['brush'] = fn.mkBrush(c)
                p['pen'] = fn.mkPen(c)

        return newData

    def setLevels(self, levels, update=True):
        """
        Set image scaling levels. Can be one of:

        * [blackLevel, whiteLevel]
        * [[minRed, maxRed], [minGreen, maxGreen], [minBlue, maxBlue]]

        Only the first format is compatible with lookup tables. See :func:`makeARGB <pyqtgraph.makeARGB>`
        for more details on how levels are applied.
        """
        if levels is not None:
            levels = np.asarray(levels)
        if not fn.eq(levels, self.levels):
            self.levels = levels
            self._effectiveLut = None
            if update and hasattr(self, 'lastargs'):
                self.setData(*self.lastargs)


# if not isinstance(spot,dict): setattr(spot,'__iter__',iter(zip(*spot.dtype.descr)[0]))

class ScatterHistogramLUTWidget(pg.HistogramLUTWidget):
    def __init__(self, *args, **kwargs):
        super(ScatterHistogramLUTWidget, self).__init__(*args, **kwargs)
        self.item = ScatterHistogramLUTItem(*args, **kwargs)
        self.setCentralItem(self.item)
        self.autoLevel = True


class ScatterHistogramLUTItem(pg.HistogramLUTItem):
    def __init__(self, *args, **kwargs):

        super(ScatterHistogramLUTItem, self).__init__(*args, **kwargs)
        reset = QAction('Reset', self.vb.menu)
        self.vb.menu.addAction(reset)
        reset.triggered.connect(self.reset)

    def reset(self):
        self.autoLevel = True
        self.plotChanged(autoLevel=True)

    def setScatterItem(self, plt):
        """Set a ScatterPlotItem to have its levels and LUT automatically controlled
        by this HistogramLUTItem.
        """
        self.plotItem = weakref.ref(plt)
        plt.sigPlotChanged.connect(lambda s: self.plotChanged())
        plt.setColorMap(self.getColorMap)  ## send function pointer, not the result
        # self.gradientChanged()
        self.regionChanged()
        self.autoLevel = True
        self.plotChanged(autoLevel=True)
        # self.vb.autoRange()

    def gradientChanged(self):
        if self.plotItem() is not None:
            self.plotItem().setLevels(self.region.getRegion())
        self.lut = None
        self.sigLookupTableChanged.emit(self)

    def regionChanged(self):
        self.autoLevel = False  # TODO: fix autoleveling being reset?
        if self.plotItem() is not None:
            self.plotItem().setLevels(self.region.getRegion())
        self.sigLevelChangeFinished.emit(self)
        # self.update()

    def plotChanged(self, autoLevel=False, autoRange=False):
        # profiler = debug.Profiler()
        data = self.plotItem().data['data']
        if len(data):
            hist, bins = np.histogram(data, 100)
            # profiler('get histogram')
            if hist[0] is None:
                return
            self.plot.setData(bins[:-1], hist)
            # profiler('set plot')
            print('autolevel:', self.autoLevel)
            if autoLevel or self.autoLevel:
                mn = bins[0]
                mx = bins[-1]
                self.region.setRegion([mn, mx])
                # profiler('set region')

    def getColorMap(self, plt=None, n=None, alpha=None):
        """Return a lookup table from the color gradient defined by this
        HistogramLUTItem.
        """
        if self.lut is None:
            self.lut = self.gradient.colorMap
        return self.lut


class LocalView(QTabWidget):
    def __init__(self):
        super(LocalView, self).__init__()

        self.view1D = pg.PlotWidget()
        self.view2D = pg.ImageView()

        self.addTab(self.view1D, '1D')
        self.addTab(self.view2D, 'Cake')

        self.setTabPosition(self.West)
        self.setTabShape(self.Triangular)

    @Slot(object,object,object)
    def plot(self, cake, Q, chi):
        '''

        Parameters
        ----------
        cake : np.ndarray
            The caked image array to be displayed

        '''
        # display cake and 1D in views
        self.view1D.clear()
        self.view1D.plot(Q.flatten(),np.sum(cake,axis=0))
        self.view2D.setImage(cake)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    # ######## SAVE THIS FOR DEBUGGING SEG FAULTS; issues are usually doing something outside the gui thread
    # import sys
    #
    #
    # def trace(frame, event, arg):
    #     print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
    #     return trace
    #
    #
    # sys.settrace(trace)

    app = QApplication([])

    ## Create window with ImageView widget
    win = QMainWindow()
    win.resize(800, 800)

    w = WaferView()
    # TODO: Looks like the three lines here are not necessary
    # csv = '/home/rp/data/HiTp/Sample14_master_metadata_high.csv'
    # csv = 'C:\\Research_FangRen\\Data\\Apr2016\\Jan_samples\\Sample1\\Sample14_master_metadata_high.csv'
    # w.redrawfromCSV(csv)

    win.setCentralWidget(w)
    win.setWindowTitle('Fitting')
    win.show()

    QApplication.instance().exec_()
