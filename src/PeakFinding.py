__author__ = 'remi'

from pylab import *
from scipy import signal
from scipy.ndimage import filters
import pyqtgraph as pg

maxfiltercoef = 5
cwtrange = np.arange(3, 100)
gaussiancentersigma = 2
gaussianwidthsigma = 5


def findpeaks(x, y):
    cwtdata = filters.gaussian_filter1d(
        filters.gaussian_filter1d(signal.cwt(y, signal.ricker, cwtrange), gaussiancentersigma, axis=1),
        gaussianwidthsigma, axis=0)
    maxima = (cwtdata == filters.maximum_filter(cwtdata, 5))
    maximaloc = np.where(maxima == 1)
    x = np.array(x)
    y = np.array(y)

    # print('before',np.array(list(np.array(np.vstack([x[maximaloc[1]], y[maximaloc[1]], maximaloc])))).shape)
    return list(np.array(np.vstack([x[maximaloc[1]], y[maximaloc[1]], maximaloc])))


# class peakplotitem(pg.PlotItem):
#    def __init__(self,x,y):
#        super(peakplotitem, self).__init__()
#        self.q, self.I, self.width, self.index = findpeaks(x,y)
#
#        self.peakdataitem=self.plot(self.q, self.I, pen=None, symbol='o')

#        self.peakdataitem.hover
#    def


class peaktooltip:
    def __init__(self, x, y, widget):
        self.q, self.I, self.width, self.index = findpeaks(x, y)
        self.scatterPoints = pg.ScatterPlotItem(self.q, self.I, size=10, pen=pg.mkPen(None),
                                                brush=pg.mkBrush(255, 255, 255, 120))
        self.display_text = pg.TextItem(text='', color=(176, 23, 31), anchor=(1, 1))
        self.display_text.hide()
        widget.addItem(self.scatterPoints)
        widget.addItem(self.display_text)
        self.scatterPoints.scene().sigMouseMoved.connect(self.onMove)

    def onMove(self, pos):
        act_pos = self.scatterPoints.mapFromScene(pos)
        p1 = self.scatterPoints.pointsAt(act_pos)
        """
        :type p1 : pg.graphicsItems.ScatterPlotItem.SpotItem
        """
        if len(p1) != 0:
            self.display_text.setText('q=%f\nI=%f' % (p1[0].pos().x(), p1[0].pos().y()))
            self.display_text.setPos(p1[0].pos())
            self.display_text.show()
        else:
            self.display_text.hide()


