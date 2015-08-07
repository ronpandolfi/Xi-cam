#from pylab import *
from scipy import signal
from scipy.ndimage import filters
from hipies import hipiesdebug
import pyqtgraph as pg
from PySide import QtCore
import inspect
import numpy as np

maxfiltercoef = 5
cwtrange = np.arange(1, 100)

maxfiltercoef = 5
cwtrange = np.arange(3, 100)
gaussiancentersigma = 2
gaussianwidthsigma = 5


@hipiesdebug.timeit
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


# def findpeaks(x, y, filtersize=(5, 5), gaussianwidthsigma=5, gaussiancentersigma=0, minimumsigma=100, snr=1.2):
# if x is None:
#         x = np.arange(len(y))
#
#     cwtdata = signal.cwt(y, signal.ricker, cwtrange)
#     cwtdata = filters.gaussian_filter1d(cwtdata, gaussiancentersigma, axis=1)
#     cwtdata = filters.gaussian_filter1d(cwtdata, gaussianwidthsigma, axis=0)
#     cwtdata = filters.minimum_filter1d(cwtdata, minimumsigma, axis=0)
#
#     maxima = (cwtdata == filters.maximum_filter(cwtdata, filtersize))
#     maximasigmas, maximaxs = np.where(maxima == 1)
#
#     x = np.array(x)
#     y = np.array(y)
#
#     peakxs = []
#     peaksigmas = []
#
#     plt.plot(maximaxs, y[maximaxs], 'or')
#
#     for mx, sigma in zip(maximaxs, maximasigmas):
#         print mx, sigma
#         print max(0, mx - filtersize[1]), min(len(y), mx + filtersize[1])
#         window = y[max(0, mx - 2 * filtersize[1]):min(len(y), mx + 2 * filtersize[1])]  # maybe scale with m's width?
#         noiselevel = stats.scoreatpercentile(window, 10)
#         print(mx, noiselevel)
#         if y[mx] > snr * noiselevel:
#             peakxs.append(mx)
#             peaksigmas.append(sigma)
#
#     # print maximaloc
#
#     # print('before',np.array(list(np.array(np.vstack([x[maximaloc[1]], y[maximaloc[1]], maximaloc])))).shape)
#     return list(np.array(np.vstack([x[peakxs], y[peakxs], [peakxs, peaksigmas]])))


# TODO: Refactor this class into hipies module so I can get rid of pyside dependency
class peaktooltip:
    def __init__(self, x, y, widget):
        self.q, self.I, self.width, self.index = findpeaks(x, y)
        self.scatterPoints = pg.PlotDataItem(self.q, self.I, size=10, pen=pg.mkPen(None),
                                             symbolPen=None, symbolBrush=pg.mkBrush(255, 255, 255, 120), symbol='o')
        self.display_text = pg.TextItem(text='', color=(176, 23, 31), anchor=(0, 1))
        self.display_text.hide()
        widget.addItem(self.scatterPoints)
        widget.addItem(self.display_text)
        self.scatterPoints.scene().sigMouseMoved.connect(self.onMove)

    def onMove(self, pixelpos):
        # act_pos = self.scatterPoints.mapFromScene(pos)
        #p1 = self.scatterPoints.pointsAt(act_pos)

        # get nearby points
        itempos = self.scatterPoints.mapFromScene(pixelpos)
        itemx = itempos.x()
        itemy = itempos.y()
        pixeldelta = 7
        delta = self.scatterPoints.mapFromScene(QtCore.QPointF(pixeldelta + pixelpos.x(), pixeldelta + pixelpos.y()))
        deltax = delta.x() - itemx
        deltay = -(delta.y() - itemy)
        p1 = [point for point in zip(self.q, self.I) if (itemx - deltax < point[0] and point[0] < itemx + deltax) and (
        itemy - deltay < point[1] and point[1] < itemy + deltay)]
        # print(self.q[1], self.I[1])
        #print(itemx - deltax, itemx + deltax, itemy - deltay, itemy + deltay)
        #print(len(p1))

        """
        :type p1 : pg.graphicsItems.ScatterPlotItem.SpotItem
        """
        if len(p1) != 0:
            self.display_text.setText('q=%f\nI=%f' % (p1[0][0], p1[0][1]))
            self.display_text.setPos(*p1[0])
            self.display_text.show()
        else:
            self.display_text.hide()


