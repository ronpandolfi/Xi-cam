# --coding: utf-8 --
import numpy as np
from PySide import QtGui, QtCore
import pyqtgraph as pg
from modpkgs import nonesigmod
from collections import OrderedDict

class timelineLinePlot(pg.PlotWidget):
    def __init__(self):
        super(timelineLinePlot, self).__init__()
        self.variationcurve = OrderedDict()

    def setData(self, t, **kwargs):
        for key in kwargs.keys():
            if key != 'y' and not self.plotItem.legend:
                self.addLegend()

            if not key in self.variationcurve:
                self.variationcurve[key] = self.plot(name=key)

            y = kwargs[key]
            l=min(len(t),len(y))
            self.variationcurve[key].setData(x=t[:l], y=y[:l])
            self.variationcurve[key].setPen(pg.mkPen(color=pg.intColor(self.variationcurve.keys().index(key),len(self.variationcurve))))



class timelineStackPlot(QtGui.QWidget):
    def __init__(self):
        super(timelineStackPlot, self).__init__()

        self.slider = QtGui.QSlider(QtCore.Qt.Vertical, self)
        self.slider.setGeometry(500, 500, 30, 10)
        # slider.setFocusPolicy(QtCore.Qt.NoFocus)

        self.plotWidget = pg.PlotWidget()

        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.plotWidget)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self.setOffset)

        self.curves = {}
        self._curvesdata = {}

    def mapSlider(self):
        return 10 * (np.exp(self.slider.value() / 100.) - 1)

    def setOffset(self, _):
        for i,curve in enumerate(self.curves.values()):
            x, y = self._curvesdata['x'],self._curvesdata['y']
            curve.setData(x[i], y[i] + i * self.mapSlider())

    def setData(self, t, x, y):
        self._curvesdata={'t':t,'x':x,'y':y}
        for i,t0 in enumerate(t):
            if t0 not in self.curves:
                self.curves[t0] = self.plotWidget.plot(x[i], y[i] + (len(self.curves) * self.mapSlider()))
        self.setColors()
        self.setOffset(0)

    def setColors(self):
        for i,curve in enumerate(self.curves.values()):
            curve.setPen(
                pg.mkPen(color=[(1 - float(i) / len(self.curves)) * 255, float(i) / len(self.curves) * 255, 255]))

    def clear(self):
        self.curves = {}
        self._curvesdata = {}
        self.plotWidget.clear()


class timelineWaterfallPlot(pg.ImageView):
    def __init__(self, *args, **kwargs):
        self.axesItem = pg.PlotItem()
        self.axesItem.setLabel('top', "Time", units='s')
        self.axesItem.setLabel('left', u'q (Å⁻¹)')
        self.axesItem.axes['left']['item'].setZValue(10)
        self.axesItem.axes['top']['item'].setZValue(10)
        self.axesItem.showAxis('bottom', False)

        kwargs['view'] = self.axesItem
        super(timelineWaterfallPlot, self).__init__(*args, **kwargs)
        self.view.setAspectLocked(False)

    def setData(self, t, x, y):
        self.setImage(np.array(y))

    def setImage(self, *args, **kwargs):
        kwargs['pos'] = [0, 0]
        super(timelineWaterfallPlot, self).setImage(*args, **kwargs)


class TimelinePlot(QtGui.QTabWidget):
    def __init__(self):
        super(TimelinePlot, self).__init__()
        self.waterfall = timelineWaterfallPlot()
        self.lineplot = timelineLinePlot()
        self.stackplot = timelineStackPlot()
        self.addTab(self.lineplot, 'Line plot')
        self.addTab(self.waterfall, 'Waterfall')
        self.addTab(self.stackplot, 'Stack plot')
        self._data = {'t': []}
        self.setTabPosition(self.West)
        self.currentChanged.connect(self.widgetChanged)

    @QtCore.Slot(object, list)
    @nonesigmod.pyside_none_deco
    def addData(self, t, *args):
        kwargs=OrderedDict()
        if isinstance(args[0],OrderedDict):
            kwargs=args[0]
        elif len(args)==1:
            kwargs['y']=args[0]
        elif len(args)==2:
            kwargs['x']=args[0]
            kwargs['y']=args[1]

        if t is None: return

        is1D = True
        if len(kwargs)==2 and 'x' in kwargs and 'y' in kwargs:
            x = kwargs['x']
            y = kwargs['y']
            is1D = False

        if len(self._data['t']) == 0:
            self.setDMode(is1D)

        # append data to currentData
        self._data['t'].append(t)
        for key,value in kwargs.items():
            if not key in self._data: self._data[key] = []
            self._data[key].append(value)

        self.setData()

    def setData(self):
        self.currentPlot().setData(**self._data)

    def setDMode(self,is1D):
        index = None
        if is1D:
            self.setCurrentIndex(0)
            self.setTabEnabled(0, True)
            self.setTabEnabled(1, False)
            self.setTabEnabled(2, False)
            index = 0  # line plot
        else:
            self.setTabEnabled(0, False)
            self.setTabEnabled(1, True)
            self.setTabEnabled(2, True)
            if self.currentIndex() == 0:
                self.setCurrentIndex(1)
            index = 1  # waterfall plot

        if index is None: index = self.currentIndex()  # otherwise do nothing
        return index

    def guessPlot(self, data):
        # if its a line plot, switch to line plot; if its a 3d plot, switch to a 3d plot or whichever is already selected
        index = None
        if type(data[1]) is tuple:
            self.setTabEnabled(0, False)
            self.setTabEnabled(1, True)
            self.setTabEnabled(2, True)
            if self.currentIndex() == 0:
                self.setCurrentIndex(1)
            index = 1  # waterfall plot

        else:
            self.setCurrentIndex(0)
            self.setTabEnabled(0, True)
            self.setTabEnabled(1, False)
            self.setTabEnabled(2, False)
            index = 0  # line plot

        if index is None: index = self.currentIndex()  # otherwise do nothing
        return index

    def currentPlot(self):
        return self.widget(self.currentIndex())

    def clearData(self):
        self._data = {'t': []}

    def widgetChanged(self, *args, **kwargs):
        self.currentPlot().clear()
        if len(self._data)>1: self.setData()


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys

    app = QtGui.QApplication([])

    ## Create window with ImageView widget
    win = QtGui.QMainWindow()
    win.resize(800, 800)

    timelineplot = TimelinePlot()
    win.setCentralWidget(timelineplot)
    win.show()
    win.setWindowTitle('TimelinePlot')

    t = 0


    def update():
        global t, timer
        if False:
            data = OrderedDict([('t', t),('FWHM',np.random.rand()),('center',t)])
            t += 1
        elif True:
            data = {'t': t, 'x': np.arange(0, 1, .01), 'y': np.sin(np.arange(0, 1, .01) * t)}
            t += 1
        else:
            data = {'t': t, 'y': np.sin(.01 * t)}
            t += 1
        timelineplot.addData(**data)

        if t == 255: timer.stop()


    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
