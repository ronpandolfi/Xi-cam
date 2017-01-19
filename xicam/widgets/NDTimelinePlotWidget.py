# --coding: utf-8 --
import numpy as np
from PySide import QtGui, QtCore
import pyqtgraph as pg
from modpkgs import nonesigmod

class timelineLinePlot(pg.PlotWidget):
    def __init__(self):
        super(timelineLinePlot, self).__init__()
        self.variationcurve = dict()

    def setData(self, t, y, color):
        colorhash = ','.join([str(c) for c in color])
        if not colorhash in self.variationcurve:
            self.variationcurve[colorhash] = self.plot()

        l=min(len(t),len(y))
        self.variationcurve[colorhash].setData(x=t[:l], y=y[:l])
        self.variationcurve[colorhash].setPen(pg.mkPen(color=color))


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

        self.curves = []
        self._curvesdata = []

    def mapSlider(self):
        return 10 * (np.exp(self.slider.value() / 100.) - 1)

    def setOffset(self, value):
        for i in range(len(self.curves)):
            x, y = self._curvesdata[i]
            self.curves[i].setData(x, y + i * self.mapSlider())

    def setData(self, t, x, y):
        if len(self.curves) == 0:
            for ycurve in y:
                self.appendcurve(t, x, ycurve)
        else:
            self.appendcurve(t, x, y[-1])
        self.setColors()

    def appendcurve(self, t, x, y):
        self._curvesdata.append((x, y))
        self.curves.append(self.plotWidget.plot(x, y + (len(self.curves) * self.mapSlider())))

    def setColors(self):
        for i in range(len(self.curves)):
            self.curves[i].setPen(
                pg.mkPen(color=[(1 - float(i) / len(self.curves)) * 255, float(i) / len(self.curves) * 255, 255]))

    def clear(self):
        self.curves = []
        self._curvesdata = []
        self.plotWidget.clear()


class timelineWaterfallPlot(pg.ImageView):
    def __init__(self, *args, **kwargs):
        self.axesItem = pg.PlotItem()
        self.axesItem.setLabel('top', "Time", units='s')
        self.axesItem.setLabel('left', 'q (Å⁻¹)')
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
        self._data = {'t': [], 'colors': []}
        self.setTabPosition(self.West)
        self.currentChanged.connect(self.widgetChanged)

    @QtCore.Slot(tuple, list)
    @nonesigmod.pyside_none_deco
    def addData(self, data, color=None):

        if data is None: return
        if color is None: color = [255, 255, 255]
        is1D = type(data[1]) is not tuple

        if is1D:
            t, y = data
        else:
            t, (x, y) = data

        if len(self._data['t']) == 0:
            self.guessPlot(data)
        # append data to currentData
        if is1D:
            colorhash = ','.join([str(c) for c in color])
            if colorhash=='255,255,255': self._data['t'].append(t)
            if colorhash not in self._data:
                self._data[colorhash] = []
                self._data['colors'].append(colorhash)
            self._data[colorhash].append(y)  # use color as hash

        else:
            self._data['t'].append(t)
            if 'y' not in self._data:
                self._data['y'] = [y]
            else:
                self._data['y'].append(y)
            if 'x' not in self._data: self._data['x'] = x  # maybe I should check that the x is the same otherwise?...
        self.setData()

    def setData(self):
        if 'x' not in self._data:
            for colorhash in self._data['colors']:
                color = map(int, colorhash.split(','))
                self.currentPlot().setData(self._data['t'], self._data[colorhash], color)
        else:
            self.currentPlot().setData(self._data['t'], self._data['x'], self._data['y'])

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
        self._data = {'t': [], 'colors': []}

    def widgetChanged(self, *args, **kwargs):
        self.currentPlot().clear()
        self.setData()


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
        if True:
            data = {'t': t, 'x': np.arange(0, 1, .01), 'y': np.sin(np.arange(0, 1, .01) * t), 'color': [255, 255, t]}
            t += 1
        else:
            data = {'t': t, 'y': np.sin(.01 * t), 'color': [255, 255, 255]}
            t += 1
        timelineplot.addData(**data)

        if t == 255: timer.stop()


    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
