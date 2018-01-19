import numpy as np
import pyqtgraph as pg

class CDRawWidget(pg.ImageView):
    pass

class CDCartoWidget(pg.ImageView):
    def __init__(self):
        self.plotitem = pg.PlotItem()
        super(CDCartoWidget, self).__init__(view=self.plotitem)

class CDModelWidget(pg.PlotWidget):
    def __init__(self):
        super(CDModelWidget, self).__init__()
        self.addLegend()
        self.orders = []
        self.orders1 = []
        for i, color in enumerate('gyrbcmkgyr'):
            self.orders.append(self.plot([], pen=pg.mkPen(color), name='Order ' + str(i)))
            self.orders1.append(self.plot([], pen=pg.mkPen(color), name='Order ' + str(i)))

class CDProfileWidget(pg.ImageView):
    pass

class CDLineProfileWidget(pg.PlotWidget):
    def __init__(self):
        super(CDLineProfileWidget, self).__init__()
        self.setAspectLocked(True)

    def plotLineProfile(self, h, w, langle, rangle=None):
        self.clear()
        x, y = self.profile(h, w, langle, rangle)
        self.plot(x, y)

    @staticmethod
    def profile(h, w, langle, rangle=None):
        if isinstance(langle, list):
            langle = np.array(langle)

        if not isinstance(langle, np.ndarray):
            raise TypeError('Angles must be a numpy.ndarray or list')

        langle = np.deg2rad(langle)
        if rangle is None:
            rangle = langle
        else:
            rangle = np.deg2rad(rangle)

        n = len(langle)
        x = np.zeros(2 * (n + 1), dtype=np.float_)
        y = np.zeros_like(x)

        dxl = np.cumsum(h / np.tan(langle))
        dxr = np.cumsum(h / np.tan(rangle))[::-1]
        x[0] = -0.5 * w
        x[-1] = 0.5 * w
        x[1:n + 1] = x[0] + dxl
        x[n + 1:-1] = x[-1] - dxr

        y[1:n + 1] = np.arange(1, n + 1) * h
        y[n + 1:-1] = np.arange(1, n + 1)[::-1] * h
        return x, y