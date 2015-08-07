import pyqtgraph as pg
from PySide import QtGui, QtCore
import numpy as np


class ArcROI(pg.ROI):
    """
    Elliptical ROI subclass with one scale handle and one rotation handle.


    ============== =============================================================
    **Arguments**
    pos            (length-2 sequence) The position of the ROI's origin.
    size           (length-2 sequence) The size of the ROI's bounding rectangle.
    \**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================

    """

    def __init__(self, pos, size, **args):
        # QtGui.QGraphicsRectItem.__init__(self, 0, 0, size[0], size[1])
        pg.ROI.__init__(self, pos, size, **args)
        #self.addRotateHandle([1.0, 0.5], [0.5, 0.5])
        #self.addScaleHandle([0.5*2.**-0.5 + 0.5, 0.5*2.**-0.5 + 0.5], [0.5, 0.5])
        self.addScaleRotateHandle([.5, 1], [.5, .5])
        self.aspectLocked = True

    def paint(self, p, opt, widget):
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.currentPen)

        p.scale(r.width(), r.height())  ## workaround for GL bug

        r = QtCore.QRectF(r.x() / r.width(), r.y() / r.height(), 1, 1)
        # p.drawEllipse(r)
        p.drawArc(r, 30 * 16, 120 * 16)



    def getArrayRegion(self, arr, img=None):
        """
        Return the result of ROI.getArrayRegion() masked by the elliptical shape
        of the ROI. Regions outside the ellipse are set to 0.
        """
        arr = pg.ROI.getArrayRegion(self, arr, img)
        if arr is None or arr.shape[0] == 0 or arr.shape[1] == 0:
            return None
        w = arr.shape[0]
        h = arr.shape[1]
        ## generate an ellipsoidal mask
        mask = np.fromfunction(
            lambda x, y: (((x + 0.5) / (w / 2.) - 1) ** 2 + ((y + 0.5) / (h / 2.) - 1) ** 2) ** 0.5 < 1, (w, h))

        return arr * mask

    def shape(self):
        self.path = QtGui.QPainterPath()
        self.path.addEllipse(self.boundingRect())
        return self.path
