import pyqtgraph as pg
from PySide import QtGui, QtCore
import numpy as np


class QCircRectF(QtCore.QRectF):
    def __init__(self, center=(0, 0), radius=1, rect=None):
        self._scale = 1.
        if rect is not None:
            self.center = rect.center()
            super(QCircRectF, self).__init__(rect)
        else:
            self.center = QtCore.QPointF(*center)

            left = self.center.x() - radius
            top = self.center.y() - radius
            bottom = self.center.y() + radius
            right = self.center.x() + radius

            super(QCircRectF, self).__init__(QtCore.QPointF(left, top), QtCore.QPointF(right, bottom))

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        self.radius *= value
        self.setLeft(self.center.x() - self._radius)
        self.setTop(self.center.y() - self._radius)
        self.setBottom(self.center.y() + self._radius)
        self.setRight(self.center.x() + self._radius)

    @property
    def radius(self):
        return (self.right() - self.left()) * .5

    @radius.setter
    def radius(self, radius):

        self.setLeft(self.center.x() - radius)
        self.setTop(self.center.y() - radius)
        self.setBottom(self.center.y() + radius)
        self.setRight(self.center.x() + radius)



class QRectF(QtCore.QRectF):
    def scale(self, ratio):
        coords = [coord * ratio for coord in self.getCoords()]

        self.setCoords(*coords)


class LinearRegionItem(pg.LinearRegionItem):
    sigRemoveRequested = QtCore.Signal(object)

    def __init__(self, *args, **kwargs):
        super(LinearRegionItem, self).__init__(*args, **kwargs)
        self.menu = None
        self.isdeleting = False

    def getArrayRegion(self,data,item): #rotate once?
        cut=np.zeros_like(data)
        if self.orientation is pg.LinearRegionItem.Horizontal:
            regionbounds = self.getRegion()

            cut[:, regionbounds[0]:regionbounds[1]] = 1
        elif self.orientation is pg.LinearRegionItem.Vertical:
            regionbounds = self.getRegion()

            cut[regionbounds[0]:regionbounds[1], :] = 1


        return (cut * data).T

    def hoverEvent(self, ev):
        hover = False
        if not ev.isExit():
            if ev.acceptDrags(QtCore.Qt.LeftButton):
                hover = True

            for btn in [QtCore.Qt.LeftButton, QtCore.Qt.RightButton, QtCore.Qt.MidButton]:
                if int(self.acceptedMouseButtons() & btn) > 0 and ev.acceptClicks(btn):
                    hover = True
            ev.acceptClicks(QtCore.Qt.RightButton)

        if hover:
            self.setMouseHover(True)

            ev.acceptClicks(
                QtCore.Qt.LeftButton)  ## If the ROI is hilighted, we should accept all clicks to avoid confusion.
            ev.acceptClicks(QtCore.Qt.RightButton)
            ev.acceptClicks(QtCore.Qt.MidButton)
        else:
            self.setMouseHover(False)

    def raiseContextMenu(self, ev):
        menu = self.getMenu()
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))


    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            self.raiseContextMenu(ev)
            ev.accept()
        elif int(ev.button() & self.acceptedMouseButtons()) > 0:
            ev.accept()
            self.sigClicked.emit(self, ev)
        else:
            ev.ignore()

    def getMenu(self):
        if self.menu is None:
            self.menu = QtGui.QMenu()
            self.menu.setTitle("ROI")
            remAct = QtGui.QAction("Remove ROI", self.menu)
            remAct.triggered.connect(self.removeClicked)
            self.menu.addAction(remAct)
            self.menu.remAct = remAct
        return self.menu

    def removeClicked(self):
        self.isdeleting = True
        QtCore.QTimer.singleShot(0, lambda: self.sigRemoveRequested.emit(self))
        # self.deleteLater()
        ## Send remove event only after we have exited the menu event handler
        #QtCore.QTimer.singleShot(0, lambda: self.sigRemoveRequested.emit(self))




class LineROI(pg.LineROI):
    def __init__(self, *args, **kwargs):
        super(LineROI, self).__init__(*args, **kwargs)
        #self.sigRemoveRequested.connect(self.deleteLater)
        self.isdeleting = False
        self.sigRemoveRequested.connect(self.delete)

    def getArrayRegion(self, data, img, axes=(0, 1), returnMappedCoords=False, **kwds):
        from skimage.draw import polygon

        shape, v, origin = self.getAffineSliceParams(data, img, axes)

        br = np.add(origin, np.multiply(v[0], shape[0]))
        tl = np.add(origin, np.multiply(v[1], shape[1]))
        bl = origin
        tr = np.sum([origin, np.multiply(v[1], shape[1]), np.multiply(v[0], shape[0])], axis=0)
        vecs = np.vstack([bl, br, tr, tl]).T
        rr, cc = polygon(vecs[0], vecs[1])
        mask = np.zeros_like(data)
        rrcc = [[r, c] for r, c in zip(rr, cc) if r < mask.shape[0] and c < mask.shape[1]]
        rr, cc = zip(*rrcc)
        mask[rr, cc] = 1

        return (data * mask).T.copy()

    def delete(self):
        self.isdeleting = True

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

    #sigRemoveRequested = QtCore.Signal()

    def __init__(self, center, radius, **args):
        # QtGui.QGraphicsRectItem.__init__(self, 0, 0, size[0], size[1])
        r = QCircRectF(center, radius)
        super(ArcROI, self).__init__(r.center, r.size(), removable=True, **args)
        #self.addRotateHandle([1.0, 0.5], [0.5, 0.5])
        #self.addScaleHandle([0.5*2.**-0.5 + 0.5, 0.5*2.**-0.5 + 0.5], [0.5, 0.5])

        self.innerhandle = self.addFreeHandle([0., .25])
        self.outerhandle = self.addFreeHandle([0., .5])
        self.lefthandle = self.addFreeHandle([.433, .25])
        self.righthandle = self.addFreeHandle([-.433, .25])

        self.startangle = 30
        self.arclength = 120

        self.aspectLocked = True
        self.translatable = False
        self.translateSnap = False
        self.removable = True

        self.cacheinner = self.innerhandle.pos()
        self.cacheouter = self.outerhandle.pos()
        self.startradius = radius
        self.startcenter = center

        self.isdeleting = False
        self.sigRemoveRequested.connect(self.delete)


    def getRadius(self):
        radius = pg.Point(self.outerhandle.pos()).length()
        # r2 = radius[1]
        #r3 = r2[0]
        return radius

    def getInnerRadius(self):
        radius = pg.Point(self.innerhandle.pos()).length()
        # r2 = radius[1]
        #r3 = r2[0]
        return radius / self.getRadius() * .5

    def boundingRect(self):
        r = self.getRadius()
        return QtCore.QRectF(-r * 1., -r * 1., 2.0 * r, 2.0 * r)

    def getCenter(self):
        r = self.boundingRect()
        r = QCircRectF(rect=r)
        return r.center
    

    def paint(self, p, opt, widget):
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(0, 255, 255))

        p.setPen(pen)

        # if self.cacheouter!=self.outerhandle.pos():
        #    self.innerhandle.setPos(self.cacheinner)


        r = self.boundingRect()
        #p.drawRect(r)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        p.scale(r.width(), r.height())  ## workaround for GL bug

        #r = QRectF(r.x() / r.width(), r.y() / r.height(), 1., 1.)


        # p.drawEllipse(r)
        r = QCircRectF(radius=0.5)
        #r.radius=self.getRadius()


        self.arclength = np.degrees(np.arctan2(self.lefthandle.pos().x(), self.lefthandle.pos().y()) -
                                    np.arctan2(self.righthandle.pos().x(), self.righthandle.pos().y()))
        self.startangle = np.degrees(np.arctan2(self.lefthandle.pos().y(), self.lefthandle.pos().x()))
        # print 'start: ',startangle
        #print 'length: ',arclength


        p.drawArc(r, -(self.startangle) * 16, -(self.arclength) * 16)


        # pos = self.mapFromView(self.innerhandle.pos())
        radius = self.getInnerRadius()

        r = QCircRectF()

        r.radius = radius
        #p.drawRect(r)
        #print r.radius
        p.drawArc(r, -(self.startangle) * 16, -self.arclength * 16)

        #p.drawLine(self.lefthandle.pos().norm()*self.innerhandle.pos().length()/2,self.lefthandle.pos().norm()/2)
        #p.drawLine(self.righthandle.pos().norm()*self.innerhandle.pos().length()/2.,self.righthandle.pos().norm()/2)



        pen.setStyle(QtCore.Qt.DashLine)
        p.setPen(pen)

        p.drawLine(QtCore.QPointF(0., 0.), self.lefthandle.pos() * 100)
        p.drawLine(QtCore.QPointF(0., 0.), self.righthandle.pos() *100)




    def getArrayRegion(self, arr, img=None):
        """
        Return the result of ROI.getArrayRegion() masked by the elliptical shape
        of the ROI. Regions outside the ellipse are set to 0.
        """
        # arr = pg.ROI.getArrayRegion(self, arr, img)
        #if arr is None or arr.shape[0] == 0 or arr.shape[1] == 0:
        #    return None
        w = arr.shape[0]
        h = arr.shape[1]
        ## generate an ellipsoidal mask
        mask = np.fromfunction(
            lambda x, y: (self.innerhandle.pos().length() < (
            (x - self.startcenter[0]) ** 2. + (y - self.startcenter[1]) ** 2.) ** .5) &
                         (((x - self.startcenter[0]) ** 2. + (
                         y - self.startcenter[1]) ** 2.) ** .5 < self.outerhandle.pos().length()) &
                         (np.degrees(np.arctan2(y - self.startcenter[1], x - self.startcenter[0])) > self.startangle) &
                         (np.degrees(np.arctan2(y - self.startcenter[1],
                                                x - self.startcenter[0])) < self.startangle + self.arclength)
            , (w, h))

        return (arr * mask).T

    def shape(self):
        self.path = QtGui.QPainterPath()
        self.path.addEllipse(self.boundingRect())
        return self.path

    def delete(self):
        self.isdeleting = True
        #self.deleteLater()