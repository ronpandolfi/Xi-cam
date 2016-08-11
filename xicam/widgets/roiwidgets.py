import numpy as np
import pyqtgraph as pg
from PySide import QtCore


class ROImageOverlay(pg.ROI):
    sigTranslated = QtCore.Signal(int, int)

    def __init__(self, data, bg_imageItem, pos, constrainX=False, constrainY=True, translateSnap=True, **kwargs):
        size = bg_imageItem.image.shape
        super(ROImageOverlay, self).__init__(pos, translateSnap=translateSnap, size=size, pen=pg.mkPen(None), **kwargs)

        self.data = data
        self.bg_imgeItem = bg_imageItem
        self._y_constrained = constrainY
        self._x_constrained = constrainX
        self._image_overlap = np.empty(size, dtype='float32')
        self._mask = np.zeros(size, dtype=bool)
        self.currentImage = None
        self.currentIndex = None
        self.flipped = False
        self.setCurrentImage(-1)
        self.flipCurrentImage()
        self.imageItem = pg.ImageItem(self.currentImage)
        self.imageItem.setParentItem(self)
        self.updateImage()

    def setCurrentImage(self, idx):
        self.currentImage = np.array(self.data[idx]).astype('float32')
        self.currentIndex = idx
        if self.flipped:
            self.flipCurrentImage(toggle=False)

    def constrainX(self, val):
        self._x_constrained = val

    def constrainY(self, val):
        self._y_constrained = val

    def flipCurrentImage(self, toggle=True):
        self.currentImage = np.flipud(self.currentImage)
        if toggle:
            self.flipped = not self.flipped

    @property
    def image_overlap(self):
        self._image_overlap.fill(0)
        x, y = self.pos()

        if x == 0:
            x_slc, bg_x_slc = None, None
        elif x < 0:
            x_slc, bg_x_slc = slice(-x, None), slice(None, x)
        elif x > 0:
            x_slc, bg_x_slc = slice(None, -x), slice(x, None)

        if y == 0:
            y_slc, bg_y_slc = None, None
        elif y < 0:
            y_slc, bg_y_slc = slice(-y, None), slice(None, y)
        elif y > 0:
            y_slc, bg_y_slc = slice(None, -y), slice(y, None)

        slc, bg_slc = (x_slc, y_slc), (bg_x_slc, bg_y_slc)

        self._image_overlap[slc] = self.bg_imgeItem.image[bg_slc]

        return self._image_overlap

    def updateImage(self, autolevels=False):
        self.imageItem.setImage(self.currentImage - self.image_overlap, autoLevels=autolevels)

    def translate(self, *args, **kwargs):
        super(ROImageOverlay, self).translate(*args, **kwargs)
        self.updateImage()
        self.sigTranslated.emit(*self.pos())

    def resetImage(self):
        self.setCurrentImage(self.currentIndex)
        self.updateImage()

    def mouseDragEvent(self, ev):
        """
        Overload ROI.mouseDragEvent to set all vertical offsets to zero and constrain dragging to horizontal axis
        """
        if ev.isStart():
            if ev.button() == QtCore.Qt.LeftButton:
                self.setSelected(True)
                if self.translatable:
                    self.isMoving = True
                    self.preMoveState = self.getState()
                    self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                    self.sigRegionChangeStarted.emit(self)
                    ev.accept()
                else:
                    ev.ignore()

        elif ev.isFinish():
            if self.translatable:
                if self.isMoving:
                    self.stateChangeFinished()
                self.isMoving = False
            return

        if self.translatable and self.isMoving and ev.buttons() == QtCore.Qt.LeftButton:
            snap = True if (ev.modifiers() & QtCore.Qt.ControlModifier) else None
            newPos = self.mapToParent(ev.pos()) + self.cursorOffset
            if self._y_constrained:
                newPos.y = self.pos().y
            if self._x_constrained:
                newPos.x = self.pos().x
            self.translate(newPos - self.pos(), snap=snap, finish=False)

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Right:
            if not self._x_constrained:
                self.translate(pg.Point((1, 0)))
        elif ev.key() == QtCore.Qt.Key_Left:
            if not self._x_constrained:
                self.translate(pg.Point((-1, 0)))
        elif ev.key() == QtCore.Qt.Key_Up:
            if not self._y_constrained:
                self.translate(pg.Point((0, 1)))
        elif ev.key() == QtCore.Qt.Key_Down:
            if not self._y_constrained:
                self.translate(pg.Point((0, -1)))
        ev.accept()