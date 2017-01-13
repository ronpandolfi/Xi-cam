

__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


import numpy as np
import scipy as sp
import pyqtgraph as pg
from PySide import QtCore


class ROImageOverlay(pg.ROI):
    """
    Class for ROI that can be added to an ImageView to overlay an image with the current image in the ImageView
    Currently the difference between the overlap is shown. To change the operations simply override the updateImage
    method


    Attributes
    ----------
    data
        The data being shown in the ImageView where the ROI is to be added
    currentImage : ndarray
        Copy of the current image in the ImageView
    currentIndex : int
        Index of the current image in the ImageView
    flipped : bool
        Specifies if the image overlay is to be flipped


    Parameters
    ----------
    data
        The data being shown in the ImageView where the ROI is to be added
    bg_imageItem
        Image item from image shown in ImageView
    pos : tuple
        Initial position where ROIImageOverlay should be displayed
    constrainX : bool, optional
        Constrains movement of ROI in x direction
    constrainY : bool, optional
        Constrains movement of ROI in y direction
    translateSnap : bool option
        Snap the ROI to image pixels
    kwargs
        Additional keyword arguments accepted by pg.ROI
    """
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
        """
        Set the currentImage attribute to the image in ImageView given by idx

        Parameters
        ----------
        idx : int
            Index of image displayed in ImageView
        """
        self.currentImage = np.array(self.data[idx]).astype('float32')
        self.currentIndex = idx
        if self.flipped:
            self.flipCurrentImage(toggle=False)

    def constrainX(self, val):
        """
        Sets the constraint of ROI movement in x direction
        """
        self._x_constrained = val

    def constrainY(self, val):
        """
        Sets the constraint of ROI movement in y direction
        """
        self._y_constrained = val

    def flipCurrentImage(self, toggle=True):
        """
        Flips the currentImage
        """
        self.currentImage = np.flipud(self.currentImage)
        if toggle:
            self.flipped = not self.flipped

    @property
    def image_overlap(self):
        """
        Returns the overlap array of the ROIImage and the background image. The returned array has the same dimensions
        As bot the ROI image and the background image with values outside the overlap set to zero.
        """
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

    def remove_outlier(self, array1, array2, total, thresh = 0.05):
        val = sp.integrate.trapz(array1, array2)
        print 1- (float(val) / total)
        if 1 - (float(val)/total) < thresh:
            return self.remove_outlier(array1[1:-1],array2[1:-1], total, thresh=thresh)
        else:
            return array1, array2

    def updateImage(self, autolevels=False, levels=None):
        """
        Updates the image shown in the ROI to the difference of the current image and the image_overlap
        """
        if levels:
            self.imageItem.setImage(self.currentImage - self.image_overlap, autoLevels=autolevels, levels=levels)
        else:
            self.imageItem.setImage(self.currentImage - self.image_overlap, autoLevels=autolevels)


    def translate(self, *args, **kwargs):
        """
        Override translate method to update the ROI image and emit the current position of the ROI image
        """
        super(ROImageOverlay, self).translate(*args, **kwargs)
        self.updateImage()
        self.sigTranslated.emit(*self.pos())

    def resetImage(self):
        """
        Resets the current image to the current index
        """
        self.setCurrentImage(self.currentIndex)
        self.updateImage()

    def mouseDragEvent(self, ev):
        """
        Override ROI.mouseDragEvent to set all vertical offsets to zero and constrain dragging to horizontal axis
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
        """
        Override the keyPressEvent to have arrow keys move the ROIImageOverlay
        """

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