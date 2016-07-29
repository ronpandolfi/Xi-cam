# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# vispy: gallery 2
#
# Adapted for use as a widget by Ron Pandolfi
# volumeViewer.getHistogram method borrowed from PyQtGraph

from collections import deque
import numpy as np
from functools import partial
from PySide import QtGui, QtCore
from vispy import scene
from vispy.color import Colormap
from pipeline import loader
import pyqtgraph as pg
import imageio
import os
import fmanager
from pipeline import msg

__author__ = "Ronald J Pandolfi"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


class TomoViewer(QtGui.QWidget):
    """
    Class that holds projection, sinogram, recon preview, and process-settings viewers for a tomography dataset.
    """

    sigReconFinished = QtCore.Signal()

    def __init__(self, paths=None, data=None, *args, **kwargs):

        if paths is None and data is None:
            raise ValueError('Either data or path to file must be provided')

        super(TomoViewer, self).__init__(*args, **kwargs)

        self._recon_path = None
        self.viewstack = QtGui.QStackedWidget(self)

        self.viewmode = QtGui.QTabBar(self)
        self.viewmode.addTab('Projection View')  # TODO: Add icons!
        self.viewmode.addTab('Sinogram View')
        self.viewmode.addTab('Slice Preview')
        self.viewmode.addTab('3D Preview')
        self.viewmode.addTab('Reconstruction View')
        self.viewmode.setShape(QtGui.QTabBar.TriangularSouth)

        if data is not None:
            self.data = data
        elif paths is not None and len(paths):
            self.data = self.loaddata(paths)

        self.cor = float(self.data.shape[1])/2.0

        self.projectionViewer = ProjectionViewer(self.data, center=self.cor, parent=self)
        self.projectionViewer.centerBox.setRange(0, self.data.shape[1])
        self.viewstack.addWidget(self.projectionViewer)

        self.sinogramViewer = StackViewer(loader.SinogramStack.cast(self.data), parent=self)
        self.sinogramViewer.setIndex(self.sinogramViewer.data.shape[0] // 2)
        self.viewstack.addWidget(self.sinogramViewer)

        self.previewViewer = PreviewViewer(self.data.shape[1], parent=self)
        self.viewstack.addWidget(self.previewViewer)

        self.preview3DViewer = Preview3DViewer(paths=paths, data=data)
        self.preview3DViewer.volumeviewer.moveGradientTick(1, 0.3)
        self.viewstack.addWidget(self.preview3DViewer)

        self.reconstructionViewer = ReconstructionViewer(parent=self)
        self.viewstack.addWidget(self.reconstructionViewer)

        v = QtGui.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.viewstack)
        v.addWidget(self.viewmode)
        self.setLayout(v)

        self.viewmode.currentChanged.connect(self.currentChanged)
        self.viewstack.currentChanged.connect(self.viewmode.setCurrentIndex)

    def wireupCenterSelection(self, recon_function):
        if recon_function is not None:
            center_param = recon_function.params.child('center')
            # Uncomment this if you want convenience of having the center parameter in pipeline connected to the
            # manual center widget, but this limits the center options to a resolution of 0.5
            # self.projectionViewer.sigCenterChanged.connect(
            #     lambda x: center_param.setValue(x)) #, blockSignal=center_param.sigValueChanged))
            self.projectionViewer.setCenterButton.clicked.connect(
                lambda: center_param.setValue(self.projectionViewer.centerBox.value()))
            center_param.sigValueChanged.connect(lambda p,v: self.projectionViewer.centerBox.setValue(v))
            center_param.sigValueChanged.connect(lambda p,v: self.projectionViewer.updateROIFromCenter(v))

    @staticmethod
    def loaddata(paths, raw=True):
        if raw:
            return loader.ProjectionStack(paths)
        else:
            return loader.StackImage(paths)

    def getsino(self, slc=None): #might need to redo the flipping and turning to get this in the right orientation
        if slc is None:
            return np.ascontiguousarray(self.sinogramViewer.currentdata[:,np.newaxis,:])
        else:
            return np.ascontiguousarray(self.data.fabimage[slc])

    def getproj(self, slc=None):
        if slc is None:
            return np.ascontiguousarray(self.projectionViewer.currentdata[np.newaxis, :, :])
        else:
            return np.ascontiguousarray(self.data.fabimage[slc])

    def getflats(self, slc=None):
        if slc is None:
            return np.ascontiguousarray(self.data.flats[:, self.sinogramViewer.currentIndex, :])
        else:
            return np.ascontiguousarray(self.data.flats[slc])

    def getdarks(self, slc=None):
        if slc is None:
            return np.ascontiguousarray(self.data.darks[: ,self.sinogramViewer.currentIndex, :])
        else:
            return np.ascontiguousarray(self.data.darks[slc])

    def getheader(self):
        return self.data.header

    def currentChanged(self, index):
        self.viewstack.setCurrentIndex(index)

    def setCorValue(self, value):
        self.cor = value

    def runSlicePreview(self):
        slice_no = self.sinogramViewer.view_spinBox.value()
        fmanager.pipeline_preview_action(self, partial(self.addSlicePreview, slice_no=slice_no))

    def run3DPreview(self):
        slc = (slice(None), slice(None, None, 8), slice(None, None, 8))
        fmanager.cor_scale = lambda x: x//8
        fmanager.pipeline_preview_action(self, self.add3DPreview, slc=slc, finish_call=msg.clearMessage)

    def runFullRecon(self, proj, sino, sino_p_chunk, ncore, update_call, interrupt_signal=None):
        fmanager.run_full_recon(self, proj, sino, sino_p_chunk, ncore, update_call, self.fullReconFinished,
                                interrupt_signal=interrupt_signal)

    def addSlicePreview(self, params, recon, slice_no=None):
        if slice_no is None:
            slice_no = self.sinogramViewer.view_spinBox.value()
        self.previewViewer.addPreview(np.rot90(recon[0],1), params, slice_no)
        self.viewstack.setCurrentWidget(self.previewViewer)
        msg.clearMessage()

    def add3DPreview(self, params, recon):
        recon = np.flipud(recon)
        self.viewstack.setCurrentWidget(self.preview3DViewer)
        self.preview3DViewer.setPreview(recon, params)
        hist = self.preview3DViewer.volumeviewer.getHistogram()
        max = hist[0][np.argmax(hist[1])]
        self.preview3DViewer.volumeviewer.setLevels([max, hist[0][-1]])

    def fullReconFinished(self):
        self.sigReconFinished.emit()
        path = fmanager.get_output_path()
        # if not extension was given assume it is a tiff directory.
        if '.' not in path:
            path = os.path.split(path)[0]
        self.reconstructionViewer.openDataset(path=path)
        self.viewstack.setCurrentWidget(self.reconstructionViewer)
        msg.clearMessage()

    def onManualCenter(self, active):
        if active:
            self.projectionViewer.showCenterDetection()
            self.viewstack.setCurrentWidget(self.projectionViewer)
        else:
            self.projectionViewer.hideCenterDetection()


class ImageView(pg.ImageView):
    """
    Subclass of PG ImageView to correct z-slider signal behavior, and add coordinate label.
    """
    sigDeletePressed = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super(ImageView, self).__init__(*args, **kwargs)
        self.scene.sigMouseMoved.connect(self.mouseMoved)

        self.coordsLabel = QtGui.QLabel(' ', parent=self)
        self.coordsLabel.setMinimumHeight(16)
        self.layout().addWidget(self.coordsLabel)
        self.coordsLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
        self.setStyleSheet("background-color: rgba(0,0,0,0%)")


    def buildMenu(self):
        super(ImageView, self).buildMenu()
        self.menu.removeAction(self.normAction)

    def keyPressEvent(self, ev):
        super(ImageView, self).keyPressEvent(ev)
        if ev.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
            self.timeLineChanged()
        elif ev.key() == QtCore.Qt.Key_Delete or ev.key() == QtCore.Qt.Key_Backspace:
            self.sigDeletePressed.emit()

    def timeIndex(self, slider):
        ## Return the time and frame index indicated by a slider
        if self.image is None:
            return (0,0)

        t = slider.value()

        xv = self.tVals
        if xv is None:
            ind = int(t)
        else:
            if len(xv) < 2:
                return (0,0)
            totTime = xv[-1] + (xv[-1]-xv[-2])
            inds = np.argwhere(xv <= t)
            if len(inds) < 1:
                return (0,t)
            ind = inds[-1,0]
        return ind, t

    def mouseMoved(self, ev):
        pos = ev
        viewBox = self.imageItem.getViewBox()
        try:
            if viewBox.sceneBoundingRect().contains(pos):
                mousePoint = viewBox.mapSceneToView(pos)
                x, y = map(int, (mousePoint.x(), mousePoint.y()))
                if (0 <= x < self.imageItem.image.shape[0]) & (0 <= y < self.imageItem.image.shape[1]):  # within bounds
                    self.coordsLabel.setText(u"<div style='font-size: 12pt;background-color:#111111;'>x={0},"
                                             u"   <span style=''>y={1}</span>,   <span style=''>I={2}</span>"\
                                             .format(x, y, self.imageItem.image[x, y]))
                else:
                    self.coordsLabel.setText(u"<div style='font-size: 12pt;background-color:#111111;'>x= ,"
                                             u"   <span style=''>y= </span>,   <span style=''>I= </span>")
        except AttributeError:
            pass


class StackViewer(ImageView):
    """
    PG ImageView subclass to view projections or sinograms of a tomography dataset
    """
    def __init__(self, data=None, view_label=None, *args, **kwargs):
        super(StackViewer, self).__init__(*args, **kwargs)

        # self.getImageItem().setAutoDownsample(True)

        self.view_label = QtGui.QLabel(self)
        self.view_label.setText('No: ')
        self.view_spinBox = QtGui.QSpinBox(self)
        self.view_spinBox.setKeyboardTracking(False)

        if data is not None:
            self.setData(data)

        l = QtGui.QHBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(self.view_label)
        l.addWidget(self.view_spinBox)
        l.addStretch(1)
        w = QtGui.QWidget()
        w.setLayout(l)
        self.ui.gridLayout.addWidget(self.view_label, 1, 1, 1, 1)
        self.ui.gridLayout.addWidget(self.view_spinBox, 1, 2, 1, 1)
        self.ui.menuBtn.setParent(None)
        self.ui.roiBtn.setParent(None)

        self.sigTimeChanged.connect(self.indexChanged)
        self.view_spinBox.valueChanged.connect(self.setCurrentIndex)

    def setData(self, data):
        self.data = data
        self.setImage(self.data)
        self.autoLevels()
        self.view_spinBox.setRange(0, self.data.shape[0] - 1)
        self.getImageItem().setRect(QtCore.QRect(0, 0, self.data.rawdata.shape[0], self.data.rawdata.shape[1]))

    def indexChanged(self, ind, time):
        self.view_spinBox.setValue(ind)

    def setIndex(self, ind):
        self.setCurrentIndex(ind)
        self.view_spinBox.setValue(ind)

    @property
    def currentdata(self):
        return self.data[self.data.currentframe].transpose()  # Maybe we need to transpose this

    def resetImage(self):
        self.setImage(self.data, autoRange=False)
        self.setIndex(self.currentIndex)


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
            self.translate(pg.Point((1, 0)))
        elif ev.key() == QtCore.Qt.Key_Left:
            self.translate(pg.Point((-1, 0)))
        elif ev.key() == QtCore.Qt.Key_Up:
            self.translate(pg.Point((0, 1)))
        elif ev.key() == QtCore.Qt.Key_Down:
            self.translate(pg.Point((0, -1)))
        ev.accept()



class ProjectionViewer(QtGui.QWidget):
    """
    Class that holds a stack viewer, an ROImageOverlay and a few widgets to allow manual center detection
    """
    sigCenterChanged = QtCore.Signal(float)

    def __init__(self, data, view_label=None, center=None, *args, **kwargs):
        super(ProjectionViewer, self).__init__(*args, **kwargs)
        self.stackViewer = StackViewer(data, view_label=view_label)
        self.imageItem = self.stackViewer.imageItem
        self.data = self.stackViewer.data
        self.normalized = False
        self.flat = np.median(self.data.flats, axis=0).transpose()
        self.dark = np.median(self.data.darks, axis=0).transpose()

        self.roi = ROImageOverlay(self.data, self.imageItem, [0, 0])
        # self.stackViewer.getHistogramWidget().setImageItem(self.roi.imageItem)
        self.imageItem.sigImageChanged.connect(self.roi.updateImage)
        self.stackViewer.view.addItem(self.roi)
        self.roi_histogram = pg.HistogramLUTWidget(image=self.roi.imageItem, parent=self)

        self.stackViewer.ui.gridLayout.addWidget(self.roi_histogram, 0, 3, 1, 2)
        self.stackViewer.keyPressEvent = self.keyPressEvent

        self.cor_widget = QtGui.QWidget(self)
        clabel = QtGui.QLabel('Rotation Center:')
        olabel = QtGui.QLabel('Offset:')
        self.centerBox = QtGui.QDoubleSpinBox(parent=self.cor_widget) #QtGui.QLabel(parent=self.cor_widget)
        self.centerBox.setDecimals(1)
        self.setCenterButton = QtGui.QToolButton()
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/check_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setCenterButton.setIcon(icon)
        self.setCenterButton.setToolTip('Set center in pipeline')
        originBox = QtGui.QLabel(parent=self.cor_widget)
        originBox.setText('x={}   y={}'.format(0, 0))
        center = center if center is not None else data.shape[1]/2.0
        self.centerBox.setValue(center) #setText(str(center))
        h1 = QtGui.QHBoxLayout()
        h1.setAlignment(QtCore.Qt.AlignLeft)
        h1.setContentsMargins(0, 0, 0, 0)
        h1.addWidget(clabel)
        h1.addWidget(self.centerBox)
        h1.addWidget(self.setCenterButton)
        h1.addWidget(olabel)
        h1.addWidget(originBox)

        plabel = QtGui.QLabel('Overlay Projection No:')
        plabel.setAlignment(QtCore.Qt.AlignRight)
        spinBox = QtGui.QSpinBox(parent=self.cor_widget)
        #TODO data shape seems to be on larger than the return from slicing it with [:-1]
        spinBox.setRange(0, data.shape[0])
        slider = QtGui.QSlider(orientation=QtCore.Qt.Horizontal, parent=self.cor_widget)
        slider.setRange(0, data.shape[0])
        spinBox.setValue(data.shape[0])
        slider.setValue(data.shape[0])
        flipCheckBox = QtGui.QCheckBox('Flip Overlay', parent=self.cor_widget)
        flipCheckBox.setChecked(True)
        constrainYCheckBox = QtGui.QCheckBox('Constrain Y', parent=self.cor_widget)
        constrainYCheckBox.setChecked(True)
        constrainXCheckBox = QtGui.QCheckBox('Constrain X', parent=self.cor_widget)
        constrainXCheckBox.setChecked(False)
        # rotateCheckBox = QtGui.QCheckBox('Enable Rotation', parent=self.cor_widget)
        # rotateCheckBox.setChecked(False)
        self.normCheckBox = QtGui.QCheckBox('Normalize', parent=self.cor_widget)
        h2 = QtGui.QHBoxLayout()
        h2.setAlignment(QtCore.Qt.AlignLeft)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(plabel)
        h2.addWidget(spinBox)
        h2.addWidget(flipCheckBox)
        h2.addWidget(constrainXCheckBox)
        h2.addWidget(constrainYCheckBox)
        # h2.addWidget(rotateCheckBox) # This needs to be implemented correctly
        h2.addWidget(self.normCheckBox)
        h2.addStretch(1)

        spinBox.setFixedWidth(spinBox.width())
        v = QtGui.QVBoxLayout(self.cor_widget)
        v.addLayout(h1)
        v.addLayout(h2)
        v.addWidget(slider)

        l = QtGui.QGridLayout(self)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(self.cor_widget)
        l.addWidget(self.stackViewer)

        slider.valueChanged.connect(spinBox.setValue)
        slider.valueChanged.connect(self.stackViewer.resetImage)
        spinBox.valueChanged.connect(self.changeOverlayProj)
        flipCheckBox.stateChanged.connect(self.flipOverlayProj)
        constrainYCheckBox.stateChanged.connect(lambda v: self.roi.constrainY(v))
        constrainXCheckBox.stateChanged.connect(lambda v: self.roi.constrainX(v))
        # rotateCheckBox.stateChanged.connect(self.addRotateHandle)
        self.normCheckBox.stateChanged.connect(self.normalize)
        self.stackViewer.sigTimeChanged.connect(lambda: self.normalize(False))
        self.roi.sigTranslated.connect(self.setCenter)
        self.roi.sigTranslated.connect(lambda x, y: originBox.setText('x={}   y={}'.format(x, y)))

        self.hideCenterDetection()

    def changeOverlayProj(self, idx):
        self.normCheckBox.setChecked(False)
        self.roi.setCurrentImage(idx)
        self.roi.updateImage()

    def setCenter(self, x, y):
        center = (self.data.shape[1] + x - 1)/2.0# subtract half a pixel out of 'some' convention?
        self.centerBox.setValue(center) # setText(str(center))
        self.sigCenterChanged.emit(center)

    def hideCenterDetection(self):
        self.normalize(False)
        self.cor_widget.hide()
        self.roi_histogram.hide()
        self.roi.setVisible(False)

    def showCenterDetection(self):
        self.cor_widget.show()
        self.roi_histogram.show()
        self.roi.setVisible(True)

    def updateROIFromCenter(self, center):
        s = self.roi.pos()[0]
        self.roi.translate(pg.Point((2 * center + 1 - self.data.shape[1] - s, 0))) # 1 again due to the so-called COR
                                                                                   # conventions...
    def flipOverlayProj(self, val):
        self.roi.flipCurrentImage()
        self.roi.updateImage()

    def addRotateHandle(self, val):
        if val:
            self.addRotateHandle.handle = self.roi.addRotateHandle([0,1], [0.2, 0.2])
        else:
            self.roi.removeHandle(self.addRotateHandle.handle)

    def normalize(self, val):
        if val and not self.normalized:
            proj = (self.imageItem.image - self.dark)/(self.flat - self.dark)
            overlay = self.roi.currentImage
            if self.roi.flipped:
                overlay = np.flipud(overlay)
            overlay = (overlay - self.dark)/(self.flat - self.dark)
            if self.roi.flipped:
                overlay = np.flipud(overlay)
            self.roi.currentImage = overlay
            self.roi.updateImage(autolevels=True)
            self.stackViewer.setImage(proj, autoRange=False, autoLevels=True)
            self.stackViewer.updateImage()
            self.normalized = True
        elif not val and self.normalized:
            self.stackViewer.resetImage()
            self.roi.resetImage()
            self.normalized = False
            self.normCheckBox.setChecked(False)

    def keyPressEvent(self, ev):
        super(ProjectionViewer, self).keyPressEvent(ev)
        if self.roi.isVisible():
            self.roi.keyPressEvent(ev)
        else:
            super(StackViewer, self.stackViewer).keyPressEvent(ev)
        ev.accept()


class PreviewViewer(QtGui.QSplitter):
    """
    Viewer class to show reconstruction previews in a PG ImageView, along with the function pipeline settings for the
    corresponding preview
    """

    def __init__(self, dim, maxpreviews=None, *args, **kwargs):
        super(PreviewViewer, self).__init__(*args, **kwargs)
        self.maxpreviews = maxpreviews if maxpreviews is not None else 10

        self.dim = dim

        self.previews = ArrayDeque(arrayshape=(dim, dim), maxlen=self.maxpreviews)
        self.datatrees = deque(maxlen=self.maxpreviews)
        self.data = deque(maxlen=self.maxpreviews)
        self.slice_numbers = deque(maxlen=self.maxpreviews)

        self.setOrientation(QtCore.Qt.Horizontal)
        self.functionform = QtGui.QStackedWidget()

        self.deleteButton = QtGui.QToolButton(self)
        self.deleteButton.setToolTip('Delete this preview')
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_40.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.deleteButton.setIcon(icon)

        self.setPipelineButton = QtGui.QToolButton(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/check_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setPipelineButton.setIcon(icon)
        self.setPipelineButton.setToolTip('Set as pipeline')

        ly = QtGui.QVBoxLayout()
        ly.setContentsMargins(0, 0, 0, 0)
        ly.setSpacing(0)
        ly.addWidget(self.functionform)
        h = QtGui.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.setPipelineButton)
        h.addWidget(self.deleteButton)
        ly.addLayout(h)
        panel = QtGui.QWidget(self)
        panel.setLayout(ly)
        self.setPipelineButton.hide()
        self.deleteButton.hide()

        self.imageview = ImageView(self)
        self.imageview.ui.roiBtn.setParent(None)
        self.imageview.ui.roiBtn.setParent(None)
        self.imageview.ui.menuBtn.setParent(None)

        self.view_label = QtGui.QLabel(self)
        self.view_label.setText('No: ')
        self.view_number = QtGui.QSpinBox(self)
        self.view_number.setReadOnly(True)
        self.view_number.setMaximum(5000) # Large enough number
        self.imageview.ui.gridLayout.addWidget(self.view_label, 1, 1, 1, 1)
        self.imageview.ui.gridLayout.addWidget(self.view_number, 1, 2, 1, 1)

        self.setCurrentIndex = self.imageview.setCurrentIndex
        self.addWidget(panel)
        self.addWidget(self.imageview)

        self.imageview.sigDeletePressed.connect(self.removePreview)
        self.setPipelineButton.clicked.connect(self.defaultsButtonClicked)
        self.deleteButton.clicked.connect(self.removePreview)
        self.imageview.sigTimeChanged.connect(self.indexChanged)

    @ QtCore.Slot(object, object)
    def indexChanged(self, index, time):
        try:
            self.functionform.setCurrentWidget(self.datatrees[index])
            self.view_number.setValue(self.slice_numbers[index])
        except IndexError as e:
            print 'index {} does not exist'.format(index)

    # Could be leaking memory if I don't explicitly delete the datatrees that are being removed
    # from the previewdata deque but are still in the functionform widget? Hopefully python gc is taking good care of me
    def addPreview(self, image, funcdata, slice_number):
        self.deleteButton.show()
        self.setPipelineButton.show()
        self.previews.appendleft(np.flipud(image))
        functree = DataTreeWidget()
        functree.setHeaderHidden(True)
        functree.setData(funcdata, hideRoot=True)
        self.data.appendleft(funcdata)
        self.datatrees.appendleft(functree)
        self.slice_numbers.appendleft(slice_number)
        self.view_number.setValue(slice_number)
        self.functionform.addWidget(functree)
        self.imageview.setImage(self.previews)
        self.functionform.setCurrentWidget(functree)

    def removePreview(self):
        if len(self.previews) > 0:
            idx = self.imageview.currentIndex
            self.functionform.removeWidget(self.datatrees[idx])
            del self.previews[idx]
            del self.datatrees[idx]
            del self.data[idx]
            del self.slice_numbers[idx]
            if len(self.previews) == 0:
                self.imageview.clear()
                self.deleteButton.hide()
                self.setPipelineButton.hide()
            else:
                self.imageview.setImage(self.previews)

    def defaultsButtonClicked(self):
        current_data = self.data[self.imageview.currentIndex]
        fmanager.set_pipeline_from_preview(current_data, setdefaults=True)


class ReconstructionViewer(QtGui.QWidget):
    def __init__(self, parent=None):
        super(ReconstructionViewer, self).__init__(parent=parent)
        self.stack_viewer = StackViewer()
        self.path_edit = QtGui.QLineEdit(parent=self)
        self.path_edit.setReadOnly(True)
        self.browse_button = QtGui.QPushButton(parent=self)
        self.browse_button.setText('Select Reconstruction')

        layout = QtGui.QGridLayout(self)
        layout.addWidget(self.path_edit, 0, 0, 1, 1)
        layout.addWidget(self.browse_button, 0, 1, 1, 1)
        layout.addWidget(self.stack_viewer, 1, 0, 2, 2)

        self.browse_button.clicked.connect(self.openDataset)

    def openDataset(self, path=None):
        if path is None:
            path = QtGui.QFileDialog.getOpenFileNames(self, 'Open Reconstruction Data', os.path.expanduser('~'))[0]
        if path:
            if len(path) == 1:
                path = path[0]
            data = loader.StackImage(path)
            self.stack_viewer.setData(data)
            if isinstance(path, list):
                path = os.path.split(path[0])[0]
            self.path_edit.setText(path)


"""
Example volume rendering

Controls:

* 1  - toggle camera between first person (fly), regular 3D (turntable) and
       arcball
* 2  - toggle between volume rendering methods
* 3  - toggle between stent-CT / brain-MRI image
* 4  - toggle between colormaps
* 0  - reset cameras
* [] - decrease/increase isosurface threshold

With fly camera:

* WASD or arrow keys - move around
* SPACE - brake
* FC - move up-down
* IJKL or mouse - look around
"""


class VolumeViewer(QtGui.QWidget):

    sigImageChanged=QtCore.Signal()

    def __init__(self,path=None,data=None,*args,**kwargs):
        super(VolumeViewer, self).__init__()

        self.levels = [0, 1]

        l = QtGui.QHBoxLayout()
        l.setContentsMargins(0,0,0,0)
        l.setSpacing(0)

        self.volumeRenderWidget=VolumeRenderWidget()
        l.addWidget(self.volumeRenderWidget.native)

        self.HistogramLUTWidget = pg.HistogramLUTWidget(image=self, parent=self)
        self.HistogramLUTWidget.setMaximumWidth(self.HistogramLUTWidget.minimumWidth()+15)# Keep static width
        self.HistogramLUTWidget.setMinimumWidth(self.HistogramLUTWidget.minimumWidth()+15)

        l.addWidget(self.HistogramLUTWidget)

        self.xregion = SliceWidget(parent=self)
        self.yregion = SliceWidget(parent=self)
        self.zregion = SliceWidget(parent=self)
        self.xregion.item.region.setRegion([0, 1000])
        self.yregion.item.region.setRegion([0, 1000])
        self.zregion.item.region.setRegion([0, 1000])
        self.xregion.sigSliceChanged.connect(self.setVolume) #change to setVolume
        self.yregion.sigSliceChanged.connect(self.setVolume)
        self.zregion.sigSliceChanged.connect(self.setVolume)
        l.addWidget(self.xregion)
        l.addWidget(self.yregion)
        l.addWidget(self.zregion)

        self.setLayout(l)

        # self.setVolume(vol=data,path=path)

        # self.volumeRenderWidget.export('video.mp4',fps=25,duration=10.)
        # self.writevideo()


    @property
    def vol(self):
        return self.volumeRenderWidget.vol

    def getSlice(self):
        xslice=self.xregion.getSlice()
        yslice=self.yregion.getSlice()
        zslice=self.zregion.getSlice()
        return xslice,yslice,zslice

    def setVolume(self, vol=None, path=None):
        sliceobj = self.getSlice()
        self.volumeRenderWidget.setVolume(vol, path, sliceobj)
        self.volumeRenderWidget.update()
        if vol is not None or path is not None:
            self.sigImageChanged.emit()
            for i, region in enumerate([self.xregion, self.yregion, self.zregion]):
                try:
                    region.item.region.setBounds([0, self.volumeRenderWidget.vol.shape[i]])
                except RuntimeError as e:
                    print e.message

    def moveGradientTick(self, idx, pos):
        tick = self.HistogramLUTWidget.item.gradient.listTicks()[idx][0]
        tick.setPos(pos, 0)
        tick.view().tickMoved(tick, QtCore.QPoint(pos*self.HistogramLUTWidget.item.gradient.length, 0))
        tick.sigMoving.emit(tick)
        tick.sigMoved.emit(tick)
        tick.view().tickMoveFinished(tick)

    def setLevels(self, levels, update=True):
        self.levels = levels
        self.setLookupTable()
        self.HistogramLUTWidget.region.setRegion(levels)
        if update:
            self.volumeRenderWidget.update()

    def setLookupTable(self, lut=None, update=True):
        try:
            table = self.HistogramLUTWidget.item.gradient.colorMap().color/256.
            pos = self.HistogramLUTWidget.item.gradient.colorMap().pos
            #table=np.clip(table*(self.levels[1]-self.levels[0])+self.levels[0],0.,1.)
            table[:, 3] = pos
            table = np.vstack([np.array([[0,0,0,0]]),table,np.array([[1,1,1,1]])])
            pos = np.hstack([[0], pos*(self.levels[1] - self.levels[0]) + self.levels[0], [1]])
            self.volumeRenderWidget.volume.cmap = Colormap(table, controls=pos)
        except AttributeError as ex:
            print ex


    def getHistogram(self, bins='auto', step='auto', targetImageSize=100, targetHistogramSize=500, **kwds):
        """Returns x and y arrays containing the histogram values for the current image.
        For an explanation of the return format, see numpy.histogram().

        The *step* argument causes pixels to be skipped when computing the histogram to save time.
        If *step* is 'auto', then a step is chosen such that the analyzed data has
        dimensions roughly *targetImageSize* for each axis.

        The *bins* argument and any extra keyword arguments are passed to
        np.histogram(). If *bins* is 'auto', then a bin number is automatically
        chosen based on the image characteristics:

        * Integer images will have approximately *targetHistogramSize* bins,
          with each bin having an integer width.
        * All other types will have *targetHistogramSize* bins.

        This method is also used when automatically computing levels.
        """
        if self.vol is None:
            return None,None
        if step == 'auto':
            step = (np.ceil(self.vol.shape[0] / targetImageSize),
                    np.ceil(self.vol.shape[1] / targetImageSize))
        if np.isscalar(step):
            step = (step, step)
        stepData = self.vol[::step[0], ::step[1]]

        if bins == 'auto':
            if stepData.dtype.kind in "ui":
                mn = stepData.min()
                mx = stepData.max()
                step = np.ceil((mx-mn) / 500.)
                bins = np.arange(mn, mx+1.01*step, step, dtype=np.int)
                if len(bins) == 0:
                    bins = [mn, mx]
            else:
                bins = 500

        kwds['bins'] = bins
        hist = np.histogram(stepData, **kwds)

        return hist[1][:-1], hist[0]

    # @volumeRenderWidget.connect
    # def on_frame(self,event):
    #     self.volumeRenderWidget.cam1.auto_roll

    def writevideo(self,fps=25):
        writer = imageio.save('foo.mp4', fps=25)
        self.volumeRenderWidget.events.draw.connect(lambda e: writer.append_data(self.render()))
        self.volumeRenderWidget.events.close.connect(lambda e: writer.close())


class VolumeRenderWidget(scene.SceneCanvas):

    def __init__(self,vol=None, path=None, size=(800,600), show=False):
        super(VolumeRenderWidget, self).__init__(keys='interactive', size=size, show=show)

        # Prepare canvas
        self.measure_fps()

        #self.unfreeze()

        # Set up a viewbox to display the image with interactive pan/zoom
        self.view = self.central_widget.add_view()

        self.vol=None
        self.setVolume(vol,path)
        self.volume=None

        # Create three cameras (Fly, Turntable and Arcball)
        fov = 60.
        self.cam1 = scene.cameras.FlyCamera(parent=self.view.scene, fov=fov, name='Fly')
        self.cam2 = scene.cameras.TurntableCamera(parent=self.view.scene, fov=fov, name='Turntable')
        self.cam3 = scene.cameras.ArcballCamera(parent=self.view.scene, fov=fov, name='Arcball')
        self.view.camera = self.cam2  # Select turntable at first


    def setVolume(self, vol=None, path=None, sliceobj=None):
        if vol is None:
            vol=self.vol

        if path is not None:
            if '*' in path:
                vol=loader.loadimageseries(path)
            elif os.path.splitext(path)[-1]=='.npy':
                vol=loader.loadimage(path)
            else:
                vol=loader.loadtiffstack(path)

        if vol is None:
            return

        self.vol = vol

        if slice is not None:
            slicevol = self.vol[sliceobj]
        else:
            slicevol = self.vol

        # Set whether we are emulating a 3D texture
        emulate_texture = False

        # Create the volume visuals
        if self.volume is None:
            self.volume = scene.visuals.Volume(slicevol, parent=self.view.scene, emulate_texture=emulate_texture)
            self.volume.method = 'translucent'
        else:
            self.volume.set_data(slicevol)
            self.volume._create_vertex_data() #TODO: Try using this instead of slicing array?

        # Translate the volume into the center of the view (axes are in strange order for unkown )
        scale = 3*(.0075,) # This works for now but might be different for different resolutions
        translate = map(lambda x: -scale[0]*x/2, reversed(vol.shape))
        self.volume.transform = scene.STTransform(translate=translate, scale=scale)

    # Implement key presses
    def on_key_press(self, event):
        if event.text == '1':
            cam_toggle = {self.cam1: self.cam2, self.cam2: self.cam3, self.cam3: self.cam1}
            self.view.camera = cam_toggle.get(self.view.camera, self.cam2)
            print(self.view.camera.name + ' camera')
        elif event.text == '2':
            pass
        elif event.text == '3':
            pass
        elif event.text == '4':
            pass
        elif event.text == '0':
            self.cam1.set_range()
            self.cam3.set_range()
        elif event.text != '' and event.text in '[]':
            s = -0.025 if event.text == '[' else 0.025
            self.volume.threshold += s
            th = self.volume.threshold
            print("Isosurface threshold: %0.3f" % th)


class SliceWidget(pg.HistogramLUTWidget):
    sigSliceChanged = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super(SliceWidget, self).__init__(*args, **kwargs)
        self.item.paint = lambda *x: None
        self.item.vb.deleteLater()
        self.item.gradient.gradRect.hide()
        self.item.gradient.allowAdd = False
        self.setMinimumWidth(70)
        self.setMaximumWidth(70)
        self.item.sigLookupTableChanged.connect(self.ticksChanged)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Expanding)

    def sizeHint(self):
        return QtCore.QSize(70, 200)

    def ticksChanged(self,LUT):
        self.sigSliceChanged.emit()
        #tuple(sorted(LUT.gradient.ticks.values()))

    def getSlice(self):
        bounds = sorted(self.item.gradient.ticks.values())
        bounds = (bounds[0]*self.item.region.getRegion()[1],bounds[1]*self.item.region.getRegion()[1])
        return slice(*bounds)


class VolumeVisual(scene.visuals.Volume):
    def set_data(self, vol, clim=None):
        """ Set the volume data.

        Parameters
        ----------
        vol : ndarray
            The 3D volume.
        clim : tuple | None
            Colormap limits to use. None will use the min and max values.
        """
        # Check volume
        if not isinstance(vol, np.ndarray):
            raise ValueError('Volume visual needs a numpy array.')
        if not ((vol.ndim == 3) or (vol.ndim == 4 and vol.shape[-1] <= 4)):
            raise ValueError('Volume visual needs a 3D image.')

        # Handle clim
        if clim is not None:
            clim = np.array(clim, float)
            if not (clim.ndim == 1 and clim.size == 2):
                raise ValueError('clim must be a 2-element array-like')
            self._clim = tuple(clim)
        self._clim = vol.min(), vol.max()   #NOTE: THIS IS MODIFIED BY RP TO RESET MIN/MAX EACH TIME

        # Apply clim
        vol = np.array(vol, dtype='float32', copy=False)
        vol -= self._clim[0]
        vol *= 1./(self._clim[1] - self._clim[0])

        # Apply to texture
        self._tex.set_data(vol)  # will be efficient if vol is same shape
        self._program['u_shape'] = vol.shape[2], vol.shape[1], vol.shape[0]
        self._vol_shape = vol.shape[:3]

        # Create vertices?
        if self._index_buffer is None:
            self._create_vertex_data()


scene.visuals.Volume = VolumeVisual


class RunViewer(QtGui.QTabWidget):
    """
    Viewer class to define run settings for a full tomography dataset reconstruction job. Has tab for local run settings
    and tab for remote job settins.
    """

    # sigRunClicked = QtCore.Signal(tuple, tuple, str, str, int, int, object)

    def __init__(self, parent=None):
        super(RunViewer, self).__init__(parent=parent)
        self.setTabPosition(QtGui.QTabWidget.West)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_41.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.local_cancelButton = QtGui.QToolButton()
        self.remote_cancelButton = QtGui.QToolButton()

        # Text Browser for local run console
        self.local_console = QtGui.QTextEdit() #Browser()
        self.local_console.setObjectName('Local')

        # Text Brower for remote run console
        self.remote_console = QtGui.QTextEdit()
        self.remote_console.setObjectName('Remote')

        for console, button in zip((self.local_console, self.remote_console),
                                   (self.local_cancelButton, self.remote_cancelButton)):
            console.setReadOnly(True)
            console.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
            button.setIcon(icon)
            button.setIconSize(QtCore.QSize(24, 24))
            button.setFixedSize(32, 32)
            button.setToolTip('Cancel current process')
            w = QtGui.QWidget()
            w.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
            w.setContentsMargins(0,0,0,0)
            l = QtGui.QGridLayout()
            l.setContentsMargins(0,0,0,0)
            l.setSpacing(0)
            l.addWidget(console, 0, 0, 2, 2)
            l.addWidget(button, 1, 2, 1, 1)
            w.setLayout(l)
            self.addTab(w, console.objectName())

    def log2local(self, msg):
        text = self.local_console.toPlainText()
        if '\n' not in msg:
            self.local_console.setText(msg + '\n\n' + text)
        else:
            topline = text.splitlines()[0]
            tail = '\n'.join(text.splitlines()[1:])
            self.local_console.setText(topline + msg + tail)
        # self.local_console.insertPlainText(msg)

    def sino_indices(self):
        return (self.reconsettings.child('Start Sinogram').value(),
                self.reconsettings.child('End Sinogram').value(),
                self.reconsettings.child('Step Sinogram').value())

    def proj_indices(self):
        return (self.reconsettings.child('Start Projection').value(),
                self.reconsettings.child('End Projection').value(),
                self.reconsettings.child('Step Projection').value())

    def runButtonClicked(self):
        self.sigRunClicked.emit(self.proj_indices(), self.sino_indices(),
                                self.reconsettings.child('Output Name').value(),
                                self.reconsettings.child('Ouput Format').value(),
                                self.localsettings.child('Sinogram Chunks').value(),
                                self.localsettings.child('Cores').value(),
                                self.log2local)


class Preview3DViewer(QtGui.QSplitter):
    def __init__(self, paths=None, data=None, *args, **kwargs):
        super(Preview3DViewer, self).__init__()
        self.setOrientation(QtCore.Qt.Horizontal)
        l = QtGui.QVBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        self.functiontree = DataTreeWidget()
        self.functiontree.setHeaderHidden(True)
        self.functiontree.clear()
        self.setPipelineButton = QtGui.QPushButton(self)
        self.setPipelineButton.setText("Set Pipeline")
        l.addWidget(self.functiontree)
        l.addWidget(self.setPipelineButton)
        panel = QtGui.QWidget(self)
        panel.setLayout(l)

        self.volumeviewer = VolumeViewer()

        self.addWidget(panel)
        self.addWidget(self.volumeviewer)

        self.funcdata = None

        self.setPipelineButton.clicked.connect(self.defaultsButtonClicked)

    def setPreview(self, recon, funcdata):
        self.functiontree.setData(funcdata, hideRoot=True)
        self.funcdata = funcdata
        self.functiontree.show()
        self.volumeviewer.setVolume(vol=recon)

    def defaultsButtonClicked(self):
        fmanager.set_pipeline_from_preview(self.funcdata)


class DataTreeWidget(QtGui.QTreeWidget):
    """
    Widget for displaying hierarchical python data structures
    (eg, nested dicts, lists, and arrays), adapted from pyqtgraph datatree.
    """

    def __init__(self, parent=None, data=None):
        QtGui.QTreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.setData(data)
        self.setColumnCount(2)
        self.setHeaderLabels(['Parameter', 'value'])

    def setData(self, data, hideRoot=False):
        """data should be a dictionary."""
        self.clear()
        self.buildTree(data, self.invisibleRootItem(), hideRoot=hideRoot)
        self.expandToDepth(3)
        self.resizeColumnToContents(0)

    def buildTree(self, data, parent, name='', hideRoot=False):
        if hideRoot:
            node = parent
        else:
            node = QtGui.QTreeWidgetItem([name, ""])
            parent.addChild(node)

        if isinstance(data, dict):
            for k in data.keys():
                self.buildTree(data[k], node, str(k))
        elif isinstance(data, list) or isinstance(data, tuple):
            for i in range(len(data)):
                self.buildTree(data[i], node, str(i))
        else:
            node.setText(1, str(data))


class ArrayDeque(deque):
    """
    Class for a numpy array deque where arrays can be appended on both ends.
    """
    def __init__(self, arraylist=[], arrayshape=None, dtype=None, maxlen=None):
        # perhaps will need to add check of datatype everytime a new array is added with extend, append, etc??
        if not arraylist and not arrayshape:
            raise ValueError('One of arraylist or arrayshape must be specified')

        super(ArrayDeque, self).__init__(iterable=arraylist, maxlen=maxlen)

        self._shape = [len(self)]
        self._dtype = dtype

        if arraylist:
            # if False in [np.array_equal(arraylist[0].shape, array.shape) for array in arraylist[1:]]:
            #     raise ValueError('All arrays in arraylist must have the same dimensions')
            # elif False in [arraylist[0].dtype == array.dtype for array in arraylist[1:]]:
            #     raise ValueError('All arrays in arraylist must have the same data type')
            map(self._shape.append, arraylist[0].shape)
        elif arrayshape:
            map(self._shape.append, arrayshape)

        self.ndim = len(self._shape)

    @property
    def shape(self):
        self._shape[0] = len(self)
        return self._shape

    @property
    def size(self):
        return np.product(self._shape)

    @property
    def dtype(self):
        if self._dtype is None and self.shape[0]:
            self._dtype = self.__getitem__(0).dtype
        return self._dtype

    @property
    def max(self):
        return np.max(max(self, key=lambda x:np.max(x)))

    @property
    def min(self):
        return np.min(min(self, key=lambda x:np.min(x)))

    def append(self, arr):
        # if arr.shape != tuple(self.shape[1:]):
        #     raise ValueError('Array shape must be {0}, got shape {1}'.format(self.shape[1:], arr.shape))
        # if self.dtype is not None and arr.dtype != self.dtype:
        #     raise ValueError('Array must be of type {}'.format(self.dtype))
        super(ArrayDeque, self).append(arr)

    def appendleft(self, arr):
        # if arr.shape != tuple(self.shape[1:]):
        #     raise ValueError('Array shape must be {0}, got shape {1}'.format(self.shape[1:], arr.shape))
        # if self.dtype is not None and arr.dtype != self.dtype:
        #     raise ValueError('Array must be of type {}'.format(self.dtype))
        super(ArrayDeque, self).appendleft(arr)

    def __getitem__(self, item):
        if type(item) is list and isinstance(item[0], slice):
            dq_item = item.pop(0)
            if isinstance(dq_item, slice):
                dq_item = dq_item.stop if dq_item.stop is not None else dq_item.start if dq_item.start is not None else 0
            return super(ArrayDeque, self).__getitem__(dq_item).__getitem__(item)
        else:
            return super(ArrayDeque, self).__getitem__(item)


# Testing
if __name__ == '__main__':
    import sys, time
    app = QtGui.QApplication(sys.argv)
    w = RunViewer()
    def foobar():
        for i in range(10000):
            w.log2local('Line {}\n\n'.format(i))
            # time.sleep(.1)
    w.local_cancelButton.clicked.connect(foobar)
    w.setWindowTitle("Test this thing")
    w.show()
    sys.exit(app.exec_())