from collections import deque
import numpy as np
import pyqtgraph as pg
from PySide import QtGui, QtCore
from pipeline import loader
from pipeline import msg
from xicam.widgets.customwidgets import DataTreeWidget, ImageView
from xicam.widgets.roiwidgets import ROImageOverlay
from xicam.widgets.imageviewers import StackViewer
from xicam.widgets.volumeviewers import VolumeViewer

__author__ = "Luis Barroso-Luque"
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

    Attributes
    ----------
    """

    sigSetDefaults = QtCore.Signal(dict)
    # sigROI = QtCore.Signal(tuple)

    def __init__(self, paths=None, data=None, *args, **kwargs):

        if paths is None and data is None:
            raise ValueError('Either data or path to file must be provided')

        super(TomoViewer, self).__init__(*args, **kwargs)

        # self._recon_path = None
        self.viewstack = QtGui.QStackedWidget(self)
        self.viewmode = QtGui.QTabBar(self)
        self.viewmode.addTab('Projection View')  # TODO: Add icons!
        self.viewmode.addTab('Sinogram View')
        self.viewmode.addTab('Slice Preview')
        self.viewmode.addTab('3D Preview')
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
        self.previewViewer.sigSetDefaults.connect(self.sigSetDefaults.emit)
        self.viewstack.addWidget(self.previewViewer)

        self.preview3DViewer = Preview3DViewer(paths=paths, data=data)
        self.preview3DViewer.volumeviewer.moveGradientTick(1, 0.3)
        self.viewstack.addWidget(self.preview3DViewer)

        v = QtGui.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.viewstack)
        v.addWidget(self.viewmode)
        self.setLayout(v)

        self.viewmode.currentChanged.connect(self.currentChanged)
        self.viewstack.currentChanged.connect(self.viewmode.setCurrentIndex)

        self.sigROI =  self.projectionViewer.sigROIChanged
        self.xbounds = self.projectionViewer.xbounds

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

    def getsino(self, slc=None):
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

    # def setCorValue(self, value):
    #     self.cor = value

    def addSlicePreview(self, params, recon, slice_no=None):
        print 'Adding slice prevvvvvs'
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

    def onManualCenter(self, active):
        if active:
            self.projectionViewer.showCenterDetection()
            self.viewstack.setCurrentWidget(self.projectionViewer)
        else:
            self.projectionViewer.hideCenterDetection()

    def centerActionActive(self):
        if self.projectionViewer.imgoverlay_roi.isVisible():
            return True
        else:
            return False

    # TODO then get the slice for that image to use in the reconstruction and fill in start, end sinogram and adjust center correspondingly when running!
    def onROIselection(self, active):
        if active:
            self.viewstack.setCurrentWidget(self.projectionViewer)
            self.projectionViewer.selection_roi.show()
        else:
            self.projectionViewer.selection_roi.setSize(self.data.shape[1:])
            self.projectionViewer.selection_roi.setPos([0,0])
            self.projectionViewer.selection_roi.hide()

    def roiActionActive(self):
        if self.projectionViewer.selection_roi.isVisible():
            return True
        else:
            return False


class ProjectionViewer(QtGui.QWidget):
    """
    Class that holds a stack viewer, an ROImageOverlay and a few widgets to allow manual center detection
    """
    sigCenterChanged = QtCore.Signal(float)
    sigROIChanged = QtCore.Signal(tuple)

    def __init__(self, data, view_label=None, center=None, *args, **kwargs):
        super(ProjectionViewer, self).__init__(*args, **kwargs)
        self.stackViewer = StackViewer(data, view_label=view_label)
        self.imageItem = self.stackViewer.imageItem
        self.data = self.stackViewer.data
        self.normalized = False
        self.xbounds = [0, self.data.shape[1]]
        self.flat = np.median(self.data.flats, axis=0).transpose()
        self.dark = np.median(self.data.darks, axis=0).transpose()

        self.imgoverlay_roi = ROImageOverlay(self.data, self.imageItem, [0, 0], parent=self.stackViewer.view)
        self.imageItem.sigImageChanged.connect(self.imgoverlay_roi.updateImage)
        self.stackViewer.view.addItem(self.imgoverlay_roi)
        self.roi_histogram = pg.HistogramLUTWidget(image=self.imgoverlay_roi.imageItem, parent=self.stackViewer)

        # roi to select region of interest
        self.selection_roi = pg.ROI([0, 0], self.data.shape[1:], removable=True)
        self.selection_roi.setPen(color=[0, 255, 255], width=1)
        self.selection_roi.addScaleHandle([1, 1], [0, 0])
        self.selection_roi.addScaleHandle([0, 0], [1, 1])
        self.selection_roi.addScaleHandle([1, 0.5], [0.5, 0.5])
        self.selection_roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.selection_roi.addScaleHandle([0.5, 0], [0.5, 1])
        self.selection_roi.addScaleHandle([0.5, 1], [0.5, 0])
        self.stackViewer.view.addItem(self.selection_roi)
        self.selection_roi.sigRegionChangeFinished.connect(self.roiChanged)
        self.selection_roi.hide()

        self.stackViewer.ui.gridLayout.addWidget(self.roi_histogram, 0, 3, 1, 2)
        self.stackViewer.keyPressEvent = self.keyPressEvent

        self.cor_widget = QtGui.QWidget(self)
        clabel = QtGui.QLabel('Rotation Center:')
        olabel = QtGui.QLabel('Offset:')
        self.centerBox = QtGui.QDoubleSpinBox(parent=self.cor_widget)
        self.centerBox.setDecimals(1)
        self.setCenterButton = QtGui.QToolButton()
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_45.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setCenterButton.setIcon(icon)
        self.setCenterButton.setToolTip('Set center in pipeline')
        originBox = QtGui.QLabel(parent=self.cor_widget)
        originBox.setText('x={}   y={}'.format(0, 0))
        center = center if center is not None else data.shape[1]/2.0
        self.centerBox.setValue(center)
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
        constrainYCheckBox.stateChanged.connect(lambda v: self.imgoverlay_roi.constrainY(v))
        constrainXCheckBox.stateChanged.connect(lambda v: self.imgoverlay_roi.constrainX(v))
        # rotateCheckBox.stateChanged.connect(self.addRotateHandle)
        self.normCheckBox.stateChanged.connect(self.normalize)
        self.stackViewer.sigTimeChanged.connect(lambda: self.normalize(False))
        self.imgoverlay_roi.sigTranslated.connect(self.setCenter)
        self.imgoverlay_roi.sigTranslated.connect(lambda x, y: originBox.setText('x={}   y={}'.format(x, y)))
        self.hideCenterDetection()

    def roiChanged(self):
        roi = self.selection_roi.getArraySlice(self.data[self.stackViewer.currentIndex], self.imageItem,
                                               returnSlice=False)
        self.sigROIChanged.emit(roi[0])
        self.xbounds = roi[0][0]

    def changeOverlayProj(self, idx):
        self.normCheckBox.setChecked(False)
        self.imgoverlay_roi.setCurrentImage(idx)
        self.imgoverlay_roi.updateImage()

    def setCenter(self, x, y):
        center = (self.data.shape[1] + x - 1)/2.0# subtract half a pixel out of 'some' convention?
        self.centerBox.setValue(center) # setText(str(center))
        self.sigCenterChanged.emit(center)

    def hideCenterDetection(self):
        self.normalize(False)
        self.cor_widget.hide()
        self.roi_histogram.hide()
        self.imgoverlay_roi.setVisible(False)

    def showCenterDetection(self):
        self.cor_widget.show()
        self.roi_histogram.show()
        self.imgoverlay_roi.setVisible(True)

    def updateROIFromCenter(self, center):
        s = self.imgoverlay_roi.pos()[0]
        self.imgoverlay_roi.translate(pg.Point((2 * center + 1 - self.data.shape[1] - s, 0))) # 1 again due to the so-called COR
                                                                                   # conventions...
    def flipOverlayProj(self, val):
        self.imgoverlay_roi.flipCurrentImage()
        self.imgoverlay_roi.updateImage()

    def addRotateHandle(self, val):
        if val:
            self.addRotateHandle.handle = self.imgoverlay_roi.addRotateHandle([0, 1], [0.2, 0.2])
        else:
            self.imgoverlay_roi.removeHandle(self.addRotateHandle.handle)

    def normalize(self, val):
        if val and not self.normalized:
            proj = (self.imageItem.image - self.dark)/(self.flat - self.dark)
            overlay = self.imgoverlay_roi.currentImage
            if self.imgoverlay_roi.flipped:
                overlay = np.flipud(overlay)
            overlay = (overlay - self.dark)/(self.flat - self.dark)
            if self.imgoverlay_roi.flipped:
                overlay = np.flipud(overlay)
            self.imgoverlay_roi.currentImage = overlay
            self.imgoverlay_roi.updateImage(autolevels=True)
            self.stackViewer.setImage(proj, autoRange=False, autoLevels=True)
            self.stackViewer.updateImage()
            self.normalized = True
        elif not val and self.normalized:
            self.stackViewer.resetImage()
            self.imgoverlay_roi.resetImage()
            self.normalized = False
            self.normCheckBox.setChecked(False)

    def keyPressEvent(self, ev):
        super(ProjectionViewer, self).keyPressEvent(ev)
        if self.imgoverlay_roi.isVisible():
            self.imgoverlay_roi.keyPressEvent(ev)
        else:
            super(StackViewer, self.stackViewer).keyPressEvent(ev)
        ev.accept()


class PreviewViewer(QtGui.QSplitter):
    """
    Viewer class to show reconstruction previews in a PG ImageView, along with the function pipeline settings for the
    corresponding preview
    """

    sigSetDefaults = QtCore.Signal(dict)

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
        icon.addPixmap(QtGui.QPixmap("gui/icons_36.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.deleteButton.setIcon(icon)

        self.setPipelineButton = QtGui.QToolButton(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_45.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
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
        self.sigSetDefaults.emit(current_data)


class Preview3DViewer(QtGui.QSplitter):

    sigSetDefaults = QtCore.Signal(dict)

    def __init__(self, paths=None, data=None, *args, **kwargs):
        super(Preview3DViewer, self).__init__()
        self.setOrientation(QtCore.Qt.Horizontal)
        l = QtGui.QVBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        self.pipelinetree = DataTreeWidget()
        self.pipelinetree.setHeaderHidden(True)
        self.pipelinetree.clear()

        self.setPipelineButton = QtGui.QToolButton(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_45.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setPipelineButton.setIcon(icon)
        self.setPipelineButton.setToolTip('Set as pipeline')

        ly = QtGui.QVBoxLayout()
        ly.setContentsMargins(0, 0, 0, 0)
        ly.setSpacing(0)
        ly.addWidget(self.pipelinetree)
        h = QtGui.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.setPipelineButton)
        ly.addLayout(h)
        panel = QtGui.QWidget(self)
        panel.setLayout(ly)

        self.volumeviewer = VolumeViewer()

        self.addWidget(panel)
        self.addWidget(self.volumeviewer)

        self.funcdata = None

        self.setPipelineButton.clicked.connect(lambda: self.sigSetDefaults.emit(self.funcdata))
        self.setPipelineButton.hide()

    def setPreview(self, recon, funcdata):
        self.pipelinetree.setData(funcdata, hideRoot=True)
        self.funcdata = funcdata
        self.pipelinetree.show()
        self.volumeviewer.setVolume(vol=recon)
        self.setPipelineButton.show()


class RunConsole(QtGui.QTabWidget):
    """
    Class to output status of a running job, and cancel the job.  Has tab for local run settings
    and can add tabs tab for remote job settings.
    """

    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap("gui/icons_51.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

    def __init__(self, parent=None):
        super(RunConsole, self).__init__(parent=parent)
        self.setTabPosition(QtGui.QTabWidget.West)

        # Text Browser for local run console
        self.local_console, self.local_cancelButton = self.addConsole('Local')
        self.local_console.setObjectName('Local')

    def addConsole(self, name):
        console = QtGui.QTextEdit()
        button = QtGui.QToolButton()
        console.setObjectName(name)
        console.setReadOnly(True)
        console.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        button.setIcon(self.icon)
        button.setIconSize(QtCore.QSize(24, 24))
        button.setFixedSize(32, 32)
        button.setToolTip('Cancel running process')
        w = QtGui.QWidget()
        w.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        w.setContentsMargins(0, 0, 0, 0)
        l = QtGui.QGridLayout()
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        l.addWidget(console, 0, 0, 2, 2)
        l.addWidget(button, 1, 2, 1, 1)
        w.setLayout(l)
        self.addTab(w, console.objectName())
        return console, button

    def log2local(self, msg):
        text = self.local_console.toPlainText()
        if '\n' not in msg:
            self.local_console.setText(msg + '\n\n' + text)
        else:
            topline = text.splitlines()[0]
            tail = '\n'.join(text.splitlines()[1:])
            self.local_console.setText(topline + msg + tail)


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
    import sys

    app = QtGui.QApplication(sys.argv)
    w = RunConsole()
    def foobar():
        for i in range(10000):
            w.log2local('Line {}\n\n'.format(i))
            # time.sleep(.1)
    w.local_cancelButton.clicked.connect(foobar)
    w.setWindowTitle("Test this thing")
    w.show()
    sys.exit(app.exec_())
