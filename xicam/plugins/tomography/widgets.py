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
import psutil
from PySide import QtGui, QtCore
from vispy import scene  # , app, io
from vispy.color import Colormap  # , BaseColormap, ColorArray
from pipeline import loader
import pyqtgraph as pg
from pyqtgraph.parametertree import ParameterTree
import imageio
import os
import fmanager

__author__ = "Ronald J Pandolfi"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


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


class TomoViewer(QtGui.QWidget):
    """
    Class that holds projection, sinogram, recon preview, and process-settings viewers for a tomography dataset.
    """
    def __init__(self, paths=None, data=None, *args, **kwargs):

        if paths is None and data is None:
            raise ValueError('Either data or path to file must be provided')

        super(TomoViewer, self).__init__(*args, **kwargs)

        self.viewstack = QtGui.QStackedWidget(self)

        self.viewmode = QtGui.QTabBar(self)
        self.viewmode.addTab('Projection')  # TODO: Add icons!
        self.viewmode.addTab('Sinogram')
        self.viewmode.addTab('Preview')
        self.viewmode.addTab('3D Preview')
        self.viewmode.addTab('Process')
        self.viewmode.addTab('Reconstruction')
        self.viewmode.setShape(QtGui.QTabBar.TriangularSouth)

        if data is not None:
            self.data = data
        elif paths is not None and len(paths):
            self.data = self.loaddata(paths)

        self.cor = self.data.shape[2]/2

        self.projectionViewer = StackViewer(self.data, parent=self)
        self.viewstack.addWidget(self.projectionViewer)

        self.sinogramViewer = StackViewer(loader.SinogramStack.cast(self.data), parent=self)
        self.sinogramViewer.setIndex(self.sinogramViewer.data.shape[0] // 2)
        self.viewstack.addWidget(self.sinogramViewer)

        self.previewViewer = PreviewViewer(self.data.shape[1], parent=self)
        self.viewstack.addWidget(self.previewViewer)

        self.preview3DViewer = ReconstructionViewer(paths=paths, data=data)
        self.viewstack.addWidget(self.preview3DViewer)

        self.processViewer = ProcessViewer(paths, self.data.shape[::2], parent=self)
        self.processViewer.sigRunClicked.connect(fmanager.run_full_recon)
        self.viewstack.addWidget(self.processViewer)

        # Make this a stack viewer with a stack of the recon
        # self.reconstructionViewer = ReconstructionViewer(paths=paths, data=data)
        # self.viewstack.addWidget(self.reconstructionViewer)

        l = QtGui.QVBoxLayout(self)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(self.viewstack)
        l.addWidget(self.viewmode)
        self.setLayout(l)

        self.viewmode.currentChanged.connect(self.currentChanged)
        self.viewstack.currentChanged.connect(self.viewmode.setCurrentIndex)

    @staticmethod
    def loaddata(paths):
        return loader.ProjectionStack(paths)

    def getsino(self, slc=None):
        if slc is None:
            return np.ascontiguousarray(self.sinogramViewer.currentdata[:,np.newaxis,:])
        else:
            return np.ascontiguousarray(self.data.fabimage.getsinogramchunk(proj_slice=slice(*slc[0]),
                                                                            sino_slc=slice(*slc[1])))

    def getflats(self, slc=None):
        if slc is None:
            return np.ascontiguousarray(self.data.flats[:, self.sinogramViewer.currentIndex, :])
        else:
            return np.ascontiguousarray(self.data.flats[slice(*slc[0]), slice(*slc[1]), :])

    def getdarks(self, slc=None):
        if slc is None:
            return np.ascontiguousarray(self.data.darks[: ,self.sinogramViewer.currentIndex, :])
        else:
            return np.ascontiguousarray(self.data.darks[slice(*slc[0]), slice(*slc[1]), :])

    def currentChanged(self, index):
        self.viewstack.setCurrentIndex(index)

    def addPreview(self, params, recon):
        npad = int((recon.shape[1] - self.data.shape[1])/2)
        recon = recon[0, npad:-npad, npad:-npad] if npad != 0 else recon[0]
        self.previewViewer.addPreview(recon, params)
        self.viewstack.setCurrentWidget(self.previewViewer)

    def setCorValue(self, value):
        self.cor = value

    def test(self, params):
        self.previewViewer.test(params)


class StackViewer(pg.ImageView):
    """
    PG ImageView subclass to view projections or sinograms of a tomography dataset
    """
    def __init__(self, data, view_label=None, *args, **kwargs):
        super(StackViewer, self).__init__(*args, **kwargs)
        self.data = data
        self.ui.roiBtn.setParent(None)
        self.setImage(self.data) # , axes={'t':0, 'x':2, 'y':1, 'c':3})
        self.getImageItem().setRect(QtCore.QRect(0, 0, self.data.rawdata.shape[0], self.data.rawdata.shape[1]))
        self.getImageItem().setAutoDownsample(True)
        self.autoLevels()
        self.getView().invertY(False)

        self.view_label = QtGui.QLabel(self)
        self.view_label.setText('No: ')
        self.view_spinBox = QtGui.QSpinBox(self)
        self.view_spinBox.setRange(0, data.shape[0] - 1)
        l = QtGui.QHBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(self.view_label)
        l.addWidget(self.view_spinBox)
        l.addStretch(1)
        w = QtGui.QWidget()
        w.setLayout(l)
        self.ui.gridLayout.addWidget(self.view_label, 1, 1, 1, 1)
        self.ui.gridLayout.addWidget(self.view_spinBox, 1, 2, 1, 1)

        self.sigTimeChanged.connect(self.indexChanged)
        self.view_spinBox.valueChanged.connect(self.setCurrentIndex)

    def indexChanged(self, ind, time):
        self.view_spinBox.setValue(ind)

    def setIndex(self, ind):
        self.setCurrentIndex(ind)
        self.view_spinBox.setValue(ind)

    @property
    def currentdata(self):
        return np.rot90(self.data[self.data.currentframe]) #these rotations are very annoying


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
        self.previewdata = deque(maxlen=self.maxpreviews)

        self.setOrientation(QtCore.Qt.Horizontal)
        self.functionform = QtGui.QStackedWidget()
        self.imageview = ImageView(self)
        self.imageview.ui.roiBtn.setParent(None)

        self.deleteButton = QtGui.QPushButton(self.imageview)
        self.deleteButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_40.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.deleteButton.setIcon(icon)
        self.imageview.ui.gridLayout.addWidget(self.deleteButton, 1, 1, 1, 1)

        self.setCurrentIndex = self.imageview.setCurrentIndex
        self.addWidget(self.functionform)
        self.addWidget(self.imageview)

        self.deleteButton.clicked.connect(self.removePreview)
        self.imageview.sigTimeChanged.connect(self.indexChanged)

    # Could be leaking memory if I don't explicitly delete the datatrees that are being removed
    # from the previewdata deque but are still in the functionform widget? Hopefully python gc is taking good care of me
    def addPreview(self, image, funcdata):
        self.previews.appendleft(image)
        functree = DataTreeWidget()
        functree.setHeaderHidden(True)
        functree.setData(funcdata, hideRoot=True)
        self.previewdata.appendleft(functree)
        self.functionform.addWidget(functree)
        self.imageview.setImage(self.previews)
        self.functionform.setCurrentWidget(functree)

    def removePreview(self):
        if len(self.previews) > 0:
            idx = self.imageview.currentIndex
            self.functionform.removeWidget(self.previewdata[idx])
            del self.previews[idx]
            del self.previewdata[idx]
            if len(self.previews) == 0:
                self.imageview.clear()
            else:
                self.imageview.setImage(self.previews)

    @QtCore.Slot(object, object)
    def indexChanged(self, index, time):
        try:
            self.functionform.setCurrentWidget(self.previewdata[index])
        except IndexError:
            print 'index {} does not exist'


class VolumeViewer(QtGui.QWidget):

    sigImageChanged=QtCore.Signal()

    def __init__(self,path=None,data=None,*args,**kwargs):
        super(VolumeViewer, self).__init__()

        self.levels=[0,1]

        l = QtGui.QHBoxLayout()
        l.setContentsMargins(0,0,0,0)
        l.setSpacing(0)

        self.volumeRenderWidget=VolumeRenderWidget()
        l.addWidget(self.volumeRenderWidget.native)

        self.HistogramLUTWidget = pg.HistogramLUTWidget(image=self)
        self.HistogramLUTWidget.setMaximumWidth(self.HistogramLUTWidget.minimumWidth()+15)# Keep static width
        self.HistogramLUTWidget.setMinimumWidth(self.HistogramLUTWidget.minimumWidth()+15)

        l.addWidget(self.HistogramLUTWidget)

        self.xregion = SliceWidget()
        self.yregion = SliceWidget()
        self.zregion = SliceWidget()
        self.xregion.item.region.setRegion([0,5000])
        self.yregion.item.region.setRegion([0,5000])
        self.zregion.item.region.setRegion([0,5000])
        self.xregion.sigSliceChanged.connect(self.setVolume) #change to setVolume
        self.yregion.sigSliceChanged.connect(self.setVolume)
        self.zregion.sigSliceChanged.connect(self.setVolume)
        l.addWidget(self.xregion)
        l.addWidget(self.yregion)
        l.addWidget(self.zregion)

        self.setLayout(l)

        self.setVolume(vol=data,path=path)

        # self.volumeRenderWidget.export('video.mp4',fps=25,duration=10.)
        # self.writevideo()


    def getSlice(self):
        xslice=self.xregion.getSlice()
        yslice=self.yregion.getSlice()
        zslice=self.zregion.getSlice()
        return xslice,yslice,zslice

    def setVolume(self,vol=None,path=None):
        sliceobj=self.getSlice()
        self.volumeRenderWidget.setVolume(vol,path,sliceobj)
        self.volumeRenderWidget.update()
        if vol is not None or path is not None:
            self.sigImageChanged.emit()
            self.xregion.item.region.setRegion([0,self.volumeRenderWidget.vol.shape[0]])
            self.yregion.item.region.setRegion([0,self.volumeRenderWidget.vol.shape[1]])
            self.zregion.item.region.setRegion([0,self.volumeRenderWidget.vol.shape[2]])

    def setLevels(self, levels, update=True):
        print 'levels:',levels
        self.levels=levels
        self.setLookupTable()

    def setLookupTable(self, lut=None, update=True):
        try:
            table=self.HistogramLUTWidget.item.gradient.colorMap().color/256.
            pos=self.HistogramLUTWidget.item.gradient.colorMap().pos

            #table=np.clip(table*(self.levels[1]-self.levels[0])+self.levels[0],0.,1.)
            table[:,3]=pos
            table=np.vstack([np.array([[0,0,0,0]]),table,np.array([[1,1,1,1]])])
            pos=np.hstack([[0],pos*(self.levels[1]-self.levels[0])+self.levels[0],[1]])

            self.volumeRenderWidget.volume.cmap = Colormap(table,controls=pos)
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

    @property
    def vol(self):
        return self.volumeRenderWidget.vol
    #
    # @volumeRenderWidget.connect
    # def on_frame(self,event):
    #     self.volumeRenderWidget.cam1.auto_roll

    def writevideo(self,fps=25):
        writer = imageio.save('foo.mp4', fps=25)
        self.volumeRenderWidget.events.draw.connect(lambda e: writer.append_data(self.render()))
        self.volumeRenderWidget.events.close.connect(lambda e: writer.close())


class VolumeRenderWidget(scene.SceneCanvas):

    def __init__(self,vol=None,path=None,size=(800,600),show=False):
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


    def setVolume(self,vol = None, path = None, sliceobj = None):
        print 'slice:',sliceobj

        if vol is None:
            vol=self.vol

        if path is not None:
            if '*' in path:
                vol=loader.loadimageseries(path)
            elif os.path.splitext(path)[-1]=='.npy':
                vol=loader.loadimage(path)
            else:
                vol=loader.loadtiffstack(path)
            self.vol=vol

        if vol is None:
            return

        if slice is not None:
            print 'preslice:',vol.shape
            slicevol=self.vol[sliceobj]
            print 'postslice:',vol.shape
        else:
            slicevol=self.vol



        # Set whether we are emulating a 3D texture
        emulate_texture = False

        # Create the volume visuals
        if self.volume is None:
            self.volume = scene.visuals.Volume(slicevol, parent=self.view.scene,emulate_texture=emulate_texture)
            self.volume.method='translucent'
        else:
            self.volume.set_data(slicevol)
            self.volume._create_vertex_data() #TODO: Try using this instead of slicing array?


        # Translate the volume into the center of the view (axes are in strange order for unkown )
        self.volume.transform = scene.STTransform(translate=(-vol.shape[2]/2,-vol.shape[1]/2,-vol.shape[0]/2))





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
        bounds=sorted(self.item.gradient.ticks.values())
        bounds=(bounds[0]*self.item.region.getRegion()[1],bounds[1]*self.item.region.getRegion()[1])
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


scene.visuals.Volume=VolumeVisual


class ProcessViewer(QtGui.QTabWidget):
    """
    Viewer class to define run settings for a full tomography dataset reconstruction job. Has tab for local run settings
    and tab for remote job settins.
    """

    sigRunClicked = QtCore.Signal(tuple, tuple, str, str, int, int)

    def __init__(self, path, dim, parent=None):
        super(ProcessViewer, self).__init__(parent=parent)
        self.setTabPosition(QtGui.QTabWidget.West)
        s = QtGui.QSplitter(QtCore.Qt.Horizontal)
        w = QtGui.QWidget()
        w.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        w.setContentsMargins(0,0,0,0)
        l = QtGui.QGridLayout()
        l.setContentsMargins(0,0,0,0)
        l.setSpacing(0)

        path, name = os.path.split(path)
        name = 'RECON_' + name.split('.')[0]
        out_path = os.path.join(path, name)

        # Create Local Parameter Tree
        self.localparamtree = pg.parametertree.ParameterTree(showHeader=False)
        precon, prun, pspecs = self.setupParams(dim, out_path)
        self.reconsettings = pg.parametertree.Parameter.create(name='Reconstruction Settings', type='group',
                                                               children=precon)
        self.localparamtree.setParameters(self.reconsettings, showTop=True)
        self.localsettings = pg.parametertree.Parameter.create(name='Run Settings', type='group', children=prun)
        self.localparamtree.addParameters(self.localsettings, showTop=True)
        self.localspecs = pg.parametertree.Parameter.create(name='Local Specifications', type='group', children=pspecs)
        self.localparamtree.addParameters(self.localspecs, showTop=True)

        l.addWidget(self.localparamtree, 0, 0, 1, 2)

        # Run and cancel push buttons
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_34.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.runButton = QtGui.QPushButton()
        self.runButton.setIcon(icon)
        self.runButton.setFlat(True)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_41.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.cancelButton = QtGui.QPushButton()
        self.cancelButton.setIcon(icon)
        self.cancelButton.setFlat(True)

        l.addWidget(self.runButton, 1, 0, 1, 1)
        l.addWidget(self.cancelButton, 1, 1, 1, 1)

        w.setLayout(l)
        s.addWidget(w)

        # Text Browser for console
        self.local_console = QtGui.QTextBrowser()
        self.local_console.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        s.addWidget(self.local_console)

        self.addTab(s, 'Local')
        self.addTab(QtGui.QWidget(), 'Remote')

        # Wire up buttons
        self.runButton.clicked.connect(self.runButtonClicked)

        # Wire up parameters
        self.reconsettings.param('Browse').sigActivated.connect(
            lambda: self.reconsettings.param('Output Name').setValue(str(QtGui.QFileDialog.getSaveFileName(self,
                                                                  'Save reconstruction as', out_path)[0])))

        sinostart = self.reconsettings.param('Start Sinogram')
        sinoend = self.reconsettings.param('End Sinogram')
        sinostep = self.reconsettings.param('Step Sinogram')
        nsino = lambda: (sinoend.value() - sinostart.value() + 1) // sinostep.value()
        chunks = self.localsettings.param('Sinogram Chunks')
        sinos = self.localsettings.param('Sinograms per Chunk')
        chunkschanged = lambda: sinos.setValue(np.round((nsino()) // chunks.value()), blockSignal=sinoschanged)
        sinoschanged = lambda: chunks.setValue((nsino() - 1)// sinos.value() + 1,  blockSignal=chunkschanged)
        chunks.sigValueChanged.connect(chunkschanged)
        sinos.sigValueChanged.connect(sinoschanged)
        sinostart.sigValueChanged.connect(chunkschanged)
        sinoend.sigValueChanged.connect(chunkschanged)
        sinostep.sigValueChanged.connect(chunkschanged)

        chunks.setValue(1)

    def setupParams(self, dim, path):
        # Local Recon Settings
        precon = [{'name': 'Start Sinogram', 'type': 'int', 'value': 0, 'default': 0},
                  {'name': 'Step Sinogram', 'type': 'int', 'value': 1, 'default': 1},
                  {'name': 'End Sinogram', 'type': 'int', 'value': dim[1], 'default': dim[1]},
                  {'name': 'Start Projection', 'type': 'int', 'value': 0, 'default': 0},
                  {'name': 'Step Projection', 'type': 'int', 'value': 1, 'default': 1},
                  {'name': 'End Projection', 'type': 'int', 'value': dim[0], 'default': dim[0]},
                  {'name': 'Ouput Format', 'type': 'list', 'values': [ 'TIFF (.tiff)', 'SPOT HDF5 (.h5)'],
                   'default': 'TIFF (.tiff)'},
                  {'name': 'Output Name', 'type': 'str', 'value': path, 'default': path},
                  {'name': 'Browse', 'type': 'action'},
                  ]
        # Local Run Settings
        total, available = self.memory()
        cores = self.cores()
        prun = [{'name': 'Cores', 'type': 'int', 'value': cores, 'default': None},
                  {'name': 'Sinogram Chunks', 'type': 'int', 'value': 0, 'default': 1},
                  {'name': 'Sinograms per Chunk', 'type': 'int', 'value': 0, 'default': 1}]
        # Local Specifications
        # siPrefix probably does not use base 2. Oh well memory will be an estimate
        pspecs = [{'name': 'Total Cores', 'type': 'int', 'value': cores, 'readonly': True},
                  {'name': 'Total Memory', 'type': 'float', 'value': total, 'suffix': 'B', 'siPrefix': True,
                   'readonly': True},
                  {'name': 'Available Memory', 'type': 'float', 'value': available, 'suffix': 'B', 'siPrefix': True,
                   'readonly': True}]
        return precon, prun, pspecs

    @staticmethod
    def memory():
        memory = psutil.virtual_memory()
        return memory.total, memory.available

    @staticmethod
    def cores():
        return psutil.cpu_count()

    def log2local(self, msg):
        self.local_console.insertPlainText(msg)

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
                                self.localsettings.child('Cores').value())


class ReconstructionViewer(VolumeViewer):
    def __init__(self, paths=None, data=None, *args, **kwargs):
        super(ReconstructionViewer, self).__init__()


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
            if False in [np.array_equal(arraylist[0].shape, array.shape) for array in arraylist[1:]]:
                raise ValueError('All arrays in arraylist must have the same dimensions')
            elif False in [arraylist[0].dtype == array.dtype for array in arraylist[1:]]:
                raise ValueError('All arrays in arraylist must have the same data type')
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
        if arr.shape != tuple(self.shape[1:]):
            raise ValueError('Array shape must be {0}, got shape {1}'.format(self.shape[1:], arr.shape))
        elif self.dtype is not None and arr.dtype != self.dtype:
            raise ValueError('Array must be of type {}'.format(self.dtype))
        super(ArrayDeque, self).append(arr)

    def appendleft(self, arr):
        if arr.shape != tuple(self.shape[1:]):
            raise ValueError('Array shape must be {0}, got shape {1}'.format(self.shape[1:], arr.shape))
        elif self.dtype is not None and arr.dtype != self.dtype:
            raise ValueError('Array must be of type {}'.format(self.dtype))
        super(ArrayDeque, self).appendleft(arr)

    def __getitem__(self, item):
        if type(item) is list and isinstance(item[0], slice):
            dq_item = item.pop(0)
            if isinstance(dq_item, slice):
                dq_item = dq_item.stop if dq_item.stop is not None else dq_item.start if dq_item.start is not None else 0
            return super(ArrayDeque, self).__getitem__(dq_item).__getitem__(item)
        else:
            return super(ArrayDeque, self).__getitem__(item)


class ImageView(pg.ImageView):
    """
    Subclass of PG ImageView to correct z-slider signal behavior.
    """
    def keyPressEvent(self, ev):
        super(ImageView, self).keyPressEvent(ev)
        self.timeLineChanged()

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
