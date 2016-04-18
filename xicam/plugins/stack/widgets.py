# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# vispy: gallery 2
#
# Adapted for use as a widget by Ron Pandolfi
# volumeViewer.getHistogram method borrowed from PyQtGraph

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

from itertools import cycle

import numpy as np

from PySide import QtGui,QtCore
from vispy import app, scene, io
from vispy.color import Colormap, BaseColormap,ColorArray
from pipeline import loader
import pyqtgraph as pg
import imageio
import os


class tomoWidget(QtGui.QWidget):

    def __init__(self,paths=None,data = None, *args,**kwargs):
        super(tomoWidget, self).__init__()

        self.paths = paths
        self.data = data

        self.viewstack = QtGui.QStackedWidget()

        self.viewmode = QtGui.QTabBar()
        self.viewmode.addTab('Projection')  # TODO: Add icons!
        self.viewmode.addTab('Sinogram')
        self.viewmode.addTab('Preview')
        self.viewmode.addTab('Process')
        self.viewmode.addTab('Reconstruction')
        self.viewmode.setShape(QtGui.QTabBar.TriangularSouth)

        self.projectionViewer = projectionViewer(paths=paths, data=data)
        self.viewstack.addWidget(self.projectionViewer)

        self.sinogramViewer = sinogramViewer(paths=paths, data=data)
        self.viewstack.addWidget(self.sinogramViewer)

        self.previewViewer = previewViewer(paths=paths, data=data)
        self.viewstack.addWidget(self.previewViewer)

        self.processViewer = processViewer(paths=paths, data=data)
        self.viewstack.addWidget(self.processViewer)

        self.reconstructionViewer = reconstructionViewer(paths=paths, data=data)
        self.viewstack.addWidget(self.reconstructionViewer)

        l = QtGui.QVBoxLayout()
        l.setContentsMargins(0,0,0,0)
        l.addWidget(self.viewstack)
        l.addWidget(self.viewmode)
        self.setLayout(l)

        self.viewmode.currentChanged.connect(self.currentChanged)

    def currentChanged(self,index):
        self.viewstack.setCurrentIndex(index)



class projectionViewer(pg.ImageView):
    xy=0
    yz=1
    xz=2

    def __init__(self, paths=None, data=None, *args, **kwargs):
        super(projectionViewer, self).__init__()

        if data is not None:
            self.data=data

        elif paths is not None and len(paths):
            self.data = loader.loadimage(paths)

        self.data = np.swapaxes(self.data,1,2) # Not sure why this is necessary

        self.setImage(self.data)#, axes={'t':0, 'x':2, 'y':1, 'c':3})

        self.autoLevels()
        self.getView().invertY(True)

    def setDisplayAxes(self,axes):
        if axes==self.yz:
            self.setImage(self.data)#, axes={'t':0, 'x':2, 'y':1, 'c':3})
        elif axes==self.xz:
            self.setImage(np.swapaxes(self.data,0,2)) #, axes={'t':2, 'x':0, 'y':1, 'c':3})


class sinogramViewer(projectionViewer):

    def __init__(self,*args,**kwargs):
        super(sinogramViewer, self).__init__(*args, **kwargs)
        self.setDisplayAxes(self.xz)

        self.timeLine.setValue(self.tVals[len(self.tVals)//2])


class volumeViewer(QtGui.QWidget):

    sigImageChanged=QtCore.Signal()

    def __init__(self,path=None,data=None,*args,**kwargs):
        super(volumeViewer, self).__init__()

        self.levels=[0,1]

        l = QtGui.QHBoxLayout()
        l.setContentsMargins(0,0,0,0)
        l.setSpacing(0)

        self.volumeRenderWidget=volumeRenderWidget()
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


class volumeRenderWidget(scene.SceneCanvas):

    def __init__(self,vol=None,path=None,size=(800,600),show=False):
        super(volumeRenderWidget, self).__init__(keys='interactive',size=size,show=show)

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

class previewViewer(pg.ImageView):
    def __init__(self, paths=None, data=None, *args, **kwargs):
        super(previewViewer, self).__init__()

class processViewer(QtGui.QWidget):
    def __init__(self, paths=None, data=None, *args, **kwargs):
        super(processViewer, self).__init__()

class reconstructionViewer(volumeViewer):
    def __init__(self, paths=None, data=None, *args, **kwargs):
        super(reconstructionViewer, self).__init__()
