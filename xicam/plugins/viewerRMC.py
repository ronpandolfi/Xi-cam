import os
from PySide import QtGui, QtCore
from xicam.plugins import base
from xicam import config
from pyqtgraph import parametertree as pt
from fabio import tifimage
from pipeline import loader, hig, msg
import pyqtgraph as pg
import numpy as np
import subprocess
import xicam.RmcView as rmc
import time
from xicam import threads
import Queue

"""
Bugs:
    1. User can resize ROI after recentering
    2. Centering/running RMC causes gui to return to original image tab, instead of staying on the current tab
        or going to the tab relevant for the button pressed (has a quickfix)
"""

class plugin(base.plugin):

    """
    HipRMC plugin class - centers images, calls HipRMC as subprocess, then displays output

    NOTE: Running HipRMC assumes that there is a HipRMC folder containing the executable in the working
          directory. If this is not the case, you will need to change the path in inOutViewer.run_RMC where
          HipRMC is called as an executable.


    Attributes
    ----------
    centerwidget : QtGui.QTabWidget
        Standard centerwidget overriding base.plugin. QTabWidget that holds instances of inOutviewer for images and
        HipRMC output
    threadWorker : threads.Worker
        Contains qtCore.QThreadpool on which to run HipRMC jobs, and a queue to hold the jobs


    Parameters
    ----------
    args
        Additional arguments. Not really used
    kwargs
        Additional keyword arguments. Not really used
    """

    name = "ViewerRMC"

    #center widget is something, rightwidget is none, and leftwidget inherits the default from parent base.plugin
    def __init__(self, *args, **kwargs):


        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabClose)
        self.rightwidget = None

        self.threadWorker = threads.Worker(Queue.Queue())
        self.threadWorker.pool.setExpiryTimeout(1)


        super(plugin, self).__init__(*args, **kwargs)


    def openfiles(self, paths):
        self.activate()
        view_widget = inOutViewer(paths = paths, worker = self.threadWorker)
        self.centerwidget.addTab(view_widget, os.path.basename(paths[0]))
        self.centerwidget.setCurrentWidget(view_widget)
        view_widget.drawCameraLocation(view_widget.orig_view,view_widget.cameraLocation)


    def tabClose(self,index):
        self.centerwidget.widget(index).deleteLater()

class LogViewer(pg.ImageView):

    def __init__(self):
        super(LogViewer, self).__init__()

    def setImage(self,*args,**kwargs):
        super(LogViewer, self).setImage(*args,**kwargs)

        levelmin = np.log(self.levelMin)
        levelmax = np.log(self.levelMax)
        if np.isnan(levelmin): levelmin = 0
        if np.isnan(levelmax): levelmax = 1
        if np.isinf(levelmin): levelmin = 0

        self.ui.histogram.setLevels(levelmin, levelmax)



class inOutViewer(QtGui.QWidget, ):
    def __init__(self, paths, worker, parent=None):
        """
           Class that holds image to be processed by HipRMC, image after it has been centered, and HipRMC output

           Attributes
           ----------
           emitter : threads.Emitter
               Holds and emits a signal when HipRMc done processing
           cameraLocation : tuple
               2-tuple (x,y) of camera location on input image
           rmcView : ximcam.RmcView.rncView
               Timeline viewer which holds and displays HipRMC output
           orig_image: np.array
               Original input image
           edited_image : np.array
               Image with camera location adjusted to its center
           orig_view : pyqtgraph.ImageView
               Holds the original image
           edited_view : pyqtgraph.ImageView
               Holds the image after camera location adjustment
           image_holder : QtGui.StackedWidget
               Main widget of plugin. Holds original and edited images, as well as HipRMC output, in tabs
           scatteringParams : pyqtgraph.parametertree
               Occupies right side of main widget. Holds configparams
           configparams : pyqtgraph.Parameter
               Class held by scatteringParams which holds parameter values for HipRMC
           output, err : str
               Output and error from HipRMC subprocess call

           Parameters
           ----------
           paths : str/list of str
               Path to input dataset
           worker: threads.Worker
               Worker which queues up jobs and runs them on a QtCore.QThreadpool
           parent : QtGui.QWidget
               parent widget
           args
               Additional arguments
           kwargs
               Additional keyword arguments
           """


        super(inOutViewer, self).__init__(parent=parent)

        self.emitter = threads.Emitter()

        layout = QtGui.QHBoxLayout()
        self.cameraLocation = config.activeExperiment.center

        # the next two will be filled as different functions are fun
        self.rmc_view= None
        self.edited_image = None

        self.worker = worker


        # load and display image
        self.orig_view = LogViewer()
        self.orig_view.setContentsMargins(0,0,0,0)
        if type(paths) == list:
            self.path = paths[0]
        else:
            self.path = paths

        self.orig_image = np.transpose(loader.loadimage(self.path))
        try:
            start_size = max(self.orig_image.shape)
        except ValueError:
            print "Image must be 2-D"


        self.image_holder = QtGui.QStackedWidget()
        self.image_holder.setContentsMargins(0,0,0,0)
        self.orig_view.setImage(self.orig_image)
        self.orig_view.autoRange()
        self.image_holder.addWidget(self.orig_view)

        # configuring right widget
        sideWidgetFormat = QtGui.QVBoxLayout()
        sideWidgetFormat.setContentsMargins(0, 0, 0, 0)


        self.scatteringParams = pt.ParameterTree()
        params = [{'name': 'Num tiles', 'type': 'int', 'value': 1, 'default': 1},
                  {'name': 'Loading factor', 'type': 'float', 'value': 0.5, 'default': 0.5},
                  {'name': 'Scale factor', 'type': 'int', 'value': 32, 'default': 32},
                  {'name': 'Numsteps factor', 'type': 'int', 'value': 100, 'default': 100},
                  {'name': 'Model start size', 'type': 'int', 'value': start_size},
                  {'name': 'Save name', 'type': 'str', 'value': 'processed'},
                  {'name': 'Mask image', 'type': 'str'}]
        self.configparams = pt.Parameter.create(name='Configuration', type='group', children=params)
        self.scatteringParams.setParameters(self.configparams, showTop=False)


        scatteringHolder = QtGui.QStackedWidget()
        scatteringHolder.addWidget(self.scatteringParams)
        scatteringHolder.setFixedHeight(300)

        centerButton = QtGui.QPushButton("Center camera location")
        runButton = QtGui.QPushButton("Run RMC processing")
        sideWidgetFormat.addWidget(scatteringHolder)
        sideWidgetFormat.addSpacing(5)
        sideWidgetFormat.addWidget(centerButton)
        sideWidgetFormat.addSpacing(5)
        sideWidgetFormat.addWidget(runButton)


        centerButton.clicked.connect(self.center)
        runButton.clicked.connect(self.runRMC)

        headings = QtGui.QTabBar(self)
        headings.addTab('Original Image')
        headings.addTab('Recentered Image')
        headings.addTab('RMC Timeline')
        headings.setShape(QtGui.QTabBar.TriangularSouth)

        self.drawROI(0,0,self.orig_image.shape[0],self.orig_image.shape[1], 'r',
                     self.orig_view.getImageItem().getViewBox())

        self.edited_view = LogViewer()
        self.image_holder.addWidget(self.edited_view)


        sidelayout = QtGui.QVBoxLayout()
        sidelayout.addWidget(self.image_holder)
        sidelayout.addWidget(headings)

        layout.addLayout(sidelayout,10)
        layout.addLayout(sideWidgetFormat,4)
        self.setLayout(layout)

        headings.currentChanged.connect(self.currentChanged)
        self.image_holder.currentChanged.connect(headings.setCurrentIndex)

    def currentChanged(self,index):
        """
        Slot to recieve centerwidgets currentchanged signal when a new tab is selected
        """

        self.image_holder.setCurrentIndex(index)


    def center(self):
        """
        Slot to receive signal when user requests a centered image. Performs centering and writes new image into
        current working directory
        """

        if self.edited_image is not None:
            self.image_holder.removeWidget(self.edited_view)
            self.edited_view = LogViewer()
            self.image_holder.addWidget(self.edited_view)

        #resize image so that it's in center
        #displays output on stackwidget


        xdim= self.orig_image.shape[0]
        ydim = self.orig_image.shape[1]

        newx = xdim + 2*abs(self.cameraLocation[0]-xdim/2)
        newy = ydim + 2*abs(self.cameraLocation[1]-ydim/2)
        self.new_dim = max(newx,newy)

        self.edited_image = np.ones((self.new_dim,self.new_dim),dtype = np.int)
        new_center = (self.new_dim/2,self.new_dim/2)

        lowleft_corner_x = new_center[0]-self.cameraLocation[0]
        lowleft_corner_y = new_center[1]-self.cameraLocation[1]

        self.edited_image[lowleft_corner_x:lowleft_corner_x+xdim,lowleft_corner_y: lowleft_corner_y+ydim] \
            = self.orig_image

        # save image
        self.write_path = self.path
        if self.write_path.endswith('.tif'):
            self.write_path = self.write_path[:-4]+'centered.tif'
        else:
            self.write_path += '_centered.tif'

        img = tifimage.tifimage(np.rot90((self.edited_image.astype(float)/
                                          self.edited_image.max()*2**16).astype(np.int16)))
        img.write(self.write_path)


        self.edited_view.setImage(self.edited_image)

        box = self.drawCameraLocation(self.edited_view,new_center)
        self.drawROI(lowleft_corner_x,lowleft_corner_y,xdim,ydim,'r', box)
        self.drawROI(0,0,self.new_dim,self.new_dim, 'b', box)

        # this is a temporary fix for a problem: pushing a button changes tab back to first
        self.image_holder.setCurrentIndex(1)

    def drawCameraLocation(self,box,location):
        """
        Draws camera location as dot on image

        Parameters
        ----------
        box : pyqtgraph.ImageView
            Class which holds the dot drawn by the function
        location: tuple
            2-tuple of camera location
        """


        cameraBox = box.getImageItem().getViewBox()
        cameraPlot = pg.ScatterPlotItem()
        cameraBox.addItem(cameraPlot)
        cameraPlot.setData([location[0]], [location[1]], pen=None,
                                symbol='o' , brush=pg.mkBrush('#FFA500'))

        return cameraBox


    def drawROI(self, xpos, ypos, xdim,ydim, color, view_box):
        """
        Draws camera location as dot on image

        Parameters
        ----------
        xpos, ypos: int
            Location of lower left corner of ROI
        xdim, ydim : int
            Size of ROI in x and y dimensions
        color : str
            Color of ROI
        view_box : pyqtgraph.ImageView
            Class which holds the ROI drawn by the function
        """

        roi = pg.RectROI((xpos,ypos),(xdim,ydim),movable = False,removable=True)
        roi.setPen(color = color)

        view_box.addItem(roi)


    def runRMC(self):
        """
        Slot to receive signal when user requests HipRMC calculation. Writes hig file of parameter values and
        calls HipRMC as subprocess
        """


        msg.showMessage('Running RMC for centered version of {}'.format(self.path), timeout=0)

        if self.rmc_view is not None:
            self.image_holder.removeWidget(self.rmc_view)

        if self.edited_image is None:
            msg.showMessage('Error: no image loaded',timeout = 0)
            msg.clearMessage()
            return



        params = self.configparams
        mask = params.child('Mask image').value()

        hig_info = {'hipRMCInput': {'instrumentation': {'inputimage': "{}".format(self.write_path),
                                             'maskimage': "{}".format(mask) if mask else '',
                                             'imagesize': [self.new_dim, self.new_dim ],
                                             'numtiles': params.child('Num tiles').value(),
                                             'loadingfactors': [params.child('Loading factor').value()]},
                         'computation': {'runname': "{}".format(params.child('Save name').value()),
                                         'modelstartsize': [params.child('Model start size').value(),
                                                            params.child('Model start size').value()],
                                         'numstepsfactor': params.child('Numsteps factor').value(),
                                         'scalefactor': params.child('Scale factor').value()}}}

        h = hig.hig(**hig_info)
        self.hig_name = './' + params.child('Save name').value()
        if not self.hig_name.endswith('.hig'):
            self.hig_name += '.hig'

        h.write(self.hig_name)
        self.save_name = self.configparams.child('Save name').value()
        self.start_time = time.time()

        process = threads.RunnableMethod(method = self.run_RMCthread,finished_slot = self.RMC_done)
        self.worker.queue.put(process)

        if not self.worker.isRunning():
            self.worker.start()

        # connects finished HipRMC to write/display
        self.emitter.sigFinished.connect(self.write_and_display)


    def run_RMCthread(self):
        """
        Slot to receive signal to run HipRMC as subprocess on background thread
        """

        proc = subprocess.Popen(['./hiprmc/bin/hiprmc', self.hig_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.output, self.err = proc.communicate()


    @QtCore.Slot()
    def RMC_done(self):
        """
        Slot to receive signal when HipRMC calculation is done. Emits signal to main thread to load output
        """
        run_time = time.time() - self.start_time
        msg.showMessage('HipRMC complete. Run time: {:.2f} s'.format(run_time))
        self.emitter.sigFinished.emit()


    @QtCore.Slot()
    def write_and_display(self):
        """
        Slot. Connected to RMC_done slot function. Loads images in rmc_viewer and writes HipRMC text output
        as txt file
        """
        # complicated way of finding and writing into folder name written by hiprmc
        ind = self.output.index(self.save_name)
        rmc_folder = './{}'.format(self.output[ind:].split("\n")[0])
        os.rename(self.hig_name, '{}/{}.hig'.format(rmc_folder, self.save_name))

        # write output of RMC to file in hiprmc output folder
        output_path = rmc_folder + "/{}_rmc_output.txt".format(self.save_name)
        with open(output_path, 'w') as txt:
            txt.write(self.output)

        # add rmcView to tabwidget
        self.rmc_view = rmc.rmcView(rmc_folder)
        self.rmc_view.findChild(QtGui.QTabBar).hide()
        self.rmc_view.setContentsMargins(0, 0, 0, 0)

        self.image_holder.addWidget(self.rmc_view)

        # this is a temporary fix for a problem: pressing either buttton changes tab back to first
        self.image_holder.setCurrentIndex(2)




