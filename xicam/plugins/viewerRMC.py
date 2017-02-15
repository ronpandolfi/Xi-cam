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
from daemon.daemon import daemon
import multiprocessing
import Queue
import glob

# TODO: Remove LUT bar from RMC Timeline (its binary anyways)
# TODO: Check mask behavior


# drag/drop taken from tomography plugin
import platform
op_sys = platform.system()

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

    name = "HipRMC"

    #center widget is something, rightwidget is none, and leftwidget inherits the default from parent base.plugin
    def __init__(self, *args, **kwargs):


        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabClose)
        self.rightwidget = None

        self.threadWorker = threads.Worker(Queue.Queue())
        self.threadWorker.pool.setExpiryTimeout(1)
        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent

        super(plugin, self).__init__(*args, **kwargs)




    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if op_sys == 'Darwin':
                fname = str(NSURL.URLWithString_(str(url.toString())).filePathURL().path())
            else:
                fname = str(url.toLocalFile())
            if os.path.isfile(fname):
                self.openfiles([fname])
            e.accept()

    def dragEnterEvent(self, e):
        e.accept()


    def openfiles(self, paths):

        """
        Overrides inherited 'openfiles' method. Used for opening single image
        """
        self.activate()
        view_widget = inOutViewer(paths = paths, worker = self.threadWorker)
        self.centerwidget.addTab(view_widget, os.path.basename(paths[0]))
        self.centerwidget.setCurrentWidget(view_widget)
        view_widget.drawCameraLocation(view_widget.orig_view, view_widget.cameraLocation)

    def opendirectory(self, folder, operation=None):
        """
        Overrides inherited 'opendirectory' method. Used for opening hiprmc output folders.
        """
        self.activate()
        if type(folder) is list:
            folder = folder[0]
        view_widget = inOutViewer(None, self.threadWorker,)
        self.centerwidget.addTab(view_widget, os.path.basename(folder))
        self.centerwidget.setCurrentWidget(view_widget)

        # check for input image and load into plugin if it exists
        input = glob.glob(os.path.join(folder, 'input_image.tif'))
        if input:
            view_widget.orig_image = np.transpose(loader.loadimage(input[0]))
            if len(view_widget.orig_image.shape) > 2: # gets ride of extra dimensions if there are any
                view_widget.orig_image = np.transpose(view_widget.orig_image).swapaxes(0,1)
                while len(view_widget.orig_image.shape) > 2:
                    view_widget.orig_image = view_widget.orig_image[:,:,0]
            view_widget.orig_view.setImage(view_widget.orig_image)
            view_widget.orig_view.autoRange()

            view_widget.drawROI(0, 0, view_widget.orig_image.shape[0], view_widget.orig_image.shape[1], 'r',
                         view_widget.orig_view.getImageItem().getViewBox())

        view_widget.rmc_view = rmc.rmcView(folder)
        view_widget.rmc_view.findChild(QtGui.QTabBar).hide()
        view_widget.rmc_view.setContentsMargins(0, 0, 0, 0)
        view_widget.image_holder.addWidget(view_widget.rmc_view)

        view_widget.fft_view = rmc.fftView()
        view_widget.fft_view.open_from_rmcView(view_widget.rmc_view.image_list)
        view_widget.fft_view.setContentsMargins(0, 0, 0, 0)
        view_widget.image_holder.addWidget(view_widget.fft_view)

        view_widget.image_holder.setCurrentIndex(2)


    def tabClose(self,index):
        self.centerwidget.widget(index).deleteLater()

class LogViewer(pg.ImageView):
    """
    Class to view images with log intensity
    """

    def setImage(self,*args,**kwargs):
        super(LogViewer, self).setImage(*args,**kwargs)

        # levelmin = np.log(self.levelMin)/np.log(1.5)
        # levelmax = np.log(self.levelMax)/np.log(1.5)
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
            Holds and emits a signal when fHipRMc done processing
        interrupt : bool
            flag - set true if rmc processing was interrupted; affects post-rmc processes
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
        headings : QtGui.QTabBar
            Displays name of corresponding tab of image_holder

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
        self.interrupt = False
        self.cameraLocation = config.activeExperiment.center
        self.rmc_view= None
        self.edited_image = None
        self.worker = worker


        # holders for original and edited images
        self.orig_view = LogViewer()
        self.orig_view.setContentsMargins(0,0,0,0)
        self.edited_view = LogViewer()
        self.edited_view.setContentsMargins(0,0,0,0)

        if type(paths) == list:
            self.path = paths[0]
        else:
            self.path = paths

        self.image_holder = QtGui.QStackedWidget()
        self.image_holder.setContentsMargins(0,0,0,0)

        # configuring right widget
        sideWidget = QtGui.QWidget()
        sideWidgetFormat = QtGui.QVBoxLayout()
        sideWidgetFormat.setContentsMargins(0, 0, 0, 0)

        # if paths is None, inOutViewer will only hold HipRMC output and the images/parameter table are not necessary
        if paths is not None:

            self.orig_image = np.transpose(loader.loadimage(self.path))
            if len(self.orig_image.shape) > 2:
                self.orig_image = np.transpose(self.orig_image).swapaxes(0,1)
                while len(self.orig_image.shape) > 2:
                    self.orig_image = self.orig_image[:,:,0]
            self.orig_view.setImage(self.orig_image)
            self.orig_view.autoRange()
            try:
                start_size = max(self.orig_image.shape)/10
            except ValueError:
                msg.showMessage("Image must be 2-D")

            scatteringHolder = QtGui.QStackedWidget()

            image_name = self.path.split('/')[-1].split('.')[0]
            self.scatteringParams = pt.ParameterTree()
            params = [{'name': 'Num tiles', 'type': 'int', 'value': 1, 'default': 1},
                      {'name': 'Loading factor', 'type': 'float', 'value': 0.5, 'default': 0.5},
                      {'name': 'Scale factor', 'type': 'int', 'value': 32, 'default': 32},
                      {'name': 'Numsteps factor', 'type': 'int', 'value': 100, 'default': 100},
                      {'name': 'Model start size', 'type': 'int', 'value': start_size},
                      {'name': 'Save name', 'type': 'str', 'value': 'hiprmc_' + image_name},
                      {'name': 'Mask image', 'type': 'str'}]
            self.configparams = pt.Parameter.create(name='Configuration', type='group', children=params)
            self.scatteringParams.setParameters(self.configparams, showTop=False)
            scatteringHolder.addWidget(self.scatteringParams)

            # # is there a better way to check for correct dimensions?
            # if len(self.orig_image.shape) > 2:
            #     shape = (self.orig_image.shape[1], self.orig_image.shape[2])
            # else:
            #     shape = self.orig_image.shape

            self.drawROI(0, 0, self.orig_image.shape[0], self.orig_image.shape[1], 'r',
                         self.orig_view.getImageItem().getViewBox())

            scatteringHolder.setFixedHeight(300)
            sideWidgetFormat.addWidget(scatteringHolder)

        centerButton = QtGui.QPushButton("Center camera location")
        runButton = QtGui.QPushButton("Run RMC processing")
        stopButton = QtGui.QPushButton("Stop RMC")
        sideWidgetFormat.addSpacing(5)
        sideWidgetFormat.addWidget(centerButton)
        sideWidgetFormat.addSpacing(5)
        sideWidgetFormat.addWidget(runButton)
        sideWidgetFormat.addSpacing(5)
        sideWidgetFormat.addWidget(stopButton)
        sideWidgetFormat.addStretch(10)
        sideWidget.setLayout(sideWidgetFormat)

        # connect buttons to processing
        centerButton.clicked.connect(self.center)
        runButton.clicked.connect(self.runRMC)
        stopButton.clicked.connect(self.stop_threads)

        # tab headings for main widget
        self.headings = QtGui.QTabBar(self)
        self.headings.addTab('Original Image')
        self.headings.addTab('Recentered Image')
        self.headings.addTab('RMC Timeline')
        self.headings.addTab('FFT RMC Timeline')
        self.headings.setShape(QtGui.QTabBar.TriangularSouth)


        leftWidget = QtGui.QWidget()
        sidelayout = QtGui.QVBoxLayout()
        sidelayout.addWidget(self.image_holder)
        sidelayout.addWidget(self.headings)
        leftWidget.setLayout(sidelayout)

        fullPlugin = QtGui.QSplitter()
        fullPlugin.addWidget(leftWidget)
        fullPlugin.addWidget(sideWidget)

        h = QtGui.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(fullPlugin)
        self.setLayout(h)

        self.image_holder.addWidget(self.orig_view)
        self.image_holder.addWidget(self.edited_view)

        self.headings.currentChanged.connect(self.currentChanged)
        self.image_holder.currentChanged.connect(self.headings.setCurrentIndex)


    def currentChanged(self, index):
        """
        Slot to receive centerwidget's currentchanged signal when a new tab is selected
        """
        if self.image_holder.widget(index):
            self.image_holder.setCurrentIndex(index)
        else:
            self.headings.setCurrentIndex(self.image_holder.currentIndex())


    def center(self, sample=True):
        """
        Slot to receive signal when user requests a centered image. Performs centering and writes new image into
        current working directory

        Parameters
        ----------
        sample: boolean, optional
            flag whether image to be centered is the original sample or the mask image
        """

        if sample:
            if self.edited_image is not None:
                msg.showMessage('Image already centered.')
                return

            image = self.orig_image
        else:
            image = self.mask

        #resize image so that it's in center and displays output if a sample image
        xdim = int(image.shape[0])
        ydim = int(image.shape[1])

        newx = xdim + 2*abs(self.cameraLocation[0]-xdim/2)
        newy = ydim + 2*abs(self.cameraLocation[1]-ydim/2)
        self.new_dim = max(newx,newy)

        self.edited_image = np.ones((self.new_dim,self.new_dim),dtype = np.int)
        new_center = (self.new_dim/2,self.new_dim/2)

        lowleft_corner_x = int(new_center[0]-self.cameraLocation[0])
        lowleft_corner_y = int(new_center[1]-self.cameraLocation[1])

        self.edited_image[lowleft_corner_x:lowleft_corner_x+xdim,lowleft_corner_y: lowleft_corner_y+ydim] = image

        # save image
        if sample:
            self.write_path = self.path

            if self.write_path.endswith('.tif'):
                self.write_path = self.write_path[:-4] + '_centered.tif'
            else:
                self.write_path += '_centered.tif'
            self.write_path_sample = self.write_path

            img = tifimage.tifimage(np.rot90((self.edited_image.astype(float) /
                                              self.edited_image.max() * 2 ** 16).astype(np.int16)))
        else:
            self.write_path = self.mask_path

            if self.write_path.endswith('.tif'):
                self.write_path = self.write_path[:-4] + '_centered.tif'
            else:
                self.write_path += '_centered.tif'
            self.write_path_mask = self.write_path

            img = tifimage.tifimage(np.rot90(self.edited_image.astype(float)))




        img.write(self.write_path)

        if sample:
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

        roi = pg.RectROI((xpos,ypos),(xdim,ydim),movable = False,removable=True, snapSize = 100000000, scaleSnap=True)
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
            msg.showMessage('Error: must center image before running HipRMC',timeout = 0)
            msg.clearMessage()
            return



        params = self.configparams

        hig_info = {'hipRMCInput': {'instrumentation': {'inputimage': "{}".format(self.write_path_sample),
                                             'imagesize': [self.new_dim, self.new_dim ],
                                             'numtiles': params.child('Num tiles').value(),
                                             'loadingfactors': [params.child('Loading factor').value()]},
                         'computation': {'runname': "{}".format(params.child('Save name').value()),
                                         'modelstartsize': [params.child('Model start size').value(),
                                                            params.child('Model start size').value()],
                                         'numstepsfactor': params.child('Numsteps factor').value(),
                                         'scalefactor': params.child('Scale factor').value()}}}

        self.mask_path = params.child('Mask image').value()
        if self.mask_path and self.mask_path != "None":
            self.mask = np.transpose(loader.loadimage(self.mask_path))
            self.center(False)
            hig_info['hipRMCInput']['instrumentation']['maskimage'] = "{}".format(self.write_path_mask)

        h = hig.hig(**hig_info)
        self.hig_name = os.path.join(os.path.abspath('.'), params.child('Save name').value())

        if not self.hig_name.endswith('.hig'):
            self.hig_name += '.hig'

        # write hig file to disk
        h.write(self.hig_name)
        self.save_name = params.child('Save name').value()
        self.start_time = time.time()

        # starts filewatcher to watch for new hiprmc folder, and the HipRMC job
        # also starts worker if it is not already running
        process = threads.RunnableMethod(method = self.run_RMCthread, finished_slot = self.RMC_done,
                                         except_slot=self.hiprmc_not_found)
        self.file_watcher = NewFolderWatcher(path=os.path.abspath("."), experiment=None)

        # when filewatcher gets rmc folder, it passes it to self.start_watcher to start another watcher
        self.file_watcher.sigFinished.connect(self.start_watcher)
        watcher = threads.RunnableMethod(method=self.file_watcher.run,)
        self.worker.queue.put(watcher)
        self.worker.queue.put(process)

        if not self.worker.isRunning():
            self.worker.start()


    def run_RMCthread(self):
        """
        Slot to receive signal to run HipRMC as subprocess on background thread
        """
        if os.path.isfile('./hiprmc/bin/hiprmc'):
            self.proc = subprocess.Popen(['./hiprmc/bin/hiprmc', self.hig_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.output, self.err = self.proc.communicate()
        else:
            raise Exception

    def hiprmc_not_found(self):
        """
        Slot to receive exception signal in case hiprmc is not found.
        """
        msg.showMessage('Cannot find HipRMC executable. Cannot run HipRMC.')
        QtGui.QMessageBox.critical(self, 'Error', 'Cannot find HipRMC executable. Cannot run HipRMC.')


    def start_watcher(self, folder):
        """
        Slot to receive signal to start looking for hiprmc output in 'folder'

        Parameters
        ----------
        folder : str, unicode
            folder to watch for hiprmc output
        """

        self.rmc_folder = folder

        # create and add rmcView to main plugin widget
        self.rmc_view = rmc.rmcView(self.rmc_folder)
        self.rmc_view.findChild(QtGui.QTabBar).hide()
        self.rmc_view.setContentsMargins(0, 0, 0, 0)
        self.image_holder.addWidget(self.rmc_view)

        # fftview to view fft transforms of output images
        self.fft_view = rmc.fftView()
        self.fft_view.open_from_rmcView(self.rmc_view.image_list)
        self.fft_view.setContentsMargins(0, 0, 0, 0)
        self.image_holder.addWidget(self.fft_view)

        # starts another filewatcher to watch rmc folder and add images to rmc_view as they are output
        self.rmc_watcher = HipRMCWatcher(path=self.rmc_folder,experiment=self.rmc_view)
        self.emitter.sigFinished.connect(self.rmc_watcher.stop)
        watchRMC = threads.RunnableMethod(method=self.rmc_watcher.run,)
        self.worker.queue.put(watchRMC)
        self.rmc_watcher.sigCallback.connect(self.add_images)
        if self.rmc_view.image_list:
           self.image_holder.setCurrentIndex(2)

        if not self.worker.isRunning():
            self.worker.start()

    def add_images(self, root):
        """
        Looks in file given by 'root' and adds images to rmcView and fftView

        Parameters
        ----------
        root : str, unicode
            path to check for images to add to rmcView and fftView
        """
        if not self.rmc_view.image_list:
            self.image_holder.setCurrentIndex(2)
        self.rmc_view.addNewImages(root=root)
        self.fft_view.open_from_rmcView(self.rmc_view.image_list)


    def stop_threads(self):
        """
        Stops all background threads (worker, any filewatcher, hiprmc
        """
        try:
            self.worker.stop()
            self.file_watcher.stop()
            self.rmc_watcher.stop()
            self.interrupt = True
            self.proc.terminate()
        except IOError:
            pass

    @QtCore.Slot()
    def RMC_done(self):
        """
        Slot to receive signal when HipRMC calculation is done. Emits signal to main thread to load output
        """
        run_time = time.time() - self.start_time

        if not self.interrupt:
            os.rename(self.hig_name, '{}/{}.hig'.format(self.rmc_folder, self.save_name))

            # write output of RMC to file in hiprmc output folder
            output_path = self.rmc_folder + "/{}_rmc_output.txt".format(self.save_name)
            with open(output_path, 'w') as txt:
                txt.write(self.output)

            msg.showMessage('HipRMC complete. Run time: {:.2f} s'.format(run_time))
            self.emitter.sigFinished.emit()
        else:
            try:
                os.rename(self.hig_name, '{}/{}.hig'.format(self.rmc_folder, self.save_name))
            except OSError:
                pass

            msg.showMessage('HipRMC interrupted by user. Run time: {:.2f} s'.format(run_time))

        self.interrupt = False


class NewFolderWatcher(daemon):

    """
    Daemon subclass to watch for hiprmc folder when it is created
    """

    sigFinished = QtCore.Signal(str)

    def process(self, path, files):

        if files:
            folder = path + '/' + files[0]
            self.sigFinished.emit(folder)
            self.stop()

    def run(self):

        if self.procold:
            self.process(self.path, self.childfiles)

        try:
            while not self.exiting:
                time.sleep(.1)
                self.checkdirectory()  # Force update; should not have to do this -.-
        except KeyboardInterrupt:
            pass

    def stop(self):
        self.exiting = True
        print ("thread stop - %s" % self.exiting)

    def __del__(self):
        self.exiting = True
        self.wait()


    def checkdirectory(self):
        """
        Checks a directory for new files, comparing what files are there now vs. before
        """
        updatedchildren = set(os.listdir(self.path))
        newchildren = updatedchildren - self.childfiles
        self.childfiles = updatedchildren
        self.process(self.path, list(newchildren))


class HipRMCWatcher(daemon):

    """
    Daemon subclass to watch hiprmc output folder. Emits sigCallback whenever folder contents change
    """

    num_cores = multiprocessing.cpu_count()
    sigCallback = QtCore.Signal(str)

    def run(self):

        if self.procold:
            self.process(self.path, self.childfiles)

        try:
            while not self.exiting:
                time.sleep(.1)
                self.checkdirectory()  # Force update; should not have to do this -.-
        except KeyboardInterrupt:
            pass

    def checkdirectory(self):
        """
        Checks a directory for new files, comparing what files are there now vs. before
        """
        updatedchildren = set(os.listdir(self.path))
        newchildren = updatedchildren - self.childfiles
        self.childfiles = updatedchildren
        self.process(self.path, list(newchildren))


    def process(self, path, files):
        if files:
            self.sigCallback.emit(path)



