import os
from PySide import QtGui, QtCore
from xicam.plugins import base
from xicam import config
from pyqtgraph import parametertree as pt
from fabio import tifimage
# from PIL import Image
from pipeline import loader, hig, msg
#from hiprmc import hiprmc
import pyqtgraph as pg
import numpy as np
import subprocess
import xicam.RmcView as rmc

"""
Bugs:
    1. User can resize ROI after recentering
    2. Centering/running RMC causes gui to return to original image tab, instead of staying on the current tab
        or going to the tab relevant for the button pressed
"""

class plugin(base.plugin):

    # name can be changed upon request
    name = "ViewerRMC"

    #center widget is something, rightwidget is none, and leftwidget inherits the default from parent base.plugin
    def __init__(self, *args, **kwargs):


        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabClose)
        self.rightwidget = None

        # DRAG-DROP
        # self.centerwidget.setAcceptDrops(True)
        # self.centerwidget.dragEnterEvent = self.dragEnterEvent
        # self.centerwidget.dropEvent = self.dropEvent




        super(plugin, self).__init__(*args, **kwargs)


    def openfiles(self, paths):
        self.activate()
        view_widget = inOutViewer(paths = paths)
        self.centerwidget.addTab(view_widget, os.path.basename(paths[0]))
        self.centerwidget.setCurrentWidget(view_widget)
        view_widget.drawCameraLocation(view_widget.orig_view,view_widget.cameraLocation)


    # Do I need these??

    # def dropEvent(self, e):
    #     for url in e.mimeData().urls():
    #         if op_sys == 'Darwin':
    #             fname = str(NSURL.URLWithString_(str(url.toString())).filePathURL().path())
    #         else:
    #             fname = str(url.toLocalFile())
    #         if os.path.isfile(fname):
    #             self.openfiles([fname])
    #         e.accept()
    #
    # def dragEnterEvent(self, e):
    #     print(e)
    #     e.accept()


    def tabClose(self,index):
        self.centerwidget.widget(index).deleteLater()



class inOutViewer(QtGui.QWidget, ):
    def __init__(self, paths, parent=None):

        super(inOutViewer, self).__init__(parent=parent)

        layout = QtGui.QHBoxLayout()
        self.cameraLocation = config.activeExperiment.center
        self.rmc_view= None
        self.edited_image = None


        # load and display image
        self.orig_view = pg.ImageView(self)
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
                  {'name': 'Save Name', 'type': 'str', 'value': 'processed'}]
        self.configparams = pt.Parameter.create(name='Configuration', type='group', children=params)
        self.scatteringParams.setParameters(self.configparams, showTop=False)


        scatteringHolder = QtGui.QStackedWidget()
        scatteringHolder.addWidget(self.scatteringParams)
        scatteringHolder.setFixedHeight(300)
        scatteringHolder.setSizePolicy(QtGui.QSizePolicy.Fixed,QtGui.QSizePolicy.Fixed)

        centerButton = QtGui.QPushButton("Center camera location")
        runButton = QtGui.QPushButton("Run RMC processing")
        sideWidgetFormat.addWidget(scatteringHolder)
        sideWidgetFormat.addSpacing(5)
        sideWidgetFormat.addWidget(centerButton)
        sideWidgetFormat.addSpacing(5)
        sideWidgetFormat.addWidget(runButton)


        centerButton.clicked.connect(self.center)
        runButton.clicked.connect(self.runRMC)

        self.headings = QtGui.QTabBar(self)
        self.headings.addTab('Original Image')
        self.headings.addTab('Recentered Image')
        self.headings.addTab('RMC Timeline')
        self.headings.setShape(QtGui.QTabBar.TriangularSouth)

        self.drawROI(0,0,self.orig_image.shape[0],self.orig_image.shape[1], 'r',
                     self.orig_view.getImageItem().getViewBox())

        self.edited_view = pg.ImageView(self)
        self.image_holder.addWidget(self.edited_view)


        sidelayout = QtGui.QVBoxLayout()
        sidelayout.addWidget(self.image_holder)
        sidelayout.addWidget(self.headings)

        layout.addLayout(sidelayout,10)
        layout.addLayout(sideWidgetFormat,4)
        self.setLayout(layout)

        self.headings.currentChanged.connect(self.currentChanged)
        self.image_holder.currentChanged.connect(self.headings.setCurrentIndex)

    def currentChanged(self,index):
        self.image_holder.setCurrentIndex(index)


    def center(self):

        if self.edited_image is not None:
            self.image_holder.removeWidget(self.edited_view)
            self.edited_view = pg.ImageView(self)
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

        # this is a temporary fix for a bug: pushing a button changes tab back to first
        self.image_holder.setCurrentIndex(1)

    def drawCameraLocation(self,imageView_item,location):

        cameraBox = imageView_item.getImageItem().getViewBox()
        cameraPlot = pg.ScatterPlotItem()
        cameraBox.addItem(cameraPlot)
        cameraPlot.setData([location[0]], [location[1]], pen=None,
                                symbol='o' , brush=pg.mkBrush('#FFA500'))

        return cameraBox


    def drawROI(self, xpos, ypos, xdim,ydim, color, view_box):

        roi = pg.RectROI((xpos,ypos),(xdim,ydim),movable = False,removable=True)
        roi.setPen(color = color)

        view_box.addItem(roi)


    def runRMC(self):
        msg.showMessage('Running RMC for centered version of {}'.format(self.path), timeout=0)

        if self.rmc_view is not None:
            self.image_holder.removeWidget(self.rmc_view)

        if self.edited_image is None:
            msg.showMessage('Error: no image loaded',timeout = 0)
            msg.clearMessage()
            return



        params = self.configparams


        hig_info = {'hipRMCInput': {'instrumentation': {'inputimage': "{}".format(self.write_path),
                                             'imagesize': [self.new_dim, self.new_dim ],
                                             'numtiles': params.child('Num tiles').value(),
                                             'loadingfactors': [params.child('Loading factor').value()]},
                         'computation': {'runname': "{}".format(params.child('Save Name').value()),
                                         'modelstartsize': [params.child('Model start size').value(),
                                                            params.child('Model start size').value()],
                                         'numstepsfactor': params.child('Numsteps factor').value(),
                                         'scalefactor': params.child('Scale factor').value()}}}

        h = hig.hig(**hig_info)
        hig_name = './' + params.child('Save Name').value()
        if not hig_name.endswith('.hig'):
            hig_name += '.hig'

        h.write(hig_name)
        proc = subprocess.Popen(['./hiprmc/bin/hiprmc', hig_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = proc.communicate()

        msg.showMessage('Done')

        # complicated way of finding and writing into folder name written by hiprmc
        ind = output.index(params.child('Save Name').value())
        rmc_folder = './{}'.format(output[ind:].split("\n")[0])
        os.rename(hig_name, '{}/{}.hig'.format(rmc_folder,params.child('Save Name').value()))

        # write output of RMC to file in hiprmc output folder
        output_path = rmc_folder + "/{}_rmc_output.txt".format(params.child('Save Name').value())
        with open(output_path, 'w') as txt:
            txt.write(output)

        # add rmcView to tabwidget
        self.rmc_view = rmc.rmcView(rmc_folder)
        self.rmc_view.findChild(QtGui.QTabBar).hide()
        self.rmc_view.setContentsMargins(0,0,0,0)
        self.image_holder.addWidget(self.rmc_view)

        # this is a temporary fix for a bug: pressing either buttton changes tab back to first
        self.image_holder.setCurrentIndex(2)


















