import os
from PySide import QtGui, QtCore
from xicam.plugins import base
from xicam import config
from pyqtgraph import parametertree as pt
# from fabio import open
# from PIL import Image
from pipeline import loader, hig
#from hiprmc import hiprmc
import pyqtgraph as pg
import numpy as np

"""
Bugs:
    1. User can resize ROI after recentering
    2. rmc does not work yet

"""

class plugin(base.plugin):

    # name can be changed upon request
    name = "HexeViewer"

    #center widget is something, rightwidget is none, and leftwidget inherits the default from parent base.plugin
    def __init__(self, *args, **kwargs):


        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)
        self.rightwidget = None

        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        # self.centerwidget.dragEnterEvent = self.dragEnterEvent
        # self.centerwidget.dropEvent = self.dropEvent



        super(plugin, self).__init__(*args, **kwargs)


    def openfiles(self, paths):
        self.activate()
        view_widget = inOutViewer(paths = paths)
        self.centerwidget.addTab(view_widget, os.path.basename(paths[0]))
        self.centerwidget.setCurrentWidget(view_widget)
        view_widget.drawCameraLocation(view_widget.view_stack,view_widget.cameraLocation)


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


    def tabCloseRequested(self,index):
        self.centerwidget.widget(index).deleteLater()



class inOutViewer(QtGui.QWidget, ):
    def __init__(self, paths, parent=None):
        super(inOutViewer, self).__init__(parent=parent)

        layout = QtGui.QHBoxLayout()
        self.cameraLocation = config.activeExperiment.center



        self.view_stack = pg.ImageView(self)
        self.path = paths[0]

        # import using pipeline function
        self.stack_image = np.transpose(loader.loadimage(self.path))


        # import using python image library (PIL)
        # self.stack_image = np.transpose(np.array(Image.open(path)))

        # configuring right widget
        sideWidgetFormat = QtGui.QVBoxLayout()
        sideWidgetFormat.setContentsMargins(0, 0, 0, 0)

        self.scatteringParams = pt.ParameterTree()
        params = [{'name': 'Num tiles', 'type': 'int', 'value': 1, 'default': 1},
                  {'name': 'Loading factor', 'type': 'int', 'value': 1},
                  {'name': 'Scale factor', 'type': 'int', 'value': 1, 'default': 1},
                  {'name': 'Numsteps factor', 'type': 'int', 'value': 100, 'default': 100},
                  {'name': 'Model start size', 'type': 'int', 'value': 1},
                  {'name': 'Save Name', 'type': 'str', 'value': 'test.tif'}]
        self.configparams = pt.Parameter.create(name='Configuration', type='group', children=params)
        self.scatteringParams.setParameters(self.configparams, showTop=False)


        scatteringHolder = QtGui.QStackedWidget()
        scatteringHolder.addWidget(self.scatteringParams)
        scatteringHolder.setFixedHeight(300)

        centerButton = QtGui.QPushButton("Center camera location")
        runButton = QtGui.QPushButton("Run RMC processing")
        saveButton = QtGui.QPushButton("Save centered image")
        sideWidgetFormat.addWidget(scatteringHolder)
        sideWidgetFormat.addSpacing(50)
        sideWidgetFormat.addWidget(centerButton)
        sideWidgetFormat.addSpacing(5)
        sideWidgetFormat.addWidget(saveButton)
        sideWidgetFormat.addSpacing(5)
        sideWidgetFormat.addWidget(runButton)

        centerButton.clicked.connect(self.center)
        runButton.clicked.connect(self.runRMC)
        saveButton.clicked.connect(self.save)

        self.heading_box = QtGui.QComboBox()
        self.heading_box.addItems(['Original Image', 'Recentered Image'])

        self.image_holder = QtGui.QStackedWidget()
        self.view_stack.setImage(self.stack_image)
        self.view_stack.autoLevels()
        self.view_stack.autoRange()
        self.image_holder.addWidget(self.view_stack)


        sidelayout = QtGui.QVBoxLayout()
        sidelayout.addWidget(self.heading_box)
        sidelayout.addSpacing(5)
        sidelayout.addWidget(self.image_holder)

        layout.addLayout(sidelayout,10)
        layout.addLayout(sideWidgetFormat,4)
        self.setLayout(layout)


        self.heading_box.activated.connect(self.image_holder.setCurrentIndex)


    def save(self):
        pass
        # if type(self.edited_image) == None:
        #     pass
        #
        # write_path = self.path
        # if write_path.endswith()
        #
        # with open(path, 'w') as f:
        #         f.write(str(self))


    def center(self):

        #resize image so that it's in center
        #displays output on stackwidget

        xdim= self.stack_image.shape[0]
        ydim = self.stack_image.shape[1]

        newx = xdim + 2*abs(self.cameraLocation[0]-xdim/2)
        newy = ydim + 2*abs(self.cameraLocation[1]-ydim/2)
        self.new_dim = max(newx,newy)

        self.edited_image = np.zeros((self.new_dim,self.new_dim),dtype = np.int)
        new_center = (self.new_dim/2,self.new_dim/2)

        lowleft_corner_x = new_center[0]-self.cameraLocation[0]
        lowleft_corner_y = new_center[1]-self.cameraLocation[1]

        self.edited_image[lowleft_corner_x:lowleft_corner_x+xdim,lowleft_corner_y: lowleft_corner_y+ydim] \
            = self.stack_image


        self.show_edited = pg.ImageView(self)
        self.show_edited.setImage(self.edited_image)
        self.image_holder.addWidget(self.show_edited)

        box = self.drawCameraLocation(self.show_edited,new_center)
        self.drawROI(lowleft_corner_x,lowleft_corner_y,xdim,ydim, box)

    def drawCameraLocation(self,imageView_item,location):

        cameraBox = imageView_item.getImageItem().getViewBox()
        cameraPlot = pg.ScatterPlotItem()
        cameraBox.addItem(cameraPlot)
        cameraPlot.setData([location[0]], [location[1]], pen=None,
                                symbol='o' , brush=pg.mkBrush('#FFA500'))

        return cameraBox


    def drawROI(self, xpos, ypos, xdim,ydim, view_box):

        roi = pg.RectROI((xpos,ypos),(xdim,ydim),movable = False,removable=True)

        # for handle in roi.getHandles():
        #     print handle.scene()
        #     roi.removeHandle(handle)

        view_box.addItem(roi)


    def runRMC(self):

        params = self.configparams


        hig_info = {'hipRMCInput': {'instrumentation': {'inputimage': "{}".format(self.path),
                                             'imagesize': [self.new_dim, self.new_dim ],
                                             'numtiles': params.child('Num tiles').value(),
                                             'loadingfactors': [params.child('Loading factor').value()]},
                         'computation': {'runname': "{}".format(params.child('Save Name').value()),
                                         'modelstartsize': [params.child('Model start size').value(),
                                                            params.child('Model start size').value()],
                                         'numstepsfactor': params.child('Numsteps factor').value(),
                                         'scalefactor': params.child('Scale factor').value()}}}

        h = hig.hig(**hig_info)
        save_name = './' + params.child('Save Name').value()
        if not save_name.endswith('.hig'):
            save_name += '.hig'


        h.write(save_name)




















