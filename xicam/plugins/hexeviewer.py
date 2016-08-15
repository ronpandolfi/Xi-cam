import os
from PySide import QtGui
from  xicam.plugins import base
from pyqtgraph import parametertree as pt
from widgets import viewer
from pipeline import loader, msg
import pyqtgraph as pg


class plugin(base.plugin):
    name = "HexeViewer"

    #center widget is something, rightwidget is none, and leftwidget inherits the default from parent base.plugin
    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)

        # configuring right widget
        self.rightwidget = QtGui.QWidget()
        rightWidgetFormat = QtGui.QVBoxLayout()
        rightWidgetFormat.setContentsMargins(0,0,0,0)

        scatteringParams = pt.ParameterTree()
        params = [{'name': 'Camera Location', 'type': 'int', 'value': 1, 'default': 0},
                  {'name': 'Param 1', 'type': 'int'},
                  {'name': 'Param 2', 'type': 'int', 'value': 1, 'default': 1},
                  {'name': 'Param 3', 'type': 'int', 'value': 0, 'default': 0},
                  {'name': 'Param 4', 'type': 'int'}]
        configparams = pt.Parameter.create(name='Configuration', type='group', children=params)
        scatteringParams.setParameters(configparams, showTop=False)

        scatteringHolder = QtGui.QStackedWidget()
        scatteringHolder.addWidget(scatteringParams)
        scatteringHolder.setFixedHeight(300)


        runButton = QtGui.QPushButton("Run")
        # runButton = QtGui.QToolButton()
        # icon = QtGui.QIcon()
        # icon.addPixmap(QtGui.QPixmap("gui/icons_45.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        # runButton.setIcon(icon)
        # runButton.setToolTip('Run scattering thing.')

        rightWidgetFormat.addWidget(scatteringHolder)
        rightWidgetFormat.addSpacing(50)
        rightWidgetFormat.addWidget(runButton)
        self.rightwidget.setLayout(rightWidgetFormat)

        runButton.clicked.connect(self.run)


       # self.inTab = imageviewer
       #  self.outTab = imagviewer


        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent



        super(plugin, self).__init__(*args, **kwargs)

    def openfiles(self, paths):
        self.activate()
        widget = viewer(paths = paths)
        self.centerwidget.addTab(widget, os.path.basename(paths[0]))

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
        print(e)
        e.accept()




class viewer(QtGui.QWidget, ):
    def __init__(self, paths, parent=None):
            super(viewer, self).__init__(parent=parent)

            self.view_stack = pg.ImageView()
            self.stack_image = loader.StackImage(paths)
            # self.stack_image.drawCenter()

            if self.stack_image is not None:
                self.view_stack.setImage(self.stack_image)
                self.view_stack.autoLevels()

            layout = QtGui.QVBoxLayout(self)
            layout.addWidget(self.view_stack)

    def run(self):
        pass

    # def drawCenter(self):
    #     shape = self.stack_image.shape
    #
    #     if len(shape)!=2:
    #         raise ValueError('Image must be 2-dimensional')
    #
