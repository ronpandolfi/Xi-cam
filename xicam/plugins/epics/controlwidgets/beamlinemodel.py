import pyqtgraph as pg
from PySide.QtGui import *
from PySide.QtCore import *

class beamlinemodel(pg.GraphicsView):
    def __init__(self,devicelist,stackplaceholder):
        super(beamlinemodel, self).__init__()

        self.view = pg.ViewBox()

        self.setCentralItem(self.view)
        self.view.setAspectLocked(True)
        self.view.setMouseEnabled(True,True)

        self.addItem(beammodel([0,10],[0,0]))
        #self.addItem(motormodel(devicelist.item(0),(4,-1),(1,2)))
        #self.addItem(motormodel(devicelist.item(1), (7, -1), (1, 2)))

    def addItem(self,*args):
        self.view.addItem(*args)

class modelitem(QGraphicsItem):

    def __init__(self,deviceitem,*args,**kwargs):
        super(modelitem, self).__init__()
        self.deviceitem=deviceitem
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        cursor = Qt.PointingHandCursor # NOTE: on Windows, this might need to be QCursor(...)
        QApplication.instance().setOverrideCursor(cursor)

    def hoverLeaveEvent(self, event):
        QApplication.instance().restoreOverrideCursor()

    def mousePressEvent(self, *args, **kwargs):
        self.deviceitem.showWidget()



class motormodel(modelitem):

    def __init__(self,deviceitem,position,size=(0,0),orientation=None,color=(200,200,200),html=None):
        '''

        Parameters
        ----------
        motor
        position    :   tuple
        '''
        super(motormodel, self).__init__(deviceitem)

        self.position = position
        self.size = size
        self.device = deviceitem.device
        self.devicepen = pg.mkPen(color='g',width=2)
        self.devicebrush = pg.mkBrush(color=[0,0,0,50])
        
        self.rectItem = QGraphicsRectItem(QRect(QPoint(*position),QSize(*size)))
        self.rectItem.setParentItem(self)
        self.rectItem.setPen(self.devicepen)
        self.rectItem.setBrush(self.devicebrush)

        self.textItem = pg.TextItem(self.device.getMnemonic(),anchor=(0,1))
        self.textItem.setParentItem(self)
        self.textItem.setPos(QPoint(*self.position))


    def boundingRect(self, *args, **kwargs):
        return QRect(QPoint(*self.position),QSize(*self.size)) # NOTE: could use QRectF.intersect here?

    def paint(self,*args,**kwargs):
        pass

class beammodel(pg.PlotDataItem):
    def __init__(self,*args,**kwargs):
        kwargs['pen']=pg.mkPen(color='g',width=2)
        super(beammodel, self).__init__(*args,**kwargs)
