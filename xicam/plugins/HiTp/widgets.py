import pyqtgraph as pg
from PySide.QtGui import *
from PySide.QtCore import *
import pandas as pd

import numpy as np

class WaferView(pg.PlotWidget):
    sigPlot = Signal(object) # emits 2-d cake array

    csvkeys = {'crystallinity':'Imax/Iave',}  # TODO: add mappings for other keys

    def __init__(self):
        super(WaferView, self).__init__()

        self.waferplotitem = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'))
        self.addItem(self.waferplotitem)
        ## Make all plots clickable
        self.lastClicked = None

        def clicked(plot, points):

            global lastClicked
            p =self.lastClicked
            if p: p.setPen(p.brush().color(), width=0)
            p=points[0]
            p.setPen('w', width=5)
            self.lastClicked = p

        self.waferplotitem.sigClicked.connect(clicked)

        self.scene().sigMouseClicked.connect(self.mouseClicked)


    def mouseClicked(self,event):
        '''

        Parameters
        ----------
        event : QMouseEvent

        '''
        print event.pos()
        #get cake data from file
        #...
        #emit cake data
        cake = np.zeros((10,10))
        self.sigPlot.emit(cake)
        event.accept()

    @Slot(str,str)
    def redrawfromCSV(self,csv,mode='crystallinity'):
        '''

        Parameters
        ----------
        csv : str
            filepath reference to CSV file to be displayed
        mode : str
            display mode; one of 'SNR','NND','TEXTURE','MAX','AVG','MAX/AVG','#PEAKS','FWHM'

        '''
        #read csv file
        #....
        #plot visualization
        #print csv
        #print 'loading csv into dataframe'
        p = pd.read_csv(csv)
        #print p
        x=np.nan_to_num(p['plate_x'])
        y=np.nan_to_num(p['plate_y'])
        z = np.nan_to_num(p[self.csvkeys[mode]])
        #print x, y
        d=(x+y).argsort()
        x,y,z = (x[d],
                 y[d],
                 z[d])
        zmin = min(z)
        zrange = np.ptp(z)
        z = (z-zmin)/zrange *256

        points = [{'pos':(x[i],y[i]),
                   'data':z[i]*100,
                   'size':30,
                   'brush': pg.intColor(z[i], 256),
                   'pen':pg.mkPen(width=0,color=pg.intColor(z[i], 256))
                   } for i in range(len(z))]

        self.waferplotitem.setPoints(points)



class LocalView(QTabWidget):
    def __init__(self):
        super(LocalView, self).__init__()

        self.view1D = pg.PlotWidget()
        self.view2D = pg.ImageView()

        self.addTab(self.view1D,'1D')
        self.addTab(self.view2D,'Cake')

        self.setTabPosition(self.West)
        self.setTabShape(self.Triangular)

    @Slot(object)
    def plot(self,cake):
        '''

        Parameters
        ----------
        cake : np.ndarray
            The caked image array to be displayed

        '''
        #display cake and 1D in views
        pass

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    # ######## SAVE THIS FOR DEBUGGING SEG FAULTS; issues are usually doing something outside the gui thread
    # import sys
    #
    #
    # def trace(frame, event, arg):
    #     print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
    #     return trace
    #
    #
    # sys.settrace(trace)

    app = QApplication([])

    ## Create window with ImageView widget
    win = QMainWindow()
    win.resize(800, 800)

    w = WaferView()
    #csv = '/home/rp/data/HiTp/Sample14_master_metadata_high.csv'
    #csv = 'C:\\Research_FangRen\\Data\\Apr2016\\Jan_samples\\Sample1\\Sample14_master_metadata_high.csv'
    w.redrawfromCSV(csv)

    win.setCentralWidget(w)
    win.setWindowTitle('Fitting')
    win.show()

    QApplication.instance().exec_()
