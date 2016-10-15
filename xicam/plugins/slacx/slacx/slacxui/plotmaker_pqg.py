import re

import numpy as np
from PySide import QtGui
import pyqtgraph as pg
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends import qt_compat
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigCanvas

from . import uitools

def display_item(item,uri,viewer,logmethod=None):
    # Don't proceed unless the item has something interesting to show.
    if logmethod:
        msg = '[{}] plotting {} item'.format(__name__,type(item).__name__)
        logmethod(msg)
    if type(item).__name__ == 'ndarray':
        dims = np.shape(item)
        if len(dims) == 2 and dims[0] > 2 and dims[1] > 2:
            plot_widget = array_plot_2d(item)
        elif len(dims) == 1:
            plot_widget = array_plot_1d(item)
    elif type(item).__name__ == 'Figure':
        plot_widget = plot_mpl_fig(item)
    else:
        plot_widget = None
    if plot_widget:
        # add a new tab to image_viewer labeled with uri of tree item 
        # embed the plot_widget in a QFrame?
        layout = QtGui.QVBoxLayout()
        layout.addWidget(plot_widget)
        frame = QtGui.QFrame()
        frame.setLayout(layout)
        #tab_indx = viewer.addTab(plot_widget,uri)
        tab_indx = viewer.addTab(frame,uri)
        viewer.setCurrentIndex(tab_indx)
    else:
        # TODO: dialog box: tell user the selected item is uninteresting.
        print '[{}]: selected item ({}) has no display method'.format(__name__,type(item).__name__)
        pass

def array_plot_2d(data_in):
    return pqg_array_plot_2d(data_in)

def array_plot_1d(data_in):
    return pqg_array_plot_1d(data_in)

def pqg_array_plot_2d(data_in):
    widg = pg.ImageView()
    widg.setImage(data_in)
    return widg 

def pqg_array_plot_1d(data_in):
    widg = pg.PlotWidget()
    widg.getPlotItem().plot(data_in)
    return widg 

def plot_mpl_fig(fig_in):
    return FigCanvas(fig_in)

