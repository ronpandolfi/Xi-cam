import re

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends import qt_compat
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigCanvas

from . import uitools

def display_item(item,uri,viewer,logmethod=None):
    # Don't proceed unless the item has something interesting to show.
    if logmethod: 
        logmethod('Log messages for image viewer not yet implemented')
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
        n_val_widgets = qlayout.count()
        # Loop through them, last to first, clear the frame
        for i in range(n_val_widgets-1,-1,-1):
            # QLayout.takeAt returns a LayoutItem
            widg = qlayout.takeAt(i)
            # get the QWidget of that LayoutItem and set it to deleteLater()
            widg.widget().deleteLater()
        qlayout.addWidget(plot_widget,0,0,1,1) 
    else:
        # TODO: dialog box: tell user the selected item is uninteresting.
        print '[{}]: selected item ({}) has no display method'.format(__name__,type(item).__name__)
        pass

def array_plot_2d(data_in):
    return mpl_array_plot_2d(data_in)

def array_plot_1d(data_in):
    return mpl_array_plot_1d(data_in)

def mpl_array_plot_1d(data_in):
    fig = Figure(figsize=(100,100))
    axes = fig.add_subplot(111)
    axes.plot(data_in)
    return FigCanvas(fig)

def mpl_array_plot_2d(data_in):
    fig = Figure(figsize=(100,100))
    axes = fig.add_subplot(111)
    axes.contour(data_in)
    # FigCanvas is a subclass of QWidget
    return FigCanvas(fig)

def plot_mpl_fig(fig_in):
    return FigCanvas(fig_in)


