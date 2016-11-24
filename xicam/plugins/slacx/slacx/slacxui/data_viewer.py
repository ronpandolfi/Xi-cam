import re

import numpy as np
from PySide import QtGui
from matplotlib.figure import Figure

from ..slacxcore.slacxtools import FileSystemIterator
from ..slacxcore.operations.optools import InputLocator
from ..slacxcore.operations.slacxop import Operation 
from . import uitools
if uitools.have_qt47:
    from . import plotmaker_pqg as plotmaker
else:
    from . import plotmaker_mpl as plotmaker
    

def display_item(item,uri,qlayout,logmethod=None):
    if logmethod: 
        logmethod('Log messages for data viewer not yet implemented')

    # Loop through the layout, last to first, clear the frame
    n_widgets = qlayout.count()
    for i in range(n_widgets-1,-1,-1):
        # QLayout.takeAt returns a LayoutItem
        widg = qlayout.takeAt(i)
        # get the QWidget of that LayoutItem and set it to deleteLater()
        widg.widget().deleteLater()

    if isinstance(item,Operation):
        op_widget = uitools.OpWidget(item)
    else:
        op_widget = None

    # Produce widgets for displaying arrays and MatPlotLib figures
    if isinstance(item,np.ndarray):
        dims = np.shape(item)
        if len(dims) == 2 and dims[0] > 2 and dims[1] > 2:
            plot_widget = plotmaker.array_plot_2d(item)
        elif len(dims) == 1 or (len(dims) == 2 and (dims[0]==2 or dims[1]==2)):
            plot_widget = plotmaker.array_plot_1d(item)
    elif isinstance(item,Figure):
        plot_widget = plotmaker.plot_mpl_fig(item)
    else:
        plot_widget = None
    
    # Produce widgets for displaying strings, dicts, etc.
    if type(item).__name__ in ['str','unicode']:
        display_text = 'str / unicode printout: <br>{}'.format(item)
    elif isinstance(item,dict):
        display_text = 'dict printout: '
        for k,v in item.items():
            display_text += '<br> {}: {}'.format(k,v)
    elif isinstance(item,list):
        display_text = 'list printout: '
        for i in range(len(item)):
            display_text += '<br> {}: {}'.format(i,item[i])
    elif isinstance(item,InputLocator):
        display_text = 'InputLocator printout: '
        display_text += '<br> src: {}'.format(item.src)
        display_text += '<br> type: {}'.format(item.tp)
        display_text += '<br> val: {}'.format(item.val)
        display_text += '<br> data: {}'.format(item.data)
    elif isinstance(item,FileSystemIterator):
        display_text = 'FileSystemIterator history: '
        for p in item.paths_done:
            display_text += '<br> {}'.format(p)
    else:
        display_text = 'object printout: <br>{}'.format(item)
    text_widget = QtGui.QTextEdit(display_text)

    # Assemble whatever widgets were produced, add them to the layout    
    if op_widget:
        qlayout.addWidget(op_widget,0,0,1,1) 
    elif plot_widget:
        qlayout.addWidget(plot_widget,0,0,1,1) 
    elif text_widget:
        # TODO: Anything else for displaying text, other than plopping it down in the center?
        qlayout.addWidget(text_widget,0,0,1,1) 
    else:
        msg = str('[{}]: selected item ({}) has no display method'.format(__name__,type(item).__name__)
            + '<br><br>Printout of item: <br>{}'.format(item))
        msg_widget = QtGui.QTextEdit(msg)
        qlayout.addWidget(msg_widget,0,0,1,1) 
        pass



