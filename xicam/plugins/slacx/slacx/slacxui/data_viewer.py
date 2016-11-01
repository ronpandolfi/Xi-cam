import re

import numpy as np
from PySide import QtGui

from . import uitools
if uitools.have_qt47:
    from . import plotmaker_pqg as plotmaker
else:
    from . import plotmaker_mpl as plotmaker
    
def display_item(item,uri,qlayout,logmethod=None):
    if logmethod: 
        logmethod('Log messages for data viewer not yet implemented')

    # Loop through the layout, last to first, clear the frame
    n_val_widgets = qlayout.count()
    for i in range(n_val_widgets-1,-1,-1):
        # QLayout.takeAt returns a LayoutItem
        widg = qlayout.takeAt(i)
        # get the QWidget of that LayoutItem and set it to deleteLater()
        widg.widget().deleteLater()

    # Produce widgets for displaying arrays and MatPlotLib figures
    if type(item).__name__ == 'ndarray':
        dims = np.shape(item)
        if len(dims) == 2 and dims[0] > 2 and dims[1] > 2:
            plot_widget = plotmaker.array_plot_2d(item)
        elif len(dims) == 1 or (len(dims) == 2 and (dims[0]==2 or dims[1]==2)):
            plot_widget = plotmaker.array_plot_1d(item)
    elif type(item).__name__ == 'Figure':
        plot_widget = plotmaker.plot_mpl_fig(item)
    else:
        plot_widget = None
    
    # Produce widgets for displaying strings, dicts, etc.
    if type(item).__name__ in ['str','unicode']:
        display_text = 'UNICODE PRINTOUT: <br>{}'.format(item)
        text_widget = QtGui.QTextEdit(display_text)
    elif type(item).__name__ == 'dict':
        display_text = 'DICT PRINTOUT: '
        for k,v in item.items():
            display_text += '<br> {}: {}'.format(k,v)
        text_widget = QtGui.QTextEdit(display_text)
    else:
        text_widget = None

    # Assemble whatever widgets were produced, add them to the layout    
    if plot_widget:
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



