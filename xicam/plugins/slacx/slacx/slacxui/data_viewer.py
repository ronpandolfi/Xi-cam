import re

import numpy as np
from PySide import QtCore, QtGui
from matplotlib.figure import Figure

from ..slacxcore.slacxtools import FileSystemIterator
from ..slacxcore.operations import optools 
from ..slacxcore.operations.slacxop import Operation 
from . import uitools
if uitools.have_qt47:
    from . import plotmaker_pqg as plotmaker
else:
    from . import plotmaker_mpl as plotmaker

unit_indent = '&nbsp;&nbsp;&nbsp;&nbsp;'

class OpWidget(QtGui.QWidget):
    
    def __init__(self,op):
        super(OpWidget,self).__init__()
        self.op = op
        #self.render_from_op()        

    def paintEvent(self,evnt):
        w = self.width()
        h = self.height()
        widgdim = float( min([w,h]) )
        # Create a painter and draw in the elements of the Operation
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen()
        qwhite = QtGui.QColor(255,255,255,255)
        pen.setColor(qwhite)
        p.setPen(pen)
        #p.setBrush()...
        p.translate(w/2, h/2)
        p.scale(widgdim/200,widgdim/200)
        rectvert = 80 
        recthorz = 50
        topleft = QtCore.QPoint(int(-1*recthorz),int(-1*rectvert))
        bottomright = QtCore.QPoint(int(recthorz),int(rectvert))
        # Large rectangle representing the Operation
        mainrec = QtCore.QRectF(topleft,bottomright)
        p.drawRect(mainrec)
        f = QtGui.QFont()
        title_hdr = QtCore.QRectF(QtCore.QPoint(-100,-1*(rectvert+10)),
                                QtCore.QPoint(100,-1*rectvert))
        #title_hdr = QtCore.QRectF(QtCore.QPoint(-30,-10),QtCore.QPoint(30,10))
        #f.setPixelSize(10)
        f.setPointSize(5)
        p.setFont(f)
        p.drawText(title_hdr,QtCore.Qt.AlignCenter,type(self.op).__name__)
        f.setPointSize(4)
        p.setFont(f)
        # Headers for input and output sides
        inphdr = QtCore.QRectF(QtCore.QPoint(-1*(recthorz+30),-1*(rectvert+10)),
                                QtCore.QPoint(-1*(recthorz+10),-1*rectvert))
        outhdr = QtCore.QRectF(QtCore.QPoint(recthorz+10,-1*(rectvert+10)),
                                QtCore.QPoint(recthorz+30,-1*rectvert))
        #outhdr = QtCore.QRectF(QtCore.QPoint(70,-90),QtCore.QPoint(90,-80))
        f.setUnderline(True)
        p.setFont(f)
        p.drawText(inphdr,QtCore.Qt.AlignCenter,optools.inputs_tag)
        p.drawText(outhdr,QtCore.Qt.AlignCenter,optools.outputs_tag)
        f.setUnderline(False)
        p.setFont(f)
        # Label the inputs
        n_inp = len(self.op.inputs)
        ispc = 2*rectvert/(2*n_inp) 
        vcrd = -1*rectvert+ispc
        for name in self.op.inputs.keys():
            il = self.op.input_locator[name]
            rec = QtCore.QRectF(QtCore.QPoint(-1*(recthorz-10),vcrd-5),QtCore.QPoint(0,vcrd+5))
            p.drawText(rec,QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter,name)
            p.drawLine(QtCore.QPoint(-1*(recthorz-5),vcrd),QtCore.QPoint(-1*(recthorz+10),vcrd))
            p.drawLine(QtCore.QPoint(-1*(recthorz+10),vcrd-10),QtCore.QPoint(-1*(recthorz+10),vcrd+10))
            ilrec = QtCore.QRectF(QtCore.QPoint(-100,vcrd-10),QtCore.QPoint(-1*(recthorz+12),vcrd+10))
            p.drawText(ilrec,QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter,#|QtCore.Qt.TextWordWrap,
            'source: {} \ntype: {} \nvalue: {}'.format(optools.input_sources[il.src],optools.input_types[il.tp],il.val))
            vcrd += 2*ispc
        # Label the outputs
        n_out = len(self.op.outputs)
        ispc = 2*rectvert/(2*n_out)
        vcrd = -1*rectvert+ispc
        for name,val in self.op.outputs.items():
            rec = QtCore.QRectF(QtCore.QPoint(0,vcrd-5),QtCore.QPoint(recthorz-10,vcrd+5))
            p.drawText(rec,QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter,name)
            p.drawLine(QtCore.QPoint(recthorz-5,vcrd),QtCore.QPoint(recthorz+10,vcrd))
            p.drawLine(QtCore.QPoint(recthorz+10,vcrd-10),QtCore.QPoint(recthorz+10,vcrd+10))
            outrec = QtCore.QRectF(QtCore.QPoint(recthorz+12,vcrd-10),QtCore.QPoint(100,vcrd+10))
            p.drawText(outrec,QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter,str(val))#|QtCore.Qt.TextWordWrap,str(val))
            vcrd += 2*ispc

def display_text(itm,indent):

    if type(itm).__name__ in ['str','unicode']:
        t = indent + '(str) <br>' + indent + '{}'.format(itm)
    elif isinstance(itm,dict):
        t = indent + '(dict)'
        for k,v in itm.items():
            t += '<br>' + indent + '{}: <br>{}'.format(k,display_text(v,indent+unit_indent))
    elif isinstance(itm,list):
        t = indent + '(list)'
        for i in range(len(itm)):
            t += '<br>' + indent + '{}: <br>{}'.format(i,display_text(itm[i],indent+unit_indent))
    elif isinstance(itm,Operation):
        t = indent + '(Operation)'
        t += '<br>' + indent + 'inputs: <br>{}'.format(display_text(itm.inputs,indent+unit_indent))
        t += '<br>' + indent + 'outputs: <br>{}'.format(display_text(itm.outputs,indent+unit_indent))
    elif isinstance(itm,optools.InputLocator):
        t = indent + '(InputLocator)'
        t += '<br>' + indent + 'src: {}'.format(optools.input_sources[itm.src])
        t += '<br>' + indent + 'type: {}'.format(optools.input_types[itm.tp])
        t += '<br>' + indent + 'val: {}'.format(itm.val)
        t += '<br>' + indent + 'data: {}'.format(itm.data)
    #elif isinstance(itm,optools.OutputContainer):
    #    t = indent + '(OutputContainer)'
    #    t += '<br>' + indent + 'data: {}'.format(itm.data)
    elif isinstance(itm,FileSystemIterator):
        t = indent + '(FileSystemIterator) - history'
        for p in itm.paths_done:
            t += '<br>' + indent + ' {}'.format(p)
    else:
        t = indent + '('+type(itm).__name__+')' + '<br>' + indent + '{}'.format(itm)
    return t
    
def display_item(itm,uri,qlayout,logmethod=None):
    if logmethod: 
        logmethod('Log messages for data viewer not yet implemented')

    # Loop through the layout, last to first, clear the frame
    n_widgets = qlayout.count()
    for i in range(n_widgets-1,-1,-1):
        # QLayout.takeAt returns a LayoutItem
        widg = qlayout.takeAt(i)
        # get the QWidget of that LayoutItem and set it to deleteLater()
        widg.widget().deleteLater()

    # If the item is an OutputContainer, unpack it.
    #if isinstance(itm,optools.OutputContainer):
    #    itm = itm.data

    if isinstance(itm,Operation):
        op_widget = OpWidget(itm)
    else:
        op_widget = None

    # Produce widgets for displaying arrays and MatPlotLib figures
    if isinstance(itm,np.ndarray):
        dims = np.shape(itm)
        if len(dims) == 2 and dims[0] > 2 and dims[1] > 2:
            plot_widget = plotmaker.array_plot_2d(itm)
        elif len(dims) == 1 or (len(dims) == 2 and (dims[0]==2 or dims[1]==2)):
            plot_widget = plotmaker.array_plot_1d(itm)
    elif isinstance(itm,Figure):
        plot_widget = plotmaker.plot_mpl_fig(itm)
    else:
        plot_widget = None
    
    # Produce widgets for displaying strings, dicts, etc.
    t = display_text(itm,unit_indent)
    text_widget = QtGui.QTextEdit(t)

    # Assemble whatever widgets were produced, add them to the layout    
    if op_widget:
        qlayout.addWidget(op_widget,0,0,1,1) 
    elif plot_widget:
        qlayout.addWidget(plot_widget,0,0,1,1) 
    elif text_widget:
        # TODO: Anything else for displaying text, other than plopping it down in the center?
        qlayout.addWidget(text_widget,0,0,1,1) 
    else:
        msg = str('[{}]: selected item ({}) has no display method'.format(__name__,type(itm).__name__)
            + '<br><br>Printout of item: <br>{}'.format(itm))
        msg_widget = QtGui.QTextEdit(msg)
        qlayout.addWidget(msg_widget,0,0,1,1) 
        pass


