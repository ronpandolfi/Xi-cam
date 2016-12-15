from PySide import QtGui, QtCore

from ...slacxcore.operations import optools 

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

