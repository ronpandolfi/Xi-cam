import os
import re
import platform
from functools import partial

from PySide import QtGui, QtCore, QtUiTools

from ..slacxcore.operations import optools
from ..slacxcore.listmodel import ListModel
from ..slacxcore import slacxtools

## Test whether we have Qt >= 4.7 
have_qt47 = True
versionReq = [4, 7]
QtVersion = QtCore.__version__ 
m = re.match(r'(\d+)\.(\d+).*', QtVersion)
if m is not None and list(map(int, m.groups())) < versionReq:
    have_qt47 = False

## Test whether we are using Windows
if platform.system() == 'Windows':
    have_windows = True
else:
    have_windows = False

def text_widget(text):
    widg = QtGui.QLineEdit(text)
    widg.setReadOnly(True)
    widg.setAlignment(QtCore.Qt.AlignHCenter)
    return widg 

def toggle_expand(trview,idx):
    trview.setExpanded(idx, not trview.isExpanded(idx))

def type_mv_widget(src,widg=None):
    if not widg:
        widg = QtGui.QComboBox()
    lm = ListModel(optools.input_types,widg)
    widg.setModel(lm)
    if src == optools.no_input:
        widg.setCurrentIndex(optools.none_type)
        for tp in [optools.auto_type,optools.str_type,optools.int_type,optools.float_type,optools.bool_type,optools.list_type]:
            lm.set_disabled(tp)
    elif src == optools.batch_input:
        widg.setCurrentIndex(optools.auto_type)
        for tp in [optools.none_type,optools.str_type,optools.int_type,optools.float_type,optools.bool_type,optools.list_type]:
            widg.model().set_disabled(tp)
    elif src == optools.user_input:
        for tp in [optools.auto_type]:
            widg.model().set_disabled(tp)
            widg.setCurrentIndex(optools.float_type)
    elif (src == optools.wf_input or src == optools.fs_input):
        for tp in [optools.str_type,optools.int_type,optools.float_type,optools.bool_type]:
            widg.model().set_disabled(tp)
            widg.setCurrentIndex(optools.auto_type)
    return widg 

#def type_selection_widget():
#    widg = QtGui.QComboBox()
#    widg.addItems(optools.input_types)
#    return widg 

def src_selection_widget():
    widg = QtGui.QComboBox()
    widg.addItems(optools.input_sources)
    return widg 

def r_hdr_widget(text):
    widg = QtGui.QLineEdit(text)
    widg.setReadOnly(True)
    widg.setAlignment(QtCore.Qt.AlignRight)
    widg.setStyleSheet( "QLineEdit { background-color: transparent }" + widg.styleSheet() )
    return widg 

def hdr_widget(text):
    widg = QtGui.QLineEdit(text)
    widg.setReadOnly(True)
    widg.setAlignment(QtCore.Qt.AlignLeft)
    widg.setStyleSheet( "QLineEdit { background-color: transparent }" + widg.styleSheet() )
    return widg 

def smalltext_widget(text):
    widg = text_widget(text)
    widg.setStyleSheet( "QLineEdit { background-color: transparent }" + widg.styleSheet() )
    return widg

def bigtext_widget(text,trunc_limit=70):
    if len(text) > trunc_limit:
        display_text = text[:trunc_limit]+'...'
    else:
        display_text = text
    widg = QtGui.QLineEdit(display_text)
    widg.setReadOnly(True)
    widg.setAlignment(QtCore.Qt.AlignLeft)
    return widg

def name_widget(name):
    name_widget = QtGui.QLineEdit(name)
    name_widget.setReadOnly(True)
    name_widget.setAlignment(QtCore.Qt.AlignRight)
    return name_widget
    
#def treesource_typval_widgets():
#    type_widget = type_mv_widget() 
#    #type_widget = type_selection_widget()
#    type_widget.setCurrentIndex(optools.auto_type)
#    val_widget = QtGui.QLineEdit('-')
#    val_widget.setReadOnly(True)
#    return type_widget, val_widget
    
def toggle_load_button(ui,txt):
    idx = ui.tree.model().index(txt)
    if (idx.isValid() and ui.tree.model().isDir(idx) 
    or not os.path.splitext(txt)[1] == '.wfl'):
        ui.load_button.setEnabled(False)
    else:
        ui.load_button.setEnabled(True)

def toggle_save_button(ui,txt):
    idx = ui.tree.model().index(ui.filename.text())
    if idx.isValid() and ui.tree.model().isDir(idx):
        ui.save_button.setEnabled(False)
    else:
        ui.save_button.setEnabled(True)

def save_path(ui,idx=QtCore.QModelIndex(),oldidx=QtCore.QModelIndex()):
    if idx.isValid():
        p = ui.tree.model().filePath(idx)
        ui.filename.setText(p)

def load_path(ui,idx=QtCore.QModelIndex()):
    if idx.isValid():
        p = ui.tree.model().filePath(idx)
        if (ui.tree.model().isDir(idx) 
        or not os.path.splitext(p)[1] == '.wfl'):
            ui.load_button.setEnabled(False)
        else:
            ui.load_button.setEnabled(True)

def stop_save_ui(ui,uiman):
    fname = ui.filename.text()
    if not os.path.splitext(fname)[1] == '.wfl':
        fname = fname + '.wfl'
    uiman.wfman.save_to_file(fname)
    ui.close()

def stop_load_ui(ui,uiman):
    #fname = ui.filename.text()
    fname = ui.tree.model().filePath(ui.tree.currentIndex())
    uiman.wfman.load_from_file(uiman.opman,fname)
    ui.close()

def start_save_ui(uiman):
    """
    Start a modal window dialog to choose a save destination for the workflow in progress 
    """
    ui_file = QtCore.QFile(slacxtools.rootdir+"/slacxui/save_browser.ui")
    ui_file.open(QtCore.QFile.ReadOnly)
    save_ui = QtUiTools.QUiLoader().load(ui_file)
    ui_file.close()
    #save_ui.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    trmod = QtGui.QFileSystemModel()
    #trmod.setRootPath('.')
    trmod.setRootPath(slacxtools.rootdir)
    trmod.setNameFilters(['*.wfl'])
    save_ui.tree_box.setTitle('Select a file to save the current workflow')
    save_ui.tree.setModel(trmod)
    save_ui.tree.hideColumn(1)
    save_ui.tree.hideColumn(3)
    save_ui.tree.setColumnWidth(0,400)
    save_ui.tree.expandAll()
    save_ui.tree.clicked.connect( partial(save_path,save_ui) )
    save_ui.tree.expanded.connect( partial(save_path,save_ui) )
    #save_ui.tree.activated.connect( save_ui.tree.setCurrentIndex )
    #save_ui.tree.selectionModel().selectionChanged.connect( save_ui.tree.selectionChanged )
    #import pdb; pdb.set_trace()
    save_ui.setParent(uiman.ui,QtCore.Qt.Window)
    save_ui.save_button.setText('&Save')
    save_ui.save_button.clicked.connect(partial(stop_save_ui,save_ui,uiman))
    #save_ui.filename.returnPressed.connect(partial(stop_save_ui,save_ui,uiman))
    save_ui.filename.textChanged.connect( partial(toggle_save_button,save_ui) )
    save_ui.filename.setText(trmod.rootPath())
    save_ui.setWindowModality(QtCore.Qt.ApplicationModal)
    save_ui.show()
    save_ui.activateWindow()

def start_load_ui(uiman):
    """
    Start a modal window dialog to load a previously saved workflow
    """
    ui_file = QtCore.QFile(slacxtools.rootdir+"/slacxui/load_browser.ui")
    ui_file.open(QtCore.QFile.ReadOnly)
    load_ui = QtUiTools.QUiLoader().load(ui_file)
    ui_file.close()
    trmod = QtGui.QFileSystemModel()
    trmod.setRootPath('.')
    trmod.setNameFilters(['*.wfl'])
    load_ui.tree_box.setTitle('Select a .wfl file to load a workflow')
    load_ui.tree.setModel(trmod)
    load_ui.tree.hideColumn(1)
    load_ui.tree.hideColumn(3)
    load_ui.tree.setColumnWidth(0,400)
    load_ui.tree.expandAll()
    load_ui.tree.clicked.connect( partial(load_path,load_ui) )
    load_ui.setParent(uiman.ui,QtCore.Qt.Window)
    load_ui.load_button.setText('&Load')
    load_ui.load_button.clicked.connect(partial(stop_load_ui,load_ui,uiman))
    #load_ui.setWindowModality(QtCore.Qt.WindowModal)
    load_ui.setWindowModality(QtCore.Qt.ApplicationModal)
    load_ui.show()
    load_ui.activateWindow()

#class ListBuildManager(object):
#    
#    def __init__(self,ui):
#        self.ui = ui 
#        super(ListBuildManager,self).__init__()
#        self.setup_ui()
#
#    def setup_ui(self):
#        self.ui.finish_button.setText('Finish')

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
        p.drawText(inphdr,QtCore.Qt.AlignCenter,'Inputs')
        p.drawText(outhdr,QtCore.Qt.AlignCenter,'Outputs')
        f.setUnderline(False)
        p.setFont(f)
        # Label the inputs
        n_inp = len(self.op.inputs)
        ispc = 2*rectvert/(2*n_inp) 
        vcrd = -1*rectvert+ispc
        for name,il in self.op.input_locator.items():
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



