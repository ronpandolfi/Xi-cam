import time
import os
from functools import partial

from PySide import QtCore, QtGui, QtUiTools
import qdarkstyle
import numpy as np

from ..slacxcore.operations import optools
from ..slacxcore.operations.slacxop import Operation 
from ..slacxcore.workflow.slacxwfman import WfManager
from ..slacxcore import slacxtools
from ..slacxcore import slacxex
from . import uitools

class OpUiManager(object):
    """
    Stores a reference to the op_builder QGroupBox, 
    performs operations on it
    """

    def __init__(self,wfman,opman):
        ui_file = QtCore.QFile(slacxtools.rootdir+"/slacxui/op_builder.ui")
        # Load the op_builder popup
        ui_file.open(QtCore.QFile.ReadOnly)
        self.ui = QtUiTools.QUiLoader().load(ui_file)
        ui_file.close()
        self.wfman = wfman 
        self.opman = opman 
        self.op = None
        self.inp_src_windows = {} 
        self.setup_ui()

    def get_op_from_tree(self,item_indx):
        x = self.opman.get_item(item_indx).data[0]
        # TODO: better type checking
        if not isinstance(x,str):  
            if issubclass(x,Operation):
                # TODO: if this op has had some set-up done, don't create a new one.
                self.create_op(x)
        else:
            self.ui.op_info.setPlainText('Operation category {}'.format(x))

    def set_op(self,op):
        """
        Load a fully or partially formed Operation into the UI.
        Set self.op to input Operation.
        Load op.description() into self.ui.op_info.
        Load op.inputs and op.outputs into self.ui.input_box and self.ui.output_box.
        """
        self.op = op
        self.ui.op_info.setPlainText(op.description())
        self.build_nameval_list()

    def setup_ui(self):
        self.ui.setWindowTitle("slacx operation builder")
        self.ui.input_box.setTitle("INPUTS")
        self.ui.output_box.setTitle("OUTPUTS")
        self.ui.finish_box.setTitle("FINISH / LOAD")
        self.ui.input_box.setMinimumWidth(600)
        self.ui.op_frame.setMinimumWidth(400)
        self.ui.op_frame.setMaximumWidth(400)
        self.ui.op_selector.setModel(self.opman)
        self.ui.op_selector.hideColumn(1)
        self.ui.op_selector.clicked.connect( partial(self.get_op_from_tree) )
        # Populate tag entry fields
        self.ui.tag_prompt.setText('operation tag:')
        self.ui.tag_prompt.setMinimumWidth(100)
        self.ui.tag_prompt.setMaximumWidth(150)
        self.ui.tag_entry.setMaximumWidth(150)
        self.ui.tag_prompt.setReadOnly(True)
        self.ui.tag_prompt.setAlignment(QtCore.Qt.AlignRight)
        # If we are editing an existing operation, use its existing tag
        if isinstance(self.opman,WfManager):    
            self.ui.tag_entry.setText(self.opman.get_item(self.ui.op_selector.currentIndex()).tag())
        else:
            self.ui.tag_entry.setText(self.default_tag())
        self.ui.test_button.setText("&Test")
        self.ui.test_button.setMinimumWidth(100)
        self.ui.test_button.clicked.connect(self.test_op)
        self.ui.finish_button.setText("&Finish")
        self.ui.finish_button.setMinimumWidth(100)
        self.ui.finish_button.clicked.connect(self.load_op)
        self.ui.finish_button.setDefault(True)
        #self.ui.returnPressed.connect(self.load_op)
        self.ui.setStyleSheet( "QLineEdit { border: none }" + self.ui.styleSheet() )
        self.ui.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def test_op(self):
        print 'Operation testing not yet implemented'

    def default_tag(self):
        indx = 0
        goodtag = False
        while not goodtag:
            testtag = 'op{}'.format(indx)
            if not testtag in self.wfman.list_tags(QtCore.QModelIndex()):
                goodtag = True
            else:
                indx += 1
        return testtag

    def create_op(self,op):
        #self.ui.op_info.setPlainText(' ')
        self.op = op()
        self.ui.op_info.setPlainText(self.op.description())
        self.build_nameval_list()

    def load_op(self):
        """
        Package self.op(Operation), ship to self.wfman
        """ 
        tag = self.ui.tag_entry.text()
        result = self.wfman.is_good_tag(tag)
        if result[0]:
            # Make sure all text inputs are loaded
            for row in range(len(self.op.inputs)):
                src = self.ui.input_layout.itemAtPosition(row,optools.src_col).widget().currentIndex()
                if src == optools.text_input:
                    name = self.ui.input_layout.itemAtPosition(row,optools.name_col).widget().text()
                    type_widget = self.ui.input_layout.itemAtPosition(row,optools.type_col).widget()
                    val_widget = self.ui.input_layout.itemAtPosition(row,optools.val_col).widget()
                    self.load_text_input(name,type_widget,val_widget) 
            self.wfman.add_op(self.op,tag) 
            self.ui.close()
        else:
            # Request a different tag
            msg_ui = slacxex.start_message_ui(slacxtools.rootdir)
            msg_ui.setParent(self.ui,QtCore.Qt.Window)
            msg_ui.setWindowTitle("Tag Error")
            msg_ui.message_box.setPlainText(
            'Tag error for {}: \n{} \n\n'.format(tag, result[1])
            + 'Enter a unique alphanumeric tag, '
            + 'using only letters, numbers, -, and _. ')
            # Set button to activate on Enter key
            msg_ui.ok_button.setFocus()
            msg_ui.show()

    def update_op(self,indx):
        """Update the operation at indx in self.wfman with self.op"""
        self.wfman.update_op(indx,self.op)
        self.ui.close()

    def clear_nameval_list(self):
        n_inp_widgets = self.ui.input_layout.count()
        for i in range(n_inp_widgets-1,-1,-1):
            item = self.ui.input_layout.takeAt(i)
            item.widget().deleteLater()
        n_out_widgets = self.ui.input_layout.count()
        for i in range(n_out_widgets-1,-1,-1):
            item = self.ui.output_layout.takeAt(i)
            item.widget().deleteLater()

    def build_nameval_list(self):
        self.clear_nameval_list()
        inp_count = len(self.op.inputs.items())
        out_count = len(self.op.outputs.items())
        self.input_rows = []
        i=0
        for name, val in self.op.inputs.items():
            self.add_nameval_widgets(name,val,i)
            # save a ref to the row of inputs for op loading later
            #self.input_rows.append(i) 
            i+=1
        i=0 
        for name, val in self.op.outputs.items():
            self.add_nameval_widgets(name,val,i,output=True)
            i+=1 

    def add_nameval_widgets(self,name,val,row,output=False):
        """a set of widgets for setting or reading input or output data"""
        #widg = QtGui.QWidget()
        name_widget = QtGui.QLineEdit(name)
        name_widget.setReadOnly(True)
        name_widget.setAlignment(QtCore.Qt.AlignRight)
        name_widget.setMinimumWidth(7*len(name))
        eq_widget = self.smalltext_widget('=')
        val_widget = QtGui.QLineEdit(str(val))
        if output:
            self.ui.output_layout.addWidget(name_widget,row,optools.name_col)
            self.ui.output_layout.addWidget(eq_widget,row,optools.eq_col)
            src_widget = QtGui.QLineEdit(self.ui.tag_entry.text())
            type_widget = QtGui.QLineEdit(type(val).__name__)
            type_widget.setReadOnly(True)
            val_widget.setReadOnly(True)
            self.ui.output_layout.addWidget(src_widget,row,optools.src_col,1,1)
            self.ui.output_layout.addWidget(type_widget,row,optools.type_col,1,1)
            self.ui.output_layout.addWidget(val_widget,row,optools.val_col,1,1)
        else:
            self.ui.input_layout.addWidget(name_widget,row,optools.name_col)
            self.ui.input_layout.addWidget(eq_widget,row,optools.eq_col)
            src_widget = self.src_selection_widget() 
            src_widget.setMinimumWidth(100)
            self.ui.input_layout.addWidget(src_widget,row,optools.src_col,1,1)
            if self.op.input_src[name]:
                self.render_input_widgets(name,row,self.op.input_src[name]) 
                src_widget.setCurrentIndex(self.op.input_src[name])
            else:
                self.render_input_widgets(name,row,0) 
            # Note the widget.activated signal sends the index of the activated item.
            # This will be passed as the next (unspecified) arg to the partial.
            src_widget.activated.connect( partial(self.render_input_widgets,name,row) )

    def render_input_widgets(self,name,row,src_indx): 
        # If input widgets exist, close them.
        for col in [optools.val_col,optools.type_col]:
            if self.ui.input_layout.itemAtPosition(row,col):
                widg = self.ui.input_layout.itemAtPosition(row,col).widget()
                widg.hide()
                widg.deleteLater()
        if src_indx == 0:
            type_widget = QtGui.QLineEdit('(type)')
            val_widget = QtGui.QLineEdit('(value)')
            type_widget.setReadOnly(True)
            val_widget.setReadOnly(True)
            btn_widget = None
        elif src_indx == optools.text_input:
            type_widget = QtGui.QComboBox()
            type_widget.addItems(optools.input_types)
            if self.op.input_type[name]:
                type_widget.setCurrentIndex(self.op.input_type[name])
            val_widget = QtGui.QLineEdit()
            if self.op.inputs[name]:
                val_widget.setText(str(self.op.inputs[name]))
            elif uitools.have_qt47:
                val_widget.setPlaceholderText('(enter value)')
            else:
                val_widget.setText(' ')
            #val_widget.returnPressed.connect( partial(self.load_text_input,name,type_widget,val_widget) )
            btn_widget = None
        elif (src_indx == optools.op_input
            or src_indx == optools.fs_input):
            type_widget = QtGui.QLineEdit('type: auto')
            type_widget.setReadOnly(True)
            val_widget = QtGui.QLineEdit('value: select ->')
            val_widget.setReadOnly(True)
            btn_widget = QtGui.QPushButton('browse...')
            btn_widget.clicked.connect( partial(self.fetch_data,name,src_indx,type_widget,val_widget) )
        else:
            msg = 'source selection {} not recognized'.format(src_indx)
            raise ValueError(msg)
        self.ui.input_layout.addWidget(type_widget,row,optools.type_col,1,1)
        self.ui.input_layout.addWidget(val_widget,row,optools.val_col,1,1)
        if btn_widget:
            self.ui.input_layout.addWidget(btn_widget,row,optools.btn_col,1,1)
        self.fetch_data(name,src_indx,type_widget,val_widget)

    def load_text_input(self,name,type_widg,val_widg,edit_text=None):
        type_indx = type_widg.currentIndex()
        if type_indx == optools.int_type:
            val = int(val_widg.text())
        elif type_indx == optools.float_type:
            val = float(val_widg.text())
        #elif src_indx == optools.array_type:
        #    val = np.array(val_widg.text())
        elif type_indx == optools.string_type:
            val = val_widg.text()
        else:
            msg = 'type selection {}, should be between 1 and {}'.format(type_indx,len(optools.valid_types))
            raise ValueError(msg)
        self.op.inputs[name] = optools.InputLocator(optools.text_input,val)
        self.update_op_info(self.op.description())

    def fetch_data(self,name,src_indx,type_widg,val_widg):
        """Use a popup to select the requested input data"""
        if name in [k for k,v in self.inp_src_windows.items()]:
            # TODO: make sure the source is still the same! If not, close existing src window and make a new one.
            if src_indx == self.inp_src_windows[name][0]:
                self.inp_src_windows[name][1].raise_()
                self.inp_src_windows[name][1].activateWindow()
                return
            else:
                self.inp_src_windows[name][1].close()
                del self.inp_src_windows[name]
        if src_indx == 0 or src_indx == optools.text_input:
            pass
        else:
            ui_file = QtCore.QFile(slacxtools.rootdir+"/slacxui/tree_browser.ui")
            ui_file.open(QtCore.QFile.ReadOnly)
            src_ui = QtUiTools.QUiLoader().load(ui_file)
            ui_file.close()
            src_ui.setAttribute(QtCore.Qt.WA_DeleteOnClose)
            src_ui.setParent(self.ui,QtCore.Qt.Window)
            if src_indx == optools.op_input:
                trmod = self.wfman
            elif src_indx == optools.fs_input:
                trmod = QtGui.QFileSystemModel()
                trmod.setRootPath('.')
            else:
                # This should never happen
                msg = '[{}] Trying to build tree for data source {}: not implemented'.format(__name__,src_indx)
                raise ValueError(msg)
            src_ui.tree.setModel(trmod)
            src_ui.tree_box.setTitle(name)
            self.inp_src_windows[name] = (src_indx,src_ui)
            src_ui.tree.expandAll()
            src_ui.tree.resizeColumnToContents(0)
            src_ui.load_button.setText('Load selected data')
            src_ui.load_button.clicked.connect(partial(self.load_from_tree,name,trmod,src_ui,src_indx,type_widg,val_widg))
            src_ui.tree.doubleClicked.connect(partial(self.load_from_tree,name,trmod,src_ui,src_indx,type_widg,val_widg))
            if src_indx == optools.fs_input:
                src_ui.tree.hideColumn(1)
                src_ui.tree.hideColumn(3)
                src_ui.tree.setColumnWidth(0,400)
            src_ui.show()
            src_ui.raise_()
            src_ui.activateWindow()

    def load_from_tree(self,name,trmod,src_ui,src_indx,type_widg,val_widg,item_indx=None):
        """
        Load the item selected in QTreeView src_ui.tree.
        Construct a unique resource identifier (uri) for that item.
        Set self.op.inputs[name] to be an optools.InputLocator(src_indx,uri).
        Also set that uri to be the text of val_widg.
        Finally, reset self.ui.op_info to reflect the changes.
        """
        trview = src_ui.tree
        if not item_indx:
            # Get the selected item in QTreeView trview:
            item_indx = trview.currentIndex()
        if src_indx == optools.fs_input:
            # Get the path of the selection
            item_uri = trmod.filePath(item_indx)
            type_widg.setText('file path')
        elif src_indx == optools.op_input:
            # Build a unique URI for this item
            item_uri = trmod.build_uri(item_indx)
            type_widg.setText(type(trmod.get_item(item_indx).data[0]).__name__)
        else:
            # This should never happen
            msg = '[{}] Trying to fetch URI for data source {}: not implemented'.format(__name__,src_indx)
            raise ValueError(msg)
        val_widg.setText(item_uri)
        val_widg.setMinimumWidth(min([10*len(item_uri),200]))
        val_widg.setMaximumWidth(200)
        self.op.inputs[name] = optools.InputLocator(src_indx,item_uri)
        self.update_op_info(self.op.description())
        #self.inp_src_windows[name][1].close()
        #del self.inp_src_windows[name]
        src_ui.close()

    def update_op_info(self,text):
        self.ui.op_info.setPlainText(text)

    def text_widget(self,text):
        widg = QtGui.QLineEdit(text)
        widg.setReadOnly(True)
        widg.setAlignment(QtCore.Qt.AlignHCenter)
        return widg 

    def src_selection_widget(self):
        widg = QtGui.QComboBox()
        widg.addItems(optools.input_sources)
        #widg.setMinimumContentsLength(20)
        return widg 

    def item_selection_widget(self):
        widg = QtGui.QPushButton('Select...')
        return widg

    def vert_hdr_widget(self,text):
        # TODO: Fix this, some day.
        widg = optools.VertQLineEdit(text)
        return widg 

    def hdr_widget(self,text):
        widg = QtGui.QLineEdit(text)
        widg.setReadOnly(True)
        widg.setAlignment(QtCore.Qt.AlignHCenter)
        widg.setStyleSheet( "QLineEdit { background-color: transparent }" + widg.styleSheet() )
        return widg 

    def smalltext_widget(self,text):
        widg = self.text_widget(text)
        widg.setMaximumWidth( 20 )
        widg.setStyleSheet( "QLineEdit { background-color: transparent }" + widg.styleSheet() )
        return widg


