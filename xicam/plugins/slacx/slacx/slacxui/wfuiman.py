import time
import os
from functools import partial

from PySide import QtCore, QtGui, QtUiTools
import qdarkstyle
import numpy as np

from ..slacxcore.listmodel import ListModel
from ..slacxcore.operations import optools
from ..slacxcore.operations.slacxop import Operation 
from ..slacxcore.workflow.slacxwfman import WfManager
from ..slacxcore import slacxtools
from ..slacxcore.operations.optools import InputLocator
from . import uitools

class WfUiManager(object):
    """
    Stores a reference to the op_builder QGroupBox, 
    performs operations on it
    """

    def __init__(self,wfman,opman):
        ui_file = QtCore.QFile(slacxtools.rootdir+"/slacxui/wf_editor.ui")
        # Load the op_builder popup
        ui_file.open(QtCore.QFile.ReadOnly)
        self.ui = QtUiTools.QUiLoader().load(ui_file)
        ui_file.close()
        self.wfman = wfman 
        self.opman = opman 
        self.op = None
        # Dicts to keep track of input widgets,
        # keyed by input variable names
        self.src_widgets = {} 
        self.type_widgets = {} 
        self.val_widgets = {} 
        self.btn_widgets = {} 
        #self.inp_src_windows = {} 
        self.setup_ui()

    def get_op(self,trmod,item_indx):
        xitem = trmod.get_item(item_indx)
        if xitem.data:
            x = xitem.data
            # TODO: cleaner type checking?
            try:
                new_op_flag = issubclass(x,Operation)
            except:
                new_op_flag = False
            try:
                existing_op_flag = isinstance(x,Operation)
            except:
                existing_op_flag = False
            if not new_op_flag and not existing_op_flag:  
                self.clear_nameval_list()
                self.ui.op_info.setPlainText('Selected item: {}'.format(x))
            elif new_op_flag:
                # Create a new Operation 
                self.create_op(x)
            elif existing_op_flag:
                # Load existing Operation
                self.set_op(x,xitem.tag())

    def set_op(self,op,uri):
        self.op = op
        self.ui.op_info.setPlainText(self.op.description())
        self.build_nameval_list()
        self.ui.uri_entry.setText(uri)
        # Don't let uri change after already being loaded.
        self.ui.uri_entry.setReadOnly(True)

    def create_op(self,op):
        self.op = op()
        self.ui.op_info.setPlainText(self.op.description())
        self.build_nameval_list()
        self.ui.uri_entry.setText(self.wfman.next_uri(type(self.op).__name__))
        self.ui.uri_entry.setReadOnly(False)

    def test_op(self):
        print 'Operation testing not yet implemented'

    def set_input(self,name,src_ui=None,item_indx=None):
        """
        Load input indicated by name into an InputLocator. 
        Store it in self.op.input_locator[name].
        """
        il = self.load_input(name,src_ui,item_indx)
        self.op.input_locator[name] = il
        #set op.input_src and op.input_type to assist in using il.val
        self.op.input_src[name] = il.src
        self.op.input_type[name] = il.tp

    def load_input(self,name,ui=None,item_indx=None):
        src = self.src_widgets[name].currentIndex()
        tp = self.type_widgets[name].currentIndex()
        if src == optools.no_input:
            il = optools.InputLocator() 
        elif src == optools.batch_input:
            val = None
            il = optools.InputLocator(src,tp,val) 
        elif src == optools.user_input:
            if tp == optools.list_type:
                val = ui.list_builder.list_data() 
            else:
                val = self.val_widgets[name].text()
            il = optools.InputLocator(src,tp,val)
        elif (src == optools.wf_input or src == optools.fs_input):
            if tp == optools.list_type:
                val = ui.list_view.model().list_data() 
                #import pdb; pdb.set_trace()
                il = optools.InputLocator(src,tp,val)
            else:
                il = self.load_from_tree(ui,src,item_indx)
            if not il: 
                if self.op.input_locator[name] is not None:
                    il = self.op.input_locator[name]
                else: 
                    val = None
                    il = optools.InputLocator(src,tp,val)
        else: 
            il = optools.InputLocator()
        self.val_widgets[name].setText(str(il.val))
        if ui:
            ui.close()
            ui.deleteLater()
        return il

    def load_from_tree(self,src_ui,src,item_indx=None):
        """
        Construct a unique resource identifier (uri) for the selected item.
        return an optools.InputLocator(src,tp,uri).
        By design this should only be called when the corresponding input source window
        (containing a TreeView widget) is open.
        """
        trview = src_ui.tree
        if not item_indx or not item_indx.isValid():
            item_indx = trview.currentIndex()
        if item_indx.isValid():
            if src == optools.fs_input:
                item_uri = trview.model().filePath(item_indx)
            elif src == optools.wf_input:
                item_uri = trview.model().build_uri(item_indx)
            il = optools.InputLocator(src,optools.auto_type,item_uri)
        else:
            il = None
        return il

    def rm_op(self):
        """
        remove the selected operation from the workflow
        """
        current_indx = self.ui.wf_selector.currentIndex()
        if current_indx.isValid(): 
            self.wfman.remove_op(current_indx)
 
    def load_op(self):
        """
        Package the finished self.op(Operation), ship to self.wfman
        """ 
        # Make sure all inputs are loaded
        for name in self.op.inputs.keys():
            # By design, load_op should only be called 
            # when (modal) input source tree browser windows are closed,
            # so skip this if src is fs or wf.
            src = self.src_widgets[name].currentIndex()
            if not src == optools.wf_input and not src == optools.fs_input:
                self.set_input(name)
            #self.op.input_locator[name] = self.load_input(name) 
        uri = self.ui.uri_entry.text()
        result = self.wfman.is_good_tag(uri)
        if result[0]:
            self.wfman.add_op(uri,self.op) 
            #self.ui.close()
            #self.ui.deleteLater()
        elif result[1] == 'Tag not unique':
            self.wfman.update_op(uri,self.op)
        else:
            # Request a different uri 
            msg_ui = slacxtools.start_message_ui()
            msg_ui.setParent(self.ui,QtCore.Qt.Window)
            msg_ui.setWindowModality(QtCore.Qt.WindowModal)
            msg_ui.setWindowTitle("Tag Error")
            msg_ui.message_box.setPlainText(
            'Tag error for {}: \n{} \n\n'.format(uri, result[1])
            + 'Enter a unique alphanumeric uri, '
            + 'using only letters, numbers, -, and _. (no periods). ')
            # Set button to activate on Enter key
            msg_ui.ok_button.setFocus()
            msg_ui.show()

    def clear_nameval_list(self):
        self.ui.op_name.setText(' ')
        n_inp_widgets = self.ui.input_layout.count()
        for i in range(n_inp_widgets-1,-1,-1):
            item = self.ui.input_layout.takeAt(i)
            item.widget().close()
            item.widget().deleteLater()
        n_out_widgets = self.ui.output_layout.count()
        for i in range(n_out_widgets-1,-1,-1):
            item = self.ui.output_layout.takeAt(i)
            item.widget().close()
            item.widget().deleteLater()

    def srcwindow_safe_close(self,widg):
        try:
            widg.close()
            widg.deleteLater()
        except RuntimeError as ex:
            print 'avoided RuntimeError while clearing widgets. Error message: {}'.format(ex.message)

    def build_nameval_list(self):
        self.clear_nameval_list()
        inp_count = len(self.op.inputs)
        out_count = len(self.op.outputs)
        self.ui.op_name.setText(type(self.op).__name__)
        if inp_count:
            self.input_header_widgets(0)
            i=1
            for name in self.op.inputs.keys():
                self.add_input_widgets(name,i)
                i+=1
        if out_count:
            self.output_header_widgets(0)
            i=1 
            for name in self.op.outputs.keys():
                self.add_output_widgets(name,i)
                i+=1 

    def input_header_widgets(self,row):
        self.ui.input_layout.addWidget(uitools.r_hdr_widget('name'),row,self.name_col,1,1)
        self.ui.input_layout.addWidget(uitools.hdr_widget('source'),row,self.src_col,1,1)
        self.ui.input_layout.addWidget(uitools.hdr_widget('type'),row,self.type_col,1,1)
        self.ui.input_layout.addWidget(uitools.hdr_widget('value'),row,self.val_col,1,1)

    def output_header_widgets(self,row):
        self.ui.output_layout.addWidget(uitools.r_hdr_widget('name'),row,self.name_col,1,1)
        self.ui.output_layout.addWidget(uitools.hdr_widget('description'),row,self.src_col,1,self.btn_col-self.src_col)

    def add_input_widgets(self,name,row):
        """Loads a set of widgets for setting or reading input or output data"""
        name_widget = uitools.name_widget(name)
        self.ui.input_layout.addWidget( name_widget,row,self.name_col,1,1 )
        eq_widget = uitools.smalltext_widget('=')
        eq_widget.setMaximumWidth(20)
        self.ui.input_layout.addWidget(eq_widget,row,self.eq_col,1,1)
        src_widget = uitools.src_selection_widget() 
        self.src_widgets[name] = src_widget 
        #val_widget = QtGui.QLineEdit(str(val))
        self.ui.input_layout.addWidget(src_widget,row,self.src_col,1,1)
        src_widget.activated.connect( partial(self.render_type_widget,name,row) )
        if self.op.input_locator[name] is not None:
            src = self.op.input_locator[name].src
        else:
            src = self.op.input_src[name]
        src_widget.setCurrentIndex(src)
        self.render_type_widget(name,row,src) 
        #name_widget.resize(10,name_widget.size().height())
        ht = name_widget.sizeHint().height()
        name_widget.sizeHint = lambda: QtCore.QSize(10*len(name_widget.text()),ht)
        name_widget.setSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Fixed)

    def add_output_widgets(self,name,row):
        name_widget = uitools.name_widget(name)
        self.ui.output_layout.addWidget(name_widget,row,self.name_col)
        eq_widget = uitools.smalltext_widget('=')
        eq_widget.setMaximumWidth(20)
        self.ui.output_layout.addWidget(eq_widget,row,self.eq_col)
        if self.op.output_doc[name]:
            desc_widget = uitools.bigtext_widget(self.op.output_doc[name])
        else:
            desc_widget = uitools.bigtext_widget('No output doc found.')
        self.ui.output_layout.addWidget(desc_widget,row,self.src_col,1,self.btn_col-self.src_col)
        ht = desc_widget.sizeHint().height()
        name_widget.sizeHint = lambda: QtCore.QSize(10*len(name_widget.text()),ht)
        desc_widget.sizeHint = lambda: QtCore.QSize(20*len(desc_widget.text()),ht)
        name_widget.setSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Fixed)
        desc_widget.setSizePolicy(QtGui.QSizePolicy.Maximum,QtGui.QSizePolicy.Fixed)

    def render_type_widget(self,name,row,src=None):
        if not src:
            src = self.src_widgets[name].currentIndex()
        if name in self.type_widgets.keys():
            if self.type_widgets[name]:
                self.type_widgets[name].close()
        # if input source windows exist, close those too.
        #if name in self.inp_src_windows.keys():
        #    self.srcwindow_safe_close(name)
        type_widget = uitools.type_mv_widget(src) 
        if src == optools.user_input:
            nonsense_types = [optools.auto_type]
            if self.op.input_locator[name]:
                if self.op.input_locator[name].tp not in nonsense_types:
                    type_widget.setCurrentIndex(self.op.input_locator[name].tp)
            elif self.op.input_type[name]:
                if self.op.input_type[name] not in nonsense_types:
                    type_widget.setCurrentIndex(self.op.input_type[name])
        elif (src == optools.wf_input or src == optools.fs_input):
            nonsense_types = [optools.str_type,optools.int_type,optools.float_type,optools.bool_type]
            if self.op.input_locator[name]:
                if self.op.input_locator[name].tp not in nonsense_types:
                    type_widget.setCurrentIndex(self.op.input_locator[name].tp)
            elif self.op.input_type[name]:
                if self.op.input_type[name] not in nonsense_types:
                    type_widget.setCurrentIndex(self.op.input_type[name])
        self.ui.input_layout.addWidget(type_widget,row,self.type_col,1,1)
        self.type_widgets[name] = type_widget
        type_widget.activated.connect( partial(self.render_val_widget,name,row) )            
        self.render_val_widget(name,row)

    def render_val_widget(self,name,row,tp=None):
        if name in self.val_widgets.keys():
            if self.val_widgets[name]:
                self.val_widgets[name].close()
        if name in self.btn_widgets.keys():
            if self.btn_widgets[name]:
                self.btn_widgets[name].close()
        src = self.src_widgets[name].currentIndex()
        if not tp:
            tp = self.type_widgets[name].currentIndex()
        #print 'render val/btn widgets for src {}, tp {}'.format(src,tp)
        btn_widget = QtGui.QPushButton()
        val_widget = QtGui.QLineEdit()
        if src == optools.no_input: 
            btn_widget.setText('no input')
            btn_widget.setEnabled(False)
            val_widget.setText('None')
            val_widget.setReadOnly(True)
        elif src == optools.batch_input:
            btn_widget.setText('auto input')
            btn_widget.setEnabled(False)
            val_widget.setText('auto')
            val_widget.setReadOnly(True)
        elif src == optools.wf_input or src == optools.fs_input:
            if tp == optools.none_type:
                btn_widget.setText('no input')
                btn_widget.setEnabled(False)
                val_widget.setText('None')
                val_widget.setReadOnly(True)
            elif tp == optools.list_type:
                btn_widget.setText("build list...")
                btn_widget.clicked.connect( partial(self.build_list,name) )
            else:
                btn_widget.setText('browse...')
                btn_widget.clicked.connect( partial(self.fetch_data,name) )
            if self.op.input_locator[name]:
                val_widget.setText(str(self.op.input_locator[name].val))
            elif self.op.inputs[name]:
                val_widget.setText(str(self.op.inputs[name]))
            val_widget.setReadOnly(True)
        elif (src == optools.user_input):
            if tp == optools.none_type:
                btn_widget.setText('no input')
                btn_widget.setEnabled(False)
                val_widget.setText('None')
                val_widget.setReadOnly(True)
            else:
                if self.op.input_locator[name]:
                    val_widget.setText(str(self.op.input_locator[name].val))
                elif self.op.inputs[name]:
                    val_widget.setText(str(self.op.inputs[name]))
                if tp == optools.list_type:
                    val_widget.setReadOnly(True)
                    btn_widget.setText("build list...")
                    btn_widget.clicked.connect( partial(self.build_list,name) )
                else:
                    btn_widget = QtGui.QPushButton('Load')
                    btn_widget.clicked.connect( partial(self.set_input,name) )
        self.ui.input_layout.addWidget(val_widget,row,self.val_col,1,1)
        self.ui.input_layout.addWidget(btn_widget,row,self.btn_col,1,1)
        self.val_widgets[name] = val_widget
        self.btn_widgets[name] = btn_widget

    def build_list(self,name):
        """Use a popup to build a list of input data"""
        src = self.src_widgets[name].currentIndex()
        ui_file = QtCore.QFile(slacxtools.rootdir+"/slacxui/list_builder.ui")
        ui_file.open(QtCore.QFile.ReadOnly)
        list_ui = QtUiTools.QUiLoader().load(ui_file)
        ui_file.close()
        list_ui.setParent(self.ui,QtCore.Qt.Window)
        list_ui.setWindowModality(QtCore.Qt.WindowModal)
        list_ui.setWindowTitle("build list from {}".format(optools.input_sources[src]))
        lm = ListModel([],list_ui)
        list_ui.list_view.setModel(lm)
        list_ui.browse_button.setText('browse...')
        list_ui.browse_button.clicked.connect( partial(self.load_from_src,src,list_ui) )
        list_ui.type_selector = uitools.type_mv_widget(src,list_ui.type_selector)
        if src == optools.user_input:
            list_ui.browse_button.setEnabled(False)
            if uitools.have_qt47:
                list_ui.value_entry.setPlaceholderText('(enter value)')
            else:
                list_ui.value_entry.setText('')
        else:
            list_ui.load_button.setEnabled(False)
            list_ui.value_entry.setReadOnly(True)
            list_ui.type_selector.model().set_disabled(optools.none_type)
        list_ui.type_selector.model().set_disabled(optools.list_type)
        list_ui.load_button.setText('Load')
        list_ui.load_button.clicked.connect( partial(self.load_value_to_list,src,list_ui) )
        list_ui.finish_button.setText('Finish')
        list_ui.finish_button.clicked.connect( partial(self.set_input,name,list_ui) )
        list_ui.remove_button.setText('Remove selected item')
        list_ui.remove_button.clicked.connect( partial(self.rm_from_list,list_ui) )
        list_ui.value_header.setText('value')
        list_ui.value_header.setStyleSheet( "QLineEdit { background-color: transparent }" + list_ui.value_header.styleSheet() )
        list_ui.type_header.setText('type')
        list_ui.type_header.setStyleSheet( "QLineEdit { background-color: transparent }" + list_ui.type_header.styleSheet() )
        list_ui.show()

    @staticmethod
    def load_path_to_list(src,src_ui,list_ui,idx=None):
        if not idx:
            idx = src_ui.tree.currentIndex()
        if idx.isValid():
            list_ui.value_entry.setText( str(src_ui.tree.model().data(idx,QtCore.Qt.DisplayRole)) )
        src_ui.close()
        src_ui.deleteLater()
        list_ui.list_view.model().append_item( str(list_ui.value_entry.text()) )

    @staticmethod
    def load_value_to_list(src,list_ui):
        # typecast and load the value_entry.text()
        tp = list_ui.type_selector.currentIndex()
        val = optools.cast_type_val(tp,list_ui.value_entry.text())
        list_ui.list_view.model().append_item(val)

    def rm_from_list(self,list_ui):
        idx = list_ui.list_view.currentIndex()
        if idx.isValid():
            row = idx.row()
            list_ui.list_view.model().remove_item(row)

    def load_from_src(self,src,list_ui):
        src_ui = self.data_fetch_ui(src,list_ui)
        src_ui.load_button.clicked.connect(partial(self.load_path_to_list,src,src_ui,list_ui))
        src_ui.tree.doubleClicked.connect(partial(self.load_path_to_list,src,src_ui,list_ui))
        src_ui.tree.clicked.connect( partial(uitools.toggle_expand,src_ui.tree) )
        if src == optools.fs_input:
            src_ui.tree.hideColumn(1)
            src_ui.tree.hideColumn(3)
            src_ui.tree.setColumnWidth(0,400)
        src_ui.show()

    def data_fetch_ui(self,src,parent=None):
        ui_file = QtCore.QFile(slacxtools.rootdir+"/slacxui/load_browser.ui")
        ui_file.open(QtCore.QFile.ReadOnly)
        src_ui = QtUiTools.QUiLoader().load(ui_file)
        ui_file.close()
        src_ui.setParent(parent,QtCore.Qt.Window)
        src_ui.setWindowModality(QtCore.Qt.WindowModal)
        src_ui.setWindowTitle("data loader")
        if src == optools.wf_input:
            trmod = self.wfman
        elif src == optools.fs_input:
            trmod = QtGui.QFileSystemModel()
            trmod.setRootPath('.')
        src_ui.tree.setModel(trmod)
        if src == optools.wf_input:
            src_ui.tree.expandToDepth(2)
        elif src == optools.fs_input:
            src_ui.tree.expandAll()
        src_ui.tree.resizeColumnToContents(0)
        src_ui.load_button.setText('Load selected data')
        src_ui.tree_box.setTitle(optools.input_sources[src])
        return src_ui

    def fetch_data(self,name):
        """Use a popup to select the input data"""
        src = self.src_widgets[name].currentIndex()
        src_ui = self.data_fetch_ui(src,self.ui)
        src_ui.load_button.clicked.connect(partial(self.set_input,name,src_ui))
        src_ui.tree.clicked.connect( partial(uitools.toggle_expand,src_ui.tree) )
        src_ui.tree.doubleClicked.connect(partial(self.set_input,name,src_ui))
        if src == optools.fs_input:
            src_ui.tree.hideColumn(1)
            src_ui.tree.hideColumn(3)
            src_ui.tree.setColumnWidth(0,400)
        src_ui.show()

    def setup_ui(self):
        self.ui.setWindowTitle("workflow setup")
        self.ui.input_box.setTitle("INPUTS")
        self.ui.output_box.setTitle("OUTPUTS")
        self.ui.finish_box.setTitle("FINISH / LOAD")
        #self.ui.input_box.setMinimumWidth(600)
        #self.ui.op_frame.setMinimumWidth(400)
        #self.ui.op_frame.setMaximumWidth(400)
        ht = self.ui.op_frame.sizeHint().height()
        self.ui.op_frame.sizeHint = lambda: QtCore.QSize(400,ht)
        self.ui.op_frame.setSizePolicy(
        QtGui.QSizePolicy.Minimum,self.ui.op_frame.sizePolicy().verticalPolicy())
        self.ui.wf_selector.setModel(self.wfman)
        #self.ui.wf_selector.hideColumn(1)
        self.ui.wf_selector.clicked.connect( partial(self.get_op,self.wfman) )
        self.ui.rm_op_button.setText("&Delete")
        self.ui.rm_op_button.clicked.connect(self.rm_op)
        self.ui.op_selector.setModel(self.opman)
        self.ui.op_selector.hideColumn(1)
        self.ui.op_selector.clicked.connect( partial(self.get_op,self.opman) )
        self.ui.op_selector.clicked.connect( partial(uitools.toggle_expand,self.ui.op_selector) ) 
        self.ui.wf_selector.clicked.connect( partial(uitools.toggle_expand,self.ui.wf_selector) )
        #self.ui.op_selector.activated.connect( partial(self.get_op,self.opman) )
        # Populate uri entry fields
        self.ui.uri_prompt.setText('operation uri:')
        #self.ui.uri_prompt.setMaximumWidth(150)
        #self.ui.uri_entry.setMaximumWidth(150)
        self.ui.uri_prompt.setReadOnly(True)
        self.ui.op_name.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.uri_prompt.setAlignment(QtCore.Qt.AlignRight)
        self.ui.uri_prompt.setStyleSheet( "QLineEdit { background-color: transparent }" 
        + self.ui.uri_prompt.styleSheet() )
        self.ui.test_button.setText("&Test")
        self.ui.test_button.setEnabled(False)
        self.ui.test_button.clicked.connect(self.test_op)
        self.ui.load_button.setText("&Load")
        self.ui.load_button.clicked.connect(self.load_op)
        self.ui.load_button.setDefault(True)
        self.ui.test_button.setMinimumWidth(100)
        self.ui.load_button.setMinimumWidth(100)
        #self.ui.exit_button.setText("E&xit")
        #self.ui.exit_button.hide()
        #self.ui.exit_button.clicked.connect(self.ui.close)
        #self.ui.exit_button.clicked.connect(self.ui.deleteLater)
        self.ui.splitter.setStretchFactor(0,1000)    
        #self.ui.returnPressed.connect(self.load_op)
        self.ui.setStyleSheet( "QLineEdit { border: none }" + self.ui.styleSheet() )
        #self.ui.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.name_col = 1
        self.eq_col = 2
        self.src_col = 3
        self.type_col = 4
        self.val_col = 5
        self.btn_col = 6


