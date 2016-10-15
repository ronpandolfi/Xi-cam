import time
import os
from functools import partial

from PySide import QtCore, QtGui, QtUiTools

from ..slacxcore.operations import optools
from ..slacxcore import slacxex
from . import uitools

class OpUiManager(object):
    """
    Stores a reference to the op_builder QGroupBox, 
    performs operations on it
    """

    def __init__(self,rootdir,wfman,imgman,opman):
        self.rootdir = rootdir
        ui_file = QtCore.QFile(self.rootdir+"/slacxui/op_builder.ui")
        # Load the op_builder popup
        ui_file.open(QtCore.QFile.ReadOnly)
        self.ui = QtUiTools.QUiLoader().load(ui_file)
        ui_file.close()
        self.wfman = wfman 
        self.imgman = imgman 
        self.opman = opman 
        self.op = None
        self.setup_ui()
        #self.fetch_op()

    def fetch_op(self):
        """Use a QtGui.QTreeView popup to select an Operation"""
        ui_file = QtCore.QFile(self.rootdir+"/slacxui/tree_browser.ui")
        ui_file.open(QtCore.QFile.ReadOnly)
        src_ui = QtUiTools.QUiLoader().load(ui_file)
        ui_file.close()
        src_ui.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        src_ui.setParent(self.ui,QtCore.Qt.Window)
        src_ui.tree.setModel(self.opman)
        src_ui.tree.resizeColumnToContents(0)
        src_ui.tree.resizeColumnToContents(1)
        #src_ui.tree.expandAll()
        src_ui.load_button.setText('Load selected Operation')
        src_ui.load_button.clicked.connect( partial(self.get_op_from_tree,src_ui) )
        src_ui.show()

    def get_op_from_tree(self,src_ui):
        # Get the selected item in QTreeView:
        item_indx = src_ui.tree.currentIndex()
        # get the Operation of this item 
        # TODO: make sure the selection is an Operation
        op = self.opman.get_item(item_indx).data[0]
        self.create_op(op)
        #self.update_op_info(self.op.description())
        src_ui.close()

    def set_op(self,op):
        """
        Load a fully or partially formed Operation into the UI.
        Set self.op to input Operation.
        Set op_selector button text to the Operation name.
        Load op.description() into self.ui.op_info.
        Load op.inputs and op.outputs into self.ui.arg_frame.
        """
        self.op = op
        self.ui.op_selector.setText(op.__name__)
        # TODO: Don't let the user change the type of operation? For their own safety? 
        self.ui.op_info.setPlainText(' ')
        self.ui.op_info.setPlainText(op.description())
        self.clear_nameval_list()
        self.build_nameval_list()

    def setup_ui(self):
        self.ui.setWindowTitle("slacx operation builder")
        self.ui.arg_frame.setMinimumWidth(700)
        self.ui.op_frame.setMinimumWidth(300)
        self.ui.op_frame.setMaximumWidth(300)
        self.ui.op_selector.setText('Click to Select Operation')
        self.ui.op_selector.clicked.connect( partial(self.fetch_op) )
        # Populate tag entry fields
        self.ui.tag_prompt.setText('enter a unique tag for this operation:')
        self.ui.tag_prompt.setMinimumWidth(200)
        self.ui.tag_prompt.setReadOnly(True)
        self.ui.tag_prompt.setAlignment(QtCore.Qt.AlignRight)
        self.ui.tag_entry.setText(self.default_tag())
        # Connect ui.finish_button with self.load_op 
        self.ui.finish_button.setText("Finish")
        self.ui.finish_button.setMinimumWidth(100)
        self.ui.finish_button.clicked.connect(self.load_op)
        # Also connect finish_button with data loader for any text inputs
        #self.ui.finish_button.clicked.connect(self.load_all_text_inputs)
        # Set button to activate on Enter key?
        self.ui.finish_button.setDefault(True)
        self.ui.tag_entry.returnPressed.connect(self.load_op)
        self.ui.setStyleSheet( "QLineEdit { border: none }" + self.ui.styleSheet() )
        # Set up the ui widget to delete itself when closed
        self.ui.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        # Define columns for i/o widget arrangement
        self.name_col = 1
        self.eq_col = 2
        self.src_col = 3
        self.type_col = 4
        self.val_col = 5
        self.btn_col = 6

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
        # Clear description window 
        self.ui.op_info.setPlainText(' ')
        # Create op 
        self.op = op()
        # Set self.ui.op_selector to operation name
        self.ui.op_selector.setText(op.__name__)
        # Set self.ui.op_info to op.description
        self.ui.op_info.setPlainText(self.op.description())
        # Clear the view of name-value widgets
        self.clear_nameval_list()
        self.build_nameval_list()

    def load_op(self):
        """
        Package self.op(Operation), ship to self.wfman
        """ 
        tag = self.ui.tag_entry.text()
        result = self.wfman.is_good_tag(tag)
        if result[0]:
            # Make sure all text inputs are loaded
            for row in self.input_rows:
                src = self.ui.nameval_layout.itemAtPosition(row,self.src_col).widget().currentIndex()
                if src == optools.text_input:
                    name = self.ui.nameval_layout.itemAtPosition(row,self.name_col).widget().text()
                    type_widget = self.ui.nameval_layout.itemAtPosition(row,self.type_col).widget()
                    val_widget = self.ui.nameval_layout.itemAtPosition(row,self.val_col).widget()
                    self.load_text_input(name,type_widget,val_widget) 
            self.wfman.add_op(self.op,tag) 
            self.ui.close()
        else:
            # Request a different tag
            msg_ui = slacxex.start_message_ui()
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
        #self.nameval_dict = {}
        # Count the items in the current layout
        n_val_widgets = self.ui.nameval_layout.count()
        # Loop through them, last to first, clear the frame
        for i in range(n_val_widgets-1,-1,-1):
            # QLayout.takeAt returns a LayoutItem
            widg = self.ui.nameval_layout.takeAt(i)
            # get the QWidget of that LayoutItem and set it to deleteLater()
            widg.widget().deleteLater()

    def build_nameval_list(self):
        # Count the items in the current layout
        n_val_widgets = self.ui.nameval_layout.count()
        # Loop through them, last to first, clear the frame
        for i in range(n_val_widgets-1,-1,-1):
            # QLayout.takeAt returns a LayoutItem- set its widget to deleteLater()
            item = self.ui.nameval_layout.takeAt(i)
            item.widget().deleteLater()
        i=0
        self.ui.nameval_layout.addWidget(self.text_widget('--- INPUTS ---'),i,0,1,6) 
        i+=1
        self.ui.nameval_layout.addWidget(self.hdr_widget('(name)'),i,self.name_col,1,1) 
        self.ui.nameval_layout.addWidget(self.hdr_widget('(source)'),i,self.src_col,1,1) 
        self.ui.nameval_layout.addWidget(self.hdr_widget('(type)'),i,self.type_col,1,1) 
        self.ui.nameval_layout.addWidget(self.hdr_widget('(value)'),i,self.val_col,1,1) 
        i+=1 
        self.input_rows = []
        for name, val in self.op.inputs.items():
            self.add_nameval_widgets(name,val,i)
            self.input_rows.append(i) 
            #src_widg,val_widg = self.add_nameval_widgets(name,val,i)
            #self.op_input_widgets[name] = val_widg
            i+=1
        self.ui.nameval_layout.addWidget(self.smalltext_widget(' '),i,0,1,4) 
        i+=1 
        self.ui.nameval_layout.addWidget(self.text_widget('--- OUTPUTS ---'),i,0,1,6) 
        i+=1 
        self.ui.nameval_layout.addWidget(self.hdr_widget('(name)'),i,self.name_col,1,1) 
        self.ui.nameval_layout.addWidget(self.hdr_widget('(source)'),i,self.src_col,1,1) 
        self.ui.nameval_layout.addWidget(self.hdr_widget('(type)'),i,self.type_col,1,1) 
        self.ui.nameval_layout.addWidget(self.hdr_widget('(value)'),i,self.val_col,1,1) 
        i+=1 
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
        self.ui.nameval_layout.addWidget(name_widget,row,self.name_col)
        eq_widget = self.smalltext_widget('=')
        self.ui.nameval_layout.addWidget(eq_widget,row,self.eq_col)
        val_widget = QtGui.QLineEdit(str(val))
        if output:
            src_widget = QtGui.QLineEdit(self.ui.tag_entry.text())
            type_widget = QtGui.QLineEdit(type(val).__name__)
            type_widget.setReadOnly(True)
            val_widget.setReadOnly(True)
            self.ui.nameval_layout.addWidget(src_widget,row,self.src_col,1,1)
            self.ui.nameval_layout.addWidget(type_widget,row,self.type_col,1,1)
            self.ui.nameval_layout.addWidget(val_widget,row,self.val_col,1,1)
        else:
            src_widget = self.src_selection_widget() 
            src_widget.setMinimumWidth(90)
            self.ui.nameval_layout.addWidget(src_widget,row,self.src_col,1,1)
            # Note the widget.activated signal sends the index of the activated item.
            # This will be passed as the next (unspecified) arg to the partial.
            src_widget.activated.connect( partial(self.render_input_widgets,name,row) )
            # Set up the input line for text input, for starters
            # TODO: A more elegant initialization
            self.render_input_widgets(name,row,0) 
        #return src_widget,val_widget

    def render_input_widgets(self,name,row,src_indx): 
        # If input widgets exist, close them.
        for col in [self.btn_col,self.val_col,self.type_col]:
            if self.ui.nameval_layout.itemAtPosition(row,col):
                widg = self.ui.nameval_layout.itemAtPosition(row,col).widget()
                widg.hide()
                widg.deleteLater()
                #widg.destroy()
        if src_indx == 0:
            type_widget = QtGui.QLineEdit('(select source)')
            val_widget = QtGui.QLineEdit('(select source)')
            type_widget.setReadOnly(True)
            val_widget.setReadOnly(True)
            btn_widget = None
        elif src_indx == optools.text_input:
            type_widget = QtGui.QComboBox()
            type_widget.addItems(optools.input_types)
            val_widget = QtGui.QLineEdit()
            if uitools.have_qt47:
                val_widget.setPlaceholderText('(enter value)')
            else:
                val_widget.setText('(enter value)')
            #val_widget.returnPressed.connect( partial(self.load_text_input,name,type_widget,val_widget) )
            #val_widget.textChanged.connect( partial(self.load_text_input,name,type_widget,val_widget) )
            #val_widget.textEdited.connect( partial(self.load_text_input,name,type_widget,val_widget) )
            # TODO: Make this a button that tests the type-casting and loading of the input?
            btn_text = 'Load text...' 
            btn_widget = QtGui.QPushButton(btn_text)
            btn_widget.clicked.connect( partial(self.load_text_input,name,type_widget,val_widget) )
            # Also connect the finish button to load the text input
            #self.ui.finish_button.clicked.connect( partial(self.load_text_input,name,type_widget,val_widget) )
        elif (src_indx == optools.image_input
            or src_indx == optools.op_input):
            btn_text = 'Select data...'
            type_widget = QtGui.QLineEdit('auto')
            type_widget.setReadOnly(True)
            val_widget = QtGui.QLineEdit('select -->')
            val_widget.setReadOnly(True)
            btn_widget = QtGui.QPushButton(btn_text)
            btn_widget.clicked.connect( partial(self.fetch_data,name,src_indx,type_widget,val_widget) )
        else:
            msg = 'source selection {} not recognized'.format(src_indx)
            raise ValueError(msg)
        self.ui.nameval_layout.addWidget(type_widget,row,self.type_col,1,1)
        self.ui.nameval_layout.addWidget(val_widget,row,self.val_col,1,1)
        if btn_widget:
            self.ui.nameval_layout.addWidget(btn_widget,row,self.btn_col,1,1)

    def load_text_input(self,name,type_widg,val_widg,edit_text=None):
        src_indx = type_widg.currentIndex()
        if src_indx == optools.int_type:
            val = int(val_widg.text())
        elif src_indx == optools.float_type:
            val = float(val_widg.text())
        elif src_indx == optools.array_type:
            val = np.array(val_widg.text())
        elif src_indx == optools.string_type:
            val = val_widg.text()
        else:
            msg = 'type selection {}, should be between 1 and {}'.format(src_indx,len(optools.valid_types))
            raise ValueError(msg)
        self.op.inputs[name] = optools.InputLocator(optools.text_input,val)
        self.update_op_info(self.op.description())

    def fetch_data(self,name,src_indx,type_widg,val_widg):
        """Use a QtGui.QTreeView popup to select the requested input data"""
        # TODO: Allow only one of these popups to exist (one per val widget).
        ui_file = QtCore.QFile(self.rootdir+"/slacxui/tree_browser.ui")
        ui_file.open(QtCore.QFile.ReadOnly)
        src_ui = QtUiTools.QUiLoader().load(ui_file)
        ui_file.close()
        src_ui.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        src_ui.setParent(self.ui,QtCore.Qt.Window)
        if src_indx == optools.image_input:
            trmod = self.imgman
        elif src_indx == optools.op_input:
            trmod = self.wfman
        src_ui.tree.setModel(trmod)
        src_ui.tree.resizeColumnToContents(0)
        src_ui.tree.resizeColumnToContents(1)
        #for idx in trmod.iter_indexes():
        #    src_ui.tree.setExpanded(idx,True)
        src_ui.tree.expandAll()
        src_ui.load_button.setText('Load selected data')
        src_ui.load_button.clicked.connect(partial(self.load_from_tree,name,trmod,src_ui,src_indx,type_widg,val_widg))
        src_ui.show()

    def load_from_tree(self,name,trmod,src_ui,src_indx,type_widg,val_widg):
        """
        Load the item selected in QTreeView src_ui.tree.
        Construct a unique resource identifier (uri) for that item.
        Set self.op.inputs[name] to be an optools.InputLocator(src_indx,uri).
        Also set that uri to be the text of val_widg.
        Finally, reset self.ui.op_info to reflect the changes.
        """
        trview = src_ui.tree
        # Get the selected item in QTreeView trview:
        item_indx = trview.currentIndex()
        # Build a unique URI for this item
        item_uri = trmod.build_uri(item_indx)
        val_widg.setText(item_uri)
        val_widg.setMinimumWidth(10*len(item_uri))
        type_widg.setText(type(trmod.get_item(item_indx).data[0]).__name__)
        self.op.inputs[name] = optools.InputLocator(src_indx,item_uri)
        #self.ui.nameval_layout.update()
        self.update_op_info(self.op.description())
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


