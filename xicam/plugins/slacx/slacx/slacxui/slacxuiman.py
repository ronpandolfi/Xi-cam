import os
import time
from functools import partial

from PySide import QtGui, QtCore, QtUiTools
import numpy as np

from . import uitools
from .wfuiman import WfUiManager
from ..slacxcore.operations.slacxop import Operation
from ..slacxcore import slacxtools
from . import data_viewer

# TODO: Make a metaclass that generates Operation subclasses.
# TODO: Use the above to make an Op development interface. 
# TODO: Consider whether this should inherit from QtCore.QObject instaed of object?

class UiManager(object):
    """
    Stores a reference to a QMainWindow,
    performs operations on it
    """

    # TODO: when the QImageView widget gets resized,
    # it will call QWidget.resizeEvent().
    # Try to use this to resize the images in the QImageView.

    def __init__(self,opman,wfman):
        """Make a UI from ui_file, save a reference to it"""
        super(UiManager,self).__init__()
        # Pick a UI definition, load it up
        ui_file = QtCore.QFile(slacxtools.rootdir+"/slacxui/basic.ui")
        ui_file.open(QtCore.QFile.ReadOnly)
        # load() produces a QMainWindow(QWidget).
        self.ui = QtUiTools.QUiLoader().load(ui_file)
        ui_file.close()
        # Set up the self.ui widget to delete itself when closed
        self.ui.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.opman = opman 
        self.wfman = wfman 

#    def run_wf(self):
#        self.wfman.run_wf()

    def edit_wf(self,trmod,item_indx=QtCore.QModelIndex()):
        """
        Interact with user to edit the workflow.
        Pass in a TreeModel and index to open the editor 
        with the item at that index loaded.
        """
        if item_indx.isValid():
            idx = item_indx
            if trmod == self.wfman:
                while idx.parent().isValid():
                    idx = idx.parent()
            x = trmod.get_item(idx).data
        else:
            x = None
            idx = self.ui.wf_tree.currentIndex()
            if idx.isValid():
                while idx.parent().isValid():
                    idx = idx.parent()
                x = self.wfman.get_item(idx).data
            else:
                idx = self.ui.op_tree.currentIndex()
                if idx.isValid():# and self.opman.get_item(idx).data is not None:
                    x = self.opman.get_item(idx).data
        existing_op_flag = isinstance(x,Operation)
        try:
            new_op_flag = issubclass(x,Operation)
        except:
            new_op_flag = False
        #print 'existing op: {}'.format(existing_op_flag)
        #print 'new op: {}'.format(new_op_flag)
        if new_op_flag: 
            #print 'new op'
            uiman = self.start_wf_editor(self.opman,idx)
            uiman.ui.op_selector.setCurrentIndex(idx)
            uiman.ui.show()
            return
        elif existing_op_flag: 
            #print 'existing op'
            uiman = self.start_wf_editor(self.wfman,idx)
            uiman.ui.wf_selector.setCurrentIndex(idx)
            uiman.ui.show()
            return
        else:
            # if we are here, there was either an invalid index selected,
            # or the selection did not point to a valid Operation
            uiman = self.start_wf_editor()
            uiman.ui.show()

    def edit_ops(self,item_indx=None):
        """
        interact with user to edit and develop new Operations 
        """
        print 'Operation editing is not yet implemented'

    def add_ops(self,item_indx=None):
        """
        interact with user to add existing Operations to the tree of available Operations 
        """
        print 'All Operations are enabled- this will change in a near future version'    

    def start_wf_editor(self,trmod=None,indx=QtCore.QModelIndex()):
        """
        Create a QFrame window from ui/wf_editor.ui, then return it
        """
        uiman = WfUiManager(self.wfman,self.opman)
        if trmod and indx.isValid():
            uiman.get_op(trmod,indx)
        uiman.ui.setParent(self.ui,QtCore.Qt.Window)
        return uiman

    def display_item(self,indx):
        """
        Display selected item from the workflow tree in image_viewer 
        """
        if indx: 
            if self.wfman.get_item(indx).data is not None:
                to_display = self.wfman.get_item(indx).data
                uri = self.wfman.build_uri(indx)
                data_viewer.display_item(to_display,uri,self.ui.image_viewer,None)

    def final_setup(self):
        # Let the message board be read-only
        self.ui.message_board.setReadOnly(True)
        # Let the message board ignore line wrapping
        #self.ui.message_board.setLineWrapMode(self.ui.message_board.NoWrap)
        # Tell the status bar that we are ready.
        self.show_status('Ready')
        # Tell the message board that we are ready.
        self.ui.message_board.insertPlainText('--- MESSAGE BOARD ---\n') 
        self.msg_board_log('slacx is ready',timestamp=slacxtools.dtstr) 
        # Clear any default tabs out of image_viewer
        #self.ui.center_frame.setMinimumWidth(200)
        self.ui.op_tree.resizeColumnToContents(0)
        self.ui.wf_tree.resizeColumnToContents(0)
        #self.ui.wf_tree.resizeColumnToContents(1)
        #self.ui.wf_tree.setColumnWidth(0,300)
        self.ui.wf_tree.hideColumn(1)
        self.ui.hsplitter.setStretchFactor(1,2)    
        #self.ui.hsplitter.setStretchFactor(2,2)    
        self.ui.vsplitter.setStretchFactor(0,1)    

    # TODO: Make this functionality work, but with signals and slots
    def run_wf(self):
    #    self.ui.run_wf_button.setText("S&top")
        self.wfman.run_wf()
    #    self.ui.run_wf_button.clicked.disconnect(self.run_wf)
    #    self.ui.run_wf_button.clicked.connect(self.wfman.stop_wf)
    #    self.ui.run_wf_button.clicked.connect(self.reset_wf_button)

    #def reset_wf_button(self):
    #    self.ui.run_wf_button.setText("&Run")
    #    self.ui.run_wf_button.clicked.disconnect(self.wfman.stop_wf)
    #    self.ui.run_wf_button.clicked.disconnect(self.reset_wf_button)
    #    self.ui.run_wf_button.clicked.connect(self.run_wf)

    def connect_actions(self):
        """Set up the works for buttons and menu items"""
        self.ui.add_op_button.setText("Add to Workflow")
        #self.ui.add_op_button.clicked.connect(self.add_ops)
        self.ui.add_op_button.clicked.connect( partial(self.edit_wf,self.opman) )
        self.ui.edit_op_button.setText("Edit Operations")
        self.ui.edit_op_button.clicked.connect(self.edit_ops)
        self.ui.load_wf_button.setText("&Load")
        self.ui.load_wf_button.clicked.connect(partial(uitools.start_load_ui,self))
        self.ui.edit_wf_button.setText("&Edit")
        self.ui.edit_wf_button.clicked.connect( partial(self.edit_wf,self.wfman) )
        self.ui.run_wf_button.setText("&Run")
        self.ui.run_wf_button.clicked.connect(self.run_wf)
        #self.reset_wf_button()
        #self.ui.run_wf_button.clicked.connect(self.reset_wf_button)
        #self.wfman.wfdone.connect(self.reset_wf_button)
        self.ui.save_wf_button.setText("&Save")
        self.ui.save_wf_button.clicked.connect(partial(uitools.start_save_ui,self))
        self.ui.wf_tree.setModel(self.wfman)
        self.ui.op_tree.setModel(self.opman)
        self.ui.op_tree.hideColumn(1)
        self.ui.op_tree.clicked.connect( partial(uitools.toggle_expand,self.ui.op_tree) ) 
        self.ui.wf_tree.clicked.connect( partial(uitools.toggle_expand,self.ui.wf_tree) )
        self.ui.wf_tree.clicked.connect( self.display_item )
        self.ui.op_tree.doubleClicked.connect( partial(self.edit_wf,self.opman) )
        self.ui.wf_tree.doubleClicked.connect( partial(self.edit_wf,self.wfman) )
        # TODO: Figure out how to get the display to follow
        # when I scroll through the tree with my arrow keys.
        #self.ui.wf_tree.activated.connect(self.display_item)
        #self.ui.wf_tree.selectionModel().selectionChanged.connect( self.ui.wf_tree.selectionChanged )

    def make_title(self):
        """Display the slacx logo in the image viewer"""
        # Load the slacx graphic  
        slacx_img_file = os.path.join(slacxtools.rootdir, "slacxui/slacx_icon_white.png")
        # Make a QtGui.QPixmap from this file
        slacx_pixmap = QtGui.QPixmap(slacx_img_file)
        # Make a QtGui.QGraphicsPixmapItem from this QPixmap
        slacx_pixmap_item = QtGui.QGraphicsPixmapItem(slacx_pixmap)
        # Add this QtGui.QGraphicsPixmapItem to a QtGui.QGraphicsScene 
        slacx_scene = QtGui.QGraphicsScene()
        slacx_scene.addItem(slacx_pixmap_item)
        qwhite = QtGui.QColor(255,255,255,255)
        textitem = slacx_scene.addText("v{}".format(slacxtools.version))
        textitem.setPos(100,35)
        textitem.setDefaultTextColor(qwhite)
        # Add the QGraphicsScene to self.ui.image_viewer layout 
        logo_view = QtGui.QGraphicsView()
        logo_view.setScene(slacx_scene)
        self.ui.image_viewer.addWidget(logo_view,0,0,1,1)
        self.ui.setWindowTitle("slacx v{}".format(slacxtools.version))
        self.ui.setWindowIcon(slacx_pixmap)

    def msg_board_log(self,msg,timestamp=slacxtools.timestr):
        """Print timestamped message to msg board"""
        self.ui.message_board.insertPlainText(
        '- ' + timestamp() + ': ' + msg + '\n') 
        self.ui.message_board.verticalScrollBar().setValue(self.ui.message_board.verticalScrollBar().maximum())
      
    def show_status(self,msg):
        self.ui.statusbar.showMessage(msg)

