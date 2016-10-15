import os
from datetime import datetime as dt
import time
from functools import partial

from PySide import QtGui, QtCore, QtUiTools
import numpy as np

from ..slacxcore import slacximg 
from . import uitools
from .opuiman import OpUiManager
from .imgloaduiman import ImgLoadUiManager

if uitools.have_qt47:
    from . import plotmaker_pqg as plotmaker
else:
    from . import plotmaker_mpl as plotmaker
    

class UiManager(object):
    """
    Stores a reference to a QMainWindow,
    performs operations on it
    """

    # TODO: when the QImageView widget gets resized,
    # it will call QWidget.resizeEvent().
    # Try to use this to resize the images in the QImageView.

    def __init__(self,rootdir):
        """Make a UI from ui_file, save a reference to it"""
        self.rootdir = rootdir
        # Pick a UI definition, load it up
        ui_file = QtCore.QFile(self.rootdir+"/slacxui/basic.ui")
        ui_file.open(QtCore.QFile.ReadOnly)
        # load() produces a QMainWindow(QWidget).
        self.ui = QtUiTools.QUiLoader().load(ui_file)
        ui_file.close()
        # Set up the self.ui widget to delete itself when closed
        self.ui.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.imgman = None    
        self.opman = None    
        self.wfman = None
        #self.op_uimans = [] 

    def apply_workflow(self):
        """
        run the workflow
        """
        self.wfman.run_wf_serial()

    def edit_op(self):
        """
        Edit the selected operation in the workflow list.
        Do this by opening an OpUiManager,
        loading it with the selected operation,
        and then setting the finish/load button
        to perform an update rather than appendage.
        """
        selected_indxs = self.ui.workflow_tree.selectedIndexes()
        if len(selected_indxs) > 0:
            uiman = self.start_op_ui_manager()
            # set OpUiManager's operation to the one selected in self.ui.workflow_tree
            uiman.set_op( self.wfman.get_item(selected_indxs[0]).data[0] )
            uiman.ui.op_selector.setEnabled(False)
            uiman.ui.tag_entry.setText(self.wfman.get_item(selected_indxs[0]).tag())
            uiman.ui.tag_entry.setReadOnly(True)
            # connect uiman.ui.finish_button to an operation updater method
            uiman.ui.finish_button.clicked.disconnect()
            uiman.ui.finish_button.clicked.connect( partial(uiman.update_op,selected_indxs[0]) )
            uiman.ui.show()
        else:
            # TODO: Inform user to select operation first
            pass

    def rm_op(self):
        """
        remove the selected operation in the workflow list from the workflow
        """
        # TODO: implement multiple selection 
        # TODO: take out the garbage
        selected_indxs = self.ui.workflow_tree.selectedIndexes()
        #for indx in selected_indxs:
        self.wfman.remove_op(selected_indxs[0])

    def add_op(self):
        """
        interact with user to build an operation into the workflow
        """
        uiman = self.start_op_ui_manager()
        uiman.ui.show()

    def start_imgload_ui_manager(self,imgfile):
        """
        Create a QFrame window from ui/tag_request.ui, then return it
        """
        ui_file = QtCore.QFile(self.rootdir+"/slacxui/tag_request.ui")
        uiman = ImgLoadUiManager(ui_file,self.imgman,imgfile)
        uiman.ui.setParent(self.ui,QtCore.Qt.Window)#|QtCore.Qt.WindowStaysOnTopHint)
        #uiman.ui.setWindowModality(QtCore.Qt.WindowModal)
        #uiman.ui.setParent(self.ui,QtCore.Qt.Popup)
        uiman.ui.activateWindow()
        #self.ui.lower()
        uiman.ui.raise_()
        self.ui.stackUnder(uiman.ui)
        #uiman.ui.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        uiman.ui.show()
        #return uiman

    def start_op_ui_manager(self):
        """
        Create a QFrame window from ui/op_builder.ui, then return it
        """
        uiman = OpUiManager(self.rootdir,self.wfman,self.imgman,self.opman)
        uiman.ui.setParent(self.ui,QtCore.Qt.Window)
        return uiman

    def close_image(self):
        """Remove selected items from the image tree"""
        # TODO: implement multiple selection  
        # TODO: take out the garbage
        selected_indxs = self.ui.image_tree.selectedIndexes()
        #for indx in selected_indxs:
        self.imgman.remove_image(selected_indxs[0])

    def open_image(self):
        """Open an image, add it to image tree"""
        # TODO: implement loading multiple images in one call? 
        # getOpenFileName(parent(Widget), caption, dir, extension(s) regexp)
        imgfile, ext = QtGui.QFileDialog.getOpenFileName(
        self.ui, 'Open file', self.rootdir, self.imgman.loader_extensions())
        if imgfile:
            # Start up a UI for tagging and loading the image
            self.start_imgload_ui_manager(imgfile)
            #self.get_img_tag(imgfile)

    def display_item(self,src='Images'):
        """
        Display selected item from the image or workflow tree 
        in a new tab in image_viewer
        """
        if src == 'Images':
            indxs = self.ui.image_tree.selectedIndexes()
            trmod = self.imgman
        elif src == 'Workflow':
            indxs = self.ui.workflow_tree.selectedIndexes()
            trmod = self.wfman
        else:
            msg = 'unrecognized image source {}'.format(src)
            raise ValueError(msg)
        #print indxs
        #for indx in indxs:
        #    print indx 
        #    print trmod.get_item(indx).data
        if len(indxs) > 0:
            indx = indxs[0]
            to_display = trmod.get_item(indx).data[0]
            uri = trmod.build_uri(indx)
            plotmaker.display_item(to_display,uri,self.ui.image_viewer,self.msg_board_log)
        else:
            # TODO: dialog box: tell user to select an item first
            pass

    def final_setup(self):
        # Let the message board be read-only
        self.ui.message_board.setReadOnly(True)
        # Let the message board ignore line wrapping
        self.ui.message_board.setLineWrapMode(self.ui.message_board.NoWrap)
        # Tell the status bar that we are ready.
        self.show_status('Ready')
        # Tell the message board that we are ready.
        self.msg_board_log('slacx is ready') 
        # Clear any default tabs out of image_viewer
        self.ui.image_viewer.clear()
        # Set image viewer tabs to be closeable
        self.ui.image_viewer.setTabsClosable(True)
        # Set the image viewer to be kinda fat
        self.ui.image_viewer.setMinimumHeight(400)
        self.ui.image_viewer.setMinimumWidth(400)
        # Leave the textual parts kinda skinny?
        #self.ui.left_panel.setMaximumWidth(400)
        self.ui.right_frame.setMinimumWidth(300)
        self.ui.right_frame.setMaximumWidth(400)
        self.ui.workflow_tree.resizeColumnToContents(0)
        self.ui.workflow_tree.resizeColumnToContents(1)
        self.ui.left_frame.setMinimumWidth(300)
        self.ui.left_frame.setMaximumWidth(400)
        self.ui.image_tree.setMinimumHeight(200)
        self.ui.title_box.setMinimumHeight(200)
        self.ui.message_board.setMinimumHeight(200)
        self.ui.image_tree.resizeColumnToContents(0)
        self.ui.image_tree.resizeColumnToContents(1)
        #self.ui.workflow_tree.setColumnWidth(0,200)
        #self.ui.workflow_tree.setColumnWidth(1,150)
        self.ui.image_tree.setColumnWidth(0,180)

    def connect_actions(self):
        """Set up the works for buttons and menu items"""
        # Connect self.ui.actionOpen (File menu):
        self.ui.actionOpen.triggered.connect(self.open_image)
        # Connect self.ui.open_images_button: 
        self.ui.open_images_button.setText("&Open images...")
        self.ui.open_images_button.clicked.connect(self.open_image)
        # Connect self.ui.close_images_button: 
        self.ui.close_images_button.setText("&Close images")
        self.ui.close_images_button.clicked.connect(self.close_image)
        # Connect self.ui.display_item_button:
        self.ui.display_item_button.setText("&Display selected item")
        self.ui.display_item_button.clicked.connect( partial(self.display_item,src='Images') )
        self.ui.display_workflowitem_button.setText("Display selected item")
        self.ui.display_workflowitem_button.clicked.connect( partial(self.display_item,src='Workflow') )
        # Connect self.ui.image_viewer tabCloseRequested to local close_tab slot
        self.ui.image_viewer.tabCloseRequested.connect(self.close_tab)
        # Make self.ui.image_viewer tabs elide (arg is a Qt.TextElideMode)
        self.ui.image_viewer.setElideMode(QtCore.Qt.ElideRight)
        # Connect self.ui.add_op_button:
        self.ui.add_op_button.setText("Add Operation")
        self.ui.add_op_button.clicked.connect(self.add_op)
        # Connect self.ui.rm_op_button:
        self.ui.rm_op_button.setText("Remove Operation")
        self.ui.rm_op_button.clicked.connect(self.rm_op)
        # Connect self.ui.edit_op_button:
        self.ui.edit_op_button.setText("Edit Operation")
        self.ui.edit_op_button.clicked.connect(self.edit_op)
        # Connect self.ui.apply_workflow_button:
        self.ui.apply_workflow_button.setText("Apply Workflow")
        self.ui.apply_workflow_button.clicked.connect(self.apply_workflow)
        # Connect self.ui.image_tree (QListView) 
        # to self.imgman (ImgManager(QAbstractListModel))
        self.ui.image_tree.setModel(self.imgman)
        # Connect self.ui.workflow_tree (QTreeView) 
        # to self.wfman (WfManager(TreeModel))
        self.ui.workflow_tree.setModel(self.wfman)

    def make_title(self):
        """Display the slacx logo in the title box"""
        # Load the slacx graphic  
        #slacx_img_file = os.path.join(self.rootdir, "ui/slacx_icon.png")
        slacx_img_file = os.path.join(self.rootdir, "slacxui/slacx_icon_white.png")
        # Make a QtGui.QPixmap from this file
        slacx_pixmap = QtGui.QPixmap(slacx_img_file)
        # Make a QtGui.QGraphicsPixmapItem from this QPixmap
        slacx_pixmap_item = QtGui.QGraphicsPixmapItem(slacx_pixmap)
        # Add this QtGui.QGraphicsPixmapItem to a QtGui.QGraphicsScene 
        slacx_scene = QtGui.QGraphicsScene()
        slacx_scene.addItem(slacx_pixmap_item)
        # Add the QGraphicsScene to the QGraphicsView
        self.ui.title_box.setScene(slacx_scene)
        # Set the main window title and icon
        self.ui.setWindowTitle("slacx")
        self.ui.setWindowIcon(slacx_pixmap)
 
    # A QtCore.Slot for closing tabs from image_viewer
    @QtCore.Slot(int)
    def close_tab(self,indx):
        self.ui.image_viewer.removeTab(indx)

    # Various simple utilities
    @staticmethod 
    def dtstr():
        """Return date and time as a string"""
        return dt.strftime(dt.now(),'%Y %m %d, %H:%M:%S')

    def msg_board_log(self,msg):
        """Print timestamped message with space to msg board"""
        self.ui.message_board.insertPlainText(
        self.dtstr() + '\n' + msg + '\n\n') 

    def show_status(self,msg):
        self.ui.statusbar.showMessage(msg)

    def export_image(self):
        """export the image in the currently selected tab"""
        pass

    def edit_image(self):
        """open an image editor for the current tab"""
        pass


