import time

from PySide import QtCore

from treemodel import TreeModel
from treeitem import TreeItem

# TODO: See note on remove_image()

class ImgManager(TreeModel):
    """
    Class for managing tree of images and image data for slacx.
    Keeps loaded images and data in an internal tree structure.
    Provides methods for adding images to and removing images from the tree.
    Renders image data/metadata/etc as tree items below parent image.
    Images resulting from some operation should be added as new top-level items.
    Data resulting from operations should be children to parent images.
    """

    def __init__(self,**kwargs):
        self._n_loaded = 0 
        #TODO: build a saved tree from kwargs
        #if 'file_list' in kwargs:
        #    for fname in file_list:
        #        self.add_image(fname)
        # Initialize the superclass TreeModel,
        # which will in turn initialize QAbstractItemModel 
        super(ImgManager,self).__init__()

    @staticmethod
    def loader_extensions():
        return str(
        "ALL (*.*);;"
        + "TIFF (*.tif *.tiff);;"
        + "RAW (*.raw);;"
        + "MAR (*.mar*)"
        )

    # add an image object to the tree as a new top-level TreeItem.
    def add_image(self,new_img,tag):
        # Count top-level rows by passing parent=QModelIndex()
        ins_row = self.rowCount(QtCore.QModelIndex())
        # Make a new TreeItem, column 0, invalid parent 
        new_treeitem = TreeItem(ins_row,0,QtCore.QModelIndex())
        new_treeitem.data.append(new_img)
        new_treeitem.set_tag( tag )
        new_treeitem.long_tag = new_img.img_url
        self.beginInsertRows(
        QtCore.QModelIndex(),ins_row,ins_row)
        # Image insertion occurs between notification methods
        self.root_items.insert(ins_row,new_treeitem)
        self.endInsertRows()
        self._n_loaded += 1

    def add_image_data(self,parent,data,tag,long_tag=None):
        """Add data as a child TreeItem to the TreeItem at parent"""
        ins_row = self.rowCount(parent)
        new_treeitem = TreeItem(ins_row,0,parent)
        new_treeitem.data.append(data)
        new_treeitem.set_tag(tag)
        if long_tag:
            new_treeitem.long_tag = long_tag 
        p_item = parent.internalPointer()
        self.beginInsertRows(parent,ins_row,ins_row)
        p_item.children.insert(ins_row,new_treeitem)
        self.endInsertRows()

    def add_image_text(self,parent,text,tag):
        """Add text as a child TreeItem to the TreeItem at parent"""
        ins_row = self.rowCount(parent)
        new_treeitem = TreeItem(ins_row,0,parent)
        new_treeitem.data.append(text)
        new_treeitem.set_tag(tag)
        new_treeitem.long_tag = text 
        p_item = parent.internalPointer()
        self.beginInsertRows(parent,ins_row,ins_row)
        p_item.children.insert(ins_row,new_treeitem)
        self.endInsertRows()

    # remove a top-level TreeItem (containing a SlacxImage) by its QModelIndex.
    def remove_image(self,removal_indx):
        # TODO: Make this a general item(+children) removal
        removal_row = removal_indx.row()
        #print '[{}] removing root_items[{}]'.format(__name__,removal_row)
        self.beginRemoveRows(
        QtCore.QModelIndex(),removal_row,removal_row)
        # Image removal occurs between notification methods
        item_removed = self.root_items.pop(removal_row)
        self.endRemoveRows()
        # TODO: wipe out display tabs if using a gui
        # TODO: update any Operations in workflow that depended on this image 

    # QAbstractItemModel subclass should implement 
    # headerData(int section,Qt.Orientation orientation[,role=Qt.DisplayRole])
    # note: section arg indicates row or column number, depending on orientation
    def headerData(self,section,orientation,data_role):
        if (data_role == QtCore.Qt.DisplayRole and section == 0):
            return "{} image(s) open".format(self.rowCount(QtCore.QModelIndex()))
        elif (data_role == QtCore.Qt.DisplayRole and section == 1):
            return "info".format(self.rowCount(QtCore.QModelIndex()))
        else:
            return None

    # Editable QAbstractItemModel subclasses must implement
    # setData(index,value[,role])
    #def setData(self,index,value):
    #    self._img_tree.insert(index,value)

    #def microtime(self):
    #    """Return the last 9 digits of the current time in microseconds"""
    #    return int(time.time()*1E6 - int(time.time()*1E-3)*1E9)
    #    #int(time.time()*1000000)


