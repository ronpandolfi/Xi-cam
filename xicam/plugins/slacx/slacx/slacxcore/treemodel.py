import string 

from PySide import QtCore

class TreeModel(QtCore.QAbstractItemModel):
    """
    Class for tree management with a QAbstractItemModel.
    Implements required virtual methods index(), parent(), rowCount().
    Other required virtual methods are columnCount() and data():
    these should be implemented by subclassing of TreeModel.
    Resizeable TreeModels must implement: 
    insertRows(), removeRows(), insertColumns(), removeColumns()
    If nicely labeled headers are desired, one should implement
    headerData().
    """

    def __init__(self):
        super(TreeModel,self).__init__()
        # keep root items in a TreeItem list
        self.root_items = []    

    def get_item(self,indx):
        return indx.internalPointer() 

    def get_item_by_tag(self,req_tag):
        for item in self.root_items:
            if item.tag() == req_tag:
                return item
        return None

    def build_uri(self,indx):
        """Build a URI for the TreeItem at indx"""
        item_ref = self.get_item(indx)
        item_uri = item_ref.tag()
        while item_ref.parent.isValid():
            item_ref = self.get_item(item_ref.parent)
            item_uri = item_ref.tag()+"."+item_uri
        return item_uri

    # get a list of tags for TreeItems under parent
    def list_tags(self,parent):
        if not parent.isValid():
            return [item.tag() for item in self.root_items]
        else:
            return [item.tag() for item in self.get_item(parent).children]
    
    # test uniqueness and good form of a tag
    def is_good_tag(self,testtag,parent=QtCore.QModelIndex()):
        spec_chars = string.punctuation 
        spec_chars = spec_chars.replace('_','')
        spec_chars = spec_chars.replace('-','')
        if not testtag:
            return (False, 'Tag is blank')
        elif testtag in self.list_tags(parent):
            return (False, 'Tag not unique')
        elif any(map(lambda s: s in testtag,[' ','\t','\n'])):
            return (False, 'Tag contains whitespace')
        elif any(map(lambda s: s in testtag,spec_chars)):
            return (False, 'Tag contains special characters')
        else:
            return (True, '')    

    # Subclass of QAbstractItemModel must implement index()
    def index(self,row,col,parent):
        """
        Returns QModelIndex address of int row, int col, under QModelIndex parent.
        If a row, column, parent combination points to an invalid index, 
        returns invalid QModelIndex().
        """
        if not parent.isValid():
            # If parent is not a valid index, a top level item is being queried.
            if row < len(self.root_items) and row >= 0:
                # Return the index
                return self.createIndex(row,col,self.root_items[row])
            else:
                # Bad row: return invalid index
                return QtCore.QModelIndex()
        else:
            # We need to grab the parent from its QModelIndex...
            p_item = parent.internalPointer()
            # and return the index of the child at row
            if row < len(p_item.children) and row >= 0:
                return self.createIndex(row,col,p_item.children[row])
            else:
                # Bad row: return invalid index
                return QtCore.QModelIndex()
                
    # Subclass of QAbstractItemModel must implement parent()
    def parent(self,index):
        """
        Returns QModelIndex of parent of item at QModelIndex index
        """
        # Grab this TreeItem from its QModelIndex
        item = index.internalPointer()
        if not item.parent.isValid():
            # We have no parent, therefore a top level item
            return QtCore.QModelIndex()
        else:
            return item.parent
        
    # Subclass of QAbstractItemModel must implement rowCount()
    def rowCount(self,parent):
        """
        Either give the number of top-level items,
        or count the children of parent
        """
        if not parent.isValid():
            # top level parent: count root items
            #print 'number of rows in root_items: {}'.format(len(self.root_items))
            return len(self.root_items)
        else:
            # count children of parent item
            parent_item = parent.internalPointer()
            return parent_item.n_children()
    
    # Subclass of QAbstractItemModel must implement columnCount()
    def columnCount(self,parent):
        """
        Let TreeModels by default have one column,
        to display the local TreeItem's tag.
        """
        return 1

    # QAbstractItemModel subclass must implement 
    # data(QModelIndex[,role=Qt.DisplayRole])
    def data(self,item_indx,data_role):
        if (not item_indx.isValid()):
            return None
        item = item_indx.internalPointer()
        #if item_indx.column() == 1:
        #    if item.data is not None:
        #        return type(item.data).__name__ 
        #    else:
        #        return ' '
        #else:
        if (data_role == QtCore.Qt.DisplayRole
        or data_role == QtCore.Qt.ToolTipRole 
        or data_role == QtCore.Qt.StatusTipRole
        or data_role == QtCore.Qt.WhatsThisRole):
            return item.tag()
        #elif (data_role == QtCore.Qt.ToolTipRole): 
        #    return item.long_tag() #+ '\n\n' + item.data_str()
        #elif (data_role == QtCore.Qt.StatusTipRole
        #    or data_role == QtCore.Qt.WhatsThisRole):
        #    return item.long_tag()
        else:
            return None

    # Expandable QAbstractItemModel subclass should implement
    # insertRows(row,count[,parent=QModelIndex()])
    def insertRows(self,row,count,parent=QtCore.QModelIndex()):
        # Signal listeners that rows are about to be born
        self.beginInsertRows(parent,row,row+count-1)
        if parent.isValid():
            # Get the TreeItem referred to by QModelIndex parent:
            item = parent.internalPointer()
            for j in range(row,row+count):
                item.children.insert(j,None)
        else:
            # Insert rows into self.root_items:
            for j in range(row,row+count):
                self.root_items.insert(j,None)
        # Signal listeners that we are done inserting rows
        self.endInsertRows()

    # Shrinkable QAbstractItemModel subclass should implement
    # removeRows(row,count[,parent=QModelIndex()])
    def removeRows(self, row, count, parent=QtCore.QModelIndex()):
        # Signal listeners that rows are about to die
        self.beginRemoveRows(parent,row,row+count-1)
        if parent.isValid():
            # Get the TreeItem referred to by QModelIndex parent:
            item = parent.internalPointer()
            for j in range(row,row+count)[::-1]:
                #del item.children[j]
                item.children.pop(j)
        else:
            for j in range(row,row+count)[::-1]:
                #del self.root_items[j]
                self.root_items.pop(j)
        # Signal listeners that we are done removing rows
        self.endRemoveRows()

    # get a TreeItem from the tree by its QModelIndex
    # QAbstractItemModel subclass should implement 
    # headerData(int section,Qt.Orientation orientation[,role=Qt.DisplayRole])
    # note: section arg indicates row or column number, depending on orientation
    def headerData(self,section,orientation,data_role):
        if (data_role == QtCore.Qt.DisplayRole and section == 0):
            return "{} item(s) open".format(self.rowCount(QtCore.QModelIndex()))
        #elif (data_role == QtCore.Qt.DisplayRole and section == 1):
        #    return "info".format(self.rowCount(QtCore.QModelIndex()))
        else:
            return None

    def iter_indexes(self,parent=QtCore.QModelIndex()):
        """Provide a list of the QModelIndexes held in the tree"""
        if parent.isValid():
            ixs = [parent]
            # Loop through children of parent
            items = self.get_item(parent).children
        else:
            ixs = []
            # Loop through root_items
            items = self.root_items
        for j in range(len(items)):
            item = items[j]
            ix = self.index(j,0,parent)
            # TODO: check for NoneType children here?
            if item.n_children > 0:
                ixs = ixs + self.iter_indexes(ix)
        return ixs

    def print_tree(self,rowprefix='',parent=QtCore.QModelIndex()):
        if parent.isValid():
            item = self.get_item(parent)
            print rowprefix+str(item.data)
            for j in range(item.n_children()):
                #print 'calling print_tree for {} more children'.format(item.n_children()-j)
                #time.sleep(1)
                self.print_tree(rowprefix+'\t',self.index(j,0,parent))
        else:
            for jroot in range(len(self.root_items)):
                #print 'calling print_tree for {} more root items'.format(len(self.root_items)-jroot)
                #time.sleep(1)
                item = self.root_items[jroot]
                #print rowprefix+str(item.data)
                self.print_tree(rowprefix,self.index(jroot,0,parent))
                #for j in range(item.n_children()):
            

    # Editable QAbstractItemModel subclasses must implement setData(index,value[,role])
    # TODO: understand whether or not this is necessary and what it means for the tree.
    def setData(self,idx,value,role=None):
        # For the TreeItem at index, set data to value
        treeitem = self.get_item(idx)
        treeitem.data = value

