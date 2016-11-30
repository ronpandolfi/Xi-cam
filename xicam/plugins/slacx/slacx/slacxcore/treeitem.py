class TreeItem(object):
    """
    This is a container for objects in a QAbstractItemModel
    to facilitate data manipulation in a tree.
    It keeps references to a parent QModelIndex,
    and to its row and column within the QAbstractItemModel structure.
    A TreeItem contains one free-form data object.
    Every TreeItem must have a tag() for display in the tree view.
    """

    def __init__(self,row,column,parent):
        self.parent = parent
        self.row = row
        self.column = column
        self.data = None        # TreeItem contains a single object as its data 
        self.children = []      # list of other TreeItems
        #self._long_tag = None 
        self._tag = None

    #def n_data(self):
    #    return len(self.data)

    def n_children(self):
        return len(self.children)

    def insert_child(self,new_child,row):
        self.children.insert(row,new_child)

    def remove_child(self,row):
        child_removed = self.children.pop(row)
    
    def tag(self):
        if not self._tag:
            return 'untagged'
        else:
            return self._tag

    #def long_tag(self):
    #    if not self._long_tag:
    #        return self._tag
    #    else:
    #        return self._long_tag

    def set_tag(self,tag_in):
        self._tag = tag_in

    #def set_long_tag(self,tag_in):
    #    self._long_tag = tag_in

    #def data_str(self):
    #    """Build a string representing self.data"""
    #    #for i in range(len(self.data)):
    #    datstr = str(self.data)
    #    return 'data:\n' + datstr[:min((len(datstr),60))]


