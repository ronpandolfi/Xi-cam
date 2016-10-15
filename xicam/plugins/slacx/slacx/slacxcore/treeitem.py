class TreeItem(object):
    """
    This is a container for objects in a QAbstractItemModel
    to facilitate data manipulation in a tree.
    It keeps references to a parent QModelIndex,
    and to its row and column within the QAbstractItemModel structure.
    Its data list elements correspond to 'columns' in the tree view.
    Every TreeItem must have a tag() for display in the tree view.
    TreeItems should have a long_tag() for other roles (e.g. tooltip).
    """

    def __init__(self,row,column,parent):
        self.parent = parent
        self.row = row
        self.column = column
        self.data = []          # list of objects, one for each column 
        self.children = []      # list of other TreeItems
        self.long_tag = 'no information'
        self._tag = None

    def n_data(self):
        return len(self.data)

    def n_children(self):
        return len(self.children)

    def insert_child(self,new_child,row):
        self.children.insert(row,new_child)

    def remove_child(self,row):
        child_removed = self.children.pop(row)
    
    def tag(self):
        if not self._tag:
            msg = str('[{}] found TreeItem with no tag. \n'
                    + 'Set a tag using TreeItem.set_tag()'.format(__name__))
            raise AttributeError(msg)
        else:
            return self._tag

    def set_tag(self,tag_in):
        self._tag = tag_in

    def data_str(self):
        """Build a string representing the items in self.data"""
        a = "data items:\n"
        for i in range(len(self.data)):
            datstr = str(self.data[i])
            a = a + '\ndata[{}]:'.format(i) + datstr[:min((len(datstr),60))] + '\n'
        return a


