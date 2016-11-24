from PySide import QtCore

class ListModel(QtCore.QAbstractListModel):
    """
    Class for list management with a QAbstractListModel.
    Implements required virtual methods rowCount() and data().
    Resizeable ListModels must implement insertRows(), removeRows().
    If a nicely labeled header is desired, implement headerData().
    """

    def __init__(self,input_list=[],parent=None):
        super(ListModel,self).__init__(parent)
        self._list_data = []
        self._enabled = []
        for thing in input_list:
            self.append_item(thing)
        
    def append_item(self,thing):
        ins_row = self.rowCount()
        self.beginInsertRows(QtCore.QModelIndex(),ins_row,ins_row)
        self._list_data.insert(ins_row,thing) 
        self.endInsertRows()
        self._enabled.insert(ins_row,True)
        idx = self.index(ins_row,0,QtCore.QModelIndex())

    def remove_item(self,row):
        self.beginRemoveRows(QtCore.QModelIndex(),row,row)
        self._list_data.pop(row) 
        self.endRemoveRows()
        self._enabled.pop(row) 

    def set_disabled(self,row):
        self._enabled[row] = False

    def list_data(self):
        return self._list_data

    def flags(self,idx):
        if self._enabled[idx.row()]:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        else:
            return QtCore.Qt.NoItemFlags

    def rowCount(self,parent=QtCore.QModelIndex()):
        return len(self._list_data)

    def columnCount(self,parent=QtCore.QModelIndex()):
        return 1

    def data(self,idx,data_role):
        if not idx.isValid():
            return None
        elif (data_role == QtCore.Qt.DisplayRole
        or data_role == QtCore.Qt.ToolTipRole 
        or data_role == QtCore.Qt.StatusTipRole
        or data_role == QtCore.Qt.WhatsThisRole):
            return str(self._list_data[idx.row()])
        else:
            return None
        #    print 'data at row {}: {}'.format(idx.row(),self._list_data[idx.row()])
        #    print 'DATA IS NONE'

    def insertRows(self,row,count):
        self.beginInsertRows(QtCore.QModelIndex(),row,row+count-1)
        for j in range(row,row+count):
            self._list_data.insert(j,None)
        self.endInsertRows()

    def removeRows(self, row, count, parent=QtCore.QModelIndex()):
        self.beginRemoveRows(parent,row,row+count-1)
        for j in range(row,row+count)[::-1]:
            self.list_items.pop(j)
        self.endRemoveRows()

    def headerData(self,section,orientation,data_role):
        return 'dummy header'

