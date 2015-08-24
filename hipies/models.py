from PySide.QtCore import Qt
from PySide import QtCore


class openfilesmodel(QtCore.QAbstractListModel):
    """
    This model creates modelitems for each open tab for navigation
    """

    def __init__(self, tabwidget):
        QtCore.QAbstractListModel.__init__(self)

        self.tabwidget = tabwidget

    def widgetchanged(self):
        self.dataChanged.emit(0, 0)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return self.tabwidget.count()

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self.tabwidget.tabText(index.row())

            # The view is asking for the actual data, so, just return the item it's asking for.
            #return self._items[index.row()]
            # elif role == Qt.BackgroundRole:
            # Here, it's asking for some background decoration.
            # Let's mix it up a bit: mod the row number to get even or odd, and return different
            # colours depending.
            #if index.row() % 2 == 0:
            #    return QColor(Qt.gray)
            #else:
            #    return QColor(Qt.darkGray)
        else:
            # We don't care about anything else, so make sure to return an empty QVariant.
            return None


class imagePropModel(QtCore.QAbstractTableModel):
    """
    This model creates modelitems for each open tab for navigation
    """

    def __init__(self, tabwidget):
        super(imagePropModel, self).__init__()

        self.tabwidget = tabwidget
        self._propdata = None

    def widgetchanged(self):
        self.modelReset.emit()
        self._propdata = None

    def rowCount(self, parent=QtCore.QModelIndex()):
        if self.tabwidget() is not None:
            count = len(self.propdata)
            # print 'ImageProp count:', count
            return count
        else:
            return 0

    def columnCount(self, *args, **kwargs):
        return 2

    @property
    def propdata(self):
        if self._propdata is None:
            self._propdata = self.tabwidget().tab.dimg.headers
        return self._propdata

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if index.column() == 0:
                return self.propdata.keys()[index.row()]
            if index.column() == 1:
                return self.propdata.values()[index.row()]

                # The view is asking for the actual data, so, just return the item it's asking for.
                # return self._items[index.row()]
                # elif role == Qt.BackgroundRole:
                # Here, it's asking for some background decoration.
                # Let's mix it up a bit: mod the row number to get even or odd, and return different
                # colours depending.
                #if index.row() % 2 == 0:
                #    return QColor(Qt.gray)
                #else:
                #    return QColor(Qt.darkGray)
        else:
            # We don't care about anything else, so make sure to return an empty QVariant.
            return None

