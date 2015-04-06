from PySide.QtCore import QAbstractListModel
from PySide.QtCore import QModelIndex
from PySide.QtGui import QColor
from PySide.QtCore import Qt
from PySide.QtGui import QItemDelegate


class openfilesmodel(QAbstractListModel):
    def __init__(self, tabwidget):
        QAbstractListModel.__init__(self)

        # Cache the passed data list as a class member.
        # self._items = mlist

        self.tabwidget = tabwidget

    def widgetchanged(self):
        self.dataChanged.emit(0, 0)

    def rowCount(self, parent=QModelIndex()):
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

