from PySide import QtCore
from PySide.QtCore import Qt
import featuremanager


class featuresModel(QtCore.QAbstractListModel):
    """
    This model displays items for each feature
    """

    def __init__(self):
        QtCore.QAbstractListModel.__init__(self)


    def widgetchanged(self):
        self.modelReset.emit()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(featuremanager.features)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return featuremanager.features[index.row()].name

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


class FeatureTree(QtCore.QAbstractItemModel):
    def __init__(self, ):
        QtCore.QAbstractItemModel.__init__(self)
        self._root = featuremanager.experiment()

    def rowCount(self, in_index):
        if in_index.isValid():
            return in_index.internalPointer().childCount()
        return self._root.childCount()

    def addChild(self, in_node, in_parent):
        if not in_parent or not in_parent.isValid():
            parent = self._root
        else:
            parent = in_parent.internalPointer()
        parent.addChild(in_node)

    def index(self, row, column, parent=None):
        if not parent.isValid():
            return self.createIndex(row, column, self._root)
        parentNode = parent.internalPointer()
        return self.createIndex(row, column, parentNode.child(row))

    def parent(self, in_index):
        if in_index.isValid():
            p = in_index.internalPointer().parent()
            if p:
                return QtCore.QAbstractItemModel.createIndex(self, p.row(), 0, p)
        return QtCore.QModelIndex()

    def columnCount(self, in_index):
        if in_index.isValid():
            return in_index.internalPointer().columnCount()
        return self._root.childCount()

    def data(self, in_index, role):
        if not in_index.isValid():
            return None
        node = in_index.internalPointer()
        if role == QtCore.Qt.DisplayRole:
            return node.name
        return None

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole and section == 0:
            return 'Name'
        return None