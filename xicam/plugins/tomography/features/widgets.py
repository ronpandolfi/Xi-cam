from PySide import QtGui, QtCore

from xicam.plugins.tomography import ui


class FeatureWidget(QtGui.QWidget):

    sigShowForm = QtCore.Signal(QtGui.QWidget)
    sigDelete = QtCore.Signal(QtGui.QWidget)


    def __init__(self, name='', checkable=True, parent=None):
        super(FeatureWidget, self).__init__(parent=parent)

        self.name = name
        self.form = QtGui.QLabel(self.name)#QtGui.QWidget()

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.frame = QtGui.QFrame(self)
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.frame)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.previewButton = QtGui.QPushButton(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.previewButton.sizePolicy().hasHeightForWidth())
        self.previewButton.setSizePolicy(sizePolicy)
        self.previewButton.setStyleSheet("margin:0 0 0 0;")
        self.previewButton.setText("")
        icon = QtGui.QIcon()
        if checkable:
            icon.addPixmap(QtGui.QPixmap("gui/icons_48.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("gui/icons_47.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
            self.previewButton.setCheckable(True)
        else:
            icon.addPixmap(QtGui.QPixmap("gui/icons_47.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.previewButton.setCheckable(False)
            self.previewButton.setChecked(True)
        self.previewButton.setIcon(icon)
        self.previewButton.setFlat(True)
        self.previewButton.setChecked(True)
        self.previewButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.horizontalLayout_2.addWidget(self.previewButton)
        self.line = QtGui.QFrame(self.frame)
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.horizontalLayout_2.addWidget(self.line)
        self.txtName = ROlineEdit(self.frame)
        self.horizontalLayout_2.addWidget(self.txtName)
        self.txtName.setText(name)
        self.line_3 = QtGui.QFrame(self.frame)
        self.line_3.setFrameShape(QtGui.QFrame.VLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.horizontalLayout_2.addWidget(self.line_3)
        self.closeButton = QtGui.QPushButton(self.frame)
        self.closeButton.setStyleSheet("margin:0 0 0 0;")
        self.closeButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("gui/icons_46.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.closeButton.setIcon(icon1)
        self.closeButton.setFlat(True)
        self.closeButton.clicked.connect(self.delete)
        self.horizontalLayout_2.addWidget(self.closeButton)
        self.verticalLayout.addWidget(self.frame)
        self.txtName.sigClicked.connect(self.mouseClicked)

        self.frame.setFrameShape(QtGui.QFrame.Box)
        self.frame.setCursor(QtCore.Qt.ArrowCursor)

    def delete(self):
        self.sigDelete.emit(self)
        # value = QtGui.QMessageBox.question(None, 'Delete this feature?',
        #                                    'Are you sure you want to delete this function?',
        #                                    (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
        # if value is QtGui.QMessageBox.Yes:
        #     manager.functions = [feature for feature in manager.functions if feature is not self]
        #     self.deleteLater()
        #     ui.showform(ui.blankform)
        pass

    def hideothers(self):
        # for item in manager.functions:
        #     if hasattr(item, 'frame_2') and item is not self and self in manager.functions:
        #             item.frame_2.hide()
        pass

    def mouseClicked(self):
        self.sigShowForm.emit(self.form)
        self.setFocus()
        self.previewButton.setFocus()
        # self.hideothers()
        # try:
        #     manager.currentindex = manager.functions.index(self)
        # except ValueError:
        #     pass

    # def showSelf(self):
    #     ui.showform(self.form)

    # def setName(self, name):
    #     self.name = name
    #     # manager.update()


class ROlineEdit(QtGui.QLineEdit):
    sigClicked = QtCore.Signal()
    def __init__(self, *args, **kwargs):
        super(ROlineEdit, self).__init__(*args, **kwargs)
        self.setReadOnly(True)
        self.setFrame(False)

    def focusOutEvent(self, *args, **kwargs):
        super(ROlineEdit, self).focusOutEvent(*args, **kwargs)
        self.setCursor(QtCore.Qt.ArrowCursor)

    def mousePressEvent(self, *args, **kwargs):
        super(ROlineEdit, self).mousePressEvent(*args, **kwargs)
        self.sigClicked.emit()

    def mouseDoubleClickEvent(self, *args, **kwargs):
        super(ROlineEdit, self).mouseDoubleClickEvent(*args, **kwargs)
        self.setFrame(True)
        self.setFocus()
        self.selectAll()