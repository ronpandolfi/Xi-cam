from PySide import QtGui, QtCore


class FeatureWidget(QtGui.QWidget):

    sigClicked = QtCore.Signal(QtGui.QWidget)
    sigDelete = QtCore.Signal(QtGui.QWidget)


    def __init__(self, name='', checkable=True, subfeatures=None, parent=None):
        super(FeatureWidget, self).__init__(parent=parent)

        self.name = name
        self.form = QtGui.QLabel(self.name) # default form

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

        self.subframe = QtGui.QFrame(self)
        self.subframe.setFrameShape(QtGui.QFrame.StyledPanel)
        self.subframe.setFrameShadow(QtGui.QFrame.Raised)
        self.subframe_layout = QtGui.QGridLayout(self.subframe)
        self.subframe_layout.setContentsMargins(0, 0, 0, 0)
        self.subframe_layout.setSpacing(0)
        self.verticalLayout.addWidget(self.subframe)
        self.subframe.hide()

        if subfeatures is not None:
            for subfeature in subfeatures:
                self.addSubFeature(subfeature)

        self.collapse()

    def addSubFeature(self, feature):
        r = self.subframe_layout.rowCount()
        spacerItem = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.subframe_layout.addItem(spacerItem, r, 0, 1, 1)
        if isinstance(feature, QtGui.QLayout):
            self.subframe_layout.addLayout(feature, r, 2, 1, 1)
        elif isinstance(feature, QtGui.QWidget):
            self.subframe_layout.addWidget(feature, r, 2, 1, 1)

    def delete(self):
        self.sigDelete.emit(self)
        # value = QtGui.QMessageBox.question(None, 'Delete this feature?',
        #                                    'Are you sure you want to delete this function?',
        #                                    (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
        # if value is QtGui.QMessageBox.Yes:
        #     manager.functions = [feature for feature in manager.functions if feature is not self]
        #     self.deleteLater()
        #     ui.showform(ui.blankform)

    def collapse(self):
        if self.subframe is not None:
            self.subframe.hide()

    def expand(self):
        if self.subframe is not None:
            self.subframe.show()

    def mouseClicked(self):
        self.sigClicked.emit(self)
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


class FeatureManager(object):
    """
    Feature Manager class to manage a list of FeatureWidgets and show the list in an appropriate layout and their
    corresponding forms in another layout. list layout must have an addWidget, removeWidget methods. Form layout
    must in addition have a setCurrentWidget method
    """

    def __init__(self, list_layout, form_layout, feature_widgets=None, blank_form=None):
        self._llayout = list_layout
        self._flayout = form_layout
        self.features = []

        if feature_widgets is not None:
            for feature in feature_widgets:
                self.addFeature(feature)

        if blank_form is not None:
            if isinstance(blank_form, str):
                self.blank_form = QtGui.QLabel(blank_form)
            else:
                self.blank_form = blank_form
        else:
            self.blank_form = QtGui.QLabel('Select a feature to view its form')
        self.blank_form.setAlignment(QtCore.Qt.AlignCenter)
        self.blank_form.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)

        self._flayout.addWidget(self.blank_form)
        self.showForm(self.blank_form)

    @property
    def count(self):
        return len(self.features)

    def addFeature(self, feature):
        self.features.append(feature)
        self._llayout.addWidget(feature)
        self._flayout.addWidget(feature.form)
        feature.sigClicked.connect(self.featureClicked)
        feature.sigDelete.connect(self.removeFeature)

    @QtCore.Slot(QtGui.QWidget)
    def featureClicked(self, feature):
        self.collapseAllFeatures()
        self.showForm(feature.form)
        feature.expand()

    def collapseAllFeatures(self):
        for feature in self.features:
            feature.collapse()

    def showForm(self, form):
        self._flayout.setCurrentWidget(form)

    def removeFeature(self, feature):
        self.features.remove(feature)
        feature.deleteLater()
        self.update()

    def removeAllFeatures(self):
        for feature in self.features:
            self.removeFeature(feature)

    def clearLayouts(self):
        for feature in self.features:
            self._flayout.removeWidget(feature)
            self._llayout.removeWidget(feature)

    def update(self):
        self.clearLayouts()
        self.showForm(self.blank_form)
        for feature in self.features:
            self._llayout.addWidget(feature)
            self._flayout.addWidget(feature.form)
        self.collapseAllFeatures()

    def swapFeatures(self, f1, f2):
        idx_1, idx_2 = self.features.index(f1), self.features.index(f2)
        self.features[idx_1], self.features[idx_2] = self.features[idx_2], self.features[idx_1]
        self.update()
