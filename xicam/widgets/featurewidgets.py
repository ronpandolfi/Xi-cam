from PySide import QtGui, QtCore


__author__ = "Luis Barroso-Luque, Ronald J Pandolfi"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


class FeatureWidget(QtGui.QWidget):
    """
    Widget that stands for a feature or function with a a preview icon and a delete icon. These are added to a layout
    to represent a set of features (as in higgisaxs) or functions (as in tomography). FeatureWidgets are normally
    managed by an instance of the FeatureManager Class


    Attributes
    ----------
    name : str
        Name to be shown in the FeatureWidget's GUI
    form
        Widget that portrays information about the widget (ie a textlabel, spinbox, ParameterTree, etc)
    subfeatures : list of FeatureWidgets/QWidgets
        Subfeatures of the widget
    previewButton : QtGui.QPushButton
        Button to call the preview action associated with this widget
    closeButton : QtGui.QPushButton
        Button that emits sigDelete

    Signals
    -------
    sigClicked(FeatureWidget)
        Signal emitted when the widget is clicked, self is emitted so that a FeatureManager has access to the sender
        Qt has a sender method which could also be used instead...
    sigDelete(FeatureWidget)
        Signal emitted when the widget's closeButton is clicked, self is emitted
    sigSubFeature(FeatureWidget)
        Signal emitted when a subfeature is added to the current widget. Emits the subfeature.


    Parameters
    ----------
    name : str
        Name to be given to widget
    checkable : bool, optional
        Boolean specifying if the previewButton is checkable (for toggling)
    closeable : bool, optional
        Boolean specifying whether the widget can be closed/deleted
    subfeatures : list of FeatureWidgets/QWidgets
        Initialization list of subfeatures. New subfeatures can be added with the addSubFeatureMethod
    parent
        Parent widget, normally the manager
    """

    sigClicked = QtCore.Signal(QtGui.QWidget)
    sigDelete = QtCore.Signal(QtGui.QWidget)
    sigSubFeature = QtCore.Signal(QtGui.QWidget)

    def __init__(self, name='', checkable=True, closeable=True, subfeatures=None, parent=None):
        super(FeatureWidget, self).__init__(parent=parent)

        self.name = name
        self.form = QtGui.QLabel(self.name) # default form
        self.subfeatures = []

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
        self.previewButton = QtGui.QPushButton(parent=self)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.previewButton.sizePolicy().hasHeightForWidth())
        self.previewButton.setSizePolicy(sizePolicy)
        self.previewButton.setStyleSheet("margin:0 0 0 0;")
        self.previewButton.setText("")
        icon = QtGui.QIcon()

        if checkable:
            icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_48.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_47.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
            self.previewButton.setCheckable(True)
        else:
            icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_47.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
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
        icon1.addPixmap(QtGui.QPixmap("xicam/gui/icons_46.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
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
        self.subframe_layout = QtGui.QVBoxLayout(self.subframe)# QtGui.QGridLayout(self.subframe)
        self.subframe_layout.setContentsMargins(0, 0, 0, 0)
        self.subframe_layout.setSpacing(0)
        self.verticalLayout.addWidget(self.subframe)
        self.subframe.hide()

        if not closeable:
            self.closeButton.hide()

        if subfeatures is not None:
            for subfeature in subfeatures:
                self.addSubFeature(subfeature)

        self.collapse()

    def addSubFeature(self, subfeature):
        """
        Adds a subfeature to the widget

        Parameters
        ----------
        subfeature : FeatureWidget/QWidget
            Widget to add as a subfeature
        """

        h = QtGui.QHBoxLayout()
        indent = QtGui.QLabel('  -   ')
        h.addWidget(indent)
        subfeature.destroyed.connect(indent.deleteLater)
        subfeature.destroyed.connect(h.deleteLater)
        if isinstance(subfeature, QtGui.QLayout):
            h.addLayout(subfeature)
        elif isinstance(subfeature, QtGui.QWidget):
            h.addWidget(subfeature)
        self.subframe_layout.addLayout(h)
        try:
            subfeature.sigDelete.connect(self.removeSubFeature)
        except AttributeError:
            pass

        self.sigSubFeature.emit(subfeature)
        self.subfeatures.append(subfeature)

    def removeSubFeature(self, subfeature):
        """
        Removes a subfeature

        Parameters
        ----------
        subfeature : FeatureWidget/QWidget
            Feature to remove
        """

        self.subfeatures.remove(subfeature)
        subfeature.deleteLater()
        del subfeature

    def delete(self):
        """
        Emits delete signal with self. Connected to deleteButton's clicked
        """

        self.sigDelete.emit(self)

    def collapse(self):
        """
        Collapses all expanded subfeatures
        """

        if self.subframe is not None:
            self.subframe.hide()

    def expand(self):
        """
        Expands subfeatures
        """

        if self.subframe is not None:
            self.subframe.show()

    def mouseClicked(self):
        """
        Slot to handle when a feature is clicked
        """

        self.sigClicked.emit(self)
        self.setFocus()
        self.previewButton.setFocus()


class ROlineEdit(QtGui.QLineEdit):
    """
    Subclass of QlineEdit used for labels in FeatureWidgets
    """

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


class FeatureManager(QtCore.QObject):
    """
    Feature Manager class to manage a list of FeatureWidgets and show the list in an appropriate layout and their
    corresponding forms in another layout. list layout must have an addWidget, removeWidget methods. Form layout
    must in addition have a setCurrentWidget method

    Attributes
    ----------
    features : list of FeatureWidgets
        List of the FeatureWidgets managed
    selectedFeature : FeatureWidget
        The currently selected feature

    Parameters
    ----------
    list_layout : QtGui.QLayout
        Layout to display the list of FeatureWidgets
    form_layout : QtGui.QLayout
        Layout to display the FeaturenWidgets form (pyqtgraph.Parameter)
    feature_widgets : list of FeatureWidgets, optional
        List with feature widgets for initialization
    blank_form : QtGui.QWidget, optional
        Widget to display in form_layout when not FunctionWidget is selected
    """

    def __init__(self, list_layout, form_layout, feature_widgets=None, blank_form=None):
        self._llayout = list_layout
        self._flayout = form_layout
        self.features = []
        self.selectedFeature = None

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
        super(FeatureManager, self).__init__()

    @property
    def count(self):
        """
        Number of features managed
        """
        return len(self.features)

    @property
    def nextFeature(self):
        """
        The feature after the selectedFeature
        """
        try:
            index = self.features.index(self.selectedFeature)
        except (ValueError, IndexError):
            return None
        if index > 0:
            return self.features[index - 1]
        else:
            return self.selectedFeature

    @property
    def previousFeature(self):
        """
        The feature before the selectedFeature
        """
        if self.selectedFeature is None:
            return None
        index = self.features.index(self.selectedFeature)
        if index < self.count - 1:
            return self.features[index + 1]
        else:
            return self.selectedFeature

    @QtCore.Slot(QtGui.QWidget)
    def featureClicked(self, feature):
        """
        Slot used to receive features sigClicked
        """
        if feature in self.features:
            self.collapseAllFeatures()
        self.showForm(feature.form)
        feature.expand()
        self.selectedFeature = feature

    @QtCore.Slot(QtGui.QWidget)
    def removeFeature(self, feature):
        """
        Slot used to receive features sigDelete
        """
        self.features.remove(feature)
        feature.deleteLater()
        del feature
        self.showForm(self.blank_form)

    @QtCore.Slot(QtGui.QWidget)
    def subFeatureAdded(self, subfeature):
        """
        Slot used to receive features sigSubFeature
        """
        try:
            subfeature.sigClicked.connect(self.featureClicked)
            subfeature.sigSubFeature.connect(self.subFeatureAdded)
            self._flayout.addWidget(subfeature.form)
        except AttributeError:
            pass

    def addFeature(self, feature):
        """
        Adds a subfeature to the given feature
        """
        self.features.append(feature)
        self._llayout.addWidget(feature)
        self._flayout.addWidget(feature.form)
        feature.sigClicked.connect(self.featureClicked)
        feature.sigDelete.connect(self.removeFeature)
        feature.sigSubFeature.connect(self.subFeatureAdded)
        if feature.subfeatures is not None:
            for subfeature in feature.subfeatures:
                self.subFeatureAdded(subfeature)

    def collapseAllFeatures(self):
        """
        Collapses all features with subfeatures
        """
        for feature in self.features:
            feature.collapse()
            if feature.subfeatures is not None:
                for subfeature in feature.subfeatures:
                    subfeature.collapse()

    def showForm(self, form):
        """
        Shows the current features form
        """
        self._flayout.setCurrentWidget(form)

    def removeAllFeatures(self):
        """
        Deletes all features
        """
        for feature in self.features:
            feature.deleteLater()
            del feature
        self.features = []
        self.showForm(self.blank_form)

    def clearLayouts(self):
        """
        Removes all features and forms from layouts
        """
        for feature in self.features:
            self._flayout.removeWidget(feature)
            self._llayout.removeWidget(feature)

    def update(self):
        """
        Updates the layouts to show the current list of features
        """
        self.clearLayouts()
        for feature in self.features:
            self._llayout.addWidget(feature)
            self._flayout.addWidget(feature.form)
        self.showForm(self.selectedFeature.form)

    def swapFeatures(self, f1, f2):
        """
        Swaps the location of two features
        """
        idx_1, idx_2 = self.features.index(f1), self.features.index(f2)
        self.features[idx_1], self.features[idx_2] = self.features[idx_2], self.features[idx_1]
        self.update()
