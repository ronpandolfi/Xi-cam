from PySide import QtCore, QtGui


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

        self._flayout.addWidget(self.blank_form)
        self.showForm(self.blank_form)

    @property
    def count(self):
        return len(self.features)

    def addFeature(self, feature):
        self.features.append(feature)
        self._llayout.addWidget(feature)
        self._flayout.addWidget(feature.form)
        feature.sigShowForm.connect(self.showForm)
        feature.sigDelete.connect(self.removeFeature)

    @QtCore.Slot(QtGui.QWidget)
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

    def swapFeatures(self, f1, f2):
        idx_1, idx_2 = self.features.index(f1), self.features.index(f2)
        self.features[idx_1], self.features[idx_2] = self.features[idx_2], self.features[idx_1]
        self.update()