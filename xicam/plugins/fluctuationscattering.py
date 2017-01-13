# avg dphi correlation (mask normalized) for each q

import base
from PySide import QtGui
import os

import widgets


class FXSPlugin(base.plugin):
    name = 'FXS'

    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        super(FXSPlugin, self).__init__(*args, **kwargs)

        # self.sigUpdateExperiment.connect()


    def tabCloseRequested(self, index):
        self.centerwidget.widget(index).deleteLater()

    def getCurrentTab(self):
        if self.centerwidget.currentWidget() is None:
            return None
        return self.centerwidget.currentWidget().widget

    def currentChanged(self, index):
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        self.centerwidget.currentWidget().load()
        self.imagePropModel.widgetchanged()


    def openfiles(self, paths=None, operation=None, operationname=None):
        self.activate()
        if type(paths) is not list:
            paths = [paths]

        widget = widgets.OOMTabItem(itemclass=widgets.fxsviewer, paths=paths, operation=operation,
                                    operationname=operationname)
        self.centerwidget.addTab(widget, os.path.basename(paths[0]))
        self.centerwidget.setCurrentWidget(widget)

    def currentImage(self):
        return self.getCurrentTab()

    def invalidatecache(self):
        self.getCurrentTab().dimg.invalidatecache()
