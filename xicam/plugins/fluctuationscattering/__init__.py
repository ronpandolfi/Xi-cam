# avg dphi correlation (mask normalized) for each q

from .. import base
from PySide.QtGui import *
import os
from .. import widgets
from fxsmainwidget import fxsmainwidget
from toolbar import fxstoolbar
from fxsplot import fxsplot


class FXSPlugin(base.plugin):
    name = 'FXS'

    def __init__(self, *args, **kwargs):

        self.centerwidget = QTabWidget()
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)
        self.rightwidget = None
        self.bottomwidget = fxsplot()
        self.toolbar = fxstoolbar()

        super(FXSPlugin, self).__init__(*args, **kwargs)

        # self.sigUpdateExperiment.connect()


    def tabCloseRequested(self, index):
        self.centerwidget.widget(index).deleteLater()

    def getCurrentTab(self):
        if self.centerwidget.currentWidget() is None:
            return None
        return self.centerwidget.currentWidget().widget

    def currentChanged(self, index):
        self.bottomwidget.clear()
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        self.centerwidget.currentWidget().load()
        self.imagePropModel.widgetchanged()


    def openfiles(self, paths=None, operation=None, operationname=None):
        self.activate()
        if type(paths) is not list:
            paths = [paths]

        widget = widgets.OOMTabItem(itemclass=fxsmainwidget, src=paths, operation=operation,
                                    operationname=operationname, toolbar=self.toolbar, plotwidget=self.bottomwidget)
        self.centerwidget.addTab(widget, os.path.basename(paths[0]))
        self.centerwidget.setCurrentWidget(widget)

    def currentImage(self):
        return self.getCurrentTab()

    def invalidatecache(self):
        self.getCurrentTab().dimg.invalidatecache()
