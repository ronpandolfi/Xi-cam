import base
from PySide import QtGui
import os

import widgets


class plugin(base.plugin):
    name = 'Viewer'

    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        self.bottomwidget = widgets.integrationwidget()

        self.toolbar = widgets.toolbar.difftoolbar()
        self.toolbar.connecttriggers(self.calibrate, self.centerfind, self.refinecenter, self.redrawcurrent,
                                     self.redrawcurrent, self.remeshmode, self.linecut, self.vertcut,
                                     self.horzcut, self.redrawcurrent, self.redrawcurrent, self.redrawcurrent,
                                     self.roi, self.arccut, self.polymask)

        super(plugin, self).__init__(*args, **kwargs)


    def tabCloseRequested(self, index):
        self.centerwidget.widget(index).deleteLater()

    def getCurrentTab(self):
        return self.centerwidget.currentWidget().widget

    def calibrate(self):
        self.getCurrentTab().calibrate()

    def centerfind(self):
        self.getCurrentTab().centerfind()

    def refinecenter(self):
        self.getCurrentTab().refinecenter()

    def redrawcurrent(self):
        self.getCurrentTab().redrawimage()

    def remeshmode(self):
        self.getCurrentTab().redrawimage()
        self.getCurrentTab().replot()

    def linecut(self):
        self.getCurrentTab().linecut()

    def vertcut(self):
        self.getCurrentTab().verticalcut()

    def horzcut(self):
        self.getCurrentTab().horizontalcut()

    def roi(self):
        self.getCurrentTab().roi()

    def arccut(self):
        self.getCurrentTab().arccut()

    def polymask(self):
        self.getCurrentTab().polymask()


    def currentChanged(self, index):
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        self.centerwidget.currentWidget().load()


    def openfiles(self, files, operation=None):
        self.activate()
        if type(files) is not list:
            files = [files]

        if len(files) == 1:
            widget = widgets.OOMTabItem(itemclass=widgets.dimgViewer, paths=files[0], plotwidget=self.bottomwidget,
                                        toolbar=self.toolbar)
            self.centerwidget.addTab(widget, os.path.basename(files[0]))
            self.centerwidget.setCurrentWidget(widget)

