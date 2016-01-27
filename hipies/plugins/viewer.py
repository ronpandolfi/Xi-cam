import base
from PySide import QtGui
import os

import widgets
import numpy as np
from pipeline.spacegroups import spacegroupwidget


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
                                     self.roi, self.arccut, self.polymask, spacegroup=self.togglespacegroup)

        super(plugin, self).__init__(*args, **kwargs)

        self.sigUpdateExperiment.connect(self.redrawcurrent)
        self.sigUpdateExperiment.connect(self.replotcurrent)
        self.sigUpdateExperiment.connect(self.invalidatecache)

        self.booltoolbar.actionAdd.triggered.connect(self.addmode)
        self.booltoolbar.actionSubtract.triggered.connect(self.subtractmode)
        self.booltoolbar.actionAdd_with_coefficient.triggered.connect(self.addwithcoefmode)
        self.booltoolbar.actionSubtract_with_coefficient.triggered.connect(self.subtractwithcoefmode)
        self.booltoolbar.actionDivide.triggered.connect(self.dividemode)
        self.booltoolbar.actionAverage.triggered.connect(self.averagemode)

        self.spacegroupwidget = spacegroupwidget()
        self.spacegroupwidget.sigDrawSGOverlay.connect(self.drawsgoverlay)
        self.placeholders[1].addWidget(self.spacegroupwidget)


    def drawsgoverlay(self, peakoverlay):
        self.getCurrentTab().drawsgoverlay(peakoverlay)

    def addmode(self):
        """
        Launch a tab as an add operation
        """
        operation = lambda m: np.sum(m, 0)
        self.launchmultimode(operation, 'Addition')

    def subtractmode(self):
        """
        Launch a tab as an sub operation
        """
        operation = lambda m: m[0] - np.sum(m[1:], 0)
        self.launchmultimode(operation, 'Subtraction')

    def addwithcoefmode(self):
        """
        Launch a tab as an add with coef operation
        """
        coef, ok = QtGui.QInputDialog.getDouble(self.ui, u'Enter scaling coefficient x (A+xB):', u'Enter coefficient')

        if coef and ok:
            operation = lambda m: m[0] + coef * np.sum(m[1:], 0)
            self.launchmultimode(operation, 'Addition with coef (x=' + coef + ')')

    def subtractwithcoefmode(self):
        """
        Launch a tab as a sub with coef operation
        """
        coef, ok = QtGui.QInputDialog.getDouble(None, u'Enter scaling coefficient x (A-xB):', u'Enter coefficient')

        if coef and ok:
            operation = lambda m: m[0] - coef * np.sum(m[1:], 0)
            self.launchmultimode(operation, 'Subtraction with coef (x=' + str(coef))

    def dividemode(self):
        """
        Launch a tab as a div operation
        """
        operation = lambda m: m[0] / m[1]
        self.launchmultimode(operation, 'Division')

    def averagemode(self):
        """
        Launch a tab as an avg operation
        """
        operation = lambda m: np.mean(m, 0)
        self.launchmultimode(operation, 'Average')

    def launchmultimode(self, operation, operationname):
        """
        Launch a tab in multi-image operation mode
        """
        self.openSelected(operation, operationname)

    def tabCloseRequested(self, index):
        self.centerwidget.widget(index).deleteLater()

    def getCurrentTab(self):
        if self.centerwidget.currentWidget() is None:
            return None
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
        self.imagePropModel.widgetchanged()


    def openfiles(self, paths=None, operation=None, operationname=None):
        self.activate()
        if type(paths) is not list:
            paths = [paths]

        widget = widgets.OOMTabItem(itemclass=widgets.dimgViewer, paths=paths, operation=operation,
                                    operationname=operationname, plotwidget=self.bottomwidget,
                                    toolbar=self.toolbar)
        self.centerwidget.addTab(widget, os.path.basename(paths[0]))
        self.centerwidget.setCurrentWidget(widget)

    def currentImage(self):
        return self.getCurrentTab()

    def replotcurrent(self):
        self.getCurrentTab().replot()

    def invalidatecache(self):
        self.getCurrentTab().dimg.invalidatecache()

    def togglespacegroup(self):
        if self.toolbar.actionSpaceGroup.isChecked():
            self.placeholders[1].setCurrentWidget(self.spacegroupwidget)
        else:
            self.placeholders[1].setCurrentWidget(self.rightwidget)