import platform
from fabio import edfimage
from pipeline import msg

# Use NSURL as a workaround to pyside/Qt4 behaviour for dragging and dropping on OSx
op_sys = platform.system()
if op_sys == 'Darwin':
    try:
        from Foundation import NSURL
    except ImportError:
        msg.logMessage('NSURL not found. Drag and drop may not work correctly',msg.WARNING)

import base
from PySide import QtGui, QtCore
import os
from pyqtgraph.parametertree import ParameterTree
import widgets
import numpy as np
from pipeline.spacegroups import spacegroupwidget
from pipeline import loader
from xicam import config
import fabio
from pipeline import calibration

from xicam.widgets.calibrationpanel import calibrationpanel

# Globals so Timeline can share the same rightmodes
configtree = ParameterTree()
configtree.setParameters(config.activeExperiment, showTop=False)


def tiltStyleMenuRequested(pos):
    config.activeExperiment.tiltStyleMenu.exec_(configtree.mapToGlobal(pos))


configtree.customContextMenuRequested.connect(tiltStyleMenuRequested)
configtree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)




rightmodes = [(configtree, QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.File))]

class ViewerPlugin(base.plugin):
    name = 'Viewer'
    sigUpdateExperiment = QtCore.Signal()


    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        self.rightmodes = rightmodes
        self.bottomwidget = widgets.integrationwidget(self.getCurrentTab)



        self.toolbar = widgets.toolbar.difftoolbar()
        self.toolbar.connecttriggers(self.calibrate, self.centerfind, self.refinecenter, self.redrawcurrent,
                                     self.redrawcurrent, self.remeshmode, self.linecut, self.vertcut,
                                     self.horzcut, self.redrawcurrent, self.redrawcurrent, self.redrawcurrent,
                                     self.roi, self.arccut, self.polymask,
                                     capture=None, removecosmics=self.removecosmics,thresholdmask=self.thresholdmask)


        self.spacegroupwidget = spacegroupwidget()
        self.spacegroupwidget.sigDrawSGOverlay.connect(self.drawsgoverlay)
        sgicon = QtGui.QIcon()
        sgicon.addPixmap(QtGui.QPixmap("xicam/gui/icons_35.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rightmodes.append((self.spacegroupwidget,sgicon))

        self.propertytable = widgets.frameproptable()
        self.rightmodes.append((self.propertytable,QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.Desktop)))

        self.calibrationPanel = calibrationpanel()
        calicon = QtGui.QIcon()
        calicon.addPixmap(QtGui.QPixmap("xicam/gui/icons_28.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rightmodes.append((self.calibrationPanel, calicon))
        self.calibrationPanel.sigCalibrate.connect(self.calibrate)

        super(ViewerPlugin, self).__init__(*args, **kwargs)

        config.activeExperiment.sigTreeStateChanged.connect(self.sigUpdateExperiment)
        self.sigUpdateExperiment.connect(self.redrawcurrent)
        self.sigUpdateExperiment.connect(self.replotcurrent)
        self.sigUpdateExperiment.connect(self.invalidatecache)

        self.booltoolbar.actionAdd.triggered.connect(self.addmode)
        self.booltoolbar.actionSubtract.triggered.connect(self.subtractmode)
        self.booltoolbar.actionAdd_with_coefficient.triggered.connect(self.addwithcoefmode)
        self.booltoolbar.actionSubtract_with_coefficient.triggered.connect(self.subtractwithcoefmode)
        self.booltoolbar.actionDivide.triggered.connect(self.dividemode)
        self.booltoolbar.actionAverage.triggered.connect(self.averagemode)



        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent


    def dragEnterEvent(self, e):
        e.accept()
        # if e.mimeData().hasFormat('text/plain'):
        # e.accept()
        # else:
        #     e.accept()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if op_sys == 'Darwin':
                fname = str(NSURL.URLWithString_(str(url.toString())).filePathURL().path())
            else:
                fname = str(url.toLocalFile())
            if os.path.isfile(fname):
                msg.logMessage(fname,msg.DEBUG)
                self.openfiles([fname])
            e.accept()




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
        if self.centerwidget.currentWidget() is None: return None
        if not hasattr(self.centerwidget.currentWidget(),'widget'): return None
        return self.centerwidget.currentWidget().widget

    def calibrate(self, algorithm=calibration.fourierAutocorrelation, calibrant='AgBh'):
        self.getCurrentTab().calibrate(algorithm, calibrant)

    def centerfind(self):
        self.getCurrentTab().centerfind()

    def refinecenter(self):
        self.getCurrentTab().refinecenter()

    def redrawcurrent(self):
        currenttab = self.getCurrentTab()
        if currenttab: self.getCurrentTab().redrawimage()


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

    def thresholdmask(self):
        self.getCurrentTab().thresholdmask()


    def currentChanged(self, index):
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        self.centerwidget.currentWidget().load()
        self.propertytable.setData(self.currentImage().dimg.headers)


    def openfiles(self, paths=None, operation=None, operationname=None):
        self.activate()
        if type(paths) is not list:
            paths = [paths]

        widget = widgets.OOMTabItem(itemclass=widgets.dimgViewer, src=paths, operation=operation,
                                    operationname=operationname, plotwidget=self.bottomwidget,
                                    toolbar=self.toolbar)
        self.centerwidget.addTab(widget, os.path.basename(paths[0]))
        self.centerwidget.setCurrentWidget(widget)

    def opendata(self, data=None, operation=None, operationname=None):
        self.activate()
        dimg = loader.datadiffimage2(data=data)
        widget = widgets.OOMTabItem(itemclass=widgets.dimgViewer, dimg=dimg, operation=operation,
                                    operationname=operationname, plotwidget=self.bottomwidget,
                                    toolbar=self.toolbar)
        self.centerwidget.addTab(widget, 'Imported data')
        self.centerwidget.setCurrentWidget(widget)

    def currentImage(self):
        return self.getCurrentTab()

    def replotcurrent(self):
        if self.getCurrentTab(): self.getCurrentTab().replot()

    def invalidatecache(self):
        if self.getCurrentTab(): self.getCurrentTab().dimg.invalidatecache()

    def togglespacegroup(self):
        if self.toolbar.actionSpaceGroup.isChecked():
            self.placeholders[1].setCurrentWidget(self.spacegroupwidget)
        else:
            self.placeholders[1].setCurrentWidget(self.rightwidget)

    def exportimage(self):
        fabimg = edfimage.edfimage(np.rot90(self.getCurrentTab().imageitem.image))
        dialog = QtGui.QFileDialog(parent=None, caption=u"Export image as EDF",
                                   directory=unicode(os.path.dirname(self.getCurrentTab().dimg.filepath)),
                                   filter=u"EDF (*.edf)")
        dialog.selectFile(unicode(os.path.dirname(self.getCurrentTab().dimg.filepath)))
        filename, ok = dialog.getSaveFileName()
        msg.logMessage(filename,msg.DEBUG)
        if ok and filename:
            fabimg.write(filename)

    def capture(self):
        self.getCurrentTab().capture()

    def maskload(self):
        filename, ok = QtGui.QFileDialog.getOpenFileName(None, 'Load Mask', os.curdir)

        if filename and ok:
            self.openfiles([filename])
        config.activeExperiment.addtomask(np.rot90(fabio.open(filename).data,3))

    def removecosmics(self):
        self.getCurrentTab().removecosmics()