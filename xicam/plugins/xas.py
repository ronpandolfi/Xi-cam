import platform
from fabio import edfimage

# Use NSURL as a workaround to pyside/Qt4 behaviour for dragging and dropping on OSx
op_sys = platform.system()
# if op_sys == 'Darwin':
#     from Foundation import NSURL

import base
from PySide import QtGui
import os

import widgets
import numpy as np
from pipeline.spacegroups import spacegroupwidget
from pipeline import loader


class XASPlugin(base.plugin):
    name = 'XAS'

    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        super(XASPlugin, self).__init__(*args, **kwargs)

        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent


    def dragEnterEvent(self, e):
        print(e)
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
                print(fname)
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
        try:
            self.getCurrentTab().redrawimage()
        except AttributeError:
            print "Using hack to bypass strange qsignal behavior. Fix this!"

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


    def openfiles(self, paths=None, operation=None, operationname=None):
        self.activate()
        if type(paths) is not list:
            paths = [paths]

        widget = widgets.OOMTabItem(itemclass=widgets.xasviewer, paths=paths)
        self.centerwidget.addTab(widget, os.path.basename(paths[0]))
        self.centerwidget.setCurrentWidget(widget)

    def opendata(self, data=None, operation=None, operationname=None):
        self.activate()
        dimg = loader.diffimage(data=data)
        widget = widgets.OOMTabItem(itemclass=widgets.dimgViewer, dimg=dimg, operation=operation,
                                    operationname=operationname, plotwidget=self.bottomwidget,
                                    toolbar=self.toolbar)
        self.centerwidget.addTab(widget, 'Imported data')
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

    def exportimage(self):
        fabimg = edfimage.edfimage(np.rot90(self.getCurrentTab().imageitem.image))
        dialog = QtGui.QFileDialog(parent=None, caption=u"Export image as EDF",
                                   directory=unicode(os.path.dirname(self.getCurrentTab().paths[0])),
                                   filter=u"EDF (*.edf)")
        dialog.selectFile(unicode(os.path.dirname(self.getCurrentTab().paths[0])))
        filename, ok = dialog.getSaveFileName()
        if ok and filename:
            fabimg.write(filename)

    def capture(self):
        self.getCurrentTab().capture()