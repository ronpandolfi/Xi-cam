#! /usr/bin/env python

__author__ = "Ronald J Pandolfi"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"

# Use NSURL as a workaround to pyside/Qt4 behaviour for dragging and dropping on OSx
import platform
op_sys = platform.system()
# if op_sys == 'Darwin':
#     from Foundation import NSURL

import os
import numpy as np
import pipeline
from pipeline import loader
from PySide import QtCore, QtGui
from xicam import xglobals
import widgets as twidgets
from xicam.plugins import widgets, explorer
from xicam.plugins import base
from PySide import QtUiTools
import fmanager
from pyqtgraph.parametertree import ParameterTree
from xicam import models
import ui


class plugin(base.plugin):
    """
    Tomography plugin class
    """
    name = "Tomography"
    def __init__(self, *args, **kwargs):

        self.leftwidget, self.centerwidget, self.rightwidget, self.bottomwidget, self.toolbar = ui.loadUi()
        self.functionwidget = ui.functionwidget

        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        # SETUP FEATURES
        fmanager.layout = self.functionwidget.functionsList
        self.functionwidget.functionsList.setAlignment(QtCore.Qt.AlignBottom)
        fmanager.load_function_pipeline('yaml/tomography/functionstack.yml')

        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent

        self.toolbar.connecttriggers(self.previewSlice, self.preview3D, self.fullReconstruction, self.manualCenter)
        super(plugin, self).__init__(*args, **kwargs)

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

    def dragEnterEvent(self, e):
        print(e)
        e.accept()

    def currentChanged(self, index):
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        try:
            self.centerwidget.currentWidget().load()
            ui.propertytable.setData(self.currentDataset().data.header.items())
            ui.propertytable.setHorizontalHeaderLabels([ 'Parameter', 'Value'])
            ui.propertytable.show()
            outname = os.path.join(os.path.dirname(self.currentDataset().data.filepath),
                                   *2*('RECON_' + os.path.split(self.currentDataset().data.filepath)[-1].split('.')[0],))
            ui.setconfigparams(int(self.currentDataset().data.header['nslices']),
                               int(self.currentDataset().data.header['nangles']),
                               outname)
            fmanager.set_function_defaults(self.currentDataset().data.header, funcs=fmanager.functions)
            # recon = fmanager.recon_function
            # if recon is not None:
            #     recon.setCenterParam(self.currentDataset().cor)
        except AttributeError as e:
            print e.message

    def tabCloseRequested(self, index):
        ui.propertytable.clear()
        ui.propertytable.hide()
        self.centerwidget.widget(index).deleteLater()

    def openfiles(self, paths,*args,**kwargs):
        self.activate()
        if type(paths) is list:
            paths = paths[0]

        widget = widgets.OOMTabItem(itemclass=twidgets.TomoViewer, paths=paths)
        self.centerwidget.addTab(widget, os.path.basename(paths))
        self.centerwidget.setCurrentWidget(widget)

    def currentDataset(self):
        try:
            return self.centerwidget.currentWidget().widget
        except AttributeError:
            print 'No dataset open.'

    def previewSlice(self):
        self.currentDataset().runSlicePreview()

    def preview3D(self):
        self.currentDataset().run3DPreview()

    def fullReconstruction(self):
        self.currentDataset().runFullRecon((ui.configparams.child('Start Projection').value(),
                                            ui.configparams.child('End Projection').value(),
                                            ui.configparams.child('Step Projection').value()),
                                           (ui.configparams.child('Start Sinogram').value(),
                                            ui.configparams.child('End Sinogram').value(),
                                            ui.configparams.child('Step Sinogram').value()),
                                           ui.configparams.child('Output Name').value(),
                                           ui.configparams.child('Ouput Format').value(),
                                           ui.configparams.child('Sinogram Chunks').value(),
                                           ui.configparams.child('Cores').value(),
                                           self.currentDataset().processViewer.log2local)

    def manualCenter(self):
        self.currentDataset().manualCenter()
