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
import ui


class plugin(base.plugin):
    """
    Tomography plugin class
    """
    name = "Tomography"
    def __init__(self, *args, **kwargs):

        self.leftwidget, self.centerwidget, self.rightwidget, self.bottomwidget, self.toolbar = ui.loadUi()
        self.functionwidget = ui.functionwidget
        self.console = self.bottomwidget
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        # SETUP FEATURES
        fmanager.layout = self.functionwidget.functionsList
        self.functionwidget.functionsList.setAlignment(QtCore.Qt.AlignBottom)
        fmanager.load_function_pipeline('yaml/tomography/default_pipeline.yml', setdefaults=True)

        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent

        self.toolbar.connecttriggers(self.previewSlice, self.preview3D, self.fullReconstruction, self.manualCenter)
        super(plugin, self).__init__(*args, **kwargs)

        self._recon_running = False

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
        self.toolbar.actionCenter.setChecked(False)
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        try:
            self.centerwidget.currentWidget().load()
            current_dataset = self.currentDataset()
            if current_dataset is not None:
                current_dataset.sigReconFinished.connect(self.fullReconstructionFinished)
                self.setPipelineValues(current_dataset)
        except AttributeError as e:
            print e.message

    def setPipelineValues(self, widget):
        ui.propertytable.setData(widget.data.header.items())
        ui.propertytable.setHorizontalHeaderLabels(['Parameter', 'Value'])
        ui.propertytable.show()
        ui.setconfigparams(int(widget.data.header['nslices']),
                           int(widget.data.header['nangles']))
        fmanager.set_function_defaults(widget.data.header, funcs=fmanager.functions)
        fmanager.update_function_parameters(funcs=fmanager.functions)
        recon = fmanager.recon_function
        if recon is not None:
            recon.setCenterParam(self.currentDataset().cor)

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
        if not self._recon_running:
            self._recon_running = True
            self.console.local_console.clear()
            start = ui.configparams.child('Start Sinogram').value()
            end = ui.configparams.child('End Sinogram').value()
            step =  ui.configparams.child('Step Sinogram').value()
            chunks = ((end - start) // step - 1) // ui.configparams.child('Sinograms/Chunk').value() + 1
            print chunks
            self.currentDataset().runFullRecon((ui.configparams.child('Start Projection').value(),
                                                ui.configparams.child('End Projection').value(),
                                                ui.configparams.child('Step Projection').value()),
                                               (start, end, step),
                                               chunks,
                                               ui.configparams.child('CPU Cores').value(),
                                               self.console.log2local)

        else:
            r = QtGui.QMessageBox.warning(self, 'Reconstruction running', 'A reconstruction is currently running.\n'
                                                                          'Are you sure you want to start another one?',
                                          (QtGui.QMessageBox.Yes | QtGui.QMessageBox.No))
            if r is QtGui.QMessageBox.Yes:
                QtGui.QMessageBox.information(self, 'Reconstruction request',
                                              'Then you should wait until the first one finishes.')

    def fullReconstructionFinished(self):
        self.console.log2local('Reconstruction complete.')
        self._recon_running = False

    def manualCenter(self, value):
        self.currentDataset().onManualCenter(value)
