# avg dphi correlation (mask normalized) for each q

from xicam.plugins import base
from PySide import QtGui
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import os
from pipeline import loader, writer
from xicam import config
import numpy as np
from xicam import xglobals
import re

from xicam.plugins import widgets
from xicam.widgets.workfloweditor import workflowEditorWidget
from pipeline import msg
import imp

class plugin(base.plugin):
    name = 'Batch'

    def __init__(self, *args, **kwargs):

        module = imp.load_source('saxsfunctions',
                                 '/home/rp/PycharmProjects/xicam/pipeline/workflowfunctions/saxsfunctions.py')
        self.leftwidget = self.workflowWidget = workflowEditorWidget('xicam/plugins/batch/defaultworkflow.yml', module)
        self.leftwidget.sigExecute.connect(self.executeBatch)

        self.centerwidget = QtGui.QWidget()
        self.fileslistwidget = widgets.filesListWidget()
        self.centerwidget.setLayout(QtGui.QVBoxLayout())
        self.centerwidget.layout().addWidget(self.fileslistwidget)

        self.rightwidget = None


        super(plugin, self).__init__(*args, **kwargs)

    def executeBatch(self):
        pathlist = self.fileslistwidget.paths
        paths = [pathlist.item(index).text() for index in xrange(pathlist.count())]

        # imageext=re.findall(r'(?<=\()\..{3}(?=\))',self.exportformat.value())[0]

        for path in paths:
            dimg = loader.loaddiffimage(path)
            self.workflowWidget.runWorkflow(dimg=dimg, rawdata=dimg.rawdata)


        msg.showMessage('Ready...')
