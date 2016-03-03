# avg dphi correlation (mask normalized) for each q

import base
from PySide import QtGui
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import os
from pipeline import loader, writer
from xicam import config
import numpy as np
from xicam import xglobals

import widgets


class plugin(base.plugin):
    name = 'Batch'

    def __init__(self, *args, **kwargs):

        self.centerwidget = QtGui.QWidget()
        self.fileslistwidget = widgets.filesListWidget()
        self.centerwidget.setLayout(QtGui.QVBoxLayout())
        self.centerwidget.layout().addWidget(self.fileslistwidget)

        self.rightwidget = ParameterTree()
        self.remeshOption = pTypes.SimpleParameter(type='bool', name='GIXS remeshing', value=False)
        self.integrateOption = pTypes.SimpleParameter(type='bool', name='Azimuthal integration', value=True)
        self.roiOption = pTypes.SimpleParameter(type='bool', name='Integrate last ROI', value=True)
        self.processButton = pTypes.ActionParameter(name='Process')
        # self.abortButton = pTypes.ActionParameter(name='Abort')
        params = [self.remeshOption, self.integrateOption, self.roiOption, self.processButton]
        paramgroup = Parameter.create(name='params', type='group', children=params)
        self.rightwidget.setParameters(paramgroup, showTop=False)

        self.processButton.sigActivated.connect(self.processfiles)

        super(plugin, self).__init__(*args, **kwargs)

    def processfiles(self):
        pathlist = self.fileslistwidget.paths
        paths = [pathlist.item(index).text() for index in xrange(pathlist.count())]
        for path in paths:

            dimg = loader.diffimage(path)

            if self.remeshOption.value():
                data = dimg.remesh
                if not writer.writeimage(data, path, suffix='remeshed'):
                    break

            if self.integrateOption.value():
                x, y, _ = dimg.integrate()
                data = np.array([x, y.data])
                if not writer.writearray(data, path, suffix=''):
                    break

            if self.roiOption.value():
                print xglobals.lastroi
                if xglobals.lastroi is not None:
                    x, y, _ = dimg.integrate(cut=xglobals.lastroi)
                    data = np.array([x, y.data])
                    if not writer.writearray(data, path, suffix='roi'):
                        break
                else:
                    pass  # No ROI was defined, hm...