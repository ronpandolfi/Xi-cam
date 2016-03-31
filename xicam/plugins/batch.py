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
import re

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
        self.logOption = pTypes.SimpleParameter(type='bool', name='Log scale image', value=False)
        self.exportformat = pTypes.ListParameter(type='list', name='Image export format', value=0, values=['EDF (.edf)','TIFF (.tif)','JPEG (.jpg)'])
        self.processButton = pTypes.ActionParameter(name='Process')
        # self.abortButton = pTypes.ActionParameter(name='Abort')
        params = [self.remeshOption, self.integrateOption, self.roiOption, self.logOption, self.exportformat, self.processButton]
        paramgroup = Parameter.create(name='params', type='group', children=params)
        self.rightwidget.setParameters(paramgroup, showTop=False)

        self.processButton.sigActivated.connect(self.processfiles)

        super(plugin, self).__init__(*args, **kwargs)

    def processfiles(self):
        pathlist = self.fileslistwidget.paths
        paths = [pathlist.item(index).text() for index in xrange(pathlist.count())]

        imageext=re.findall(r'(?<=\()\..{3}(?=\))',self.exportformat.value())[0]

        for path in paths:

            xglobals.statusbar.showMessage('Processing item ' + str(paths.index(path)+1) + ' of ' + str(len(paths))+ '...')
            xglobals.app.processEvents()

            dimg = loader.diffimage(path)

            if self.remeshOption.value():
                data = dimg.remesh
                if not writer.writeimage(data if not self.logOption.value() else (np.log(data * (data > 0) + (data < 1))), path, suffix='_remeshed', ext=imageext):
                    break

            if self.integrateOption.value():
                x, y, _ = dimg.integrate()
                data = np.array([x, y])
                if not writer.writearray(data, path, suffix=''):
                    break

            if self.roiOption.value():
                print 'lastroi:',xglobals.lastroi
                if xglobals.lastroi is not None:
                    # lastroi is a tuple with an ROI item and an imageitem (both are need to get a cut array)
                    cut = (xglobals.lastroi[0].getArrayRegion(np.ones_like(dimg.data), xglobals.lastroi[1])).T
                    x, y, _ = dimg.integrate(cut=cut)
                    data = np.array([x, y])
                    if not writer.writearray(data, path, suffix='_roi'):
                        break
                else:
                    pass  # No ROI was defined, hm...

            if os.path.splitext(path)[-1] != imageext:
                writer.writeimage(dimg.data, path, ext=imageext)

        xglobals.statusbar.showMessage('Ready...')