from .. import base


import matplotlib.pyplot as plt

import numpy as np

from larch import Interpreter
from larch_plugins.math.mathutils import index_of
from larch_plugins.xafs import pre_edge

import xasloader
from xicam.widgets.NDTimelinePlotWidget import XASTimelineWidget


def runtest():
    EZTest.bottomwidget.clear()

def openfiles(filepaths):
    for path in filepaths:
        spectra = xasloader.open(path)
        spectra.treat()

        for t,scan in enumerate(spectra.scans):
            EZTest.plot((t,(np.array(scan.Energy), np.array(scan.I_norm))))




EZTest=base.EZplugin(name='XAS',
                     toolbuttons=[('xicam/gui/icons_34.png',runtest)],
                     parameters=[{'name':'Pre-edge Min','value':10,'type':'int'},
                                 {'name':'Pre-edge Max','value':30,'type':'int'}],
                     openfileshandler=openfiles,
                     centerwidget=None,
                     bottomwidget=XASTimelineWidget)