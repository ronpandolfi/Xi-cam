from .. import base


import matplotlib.pyplot as plt

import numpy as np

import xasloader
from xicam.widgets.NDTimelinePlotWidget import XASTimelineWidget
from xicam.widgets import fitting
from PySide.QtGui import *

def runtest():
    EZTest.bottomwidget.clear()

def openfiles(filepaths):
    for path in filepaths:
        spectra = xasloader.open(path)
        spectra.treat()

        for t,scan in enumerate(spectra.scans):
            EZTest.plot((t,(np.array(scan.Energy), np.array(scan.I_norm))))


XASplot = XASTimelineWidget()

EZTest=base.EZplugin(name='XAS',
                     toolbuttons=[('xicam/gui/icons_34.png',runtest)],
                     parameters=[{'name':'Pre-edge Min','value':10,'type':'int'},
                                 {'name':'Pre-edge Max','value':30,'type':'int'},
                                 fitting.FitParameter(XASplot.stackplot.plotWidget)],
                     openfileshandler=openfiles,
                     centerwidget=None,
                     bottomwidget=XASplot)