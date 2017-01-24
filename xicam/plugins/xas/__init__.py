from .. import base


import matplotlib.pyplot as plt

import numpy as np

from larch import Interpreter
from larch_plugins.math.mathutils import index_of
from larch_plugins.xafs import pre_edge

import xasloader



def runtest():
    import numpy as np

    img = np.random.random((100,100,100))
    EZTest.setImage(img)

    hist = np.histogram(img,100)
    EZTest.plot(hist[1][:-1],hist[0])

def openfiles(filepaths):
    for path in filepaths:
        spectra = xasloader.open(path)

        for scan in spectra.scans:
            EZTest.plot(np.array(scan.e), np.array(scan.y))




EZTest=base.EZplugin(name='XAS',toolbuttons=[('xicam/gui/icons_34.png',runtest)],parameters=[{'name':'Test','value':10,'type':'int'}],openfileshandler=openfiles,imageenabled=False)