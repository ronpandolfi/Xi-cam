import nexpy.api.nexus as nx
import numpy as np
import scipy.ndimage
from PySide import QtCore
from hipies import debug
import multiprocessing
import time
import os


class nexusmerger(QtCore.QThread):
    def __init__(self, *args, **kwargs):
        super(nexusmerger, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def run(self):
        p = multiprocessing.Process(target=mergenexus, kwargs=self.kwargs)
        self.job = p
        p.start()


def mergenexus(**kwargs):
    path = kwargs['path']

    if os.path.isfile(path):
        nxroot = nx.load(path, mode='rw')
    else:
        nxroot = nx.NXroot(nx.NXdata(kwargs['img']))

    if not hasattr(nxroot.data, 'rawfile'):
        nxroot.data.rawfile = kwargs['rawpath']
    if not hasattr(nxroot.data, 'thumb'):
        nxroot.data.thumbnail = kwargs['thumb']
    # TODO: merge variation
    if not hasattr(nxroot.data, 'variation'):
        nxroot.data.variation = kwargs['variation'].items()

    writenexus(nxroot, kwargs['path'])


def writenexus(nexroot, path):
    nexroot.save(path)

def thumbnail(img, size=160.):
    """
    Generate a thumbnail from an image
    """
    size = float(size)
    img = np.log(img * (img > 0) + 1.)
    img *= 255 / np.max(np.asarray(img))

    desiredsize = np.array([size, size])

    zoomfactor = np.max(desiredsize / np.array(img.shape))

    # OVERRIDE!
    zoomfactor = 0.1

    img = scipy.ndimage.zoom(img, zoomfactor, order=2) / 100
    img = img.astype(np.uint8)
    return img