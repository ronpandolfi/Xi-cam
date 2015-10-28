import nexpy.api.nexus as nx
import numpy as np
import scipy.ndimage
from PySide import QtCore
from hipies import debugtools
import multiprocessing
import time
import os
from hipies import debugtools
from PIL import Image


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

    newfile = not os.path.isfile(path)
    if newfile:
        nxroot = nx.NXroot(nx.NXdata(kwargs['img']))
    else:
        nxroot = nx.load(path, mode='rw')


    if not hasattr(nxroot.data, 'rawfile'):
        nxroot.data.rawfile = kwargs['rawpath']
    if not hasattr(nxroot.data, 'thumb'):
        nxroot.data.thumbnail = kwargs['thumb']
    # TODO: merge variation
    if not hasattr(nxroot.data, 'variation'):
        nxroot.data.variation = kwargs['variation'].items()

    if newfile:
        writenexus(nxroot, kwargs['path'])


def writenexus(nexroot, path):
    try:
        nexroot.save(path)
    except IOError:
        print('IOError: Check that you have write permissions.')


@debugtools.timeit
def thumbnail(img, factor=10):
    """
    Generate a thumbnail from an image
    """

    shape = img.shape[::-1]
    img = Image.fromarray(img)
    img.thumbnail(np.divide(shape, factor))
    # print 'thumb:', shape, np.array(img).shape
    return np.array(img)




def blockshaped(arr, factor):
    firstslice = np.array_split(arr, arr.shape[0] // factor)
    secondslice = map(lambda x: np.array_split(x, arr.shape[1] // factor, axis=1), firstslice)
    return np.array(secondslice)