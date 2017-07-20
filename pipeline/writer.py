from __future__ import absolute_import
from __future__ import unicode_literals
#import nexpy.api.nexus as nx

from future import standard_library
standard_library.install_aliases()
import numpy as nx

import numpy as np
import scipy.ndimage
from PySide import QtCore
from xicam import debugtools, dialogs
import multiprocessing
import time
import os
from PIL import Image
from fabio import edfimage, tifimage
import scipy.misc
from . import msg


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
        nxroot.data.variation = list(kwargs['variation'].items())

    if newfile:
        writenexus(nxroot, kwargs['path'])


def writenexus(nexroot, path):
    try:
        nexroot.save(path)
    except IOError:
        msg.logMessage('IOError: Check that you have write permissions.',msg.ERROR)


# @debugtools.timeit
def thumbnail(img, factor=5):
    """
    Generate a thumbnail from an image
    """

    shape = img.shape[::-1]
    img = Image.fromarray(img).convert('I')
    img.thumbnail(np.divide(shape, factor))
    # print 'thumb:', shape, np.array(img).shape
    return np.array(img)


import io


@debugtools.timeit
def jpeg(img):
    buffer = io.StringIO()
    pilImage = Image.fromarray(img)
    pilImage.save(buffer, "JPEG", quality=85)
    msg.logMessage(('JPEG buffer size (bytes):', buffer.len),msg.DEBUG)
    return buffer


def blockshaped(arr, factor):
    firstslice = np.array_split(arr, arr.shape[0] // factor)
    secondslice = [np.array_split(x, arr.shape[1] // factor, axis=1) for x in firstslice]
    return np.array(secondslice)


def writeimage(image, path, mask=None, headers=None, suffix='',ext=None, dialog=False):
    if dialog:
        filename,ok = dialogs.savedatadialog(guesspath=path,caption="Save data to "+ext)
        if filename and ok:
            writeimage(image, filename, headers)
            if mask is not None:
                maskname = ''.join(os.path.splitext(filename)[:-1]) + "_mask" + os.path.splitext(filename)[-1]
                writeimage(mask, maskname, headers)

    if headers is None: headers = dict()
    if ext is None: ext = os.path.splitext(path)[-1]

    path = ''.join(os.path.splitext(path)[:-1]) + suffix + ext
    if notexitsoroverwrite(path):
        if ext.lower() == '.edf':
            fabimg = edfimage.edfimage(np.rot90(image), header=headers)
            fabimg.write(path)
        elif ext.lower() == '.tif':
            fabimg = tifimage.tifimage(np.rot90((image.astype(float)/image.max()*2**16).astype(np.int16)), header=headers)
            fabimg.write(path)
        elif ext.lower() == '.png':
            raise NotImplementedError
        elif ext.lower() == '.jpg':
            scipy.misc.imsave(path,np.rot90(image))

    else:
        return False
    return True


def writearray(data, path, headers=None, suffix=''):
    if headers is None: headers = dict()
    ext = '.csv'
    path = ''.join(os.path.splitext(path)[:-1]) + suffix + ext

    if notexitsoroverwrite(path):
        np.savetxt(path, np.array(data), header=headers)
        return True
    else:
        return False


def notexitsoroverwrite(path):
    if os.path.isfile(path):
        if not dialogs.checkoverwrite(path): return False
    return True