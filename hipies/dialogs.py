from PySide import QtGui, QtCore
from fabio import edfimage, tifimage
import numpy as np
import os


def savedatadialog(data, mask=None, guesspath="", caption="Save data to EDF", headers=None):
    if headers is None:
        headers = dict()

    dialog = QtGui.QFileDialog(parent=None, caption=caption, directory=os.path.dirname(guesspath),
                               filter=u"EDF (*.edf);;PNG (*.png);;TIFF (*.tif)")
    dialog.selectFile(os.path.basename(guesspath))
    filename, ok = dialog.getSaveFileName()

    if filename and ok:
        writefile(data, filename, headers)
        if mask is not None:
            maskname = ''.join(os.path.splitext(filename)[:-1]) + "_mask" + os.path.splitext(filename)[-1]
            writefile(mask, maskname, headers)

    return filename, ok


def writefile(image, path, headers):
    if os.path.splitext(path)[-1].lower() == '.edf':
        fabimg = edfimage.edfimage(np.rot90(image), header=headers)
        fabimg.write(path)
    elif os.path.splitext(path)[-1].lower() == '.tif':
        fabimg = tifimage.tifimage(np.rot90(image), header=headers)
        fabimg.write(path)
    elif os.path.splitext(path)[-1].lower() == '.png':
        raise NotImplementedError