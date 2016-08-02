from PySide import QtGui, QtCore
from fabio import edfimage, tifimage
import numpy as np
import os
from pipeline import writer

def savedatadialog(data, mask=None, guesspath="", caption="Save data to EDF", headers=None):
    if headers is None:
        headers = dict()

    dialog = QtGui.QFileDialog(parent=None, caption=caption, directory=os.path.dirname(guesspath),
                               filter=u"EDF (*.edf);;PNG (*.png);;TIFF (*.tif)")
    dialog.selectFile(os.path.basename(guesspath))
    filename, ok = dialog.getSaveFileName()

    if filename and ok:
        writer.writefile(data, filename, headers)
        if mask is not None:
            maskname = ''.join(os.path.splitext(filename)[:-1]) + "_mask" + os.path.splitext(filename)[-1]
            writer.writefile(mask, maskname, headers)

    return filename, ok

def infodialog(msg,shortmsg):
    msgBox = QtGui.QMessageBox()
    msgBox.setText(msg)
    msgBox.setInformativeText(shortmsg)
    msgBox.setStandardButtons(QtGui.QMessageBox.Close)
    msgBox.setDefaultButton(QtGui.QMessageBox.Close)

    response = msgBox.exec_()

def checkoverwrite(path):
    return True