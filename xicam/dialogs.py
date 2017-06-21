from __future__ import unicode_literals
from PySide import QtGui, QtCore
from fabio import edfimage, tifimage
import numpy as np
import os

def savedatadialog(guesspath="", caption="Save data to EDF", headers=None):
    if headers is None:
        headers = dict()

    dialog = QtGui.QFileDialog(parent=None, caption=caption, directory=os.path.dirname(guesspath),
                               filter=u"EDF (*.edf);;PNG (*.png);;TIFF (*.tif)")
    dialog.selectFile(os.path.basename(guesspath))
    filename, ok = dialog.getSaveFileName()



    return filename, ok

def infodialog(msg,shortmsg):
    msgBox = QtGui.QMessageBox()
    msgBox.setText(msg)
    msgBox.setInformativeText(shortmsg)
    msgBox.setStandardButtons(QtGui.QMessageBox.Close)
    msgBox.setDefaultButton(QtGui.QMessageBox.Close)

    response = msgBox.exec_()

def checkoverwrite(path):
    msgBox = QtGui.QMessageBox()
    msgBox.setText('Are you sure you want to overwrite {}?'.format(path))
    # msgBox.setInformativeText(shortmsg)
    msgBox.setStandardButtons(QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel)
    msgBox.setDefaultButton(QtGui.QMessageBox.Cancel)

    response = msgBox.exec_()
    return response==QtGui.QMessageBox.Ok