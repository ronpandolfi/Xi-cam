import os

from PySide import QtCore

version='0.0.1'

qdir = QtCore.QDir(__file__)
qdir.cdUp()
rootdir = os.path.split( qdir.absolutePath() )[0]#+'/slacx'

