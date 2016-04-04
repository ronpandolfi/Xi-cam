__author__ = "Ronald J Pandolfi"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"

from PySide import QtGui
from PySide import QtCore
from PySide.QtCore import Qt
import os


class newfilewatcher(QtCore.QFileSystemWatcher):
    newFilesDetected = QtCore.Signal(str, list)

    def __init__(self):
        super(newfilewatcher, self).__init__()
        self.childrendict = dict()
        self.directoryChanged.connect(self.checkdirectory)

    def addPath(self, path):
        super(newfilewatcher, self).addPath(path)
        self.childrendict[path] = set(os.listdir(path))

    def checkdirectory(self, path):
        # print(path)
        updatedchildren = set(os.listdir(path))
        newchildren = updatedchildren - self.childrendict[path]
        self.childrendict[path] = updatedchildren
        self.newFilesDetected.emit(path, list(newchildren))
