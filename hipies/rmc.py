import os
from PySide import QtGui


class gui():
    def __init__(self, ui):
        self.ui = ui

        self.ui.rmcbutton.clicked.connect(self.showrmc)

        self.ui.rmcaddfiles.clicked.connect(self.addfiles)
        self.ui.rmcremovefiles.clicked.connect(self.removefiles)


    def showrmc(self):
        """
        switch to timeline view
        """
        self.ui.viewmode.setCurrentIndex(3)
        self.ui.sidemode.setCurrentIndex(1)

    def addfiles(self):
        paths, ok = QtGui.QFileDialog.getOpenFileNames(self.ui, 'Add files to RMC', os.curdir,
                                                       "*.tif *.edf *.fits *.tif")
        self.ui.rmcinputpaths.addItems(paths)

    def removefiles(self):
        a = self.ui.rmcinputpaths
        '''
        :type a : PySide.QtGui.QListWidget
        '''
        a.QListWidget.selectedItems()
        for index in self.ui.rmcinputpaths.selectedIndexes():
            item = self.ui.rmcinputpaths.takeItem(index)
            item = None