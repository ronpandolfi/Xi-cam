import os

from PySide import QtGui,QtCore
from pipeline import hig,loader


class gui():
    def __init__(self, ui):
        self.ui = ui

        self.ui.rmcbutton.clicked.connect(self.showrmc)

        self.ui.rmcaddfiles.clicked.connect(self.addfiles)
        self.ui.rmcremovefiles.clicked.connect(self.removefiles)
        self.ui.rmcExecute.clicked.connect(self.execute)


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

    def execute(self):
        tiles = self.ui.rmcTiles.value()
        steps = self.ui.rmcSteps.value()
        scalefactor = self.ui.rmcScalefactor.value()
        modlestartsize = self.ui.rmcModlestartsize.value()
        lodingfactors_1 = self.ui.rmcLoadingfactors_1.value()
        lodingfactors_2 = self.ui.rmcLoadingfactors_2.value()
        lodingfactors_3 = self.ui.rmcLoadingfactors_3.value()
        rip=self.ui.rmcinputpaths
        inputpaths = [rip.item(index).text() for index in xrange(rip.count())]

        d = {'hipRMCInput': {'instrumentation': {'inputimage': inputpaths[0],
                                             'imagesize': loader.loadimage(inputpaths[0]).shape,
                                             'numtiles': tiles,
                                             'loadingfactors': [lodingfactors_1, lodingfactors_2, lodingfactors_3 ]},
                         'computation': {'runname': "test",
                                         'modelstartsize': [modlestartsize, modlestartsize],
                                         'numstepsfactor': steps,
                                         'scalefactor': scalefactor}}}
        h = hig.hig(**d)
        h.write("test.hig")
