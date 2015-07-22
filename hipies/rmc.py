import os
import RmcView
from PySide import QtGui,QtCore
from pipeline import hig,loader


class gui():
    def __init__(self, ui):
        self.ui = ui

        self.ui.rmcbutton.clicked.connect(self.showrmc)

        self.ui.rmcaddfiles.clicked.connect(self.addfiles)
        self.ui.rmcremovefiles.clicked.connect(self.removefiles)
        self.ui.rmcExecute.clicked.connect(self.execute)
        self.ui.rmcAddloadingfactor.clicked.connect(self.addloadingfactor)
        self.ui.rmcSubtractloadingfactor.clicked.connect(self.subtractloadingfactor)
        self.ui.rmcopen.clicked.connect(self.open)
        self.ui.rmcreset.clicked.connect(self.reset)
        self.ui.rmcreset_2.clicked.connect(self.reset2)


    def open(self):
        Outputdirectory= QtGui.QFileDialog.getExistingDirectory(self.ui,"Select an Output directory")
        self.ui.rmcoutput.setText(Outputdirectory)

    def reset(self):
        self.ui.rmcoutput.setText("")

    def reset2(self):
        self.ui.rmcoutput.setText("")
        self.ui.rmcLoadingfactors.clear()
        self.ui.rmcTiles.setValue(1)
        self.ui.rmcSteps.setValue(99)
        self.ui.rmcScalefactor.setValue(1)
        self.ui.rmcModlestartsize.setValue(1)


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
        for index in self.ui.rmcinputpaths.selectedIndexes():
            item = self.ui.rmcinputpaths.takeItem(index.row())
            item = None

    def addloadingfactor(self):
        loadingfactor,_=QtGui.QInputDialog.getDouble(self.ui, "Loading Factor", "Enter a loading factor:", value=0, minValue=-0, maxValue=1000, decimals=3 )
        newItem = QtGui.QListWidgetItem()
        newItem.setText(str(loadingfactor))
        self.ui.rmcLoadingfactors.addItem(newItem)

    def subtractloadingfactor(self):
        for index in self.ui.rmcLoadingfactors.selectedIndexes():
            item = self.ui.rmcLoadingfactors.takeItem(index.row())
            item = None

    def execute(self):
        tiles = self.ui.rmcTiles.value()
        steps = self.ui.rmcSteps.value()
        scalefactor = self.ui.rmcScalefactor.value()
        modlestartsize = self.ui.rmcModlestartsize.value()

        loadingfactors = []
        all_items = self.ui.rmcLoadingfactors.findItems('', QtCore.Qt.MatchRegExp)
        for item in all_items:
            loadingfactors.append(item.text())


        rip=self.ui.rmcinputpaths
        inputpaths = [rip.item(index).text() for index in xrange(rip.count())]

        d = {'hipRMCInput': {'instrumentation': {'inputimage': inputpaths[0],
                                                 'imagesize': loader.loadimage(inputpaths[0]).shape,
                                                 'numtiles': tiles,
                                                 'loadingfactors': loadingfactors,
                                                 'maskimage': ["data/mask.tif"]},  # optional
                             'computation': {'runname': self.ui.rmcoutput.text(),
                                         'modelstartsize': [modlestartsize, modlestartsize],
                                         'numstepsfactor': steps,
                                             'scalefactor': scalefactor}}},
        h = hig.hig(**d)
        h.write("test_input.hig")
        os.system("./hiprmc test_input.hig")


    def displayoutput(self):
        path = self.ui.rmcoutput.text()

        layout = self.ui.rmclayout
        layout.addWidget(RmcView.rmcView(path))
