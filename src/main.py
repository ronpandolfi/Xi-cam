import sys
import os

import qdarkstyle
from scipy import misc
from pyqtgraph.parametertree import ParameterTree

from PySide.QtUiTools import QUiLoader
from PySide.QtCore import QFile
from PySide.QtGui import QApplication
from PySide.QtGui import QTabWidget
from PySide.QtGui import QFileDialog
from PySide.QtGui import QAction
from PySide.QtGui import QVBoxLayout
from config import experiment
from graphics import imageTab


class MyMainWindow():
    def __init__(self):
        app = QApplication(sys.argv)
        app.setStyleSheet(qdarkstyle.load_stylesheet())
        # QMainWindow.__init__(self, parent)
        loader = QUiLoader()
        file = QFile("../gui/mainwindow.ui")
        file.open(QFile.ReadOnly)
        self.ui = loader.load(file)
        file.close()

        actionOpen = self.ui.findChild(QAction, 'actionOpen')
        actionOpen.triggered.connect(self.dialogopen)

        actionOpen = self.ui.findChild(QAction, 'actionCenterFind')
        actionOpen.triggered.connect(self.centerfind)

        tabWidget = self.ui.findChild(QTabWidget, 'tabWidget')
        tabWidget.tabCloseRequested.connect(self.tabCloseRequested)

        self.experiment = experiment()

        experimentTree = ParameterTree()
        experimentTree.setParameters(self.experiment, showTop=False)
        self.experiment.sigTreeStateChanged.connect(self.experiment.save)

        settingsList = self.ui.findChild(QVBoxLayout, 'propertiesBox')
        settingsList.addWidget(experimentTree)

        ##
        self.openimage('../samples/AgB_saxs_00010.edf_mod.tif')
        ##

        self.ui.show()
        sys.exit(app.exec_())


    def tabCloseRequested(self, index):
        self.ui.findChild(QTabWidget, 'tabWidget').widget(index).deleteLater()


    def dialogopen(self):
        filename, _ = QFileDialog.getOpenFileName(self.ui, 'Open file', os.curdir, "*.tif")
        print(filename)
        self.openimage(filename)

    def openimage(self, path):
        imgdata = misc.imread(path)
        newimagetab = imageTab(imgdata, self.experiment)

        tabWidget = self.ui.findChild(QTabWidget, 'tabWidget')
        tabWidget.addTab(newimagetab, path)

    def centerfind(self):
        tabwidget = self.ui.findChild(QTabWidget, 'tabWidget')
        tabwidget.widget(tabwidget.currentIndex()).findcenter()


if __name__ == '__main__':
    window = MyMainWindow()