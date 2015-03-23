import sys
import os

import qdarkstyle
from pyqtgraph.parametertree import ParameterTree
import fabio

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
        # Initialize PySide app with dark stylesheet
        app = QApplication(sys.argv)
        app.setStyleSheet(qdarkstyle.load_stylesheet())

        # Load the gui from file
        loader = QUiLoader()
        file = QFile("../gui/mainwindow.ui")
        file.open(QFile.ReadOnly)
        self.ui = loader.load(file)
        file.close()


        # Wire up action buttons
        actionOpen = self.ui.findChild(QAction, 'actionOpen')
        actionOpen.triggered.connect(self.dialogopen)
        actionOpen = self.ui.findChild(QAction, 'actionCenterFind')
        actionOpen.triggered.connect(self.centerfind)
        actionOpen = self.ui.findChild(QAction, 'actionPolyMask')
        actionOpen.triggered.connect(self.polymask)
        tabWidget = self.ui.findChild(QTabWidget, 'tabWidget')
        tabWidget.tabCloseRequested.connect(self.tabCloseRequested)

        # Initialize an empty experiment
        self.experiment = experiment()

        # Connect the experiment tree to a pg tree view and wire up
        experimentTree = ParameterTree()
        experimentTree.setParameters(self.experiment, showTop=False)
        self.experiment.sigTreeStateChanged.connect(self.experiment.save)
        settingsList = self.ui.findChild(QVBoxLayout, 'propertiesBox')
        settingsList.addWidget(experimentTree)


        ##
        self.openimage('../samples/AgB_saxs_00010.edf_mod.tif')
        ##

        # Show UI and end app when it closes
        self.ui.show()
        sys.exit(app.exec_())

    def load_image(self, path):
        # Load an image path with fabio
        return fabio.open(path).data

    def currentImageTab(self):
        # Get the currently shown image tab
        tabwidget = self.ui.findChild(QTabWidget, 'tabWidget')
        return tabwidget.widget(tabwidget.currentIndex())

    def viewmask(self):
        # Show the mask overlay
        self.currentImageTab().viewmask()

    def tabCloseRequested(self, index):
        # Delete a tab from the tab view upon request
        self.ui.findChild(QTabWidget, 'tabWidget').widget(index).deleteLater()

    def polymask(self):
        # Add a polygon mask ROI to the tab
        self.currentImageTab().polymask()

    def dialogopen(self):
        # Open a file dialog then open that image
        filename, _ = QFileDialog.getOpenFileName(self.ui, 'Open file', os.curdir, "*.tif *.edf")
        print(filename)
        self.openimage(filename)

    def openimage(self, path):
        # Load the image path with Fabio
        imgdata = self.load_image(path)

        # Make an image tab for that file and add it to the tab view
        newimagetab = imageTab(imgdata, self.experiment)
        tabWidget = self.ui.findChild(QTabWidget, 'tabWidget')
        tabWidget.addTab(newimagetab, path)

    def centerfind(self):
        # find the center of the current tab
        self.currentImageTab().findcenter()


if __name__ == '__main__':
    window = MyMainWindow()