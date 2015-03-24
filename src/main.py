# TODO: Add calibrant selection
# TODO: Add calibration button
# TODO: Make experiment save/load
# TODO: Add peak marking
# TODO: Add q trace
# TODO: Confirm q calibration
# TODO: Add caking
# TODO: Synchronize tabs
# TODO: Add mask clear
# TODO: Clean tab names
# TODO: Add arc ROI






import sys
import os

import qdarkstyle
import fabio
from pyqtgraph.parametertree import ParameterTree

from PySide.QtUiTools import QUiLoader
from PySide.QtCore import QFile
from PySide.QtGui import QApplication
from PySide.QtGui import QTabWidget
from PySide.QtGui import QFileDialog
from PySide.QtGui import QAction
from PySide.QtGui import QVBoxLayout
from PySide.QtGui import QMenu
from PySide.QtGui import QToolButton
from PySide.QtGui import QToolBar
from config import experiment
from graphics import imageTab


sys.path.append("../gui/")

class MyMainWindow():
    def __init__(self):
        # Initialize PySide app with dark stylesheet
        self.app = QApplication(sys.argv)
        self.app.setStyleSheet(qdarkstyle.load_stylesheet())

        # Load the gui from file
        loader = QUiLoader()
        file = QFile("../gui/mainwindow.ui")
        file.open(QFile.ReadOnly)
        self.ui = loader.load(file)
        file.close()


        # Wire up action buttons
        self.ui.findChild(QAction, 'actionOpen').triggered.connect(self.dialogopen)
        self.ui.findChild(QAction, 'actionCenterFind').triggered.connect(self.centerfind)
        self.ui.findChild(QAction, 'actionPolyMask').triggered.connect(self.polymask)
        self.ui.findChild(QAction, 'actionLog_Intensity').triggered.connect(self.logintensity)
        self.ui.findChild(QAction, 'actionRemove_Cosmics').triggered.connect(self.removecosmics)
        self.ui.findChild(QAction, 'actionMultiPlot').triggered.connect(self.multiplottoggle)
        self.ui.findChild(QAction, 'actionMaskLoad').triggered.connect(self.maskload)
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

        menu = QMenu()
        actionMasking = self.ui.findChild(QAction, 'actionMasking')
        actionPolyMask = self.ui.findChild(QAction, 'actionPolyMask')
        menu.addAction(actionPolyMask)
        menu.addAction(self.ui.findChild(QAction, 'actionRemove_Cosmics'))
        menu.addAction(self.ui.findChild(QAction, 'actionMaskLoad'))
        toolbuttonMasking = QToolButton()
        toolbuttonMasking.setDefaultAction(actionMasking)
        toolbuttonMasking.setMenu(menu)
        toolbuttonMasking.setPopupMode(QToolButton.InstantPopup)
        self.ui.findChild(QToolBar, 'toolBar').addWidget(toolbuttonMasking)

        self.statusbar = self.ui.statusbar

        self.statusbar.showMessage('Ready...')
        self.app.processEvents()
        ##
        self.openimage('../samples/AgB_saxs_00010.edf_mod.tif')
        ##

        # Show UI and end app when it closes
        self.ui.show()
        sys.exit(self.app.exec_())

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
        self.statusbar.showMessage('Loading image...')
        self.app.processEvents()
        # Load the image path with Fabio
        imgdata = self.load_image(path)

        # Make an image tab for that file and add it to the tab view
        newimagetab = imageTab(imgdata, self.experiment, self)
        self.ui.findChild(QTabWidget, 'tabWidget').addTab(newimagetab, path)
        self.statusbar.showMessage('Ready...')

    def centerfind(self):
        self.statusbar.showMessage('Finding center...')
        self.app.processEvents()
        # find the center of the current tab
        self.currentImageTab().findcenter()
        self.statusbar.showMessage('Ready...')

    def logintensity(self):
        self.currentImageTab().logintensity(toggle=True)

    def removecosmics(self):
        self.statusbar.showMessage('Removing cosmic rays...')
        self.app.processEvents()
        self.currentImageTab().removecosmics()
        self.statusbar.showMessage('Ready...')

    def multiplottoggle(self):
        self.currentImageTab().radialintegrate()

    def maskload(self):
        path, _ = QFileDialog.getOpenFileName(self.ui, 'Open file', os.curdir, "*.tif *.edf")
        mask = self.load_image(path)
        self.experiment.addtomask(mask)


if __name__ == '__main__':
    window = MyMainWindow()