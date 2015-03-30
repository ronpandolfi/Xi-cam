### TODO: Add calibrant selection
# TODO: Add calibration button
# TODO: Make experiment save/load
### TODO: Add peak marking
### TODO: Add q trace
# TODO: Confirm q calibration
### TODO: Add caking
# TODO: Synchronize tabs
## TODO: Add mask clear
### TODO: Clean tab names
### TODO: Add arc ROI
## TODO: Use detector mask in centerfinder






import sys
import os

import qdarkstyle
import fabio


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
from PySide.QtGui import QMessageBox
from PySide.QtGui import QSplitter
from PySide.QtGui import QLayout
from PySide.QtGui import QFileSystemModel
from PySide.QtGui import QTreeView
from PySide.QtCore import QDir
from PySide.QtGui import QListView
from PySide.QtGui import QStandardItemModel
from PySide.QtGui import QStandardItem
# from PySide.QtCore import QStringList
from config import experiment
from graphics import imageTab
from graphics import smallview
from pyqtgraph.parametertree import \
    ParameterTree  # IF THIS IS LOADED BEFORE PYSIDE, BAD THINGS HAPPEN; pycharm insists I'm wrong...
import pyqtgraph as pg
import models


# sys.path.append("../gui/")

class MyMainWindow():
    def __init__(self):
        # Initialize PySide app with dark stylesheet
        self.app = QApplication(sys.argv)
        # self.app.setStyle('Plastique')
        self.app.setStyleSheet(qdarkstyle.load_stylesheet() + """
        QSplitter::handle {
            background-color:gray;
        }
        QListView {
            margin:0px;
            spacing:0px;
            padding:0px;
        }

        """)
        #self.app.setStyle('plastique')

        # Load the gui from file
        loader = QUiLoader()
        file = QFile("../gui/mainwindow.ui")
        file.open(QFile.ReadOnly)
        self.ui = loader.load(file)
        file.close()

        # Initialize an empty experiment
        self.experiment = experiment()

        # Connect the experiment tree to a pg tree view and wire up
        self.experimentTree = ParameterTree()
        self.bindexperiment()
        settingsList = self.ui.findChild(QVBoxLayout, 'propertiesBox')
        settingsList.addWidget(self.experimentTree)


        # Wire up action buttons
        self.ui.findChild(QAction, 'actionOpen').triggered.connect(self.dialogopen)
        self.ui.findChild(QAction, 'actionCenterFind').triggered.connect(self.centerfind)
        self.ui.findChild(QAction, 'actionPolyMask').triggered.connect(self.polymask)
        self.ui.findChild(QAction, 'actionLog_Intensity').triggered.connect(self.logintensity)
        self.ui.findChild(QAction, 'actionRemove_Cosmics').triggered.connect(self.removecosmics)
        self.ui.findChild(QAction, 'actionMultiPlot').triggered.connect(self.multiplottoggle)
        self.ui.findChild(QAction, 'actionMaskLoad').triggered.connect(self.maskload)
        self.ui.findChild(QAction, 'actionSaveExperiment').triggered.connect(self.experiment.save)
        self.ui.findChild(QAction, 'actionLoadExperiment').triggered.connect(self.loadexperiment)
        tabWidget = self.ui.findChild(QTabWidget, 'tabWidget')
        tabWidget.tabCloseRequested.connect(self.tabCloseRequested)

        model = QFileSystemModel()

        tree = self.ui.findChild(QTreeView, 'treebrowser')
        tree.setModel(model)
        parent = QDir()
        parent.cdUp()
        model.setRootPath(parent.absolutePath())
        tree.setRootIndex(model.index(parent.absolutePath()))
        header = tree.header()
        tree.setHeaderHidden(True)
        for i in range(1, 4):
            header.hideSection(i)
        filter = ["*.tif", "*.edf"]
        model.setNameFilters(filter)
        tree.show()

        list = ['test', 'test2', 'test3']
        listview = self.ui.findChild(QListView, 'openfileslist')
        m = models.openfilesmodel(list)
        listview.setModel(m)

        self.smallimageview = smallview(model=m)
        smallimagebox = self.ui.getChild(QVBoxLayout, 'smallimageview')
        smallimagebox.addWidget(self.smallimageview)
        listview.clicked.connect(self.smallimageview.loaditem)



        # Add a plot widget to the splitter for integration
        integrationwidget = pg.PlotWidget()
        self.integration = integrationwidget.getPlotItem()
        self.integration.setLabel('bottom', u'q (\u212B\u207B\u00B9)', '')
        self.ui.findChild(QVBoxLayout, 'plotholder').addWidget(integrationwidget)

        splitter = self.ui.findChild(QSplitter, 'splitter')
        splitter.moveSplitter(0,0)


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
        self.openimage('../samples/AgB_00001.edf')
        self.calibrate()
        ##

        # Show UI and end app when it closes
        for layout in self.ui.findChildren(QLayout):
            try:
                layout.setSpacing(0)
                layout.setContentsMargin(0, 0, 0, 0)
            except:
                pass



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
        if filename is not u'':
            if self.experiment.iscalibrated:
                self.openimage(filename)
            else:
                msgBox = QMessageBox()
                msgBox.setText("The current experiment has not yet been calibrated. ")
                msgBox.setInformativeText("Use this image as a calibrant (AgBe)?")
                msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                msgBox.setDefaultButton(QMessageBox.Yes)

                response = msgBox.exec_()

                if response == QMessageBox.Yes:
                    self.openimage(filename)
                    self.calibrate()
                elif response == QMessageBox.No:
                    self.openimage(filename)
                elif response == QMessageBox.Cancel:
                    return None

    def calibrate(self):
        self.currentImageTab().calibrate()

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

    def loadexperiment(self):
        path, _ = QFileDialog.getOpenFileName(self.ui, 'Open file', os.curdir, "*.exp")
        self.experiment = experiment(path)

    def bindexperiment(self):
        self.experimentTree.setParameters(self.experiment, showTop=False)
        self.experiment.sigTreeStateChanged.connect(self.experiment.save)


if __name__ == '__main__':
    window = MyMainWindow()