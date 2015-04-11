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
from PySide.QtGui import QInputDialog
from PySide.QtGui import QFileSystemModel
from PySide.QtGui import QTreeView
from PySide.QtCore import QDir
from PySide.QtCore import QSize
from PySide.QtGui import QListView
from PySide.QtGui import QTableView
from PySide.QtGui import QStandardItem
from PySide.QtGui import QWidget
from PySide.QtGui import QFont
from PySide.QtGui import QPushButton
from PySide.QtGui import QStackedWidget
# from PySide.QtCore import QStringList
from config import experiment
from graphics import imageTabTracker
from graphics import smallimageview
from pyqtgraph.parametertree import \
    ParameterTree  # IF THIS IS LOADED BEFORE PYSIDE, BAD THINGS HAPPEN; pycharm insists I'm wrong...
import pyqtgraph as pg
import models
from thumbwidget import thumbwidgetcollection
from PySide.QtGui import QCheckBox
import numpy as np


# sys.path.append("../gui/")

class MyMainWindow():
    def __init__(self):
        # Initialize PySide app with dark stylesheet
        self.app = QApplication(sys.argv)
        self.app.setStyle('Plastique')
        with open('../gui/style.stylesheet', 'r') as f:
            self.app.setStyleSheet(f.read())
        # print(qdarkstyle.load_stylesheet())
        #self.app.setStyle('plastique')
        font = self.app.font()
        font.setStyleStrategy(QFont.PreferAntialias)
        self.app.setFont(font)

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
        self.ui.findChild(QAction, 'actionLog_Intensity').triggered.connect(self.redrawcurrent)
        self.ui.findChild(QAction, 'actionRemove_Cosmics').triggered.connect(self.removecosmics)
        self.ui.findChild(QAction, 'actionMultiPlot').triggered.connect(self.multiplottoggle)
        self.ui.findChild(QAction, 'actionMaskLoad').triggered.connect(self.maskload)
        self.ui.findChild(QAction, 'actionSaveExperiment').triggered.connect(self.experiment.save)
        self.ui.findChild(QAction, 'actionLoadExperiment').triggered.connect(self.loadexperiment)
        self.ui.findChild(QAction, 'actionRadial_Symmetry').triggered.connect(self.redrawcurrent)
        self.ui.findChild(QAction, 'actionMirror_Symmetry').triggered.connect(self.redrawcurrent)
        self.ui.findChild(QAction, 'actionShow_Mask').triggered.connect(self.redrawcurrent)
        # self.ui.findChild(QAction, 'actionVertical_Cut').triggered.connect(self.verticalcut)
        #self.ui.findChild(QAction, 'actionHorizontal_Cut').triggered.connect(self.horizontalcut)
        self.ui.findChild(QAction, 'actionLine_Cut').triggered.connect(self.linecut)
        tabWidget = self.ui.findChild(QTabWidget, 'tabWidget')
        tabWidget.tabCloseRequested.connect(self.tabCloseRequested)
        tabWidget.currentChanged.connect(self.currentchanged)
        self.previoustabindex = -1

        self.treemodel = QFileSystemModel()

        tree = self.ui.findChild(QTreeView, 'treebrowser')
        tree.setModel(self.treemodel)
        parent = QDir()
        parent.cdUp()
        self.treemodel.setRootPath(parent.absolutePath())
        tree.setRootIndex(self.treemodel.index(parent.absolutePath()))
        header = tree.header()
        tree.setHeaderHidden(True)
        for i in range(1, 4):
            header.hideSection(i)
        filter = ["*.tif", "*.edf"]
        self.treemodel.setNameFilters(filter)
        tree.show()
        self.smallimageview = smallimageview(self.treemodel)
        smallimagebox = self.ui.findChild(QVBoxLayout, 'smallimageview')
        smallimagebox.addWidget(self.smallimageview)
        tree.clicked.connect(self.smallimageview.loaditem)
        tree.doubleClicked.connect(self.itemopen)

        self.thumbwidgets = thumbwidgetcollection()
        self.ui.findChild(QWidget, 'thumbbox').setLayout(self.thumbwidgets)



        listview = self.ui.findChild(QListView, 'openfileslist')
        self.listmodel = models.openfilesmodel(tabWidget)
        listview.setModel(self.listmodel)
        listview.doubleClicked.connect(self.switchtotab)


        # imagemodel = QFileSystemModel()
        #self.imgbrowser=self.ui.findChild(QTableView,'imgbrowser')
        #self.imgbrowser.setModel(imagemodel)
        #imagemodel.setRootPath(parent.absolutePath())
        #self.imgbrowser.setRootIndex(imagemodel.index(parent.absolutePath()))

        self.ui.findChild(QCheckBox, 'filebrowsercheck').stateChanged.connect(self.filebrowserpanetoggle)
        self.ui.findChild(QCheckBox, 'openfilescheck').stateChanged.connect(self.openfilestoggle)

        self.ui.findChild(QPushButton, 'librarybutton').clicked.connect(self.showlibrary)
        self.ui.findChild(QPushButton, 'viewerbutton').clicked.connect(self.showviewer)


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
        # self.ui.findChild(QToolBar, 'toolBar').addWidget(toolbuttonMasking)
        self.difftoolbar = QToolBar()
        self.difftoolbar.addWidget(toolbuttonMasking)
        self.difftoolbar.addAction(self.ui.findChild(QAction, 'actionLog_Intensity'))
        self.difftoolbar.addAction(self.ui.findChild(QAction, 'actionCenterFind'))
        self.difftoolbar.addAction(self.ui.findChild(QAction, 'actionCake'))
        self.difftoolbar.addAction(self.ui.findChild(QAction, 'actionRadial_Symmetry'))
        self.difftoolbar.addAction(self.ui.findChild(QAction, 'actionMirror_Symmetry'))
        self.difftoolbar.addAction(self.ui.findChild(QAction, 'actionShow_Mask'))
        # self.difftoolbar.addAction(self.ui.findChild(QAction, 'actionVertical_Cut'))
        #self.difftoolbar.addAction(self.ui.findChild(QAction, 'actionHorizontal_Cut'))
        self.difftoolbar.addAction(self.ui.findChild(QAction, 'actionLine_Cut'))
        self.difftoolbar.setIconSize(QSize(32, 32))
        self.ui.findChild(QVBoxLayout, 'diffbox').addWidget(self.difftoolbar)

        self.booltoolbar = QToolBar()
        self.booltoolbar.addAction(self.ui.findChild(QAction, 'actionAdd'))
        self.booltoolbar.addAction(self.ui.findChild(QAction, 'actionSubtract'))
        self.booltoolbar.addAction(self.ui.findChild(QAction, 'actionAdd_with_coefficient'))
        self.booltoolbar.addAction(self.ui.findChild(QAction, 'actionSubtract_with_coefficient'))
        self.booltoolbar.addAction(self.ui.findChild(QAction, 'actionDivide'))
        self.booltoolbar.addAction(self.ui.findChild(QAction, 'actionAverage'))
        self.ui.findChild(QAction, 'actionAdd').triggered.connect(self.addmode)
        self.ui.findChild(QAction, 'actionSubtract').triggered.connect(self.subtractmode)
        self.ui.findChild(QAction, 'actionAdd_with_coefficient').triggered.connect(self.addwithcoefmode)
        self.ui.findChild(QAction, 'actionSubtract_with_coefficient').triggered.connect(self.subtractwithcoefmode)
        self.ui.findChild(QAction, 'actionDivide').triggered.connect(self.dividemode)
        self.ui.findChild(QAction, 'actionAverage').triggered.connect(self.averagemode)
        self.booltoolbar.setIconSize(QSize(32, 32))
        self.ui.findChild(QVBoxLayout, 'leftpanelayout').addWidget(self.booltoolbar)


        self.statusbar = self.ui.statusbar

        self.statusbar.showMessage('Ready...')
        self.app.processEvents()
        ##
        self.openimage('../samples/AgB_1s_2m.edf')
        self.calibrate()
        ##

        # Show UI and end app when it closes
        # for layout in self.ui.findChildren(QLayout):
        #    try:
        #        layout.setSpacing(0)
        #        layout.setContentsMargin(0, 0, 0, 0)
        #    except:
        #        pass



        self.ui.show()
        sys.exit(self.app.exec_())

    def linecut(self):
        self.currentImageTab().tab.linecut()

    def switchtotab(self, index):
        self.ui.findChild(QTabWidget, 'tabWidget').setCurrentIndex(index.row())

    def addmode(self):
        operation = lambda m: np.sum(m, (0))
        self.launchmultimode(operation, 'Addition')

    def subtractmode(self):
        operation = lambda m: m[0] - np.sum(m[1:], (0))
        self.launchmultimode(operation, 'Subtraction')

    def addwithcoefmode(self):
        coef, ok = QInputDialog.getDouble(self.ui, u'Enter scaling coefficient x (A+xB):', u'Enter coefficient')

        if coef and ok:
            operation = lambda m: m[0] + coef * np.sum(m[1:], (0))
            self.launchmultimode(operation, 'Addition with coef (x=' + coef + ')')

    def subtractwithcoefmode(self):
        coef, ok = QInputDialog.getDouble(self.ui, u'Enter scaling coefficient x (A-xB):', u'Enter coefficient')

        if coef and ok:
            operation = lambda m: m[0] - coef * np.sum(m[1:], (0))
            self.launchmultimode(operation, 'Subtraction with coef (x=' + coef)

    def dividemode(self):
        operation = lambda m: m[0] / m[1]
        self.launchmultimode(operation, 'Division')

    def averagemode(self):
        operation = lambda m: np.mean(m, 0)
        self.launchmultimode(operation, 'Average')


    def launchmultimode(self, operation, operationname):
        indices = self.ui.findChild(QTreeView, 'treebrowser').selectedIndexes()
        paths = [self.treemodel.filePath(index) for index in indices]
        newimagetab = imageTabTracker(paths, self.experiment, self, operation=operation)
        filenames = [path.split('/')[-1] for path in paths]
        self.ui.findChild(QTabWidget, 'tabWidget').addTab(newimagetab, operationname + ': ' + ', '.join(filenames))

    def currentchanged(self, index):
        print('Changing from', self.previoustabindex, 'to', index)
        if index > -1:
            tabwidget = self.ui.findChild(QTabWidget, 'tabWidget')

            try:

                tabwidget.widget(self.previoustabindex).unload()
            except AttributeError:
                print('AttributeError intercepted in currentchanged()')
            tabwidget.widget(index).load()
        self.previoustabindex = index


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
        self.listmodel.widgetchanged()

    def polymask(self):
        # Add a polygon mask ROI to the tab
        self.currentImageTab().tab.polymask()

    def dialogopen(self):
        # Open a file dialog then open that image
        filename, _ = QFileDialog.getOpenFileName(self.ui, 'Open file', os.curdir, "*.tif *.edf")
        self.openfile(filename)

    def itemopen(self, index):
        path = self.treemodel.filePath(index)
        self.openfile(path)

    def openfile(self, filename):
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
        self.currentImageTab().load()
        self.currentImageTab().tab.calibrate()

    def openimage(self, path):
        self.statusbar.showMessage('Loading image...')
        self.app.processEvents()
        # Load the image path with Fabio
        # imgdata = self.load_image(path)

        # Make an image tab for that file and add it to the tab view
        newimagetab = imageTabTracker(path, self.experiment, self)
        tabwidget = self.ui.findChild(QTabWidget, 'tabWidget')
        tabwidget.setCurrentIndex(tabwidget.addTab(newimagetab, path.split('/')[-1]))

        self.statusbar.showMessage('Ready...')

    def centerfind(self):
        self.statusbar.showMessage('Finding center...')
        self.app.processEvents()
        # find the center of the current tab
        self.currentImageTab().findcenter()
        self.statusbar.showMessage('Ready...')

    def redrawcurrent(self):
        self.currentImageTab().tab.redrawimage()


    def removecosmics(self):
        self.statusbar.showMessage('Removing cosmic rays...')
        self.app.processEvents()
        self.currentImageTab().tab.removecosmics()
        self.statusbar.showMessage('Ready...')

    def multiplottoggle(self):
        self.currentImageTab().replot()

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

    def filebrowserpanetoggle(self):
        pane = self.ui.findChild(QTreeView, 'treebrowser')
        pane.setHidden(not pane.isHidden())

    def openfilestoggle(self):
        pane = self.ui.findChild(QListView, 'openfileslist')
        pane.setHidden(not pane.isHidden())

    def showlibrary(self):
        self.ui.findChild(QStackedWidget, 'viewmode').setCurrentIndex(1)

    def showviewer(self):
        self.ui.findChild(QStackedWidget, 'viewmode').setCurrentIndex(0)


if __name__ == '__main__':
    window = MyMainWindow()