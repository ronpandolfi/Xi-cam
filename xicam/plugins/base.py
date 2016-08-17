from PySide import QtCore, QtGui
import widgets
from xicam.widgets import explorer, login

activeplugin = None

# Base DEFAULTS
w = QtGui.QSplitter()
w.setOrientation(QtCore.Qt.Vertical)
w.setContentsMargins(0, 0, 0, 0)
l = QtGui.QVBoxLayout()
l.setContentsMargins(0, 10, 0, 0)
l.setSpacing(0)

loginwidget = login.LoginDialog()

preview = widgets.previewwidget()
w.addWidget(preview)

fileexplorer = explorer.MultipleFileExplorer(w)
filetree = fileexplorer.explorers['Local'].file_view

fileexplorer.sigLoginRequest.connect(loginwidget.loginRequest)
fileexplorer.sigLoginSuccess.connect(loginwidget.loginResult)
fileexplorer.sigPreview.connect(preview.loaditem)

l.addWidget(loginwidget)
l.addWidget(fileexplorer)

booltoolbar = QtGui.QToolBar()

booltoolbar.actionTimeline = QtGui.QAction(QtGui.QIcon('gui/icons_26.png'), 'Timeline', w)
booltoolbar.actionAdd = QtGui.QAction(QtGui.QIcon('gui/icons_11.png'), 'actionAdd', w)
booltoolbar.actionSubtract = QtGui.QAction(QtGui.QIcon('gui/icons_13.png'), 'actionSubtract', w)
booltoolbar.actionAdd_with_coefficient = QtGui.QAction(QtGui.QIcon('gui/icons_14.png'), 'actionAdd_with_coefficient', w)
booltoolbar.actionSubtract_with_coefficient = QtGui.QAction(QtGui.QIcon('gui/icons_15.png'),
                                                            'actionSubtract_with_coefficient', w)
booltoolbar.actionDivide = QtGui.QAction(QtGui.QIcon('gui/icons_12.png'), 'actionDivide', w)
booltoolbar.actionAverage = QtGui.QAction(QtGui.QIcon('gui/icons_16.png'), 'actionAverage', w)

booltoolbar.addAction(booltoolbar.actionTimeline)
booltoolbar.addAction(booltoolbar.actionAdd)
booltoolbar.addAction(booltoolbar.actionSubtract)
booltoolbar.addAction(booltoolbar.actionAdd_with_coefficient)
booltoolbar.addAction(booltoolbar.actionSubtract_with_coefficient)
booltoolbar.addAction(booltoolbar.actionDivide)
booltoolbar.addAction(booltoolbar.actionAverage)
booltoolbar.setIconSize(QtCore.QSize(32, 32))
l.addWidget(booltoolbar)

panelwidget = QtGui.QWidget()
panelwidget.setLayout(l)
w.addWidget(panelwidget)
w.setSizes([250, w.height() - 250])


class IconTabBar(QtGui.QTabBar):
    def tabSizeHint(self, index):
        return QtCore.QSize(32+12, 32+12)

class IconTabWidget(QtGui.QTabWidget):
    def __init__(self):
        super(IconTabWidget, self).__init__()
        self.setTabBar(IconTabBar())
        self.setIconSize(QtCore.QSize(32, 32))


leftwidget = IconTabWidget()
leftwidget.addTab(w, QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.Folder), '')

rightwidget = IconTabWidget()

class plugin(QtCore.QObject):
    name = 'Unnamed Plugin'

    hidden = False

    def __init__(self, placeholders):
        super(plugin, self).__init__()

        self.placeholders = placeholders

        if not hasattr(self, 'centerwidget'):
            self.centerwidget = None

        if not hasattr(self, 'rightwidget'):
            self.rightwidget = rightwidget

        if not hasattr(self, 'bottomwidget'):
            self.bottomwidget = None

        if not hasattr(self, 'leftwidget'):
            self.leftwidget = leftwidget
            self.booltoolbar = booltoolbar
            self.filetree = filetree

        if not hasattr(self, 'toolbar'):
            self.toolbar = None

        for widget, placeholder in zip(
                [self.centerwidget, self.rightwidget, self.bottomwidget, self.toolbar, self.leftwidget],
                self.placeholders):
            if widget is not None and placeholder is not None:
                placeholder.addWidget(widget)

    def openSelected(self, operation=None, operationname=None):
        indices = self.filetree.selectedIndexes()
        paths = [self.filetree.filetreemodel.filePath(index) for index in indices]
        self.openfiles(paths, operation, operationname)

    def openfiles(self, files, operation=None, operationname=None):
        pass

    @property
    def isActive(self):
        return activeplugin == self

    def opendirectory(files, operation=None):
        pass

    def addfiles(files, operation=None):
        pass

    def calibrate(self):
        self.centerwidget.currentWidget().widget.calibrate()

    def activate(self):

        for widget, placeholder in zip(
                [self.centerwidget, self.rightwidget, self.bottomwidget, self.toolbar, self.leftwidget],
                self.placeholders):
            if widget is not None and placeholder is not None:
                placeholder.setCurrentWidget(widget)
                placeholder.show()
            if widget is None and placeholder is not None:
                placeholder.hide()

        global leftwidget, rightwidget  # if these will become attributes then the check will need to be different
        if self.leftwidget is leftwidget:
            if self.leftwidget.count() > 1:
                for idx in reversed(range(self.leftwidget.count() - 1)):
                    self.leftwidget.removeTab(idx + 1)
            if hasattr(self, 'leftmodes'):
                for widget, icon in self.leftmodes:
                    self.leftwidget.addTab(widget, icon, '')
                self.leftwidget.tabBar().show()
            else:
                self.leftwidget.tabBar().hide()

        if self.rightwidget is rightwidget:
            for idx in reversed(range(self.rightwidget.count())):
                self.rightwidget.removeTab(idx)
            if hasattr(self, 'rightmodes'):
                for widget, icon in self.rightmodes:
                    self.rightwidget.addTab(widget, icon, '')
                if self.rightwidget.count() > 1:
                    self.rightwidget.tabBar().show()
                else:
                    self.rightwidget.tabBar().hide()
            else:
                self.rightwidget.hide()

        global activeplugin
        activeplugin = self

    def currentImage(self):
        pass
