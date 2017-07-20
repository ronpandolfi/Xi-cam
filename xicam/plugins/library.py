from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from past.utils import old_div

from PySide import QtGui
import sys
import re
from PySide import QtGui
from PySide import QtCore
from PySide.QtCore import Qt
import os
import pipeline
import numpy as np
import pyqtgraph as pg
import fabio

from . import base
from . import viewer
from xicam.widgets import featurewidgets as fw
from xicam.widgets import customwidgets as cw
from xicam.widgets import explorer
from xicam.widgets.stardelegate import stardelegate

class LibraryPlugin(base.plugin):
    name = 'Library'

    def __init__(self, *args, **kwargs):


        if sys.platform == 'win32':
            libraryview = LibraryWidget(self, 'C://')
        else:
            libraryview = LibraryWidget(self, pipeline.pathtools.getRoot())


        self.centerwidget = QtGui.QTabWidget()
        self.tabBar = explorer.TabBarPlus()
        self.centerwidget.setTabBar(self.tabBar)

        # self.centerwidget.addTab(libraryview, 'Local')
        self.addDatabase(libraryview, 'Local')
        self.tabBar.plusClicked.connect(self.createNewLibrary)


        libraryview.sigOpenFile.connect(self.addViewer)

        self.newtabmenu = QtGui.QMenu(None)
        adddatabroker = QtGui.QAction('Data Broker', self.newtabmenu)
        # addspot = QtGui.QAction('SPOT', self.newtabmenu)
        # addcori = QtGui.QAction('Cori', self.newtabmenu)
        # addedison = QtGui.QAction('Edison', self.newtabmenu)
        # addbragg = QtGui.QAction('Bragg', self.newtabmenu)
        # addsftp = QtGui.QAction('SFTP Connection', self.newtabmenu)
        # showjobtab = QtGui.QAction('Jobs', self.newtabmenu)

        self.newtabmenu.addActions([adddatabroker])
        # self.newtabmenu.addActions([adddatabroker, addspot, addcori, addedison, addbragg])
        adddatabroker.triggered.connect(self.addDataBrokerTab)
        # addspot.triggered.connect(self.addSPOTTab)
        # addedison.triggered.connect(lambda: self.addHPCTab('Edison'))
        # addcori.triggered.connect(lambda: self.addHPCTab('Cori'))
        # addbragg.triggered.connect(lambda: self.addHPCTab('Bragg'))
        # addsftp.triggered.connect(self.addSFTPTab)
        # showjobtab.triggered.connect(lambda: self.addTab(self.jobtab, 'Jobs'))

        self.rightwidget = None

        self.featureform = None

        self.bottomwidget = None

        self.toolbar = None

        super(LibraryPlugin, self).__init__(*args, **kwargs)

        # self.centerwidget.sigOpenFile.connect(viewer.plugininstance.openfiles)

    def addDatabase(self, library, name):
        window = QtGui.QWidget()
        stack = QtGui.QStackedWidget()
        stack.addWidget(library)

        # set up toolbar
        toolbar = QtGui.QWidget()
        toolbar.setMaximumHeight(60)
        grid_button = QtGui.QToolButton()
        grid_button.setToolTip('Folder view')
        grid_button.resize(QtCore.QSize(40, 40))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_65.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        grid_button.setIcon(icon)
        grid_button.setIconSize(QtCore.QSize(40, 40))
        grid_button.clicked.connect(self.showGridWindow)
        image_button = QtGui.QToolButton()
        image_button.setToolTip('Image view')
        image_button.resize(QtCore.QSize(40, 40))
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_66.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        image_button.setIcon(icon)
        image_button.setIconSize(QtCore.QSize(40, 40))
        image_button.clicked.connect(self.showImageWindow)

        h = QtGui.QHBoxLayout()
        grid = QtGui.QGridLayout()
        grid.addWidget(grid_button, 0, 0)
        grid.addWidget(image_button, 0, 1)
        grid.addItem(QtGui.QSpacerItem(1, 1), 0, 2, 1, 10)
        h.addLayout(grid)
        toolbar.setLayout(h)


        l = QtGui.QVBoxLayout()
        l.addWidget(stack)
        l.addWidget(toolbar)
        window.setLayout(l)
        self.centerwidget.addTab(window, name)

    def showGridWindow(self):
        self.currentStack().setCurrentWidget(self.currentStack().widget(0))

    def showImageWindow(self):
        if self.currentStack().count() > 1:
            self.currentStack().setCurrentWidget(self.currentStack().widget(1))

    def addViewer(self, widget):
        view = LibraryItem(widget)
        stack = self.currentStack()
        if stack.count() > 1:
            stack.removeWidget(stack.widget(1))
        stack.addWidget(view)
        stack.setCurrentWidget(view)

    def currentStack(self):
        return self.centerwidget.currentWidget().layout().itemAt(0).widget()


    # TODO: fill in next four functions to interface with remote databases
    def createNewLibrary(self):
        self.newtabmenu.popup(QtGui.QCursor.pos())

    def addDataBrokerTab(self):
        print('databroker')

    def addSPOTTab(self):
        print('spot')

    def addHPCTab(self, system):
        print(system)




class LibraryWidget(QtGui.QScrollArea):
    sigOpenFile = QtCore.Signal(QtGui.QWidget)

    def __init__(self, *args, **kwargs):
        super(LibraryWidget, self).__init__()
        self.setWidgetResizable(True)
        self.setFocusPolicy(Qt.NoFocus)
        l = librarylayout(*args, **kwargs)
        w = QtGui.QWidget()
        w.setLayout(l)
        w.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.setWidget(w)
        l.sigOpenFile.connect(self.sigOpenFile)


class FlowLayout(QtGui.QLayout):
    """
    A grid layout that wraps and adjusts to its items
    """

    def __init__(self, parent=None, margin=5, spacing=-1):
        super(FlowLayout, self).__init__(parent)

        self.margin = margin
        self.setSpacing(spacing)

        self.itemList = []


    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList[index]

        return None

    def takeAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList.pop(index)

        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QtCore.QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()

        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())

        size += QtCore.QSize(2 * self.margin, 2 * self.margin)
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing() + wid.style().layoutSpacing(QtGui.QSizePolicy.PushButton,
                                                                QtGui.QSizePolicy.PushButton,
                                                                Qt.Horizontal)
            spaceY = self.spacing() + wid.style().layoutSpacing(QtGui.QSizePolicy.PushButton,
                                                                QtGui.QSizePolicy.PushButton,
                                                                Qt.Vertical)
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()


class librarylayout(FlowLayout):
    """
    Extend the flow layout to fill it with thumbwidgetitems representing files/folders
    """

    sigOpenFile = QtCore.Signal(QtGui.QWidget)

    def __init__(self, parentwindow, path='/Volumes'):
        super(librarylayout, self).__init__()
        self.parent = None
        self.parentwindow = parentwindow
        self.populate(path)

    def chdir(self, path):
        self.clear()
        self.populate(os.path.normpath(path))

    def populate(self, path):
        self.parent = QtCore.QDir()
        self.parent.cd(path)

        dir = QtCore.QDir(self.parent)
        entries = dir.entryList()

        for entry in entries:
            # print fileinfo.fileName()
            if not (entry == '..' and os.path.normpath(path) == pipeline.pathtools.getRoot()) and not entry == '.':
                nameoverride = entry if entry == '..' else None
                w = thumbwidgetitem(os.path.join(path, entry), parentwindow=self.parentwindow,
                                    nameoverride=nameoverride)
                self.addWidget(w)
                w.sigOpenFile.connect(self.sigOpenFile)
                w.sigChangeRoot.connect(self.chdir)


    def clear(self):
        while self.count():
            item = self.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

# class for showing tags in their own 'boxes'. Might be more trouble than it's worth
class tagItem(QtGui.QFrame):

    def __init__(self, text):
        super(tagItem, self).__init__()
        self.text = text

class LibraryItem(QtGui.QSplitter):

    def __init__(self, thumbwidget, *args, **kwargs):
        super(LibraryItem, self).__init__(*args, **kwargs)

        # image item from thumbnail
        self.image = cw.ImageView()
        self.image.setImage(np.rot90(thumbwidget.thumb, 3))
        self.hist = self.image.getHistogramWidget()
        set_button = cw.histDialogButton('Set', parent=self)
        set_button.connectToHistWidget(self.hist)
        imageWidget = pg.GraphicsLayoutWidget()
        vb = imageWidget.addViewBox(row=1, col=1)
        vb.addItem(self.image.getImageItem())
        vb.setAspectLocked(True)

        self.attrs = attributeViewer()
        hist_widget = QtGui.QWidget()
        v = QtGui.QVBoxLayout()
        v.addWidget(self.hist)
        v.addWidget(set_button)
        hist_widget.setLayout(v)
        self.attrs.addAttribute('Histogram', hist_widget)

        self.keywords = keywordsWidget()
        self.attrs.addAttribute('Keywords', self.keywords)

        self.metadata = pg.TableWidget()
        self.metadata.verticalHeader().hide()
        self.metadata.horizontalHeader().setStretchLastSection(True)
        self.metadata.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.metadata.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.metadata.show()
        self.attrs.addAttribute('Metadata', self.metadata)
        self.attrs.layout.setContentsMargins(0, 0, 0, 0)


        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(imageWidget)
        layout.addWidget(self.attrs)
        self.setLayout(layout)

class attributeViewer(QtGui.QWidget):

    def __init__(self):
        super(attributeViewer, self).__init__()
        self.layout = QtGui.QVBoxLayout()
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(self.layout)
        self.setMinimumWidth(140)
        self.attributes = []

    def addAttribute(self, name, attribute):
        # attribute is type QtGui.QWidget

        widget = attributeWidget(name)
        widget.previewButton.hide()
        widget.closeButton.hide()
        widget.addSubFeature(attribute)
        widget.sigClicked.connect(self.attributeClicked)
        self.attributes.append(widget)
        self.layout.addWidget(widget)

    def attributeClicked(self, attribute):

        hidden = attribute.hidden

        if attribute in self.attributes:
            self.collapseAllAttributes()

        if attribute.subframe and not hidden:
            attribute.collapse()
        else:
            attribute.expand()

    def collapseAllAttributes(self):
        for attr in self.attributes:
            attr.collapse()
            if attr.subfeatures is not None:
                for subattr in attr.subfeatures:
                    try:
                        subattr.collapse()
                    except AttributeError:
                        pass

class keywordsWidget(QtGui.QWidget):

    sigRatingChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        super(keywordsWidget, self).__init__(parent=parent)

        # sets up keywords textedit box
        self.keywords = set([])
        self.textbox = QtGui.QTextEdit()
        label_1 = QtGui.QLabel('Keywords: ')
        self.textbox.textChanged.connect(self.updateKeywords)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(label_1)
        layout.addWidget(self.textbox)

        # sets up star rating box
        label_2 = QtGui.QLabel('Rating: ')
        t = QtGui.QTableWidget(1, 1)
        t.horizontalHeader().hide()
        t.verticalHeader().hide()
        t.setItemDelegate(stardelegate.StarDelegate())
        t.setEditTriggers(QtGui.QAbstractItemView.DoubleClicked |
                                    QtGui.QAbstractItemView.SelectedClicked)
        self.rating = QtGui.QTableWidgetItem()
        self.rating.setData(0, 0)
        t.itemChanged.connect(lambda x: self.sigRatingChanged.emit(x.data(0) + 1))
        t.setItem(0, 0, self.rating)
        t.horizontalHeader().setStretchLastSection(True)
        t.setMaximumHeight(30)
        t.setMaximumWidth(110)
        t.show()
        layout.addWidget(label_2)
        layout.addWidget(t)

        self.setLayout(layout)


    def updateKeywords(self):
        self.keywords = set([])
        for item in filter(None, re.split("[,:;]", self.textbox.toPlainText())):
            self.keywords.add(item.strip())



class attributeWidget(fw.FeatureWidget):

    def __init__(self, name='', checkable=True, closeable=True, subfeatures=None, parent=None):
        super(attributeWidget, self).__init__(name=name, checkable=checkable, closeable=closeable,
                                              subfeatures=subfeatures, parent=parent)
        self.hidden = False

    def addSubFeature(self, subfeature):
        """
        Adds a subfeature to the widget

        Parameters
        ----------
        subfeature : FeatureWidget/QWidget
            Widget to add as a subfeature
        """

        h = QtGui.QHBoxLayout()
        subfeature.destroyed.connect(h.deleteLater)
        if isinstance(subfeature, QtGui.QLayout):
            h.addLayout(subfeature)
        elif isinstance(subfeature, QtGui.QWidget):
            h.addWidget(subfeature)
        self.subframe_layout.addLayout(h)
        try:
            subfeature.sigDelete.connect(self.removeSubFeature)
        except AttributeError:
            pass

        self.sigSubFeature.emit(subfeature)
        self.subfeatures.append(subfeature)

    def collapse(self):
        """
        Collapses all expanded subfeatures
        """

        if self.subframe is not None:
            self.subframe.hide()
            self.hidden = True

    def expand(self):
        """
        Expands subfeatures
        """

        if self.subframe is not None:
            self.subframe.show()
            self.hidden = False




class thumbwidgetitem(QtGui.QFrame):
    """
    A single icon representing a file/folder that can be accessed/opened
    """
    sigChangeRoot = QtCore.Signal(str)
    sigOpenFile = QtCore.Signal(QtGui.QWidget)


    def __init__(self, path, parentwindow, nameoverride=None):
        path = os.path.normpath(path)

        self.foldericon = QtGui.QImage()
        self.foldericon.load('xicam/gui/GenericFolderIcon.png')

        self.fileicon = QtGui.QImage()
        self.fileicon.load('xicam/gui/post-360412-0-09676400-1365986245.png')

        print('Library widget generated for ' + path)
        super(thumbwidgetitem, self).__init__()
        self.parentwindow = parentwindow
        self.setObjectName('thumb')
        desiredsize = QtCore.QSize(160, 200)

        self.setFixedSize(desiredsize)
        self.setAutoFillBackground(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.layout = QtGui.QVBoxLayout(self)  # .frame

        self.path = path
        # print path
        # data = pipeline.loader.loadimage(self.path)
        # dimg = pipeline.loader.diffimage(filepath=self.path)

        self.image = QtGui.QImage()
        # print os.path.splitext(path)[1]
        if os.path.isdir(path):
            self.image = self.foldericon
        elif os.path.splitext(path)[1] in pipeline.loader.acceptableexts:


            try:
                img = fabio.open(path).data
                thumbnail = pipeline.writer.thumbnail(img)
                print(path)
                self.thumb = (np.log(thumbnail * (thumbnail > 0) + (thumbnail < 1))).copy()
                # self.thumb = np.rot90(np.log(dimg.thumbnail * (dimg.thumbnail > 0) + (dimg.thumbnail < 1))).copy()
                print('thumbmax:', np.max(self.thumb))

                # convert to 8 bit value scale
                a = 255./(np.max(self.thumb)-np.min(self.thumb))
                b = -a * np.min(self.thumb)
                self.thumb = a * self.thumb + b

                # self.thumb *= old_div(255., np.max(self.thumb))

                # if self.thumb is not None:
                self.image = QtGui.QImage(self.thumb.astype(np.uint8), self.thumb.shape[1], self.thumb.shape[0],
                                          self.thumb.shape[1],
                                          QtGui.QImage.Format_Indexed8)
            except Exception as ex:
                print(ex)

                self.image = self.fileicon


        else:
            self.image = self.fileicon

        image_label = ScaledLabel(self.image)
        self.layout.addWidget(image_label)

        self.namelabel = QtGui.QLabel(os.path.basename(path) if nameoverride is None else nameoverride)
        self.namelabel.setAlignment(Qt.AlignHCenter)
        # self.namelabel.setWordWrap(True)
        self.namelabel.setToolTip(self.namelabel.text())
        self.namelabel.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum)

        line = QtGui.QFrame()
        line.setGeometry(QtCore.QRect(0, 0, desiredsize.width() * 9 / 10, 3))
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        self.layout.addWidget(line)

        self.layout.addWidget(self.namelabel)
        self.setLayout(self.layout)

    def enterEvent(self, *args, **kwargs):
        # self.frame.setFrameStyle(QFrame.Raised)
        pass

    # def leaveEvent(self, *args, **kwargs):
    # self.frame.setFrameStyle(QFrame.Plain)

    # def mousePressEvent(self, *args, **kwargs):
    # self.frame.setFrameStyle(QFrame.Sunken)

    def mouseDoubleClickEvent(self, *args, **kwargs):
        # self.sigOpenFile.emit(self)
        #
        if os.path.isdir(self.path):
            self.sigChangeRoot.emit(self.path)
        else:
            # self.sigOpenFile.emit(self.path)
            self.sigOpenFile.emit(self)


class ScaledLabel(QtGui.QLabel):
    """
    A label that scales with the constrained dimension; for thumbnails
    """

    def __init__(self, image):
        super(ScaledLabel, self).__init__()
        self._pixmap = QtGui.QPixmap.fromImage(image)
        self._pixmap = QtGui.QPixmap(self._pixmap)
        self.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        self.setPixmap(self._pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio))