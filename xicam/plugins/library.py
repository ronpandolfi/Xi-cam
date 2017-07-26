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

class LibraryPlugin(base.plugin):

    name = 'Library'

    sigUpdateTags = QtCore.Signal(str, str, object, str)

    def __init__(self, *args, **kwargs):


        if sys.platform == 'win32':
            libraryview = LibraryWidget(self, 'C://')
        else:
            libraryview = LibraryWidget(self, pipeline.pathtools.getRoot())


        self.centerwidget = QtGui.QTabWidget()
        self.tabBar = explorer.TabBarPlus()
        self.centerwidget.setTabBar(self.tabBar)

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

        self.sigUpdateTags.connect(self.printit)


        # self.centerwidget.sigOpenFile.connect(viewer.plugininstance.openfiles)


    def printit(self, *x):
        print(*x)

    def addDatabase(self, library, name):
        window = QtGui.QWidget()

        # set up thumbwidget library view
        stack = QtGui.QStackedWidget()
        stack.addWidget(library)
        library.sigUpdateTags.connect(lambda w, x, y, z: self.sigUpdateTags.emit(w, x, y, z))

        # set up toolbar, connect signals
        toolbar = LibraryToolbar()
        toolbar.grid_button.clicked.connect(self.showGridWindow)
        toolbar.image_button.clicked.connect(self.showImageWindow)
        toolbar.rows_button.clicked.connect(self.showRowsWindow)
        toolbar.check_all.clicked.connect(self.checkAll)
        toolbar.uncheck_all.clicked.connect(self.uncheckAll)

        # set up columnViewer view
        column_view = columnView()
        column_view.sigUpdateTags.connect(lambda w, x, y, z: self.sigUpdateTags.emit(w, x, y, z))
        stack.addWidget(column_view)
        column_view.sigOpenFile.connect(self.addViewer)

        l = QtGui.QVBoxLayout()
        l.addWidget(stack)
        l.addWidget(toolbar)
        window.setLayout(l)
        self.centerwidget.addTab(window, name)

    def showGridWindow(self):
        self.currentStack().setCurrentWidget(self.currentStack().widget(0))

    def showImageWindow(self):
        if self.currentStack().count() > 2:
            self.currentStack().setCurrentWidget(self.currentStack().widget(2))

    def showRowsWindow(self):
        columns = self.currentStack().widget(1)
        if not columns.populated:
            columns.populateFromThumbwidgets(self.currentStack().widget(0))
        self.currentStack().setCurrentWidget(columns)

    def checkAll(self):
        library = self.currentStack().widget(0)
        for child in library.widget().children(): # loop over all children of LibraryWidget
            if isinstance(child, thumbwidgetitem): # only look for checkbox if child is a thumbwidgetitem
                # first clause confirms that thumbwidgetitem has checkbox
                # second clause excludes folders from being checked
                if hasattr(child, 'checkbox') and not child.foldericon == child.image:
                    child.checkbox.setChecked(True)

    def uncheckAll(self):
        library = self.currentStack().widget(0)
        for child in library.widget().children(): # loop over all children of LibraryWidget
            if isinstance(child, thumbwidgetitem): # only look for checkbox if child is a thumbwidgetitem
                if hasattr(child, 'checkbox'): # uncheck all thumbwidgetitems with checkbox, including folders
                    child.checkbox.setChecked(False)

    def addViewer(self, widget):
        view = LibraryItem(widget)
        view.sigUpdateTags.connect(lambda w, x, y, z: self.sigUpdateTags.emit(w, x, y, z))
        stack = self.currentStack()

        # TODO: check for unique ID here, since you can open images from both grid and row view
        # TODO: if unique ID is the same, don't remove
        if stack.count() > 2:
            stack.removeWidget(stack.widget(2))

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

class columnView(QtGui.QWidget):

    sigOpenFile = QtCore.Signal(QtGui.QWidget)
    sigUpdateTags = QtCore.Signal(str, str, object, str)

    def __init__(self):
        super(columnView, self).__init__()

        # TODO: enable internal drag/drop of columns
        # self.setDragEnabled(True)
        # self.setAcceptDrops(True)
        # self.viewport().setAcceptDrops(True)
        # self.setDragDropOverwriteMode(False)
        # self.setDropIndicatorShown(True)
        # self.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        # self.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        # self.setSelectionBehavior(QtGui.QAbstractItemView.SelectColumns)

        self.addButton = QtGui.QPushButton('+')
        self.addButton.setMaximumHeight(20)
        self.addButton.setMaximumWidth(20)
        self.addButton.clicked.connect(self.changeColumns)
        tmp_layout = QtGui.QHBoxLayout()
        tmp_layout.setAlignment(Qt.AlignRight)
        tmp_layout.addWidget(self.addButton)

        self.table = QtGui.QTableWidget(0, 2)
        self.table.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().sectionClicked.connect(self.sortColumn)
        self.table.horizontalHeader().setSortIndicatorShown(True)

        self.headers = ['Name', 'Thumb']
        self.table.setHorizontalHeaderLabels(self.headers)
        self.populated = False

        self.layout = QtGui.QVBoxLayout()
        self.layout.addLayout(tmp_layout)
        self.layout.addWidget(self.table)
        self.setLayout(self.layout)


    def changeColumns(self):

        # this dictionary should be generated in the __init___ by querying the database
        fields = {'Name': False, 'Thumb': False, 'Rating': False}

        # set boxes to be checked for columns that are already present
        for key in self.headers:
            if key in fields.keys():
                fields[key] = True

        dialog = cw.checkBoxDialog(fields=fields, title='Select fields to sort by')
        headers, accepted = dialog.getHeaders()

        if accepted:

            self.headers = headers

            # remove current table and create a new one to repopulate
            self.layout.removeWidget(self.table)
            del self.table
            self.table = QtGui.QTableWidget(0, len(self.headers))
            self.table.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
            self.table.horizontalHeader().sectionClicked.connect(self.sortColumn)
            self.table.horizontalHeader().setSortIndicatorShown(True)
            self.layout.addWidget(self.table)

            # set headers on new table and populate it
            self.table.setHorizontalHeaderLabels(self.headers)
            self.setUnpopulated()
            self.populateFromThumbwidgets()



    def sortColumn(self, index):

        # sort column by clicking on header, while ignoring image-based sorting (for thumbnails)

        if self.table.horizontalHeaderItem(index).text() != 'Thumb':
            if self.table.horizontalHeader().sortIndicatorOrder() == Qt.DescendingOrder:
                self.table.sortItems(index, order = Qt.DescendingOrder)
            else:
                self.table.sortItems(index, order = Qt.AscendingOrder)

    def setItem(self, widget, row, header_name):

        # TODO: this will have to be made more general in the future by querying the database for
        # TODO: values to be set for a row given a header_name
        if header_name == 'Name':
            item = QtGui.QTableWidgetItem(os.path.basename(widget.path))
            self.table.setItem(row, self.headers.index(header_name), item)
        elif header_name == 'Thumb':
            w = ScaledColumnLabel(widget.image, widget)
            w.sigOpenFile.connect(self.sigOpenFile.emit)
            self.table.setCellWidget(row, self.headers.index(header_name), w)
        elif header_name == 'Rating':
            w = cw.starRating()
            w.setRating(widget.starRating.stars())
            w.sigRatingChanged.connect(lambda rating: self.sigUpdateTags.emit(widget.path, 'Rating',
                                                                              rating, 'replace'))

            # connect signals - should be changed to database query when database is implemented
            # at this point, the signals are probably broken
            w.sigRatingChanged.connect(widget.starRating._setRating)
            widget.starRating.sigRatingChanged.connect(w._setRating)

            self.table.setCellWidget(row, self.headers.index(header_name), w)


    def populateFromThumbwidgets(self, libraryWidget=None):

        if libraryWidget:
            self.libraryWidget = libraryWidget
        for child in self.libraryWidget.widget().children():
            if isinstance(child, thumbwidgetitem):
                if child.foldericon != child.image:
                    r = self.table.rowCount()
                    self.table.insertRow(r)
                    for header in self.headers:
                        self.setItem(child, r, header)

                    # TODO: here query database for other metadata to sort by
                else:
                    child.sigChangeRoot.connect(self.setUnpopulated)

        self.populated = True

    def setUnpopulated(self, *args):
        self.populated = False
        count = self.table.rowCount()
        if count > 0:
            for i in range(count):
                self.table.removeRow(0)



class LibraryToolbar(QtGui.QWidget):

    def __init__(self):
        super(LibraryToolbar, self).__init__()
        self.setMaximumHeight(60)

        # button for viewing grid of images
        self.grid_button = QtGui.QToolButton()
        self.grid_button.setToolTip('Folder view')
        self.grid_button.resize(QtCore.QSize(40, 40))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_65.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.grid_button.setIcon(icon)
        self.grid_button.setIconSize(QtCore.QSize(40, 40))
        # self.grid_button.clicked.connect(self.showGridWindow)

        # button for viewing single image
        self.image_button = QtGui.QToolButton()
        self.image_button.setToolTip('Image view')
        self.image_button.resize(QtCore.QSize(40, 40))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_66.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.image_button.setIcon(icon)
        self.image_button.setIconSize(QtCore.QSize(40, 40))
        # self.image_button.clicked.connect(self.showImageWindow)

        # button for rows view
        self.rows_button = QtGui.QToolButton()
        self.rows_button.setToolTip('Rows view')
        self.rows_button.resize(QtCore.QSize(40, 40))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_67.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.rows_button.setIcon(icon)
        self.rows_button.setIconSize(QtCore.QSize(40, 40))

        # buttons for checking/unchecking all boxes
        self.check_all = QtGui.QPushButton('Check all')
        self.uncheck_all = QtGui.QPushButton('Uncheck all')


        h = QtGui.QHBoxLayout()
        grid = QtGui.QGridLayout()
        grid.addWidget(self.grid_button, 0, 0)
        grid.addWidget(self.rows_button, 0, 1)
        grid.addWidget(self.image_button, 0, 2)
        grid.addWidget(self.check_all, 0, 3)
        grid.addWidget(self.uncheck_all, 0, 4)
        grid.addItem(QtGui.QSpacerItem(1, 1), 0, 5, 1, 10)
        h.addLayout(grid)
        self.setLayout(h)


class LibraryWidget(QtGui.QScrollArea):
    sigOpenFile = QtCore.Signal(QtGui.QWidget)
    sigUpdateTags = QtCore.Signal(str, str, object, str)

    def __init__(self, *args, **kwargs):
        super(LibraryWidget, self).__init__()
        self.setWidgetResizable(True)
        self.setFocusPolicy(Qt.NoFocus)
        l = librarylayout(*args, **kwargs)
        l.sigUpdateTags.connect(lambda w, x, y, z: self.sigUpdateTags.emit(w, x, y, z))
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
    sigUpdateTags = QtCore.Signal(str, str, object, str)

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
                w.sigUpdateTags.connect(lambda w, x, y, z: self.sigUpdateTags.emit(w, x, y, z))
                self.addWidget(w)
                w.sigOpenFile.connect(self.sigOpenFile)
                w.sigChangeRoot.connect(self.chdir)


    def clear(self):
        while self.count():
            item = self.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

class LibraryItem(QtGui.QSplitter):

    sigUpdateTags = QtCore.Signal(str, str, object, str)

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
        self.keywords.starRating.setRating(thumbwidget.starRating.stars())

        # connect keywords stars to thumbwidget stars
        self.keywords.starRating.sigRatingChanged.connect(thumbwidget.starRating._setRating)

        # connect stars to database
        self.keywords.sigUpdateTags.connect(lambda w, x, y, z: self.sigUpdateTags.emit(
            thumbwidget.path, x, y, z))


        # connect thumbwidget stars to keywords stars
        thumbwidget.starRating.sigRatingChanged.connect(self.keywords.starRating._setRating)
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
        self.setMinimumWidth(200)
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

#TODO: this class should also be connected to the database since it can change metadata
class keywordsWidget(QtGui.QWidget):

    sigUpdateTags = QtCore.Signal(str, str, object, str)

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
        tmp_layout = QtGui.QHBoxLayout()
        label_2 = QtGui.QLabel('Rating: ')
        self.starRating = cw.starRating()
        self.starRating.sigRatingChanged.connect(lambda rating: self.sigUpdateTags.emit(
            '', 'Rating', rating, 'replace'))
        # layout.addWidget(label_2)
        # layout.addWidget(self.starRating)
        tmp_layout.addWidget(label_2)
        tmp_layout.setSpacing(20)
        tmp_layout.addWidget(self.starRating)
        layout.addLayout(tmp_layout)

        # this is proof of concept for another type of user-generated metadata
        # in the future, this __init__ will generate the necessary fields and add them by querying the database
        # in the meantime, it has to be done manually
        tmp_layout = QtGui.QHBoxLayout()
        label_3 = QtGui.QLabel('Field: ')
        self.field = QtGui.QComboBox()
        self.field.currentIndexChanged.connect(self.emitFieldChanges)
        self.field.addItems(['Physics', 'Geology', 'Biology', 'Materials Science'])
        tmp_layout.addWidget(label_3)
        tmp_layout.addWidget(self.field)
        layout.addLayout(tmp_layout)

        self.setLayout(layout)

    def emitFieldChanges(self, index):
        text = self.field.itemText(index)
        self.sigUpdateTags.emit('', 'Field', text, 'replace')

    def updateKeywords(self):
        self.keywords = set([])
        for item in filter(None, re.split("[,:;]", self.textbox.toPlainText())):
            self.keywords.add(item.strip())

        self.sigUpdateTags.emit('', 'keywords', list(self.keywords), 'replace')


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
    sigUpdateTags = QtCore.Signal(str, str, object, str)


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

        # sets up text label containing dataset/folder name
        self.namelabel = QtGui.QLabel(os.path.basename(path) if nameoverride is None else nameoverride)
        self.namelabel.setAlignment(Qt.AlignHCenter)
        # self.namelabel.setWordWrap(True)
        self.namelabel.setToolTip(self.namelabel.text())
        self.namelabel.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum)

        # checkbox for marking selected thumbnails
        self.checkbox = QtGui.QCheckBox()
        tmp_layout = QtGui.QHBoxLayout()
        tmp_layout.addWidget(self.checkbox)
        tmp_layout.addWidget(self.namelabel)
        tmp_layout.setAlignment(Qt.AlignLeft)
        self.layout.addLayout(tmp_layout)

        # sets up image in thumbnail
        image_label = ScaledLabel(self.image)
        self.layout.addWidget(image_label)



        # sets up line separating the name and image labels
        line = QtGui.QFrame()
        line.setGeometry(QtCore.QRect(0, 0, desiredsize.width() * 9 / 10, 3))
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        self.layout.addWidget(line)

        # sets up starRating widget to keep track of rating on thumbwidgets
        if self.image != self.foldericon:
            self.starRating = cw.starRating()
            self.starRating.sigRatingChanged.connect(lambda rating: self.sigUpdateTags.emit(self.path,
                                                                            'Rating', rating, 'replace'))
            tmp_layout_2 = QtGui.QHBoxLayout()
            tmp_layout_2.setAlignment(Qt.AlignHCenter)
            tmp_layout_2.addWidget(self.starRating)
            self.layout.addLayout(tmp_layout_2)

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

class ScaledColumnLabel(ScaledLabel):
    """
    A label that scales with dimension; for thumbnails in COLUMN view (need specialized signal)
    """

    sigOpenFile = QtCore.Signal(QtGui.QWidget)

    def __init__(self, image, thumbwidget):
        super(ScaledColumnLabel, self).__init__(image=image)
        self.thumb = thumbwidget

    def mouseDoubleClickEvent(self, event):
        self.sigOpenFile.emit(self.thumb)

