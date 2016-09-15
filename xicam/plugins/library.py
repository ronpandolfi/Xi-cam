from PySide import QtGui
import sys
import base
import viewer


class LibraryPlugin(base.plugin):
    name = 'Library'

    def __init__(self, *args, **kwargs):


        if sys.platform == 'win32':
            libraryview = LibraryWidget(self, 'C://')
        else:
            libraryview = LibraryWidget(self, pipeline.pathtools.getRoot())

        self.centerwidget = libraryview

        self.rightwidget = None

        self.featureform = None

        self.bottomwidget = None

        self.toolbar = None

        super(plugin, self).__init__(*args, **kwargs)

        # self.centerwidget.sigOpenFile.connect(viewer.plugininstance.openfiles)


from PySide import QtGui
from PySide import QtCore
from PySide.QtCore import Qt
import os
import pipeline
import numpy as np


class LibraryWidget(QtGui.QScrollArea):
    sigOpenFile = QtCore.Signal(str)

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

    sigOpenFile = QtCore.Signal(str)

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


class thumbwidgetitem(QtGui.QFrame):
    """
    A single icon representing a file/folder that can be accessed/opened
    """
    sigChangeRoot = QtCore.Signal(str)
    sigOpenFile = QtCore.Signal(str)


    def __init__(self, path, parentwindow, nameoverride=None):
        path = os.path.normpath(path)

        self.foldericon = QtGui.QImage()
        self.foldericon.load('xicam/gui/GenericFolderIcon.png')

        self.fileicon = QtGui.QImage()
        self.fileicon.load('xicam/gui/post-360412-0-09676400-1365986245.png')

        print 'Library widget generated for ' + path
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
        dimg = pipeline.loader.diffimage(filepath=self.path)
        self.image = QtGui.QImage()
        # print os.path.splitext(path)[1]
        if os.path.isdir(path):
            self.image = self.foldericon
        elif os.path.splitext(path)[1] in pipeline.loader.acceptableexts:


            try:
                self.thumb = np.rot90(np.log(dimg.thumbnail * (dimg.thumbnail > 0) + (dimg.thumbnail < 1))).copy()
                print 'thumbmax:', np.max(self.thumb)
                self.thumb *= 255. / np.max(self.thumb)



                # if self.thumb is not None:
                self.image = QtGui.QImage(self.thumb.astype(np.uint8), self.thumb.shape[1], self.thumb.shape[0],
                                          self.thumb.shape[1],
                                          QtGui.QImage.Format_Indexed8)
            except Exception as ex:
                print ex.message

                self.image = self.fileicon


        else:
            self.image = self.fileicon

        image_label = ScaledLabel(self.image)
        image_label.setAlignment(Qt.AlignHCenter)
        self.layout.addWidget(image_label)

        self.namelabel = QtGui.QLabel(os.path.basename(path) if nameoverride is None else nameoverride)
        self.namelabel.setAlignment(Qt.AlignHCenter)
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
        if os.path.isdir(self.path):
            self.sigChangeRoot.emit(self.path)
        else:
            self.sigOpenFile.emit(self.path)


class ScaledLabel(QtGui.QLabel):
    """
    A label that scales with the constrained dimension; for thumbnails
    """

    def __init__(self, image):
        super(ScaledLabel, self).__init__()
        self._pixmap = QtGui.QPixmap.fromImage(image)
        self._pixmap = QtGui.QPixmap(self._pixmap)

    def resizeEvent(self, event):
        self.setPixmap(self._pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio))