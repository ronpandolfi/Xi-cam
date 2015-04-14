from PySide.QtGui import QWidget
from PySide.QtGui import QLabel
from PySide.QtGui import QVBoxLayout
from PySide.QtGui import QLayout
from PySide.QtCore import QDir
from PySide.QtCore import QDirIterator
from PySide.QtGui import QImage
from PySide.QtGui import QPixmap
from PySide.QtCore import Qt
from PySide.QtCore import QRect
from PySide.QtCore import QSize
from PySide.QtCore import QPoint
from PySide.QtGui import QSizePolicy
from PySide.QtGui import QFrame
import fabio
from scipy.misc import imresize
import numpy as np
from PIL import Image


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=5, spacing=-1):
        super(FlowLayout, self).__init__(parent)

        # if parent is not None:
        # self.margin = margin
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
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()

        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())

        size += QSize(2 * self.margin, 2 * self.margin)
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing() + wid.style().layoutSpacing(QSizePolicy.PushButton, QSizePolicy.PushButton,
                                                                Qt.Horizontal)
            spaceY = self.spacing() + wid.style().layoutSpacing(QSizePolicy.PushButton, QSizePolicy.PushButton,
                                                                Qt.Vertical)
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()


class thumbwidgetcollection(FlowLayout):
    def __init__(self):
        super(thumbwidgetcollection, self).__init__()

        parent = QDir()
        parent.cdUp()
        parent.cd('samples/')
        print parent.absolutePath()

        diriterator = QDirIterator(parent)

        while diriterator.hasNext():
            print(diriterator.fileName())
            if diriterator.fileInfo().isFile():
                self.addWidget(thumbwidgetitem(diriterator.filePath()))
            diriterator.next()


class folderwidget(QWidget):  # WIP
    def __init__(self, path):
        super(folderwidget, self).__init__()

        self.layout = QVBoxLayout()
        self.path = path
        self.imgdata = fabio.open(path).data
        self.imgdata = np.log(self.imgdata * (self.imgdata > 0) + (self.imgdata < 1))
        self.imgdata *= 255 / np.max(self.imgdata)
        self.imgdata = self.imgdata.astype(np.uint8)

        dims = (min(desiredsize, self.imgdata.shape[0] * desiredsize / self.imgdata.shape[1]),
                min(desiredsize, self.imgdata.shape[1] * desiredsize / self.imgdata.shape[0]))
        # dims=(220,230)
        #print(dims)
        #print self.imgdata
        #self.imgdata = imresize(self.imgdata, (dims[0],dims[1]))
        #print self.imgdata

        im = Image.fromarray(self.imgdata, 'L')
        im.thumbnail((150, 150))
        print(im.size)

        self.namelabel = QLabel(path)
        self.image = QImage(im.tobytes('raw', 'L'), im.size[0], im.size[1], im.size[0],
                            QImage.Format_Indexed8)
        image_label = QLabel(" ")
        #image_label.setMaximumSize(200,200)
        image_label.setScaledContents(True)
        image_label.setPixmap(QPixmap.fromImage(self.image))
        self.layout.addWidget(image_label)
        # self.layout.addWidget(self.namelabel)
        self.setLayout(self.layout)


class thumbwidgetitem(QFrame):
    def __init__(self, path):
        super(thumbwidgetitem, self).__init__()
        self.setObjectName('thumb')
        desiredsize = QSize(250, 300)

        # toplayout = QVBoxLayout(self)
        #self.frame = QFrame(self)


        self.setFixedSize(desiredsize)
        self.setAutoFillBackground(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # self.frame.setFixedSize(desiredsize)
        #self.frame.setFrameStyle(QFrame.Plain)
        #self.frame.setFrameShape(QFrame.StyledPanel)
        #self.setStyle('background-color:#999999')



        self.layout = QVBoxLayout(self)  #.frame

        self.path = path
        self.imgdata = fabio.open(path).data
        self.imgdata = np.log(self.imgdata * (self.imgdata > 0) + (self.imgdata < 1))
        self.imgdata *= 255 / np.max(self.imgdata)
        self.imgdata = self.imgdata.astype(np.uint8)

        # dims = (min(desiredsize, self.imgdata.shape[0] * desiredsize / self.imgdata.shape[1]),
        #        min(desiredsize, self.imgdata.shape[1] * desiredsize / self.imgdata.shape[0]))
        # dims=(220,230)
        #print(dims)
        #print self.imgdata
        #self.imgdata = imresize(self.imgdata, (dims[0],dims[1]))
        #print self.imgdata

        im = Image.fromarray(self.imgdata, 'L')
        #im.thumbnail((150, 150))
        print(im.size)

        self.namelabel = QLabel(path.split('/')[-1])
        self.namelabel.setAlignment(Qt.AlignHCenter)
        self.namelabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.image = QImage(im.tobytes('raw', 'L'), im.size[0], im.size[1], im.size[0],
                            QImage.Format_Indexed8)
        image_label = ScaledLabel(self.image)
        image_label.setAlignment(Qt.AlignHCenter)

        self.layout.addWidget(image_label)

        line = QFrame()
        line.setGeometry(QRect(0, 0, desiredsize.width() * 9 / 10, 3))
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(line)

        self.layout.addWidget(self.namelabel)
        self.setLayout(self.layout)

    def enterEvent(self, *args, **kwargs):
        # self.frame.setFrameStyle(QFrame.Raised)
        pass

        #def leaveEvent(self, *args, **kwargs):
        #    self.frame.setFrameStyle(QFrame.Plain)

        #def mousePressEvent(self, *args, **kwargs):
        #    self.frame.setFrameStyle(QFrame.Sunken)


class ScaledLabel(QLabel):
    def __init__(self, image):
        super(ScaledLabel, self).__init__()
        self._pixmap = QPixmap.fromImage(image)
        self._pixmap = QPixmap(self._pixmap)

    def resizeEvent(self, event):
        self.setPixmap(self._pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio))