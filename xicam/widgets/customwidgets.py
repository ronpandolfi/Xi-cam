from __future__ import absolute_import
from __future__ import unicode_literals


from builtins import str
from builtins import map
from builtins import range
__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


import numpy as np
import pyqtgraph as pg
from . import explorer
from PySide import QtGui, QtCore
from xicam.widgets.stardelegate import stardelegate



class DataTreeWidget(QtGui.QTreeWidget):
    """
    Widget for displaying hierarchical python data structures
    (eg, nested dicts, lists, and arrays), adapted from pyqtgraph datatree.
    """

    def __init__(self, parent=None, data=None):
        QtGui.QTreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.setData(data)
        self.setColumnCount(2)
        self.setHeaderLabels(['Parameter', 'value'])

    def setData(self, data, hideRoot=False):
        """data should be a dictionary."""
        self.clear()
        self.buildTree(data, self.invisibleRootItem(), hideRoot=hideRoot)
        self.expandToDepth(3)
        self.resizeColumnToContents(0)

    def buildTree(self, data, parent, name='', hideRoot=False):
        if hideRoot:
            node = parent
        else:
            node = QtGui.QTreeWidgetItem([name, ""])
            parent.addChild(node)

        if isinstance(data, dict):
            for k in list(data.keys()):
                self.buildTree(data[k], node, str(k))
        elif isinstance(data, list) or isinstance(data, tuple):
            for i in range(len(data)):
                self.buildTree(data[i], node, str(i))
        else:
            node.setText(1, str(data))


class ImageView(pg.ImageView):
    """
    Subclass of PG ImageView to correct z-slider signal behavior, and add coordinate label.
    See pygtgraph.ImageView for documentation
    """
    sigDeletePressed = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super(ImageView, self).__init__(*args, **kwargs)
        self.scene.sigMouseMoved.connect(self.mouseMoved)

        self.coordsLabel = QtGui.QLabel(' ', parent=self)
        self.coordsLabel.setMinimumHeight(16)
        self.layout().addWidget(self.coordsLabel)
        self.coordsLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
        self.setStyleSheet("background-color: rgba(0,0,0,0%)")


    def buildMenu(self):
        super(ImageView, self).buildMenu()
        self.menu.removeAction(self.normAction)

    def keyPressEvent(self, ev):
        super(ImageView, self).keyPressEvent(ev)
        if ev.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
            self.timeLineChanged()
        elif ev.key() == QtCore.Qt.Key_Delete or ev.key() == QtCore.Qt.Key_Backspace:
            self.sigDeletePressed.emit()

    def timeIndex(self, slider):
        ## Return the time and frame index indicated by a slider
        if self.image is None:
            return (0,0)

        t = slider.value()

        xv = self.tVals
        if xv is None:
            ind = int(t)
        else:
            if len(xv) < 2:
                return (0,0)
            totTime = xv[-1] + (xv[-1]-xv[-2])
            inds = np.argwhere(xv <= t)
            if len(inds) < 1:
                return (0,t)
            ind = inds[-1,0]
        return ind, t

    def mouseMoved(self, ev):
        pos = ev
        viewBox = self.imageItem.getViewBox()
        try:
            if viewBox.sceneBoundingRect().contains(pos):
                mousePoint = viewBox.mapSceneToView(pos)
                x, y = list(map(int, (mousePoint.x(), mousePoint.y())))
                if (0 <= x < self.imageItem.image.shape[0]) & (0 <= y < self.imageItem.image.shape[1]):  # within bounds
                    self.coordsLabel.setText(u"<div style='font-size: 12pt;background-color:#111111;'>x={0},"
                                             u"   <span style=''>y={1}</span>,   <span style=''>I={2}</span>"\
                                             .format(x, y, self.imageItem.image[x, y]))
                else:
                    self.coordsLabel.setText(u"<div style='font-size: 12pt;background-color:#111111;'>x= ,"
                                             u"   <span style=''>y= </span>,   <span style=''>I= </span>")
        except AttributeError:
            pass

class starRating(QtGui.QTableWidget):
    """
    Sets up a one-cell tablewidget containing a place for selecting between 0 to 5 stars
    """

    sigRatingChanged = QtCore.Signal(int)

    def __init__(self):
        super(starRating, self).__init__(1, 1)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setItemDelegate(stardelegate.StarDelegate())
        self.setEditTriggers(QtGui.QAbstractItemView.DoubleClicked |
                      QtGui.QAbstractItemView.SelectedClicked)
        self.rating = QtGui.QTableWidgetItem()
        self.rating.setData(0, 0)
        self.itemChanged.connect(lambda x: self.sigRatingChanged.emit(x.data(0) + 1))
        self.setItem(0, 0, self.rating)
        self.horizontalHeader().setStretchLastSection(True)
        self.setMaximumHeight(30)
        self.setMaximumWidth(110)

    def stars(self):
        """
        Number of stars shown in widget. Accounts for weird offset introduced in stardelegate code
        """
        rating = self.rating.data(0)
        return rating + 1 if rating else rating

    def setRating(self, val):
        """
        Sets star rating programmatically
        """
        self.rating.setData(0, int(val))

    def _setRating(self, val):
        """
        Sets star rating while accounting for the float values that the star rating item delegate accepts
        """
        self.blockSignals(True)
        if val < 0:
            if self.rating.data(0) != 0:
                self.rating.setData(0, 0)
        else:
            if int(np.ceil(val)) != self.rating.data(0):
                self.rating.setData(0, int(np.ceil(val)))
        self.blockSignals(False)

class histDialogButton(QtGui.QPushButton):
    """
    Button to connect to a pyqtgraph.HistogramLUTWidget. Sets the maximum and minimum hist value based on
    values provided by user from popup dialog
    """

    def __init__(self, label, *args, **kwargs):
        super(histDialogButton, self).__init__(label, *args, **kwargs)

        self.value = None

    def showDialog(self, min_default, max_default):

        self.dialog = QtGui.QDialog(parent=self.window())
        layout = QtGui.QVBoxLayout()
        text = QtGui.QLabel("Set maximum and minimum values for histogram.")
        layout.addWidget(text)
        layout.addSpacing(5)

        label1 = QtGui.QLabel("Max: ")
        label2 = QtGui.QLabel("Min: ")

        self.field1 = QtGui.QDoubleSpinBox()
        self.field2 = QtGui.QDoubleSpinBox()
        self.field1.setRange(-1000000, 1000000)
        self.field2.setRange(-1000000, 1000000)
        self.field1.setValue(max_default)
        self.field2.setValue(min_default)

        h1 = QtGui.QHBoxLayout()
        h1.addWidget(label1)
        h1.addWidget(self.field1)
        h2 = QtGui.QHBoxLayout()
        h2.addWidget(label2)
        h2.addWidget(self.field2)

        valueButton = QtGui.QPushButton("Ok")
        valueButton.setDefault(True)
        cancelButton = QtGui.QPushButton("Cancel")
        valueButton.clicked.connect(self.return_vals)
        cancelButton.clicked.connect(self.cancel)

        h3 = QtGui.QHBoxLayout()
        h3.addWidget(cancelButton)
        h3.addWidget(valueButton)

        layout.addLayout(h1)
        layout.addLayout(h2)
        layout.addLayout(h3)
        self.dialog.setLayout(layout)
        self.dialog.setWindowTitle("Set histogram bounds")
        self.dialog.exec_()

    def return_vals(self):
        self.value = [self.field2.value(), self.field1.value()]
        self.dialog.accept()


    def cancel(self):
        self.value = None
        self.dialog.reject()

    def connectToHistWidget(self, hist):

        self.hist = hist
        self.clicked.connect(self.setHistValues)

    def setHistValues(self):

        defaults = self.hist.getLevels()
        self.showDialog(defaults[0], defaults[1])

        if self.value:
            self.hist.setLevels(self.value[0], self.value[1])


class dataDialog(QtGui.QDialog):

    """
    Subclass of QDialog to allow for inputs for filenames
    """

    def __init__(self, parent=None, msg=None):
        super(dataDialog, self).__init__(parent)

        # text at top of box
        text = QtGui.QLabel(msg)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(text)
        layout.addSpacing(5)

        # dialog for searching through files
        dialog = explorer.MultipleFileExplorer(self)
        layout.addWidget(dialog)

        horiz_layout = QtGui.QHBoxLayout()
        # horiz_layout.setContentsMargins(0,0,0,0)

        # ok and cancel buttons for user interaction
        button_box = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)

        # file name viewer
        file_label = QtGui.QLabel("File name: ")
        file_name = QtGui.QLineEdit()
        file_name.setReadOnly(True)
        file_name.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)

        horiz_layout.addWidget(file_label)
        horiz_layout.addWidget(file_name)
        horiz_layout.addWidget(button_box)

        layout.addSpacing(5)
        layout.addLayout(horiz_layout)
        self.setLayout(layout)
        self.exec_()

class checkBoxDialog(QtGui.QDialog):
    """
    Subclass of QDialog that lists fields that can be checked or unchecked. Returns list of checked fields
    """

    def __init__(self, parent=None, fields={}, title=''):
        # fields are key/value pairs of "field" and "checked/unchecked"

        super(checkBoxDialog, self).__init__(parent)
        self.setWindowTitle(title)
        self.fields = fields.copy()
        self.widgets = {}

        layout = QtGui.QGridLayout()
        row, col = 0, 0

        for key, value in fields.items():
            widget = QtGui.QWidget()
            tmp_layout = QtGui.QHBoxLayout()
            tmp_layout.setAlignment(QtCore.Qt.AlignLeft)
            box = QtGui.QCheckBox()
            box.setChecked(value)
            name = QtGui.QLabel(key)
            tmp_layout.addWidget(box)
            tmp_layout.addWidget(name)
            widget.setLayout(tmp_layout)
            self.widgets[box] = name
            layout.addWidget(widget, row, col)

            col = (col + 1)%3
            if col == 0:
                row += 1

        ok_button = QtGui.QPushButton('Ok')
        ok_button.clicked.connect(self.accept)
        cancel_button = QtGui.QPushButton('Cancel')
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button, row + 1, 1)
        layout.addWidget(ok_button, row + 1, 2)

        self.setLayout(layout)

    def _getHeaders(self):

        headers = []
        for widget in self.widgets.keys():
            if widget.isChecked():
                headers.append(self.widgets[widget].text())
        return headers

    def getHeaders(self):

        result = self.exec_()
        headers = self._getHeaders()

        return (headers, result == QtGui.QDialog.Accepted)


class sliceDialog(QtGui.QDialog):

    """
    Subclass of QDialog to allow for more diverse returns from input dialog popups, specifically
    relating to tomography slice previews

    Attributes
    ----------
    field1: int
        integer representing lowest slice number to reconstruct
    field2: int
        integer representing highest slice number to reconstruct
    """

    def __init__(self, parent=None, val1=0, val2=0, maximum=1000):

        super(sliceDialog, self).__init__(parent)

        text = QtGui.QLabel("Which slice(s) would you like to reconstruct?")
        layout = QtGui.QVBoxLayout()
        layout.addWidget(text)

        label1 = QtGui.QLabel("Start slice: ")
        label2 = QtGui.QLabel("End slice: ")
        self.field1 = QtGui.QSpinBox()
        self.field1.setRange(0,maximum)
        self.field1.setValue(val1)
        self.field2 = QtGui.QSpinBox()
        self.field2.setRange(0,maximum)
        self.field2.setValue(val2)


        h1 = QtGui.QHBoxLayout()
        h1.addWidget(label1)
        h1.addWidget(self.field1)
        h2 = QtGui.QHBoxLayout()
        h2.addWidget(label2)
        h2.addWidget(self.field2)

        valueButton = QtGui.QPushButton("Ok")
        valueButton.setDefault(True)
        cancelButton = QtGui.QPushButton("Cancel")
        valueButton.clicked.connect(self.return_vals)
        cancelButton.clicked.connect(self.cancel)
        h3 = QtGui.QHBoxLayout()
        h3.addWidget(cancelButton)
        h3.addWidget(valueButton)

        layout.addLayout(h1)
        layout.addLayout(h2)
        layout.addLayout(h3)
        self.setLayout(layout)
        self.setWindowTitle("Slice Preview")
        self.exec_()

    def return_vals(self):
        try:
            val1 = int(self.field1.text())
            val2 = int(self.field2.text())
            if val1>val2:
                self.value = [val2,val1]
            else:
                self.value = [val1,val2]
        except ValueError:
            self.value = "Slice values must be integers"

        self.accept()


    def cancel(self):
        self.value = None
        self.reject()
