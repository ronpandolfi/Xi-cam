from PySide.QtGui import *
from PySide.QtCore import *
from collections import Iterable, OrderedDict

from datetime import datetime

from xicam import config

class MetaDataWidget(QTreeWidget):
    """
    Widget for displaying hierarchical python data structures
    (eg, nested dicts, lists, and arrays).
    """

    def __init__(self, parent=None, data=None):
        super(MetaDataWidget, self).__init__(parent)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.setData(data)
        self.setSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.MinimumExpanding)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setColumnCount(2)
        self.setHeaderLabels(('Key','Value'))
        self.header().setStretchLastSection(True)
        self.setColumnWidth(0,100)


        self.contextMenu = QMenu()
        useAsMenu = QMenu(u'Use as...',parent=self.contextMenu)
        useAsMenu.addAction('Beam Energy').triggered.connect(self.useAsEnergy)
        useAsMenu.addAction('Downstream Intensity').triggered.connect(self.useAsI1)
        useAsMenu.addAction('Timeline Axis').triggered.connect(self.useAsTimeline)
        self.contextMenu.addMenu(useAsMenu)

    def mousePressEvent(self,ev):
        if ev.button()==Qt.RightButton:
            self.contextMenu.popup(self.mapToGlobal(ev.pos()))

    def setData(self,data):
        self(data)

    def useAsI1(self):
        config.activeExperiment.setHeaderMap('I1 AI',self.getSelectedKey())

    def useAsEnergy(self):
        config.activeExperiment.setHeaderMap('Beam Energy',self.getSelectedKey())

    def useAsTimeline(self):
        config.activeExperiment.setHeaderMap('Timeline Axis', self.getSelectedKey())

    def getSelectedKey(self):
        return self.item(self.selectedIndexes()[0].row(),0).value

    def __call__(self, header):
        self.fill(header)

    def fill(self, value):
        self.clear()
        fill_item(self.invisibleRootItem(), value)

def fill_item(item, value):
    """
    Display a dictionary as a QtWidgets.QtTreeWidget

    adapted from http://stackoverflow.com/a/21806048/1221924
    """
    item.setExpanded(True)
    if hasattr(value, 'items'):
        for key, val in sorted(value.items()):
            child = QTreeWidgetItem()
            # val is dict or a list -> recurse
            if hasattr(val, 'items') or _listlike(val):
                child.setText(0, _short_repr(key).strip("'"))
                item.addChild(child)
                fill_item(child, val)
                if key == 'descriptors':
                    child.setExpanded(False)
            # val is not iterable -> show key and val on one line
            else:
                # Show human-readable datetime alongside raw timestamp.
                # 1484948553.567529 > '[2017-01-20 16:42:33] 1484948553.567529'
                if (key == 'time') and isinstance(val, float):
                    FMT = '%Y-%m-%d %H:%M:%S'
                    ts = datetime.fromtimestamp(val).strftime(FMT)
                    text = "time"," [{}] {}".format(ts, val)
                else:
                    text = _short_repr(key).strip("'"),_short_repr(val)
                child.setText(0, text[0])
                child.setText(1, text[1])
                item.addChild(child)

    elif type(value) is list:
        for val in value:
            if hasattr(val, 'items'):
                fill_item(item, val)
            elif _listlike(val):
                fill_item(item, val)
            else:
                child = QTreeWidgetItem()
                item.addChild(child)
                child.setExpanded(True)
                child.setText(0, _short_repr(val))
    else:
        child = QTreeWidgetItem()
        child.setText(0, _short_repr(value))
        item.addChild(child)


def _listlike(val):
    return isinstance(val, Iterable) and not isinstance(val, str)


def _short_repr(text):
    r = repr(text)
    if len(r) > 82:
        r = r[:27] + '...'
    return r