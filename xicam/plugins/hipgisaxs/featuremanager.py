from PySide.QtUiTools import QUiLoader
from PySide import QtGui
from PySide import QtCore

import models
import ui
import customwidgets
import display
import hig


features = []
functionTree = None
layout = None

def clearFeatures():
    global features
    while features:
        feature = features[-1]
        features.remove(feature)
        feature.deleteLater()


def addSubstrate():
    global features
    if not sum([type(feature) is customwidgets.substrate for feature in features]):
        features.insert(0, customwidgets.substrate())
    update()


def addLayer():
    global features
    features.append(customwidgets.layer())
    update()


def addParticle():
    global features
    features.append(customwidgets.particle())
    update()


def removeFeature(index):
    global features
    del features[index]
    update()


def layercount():
    return sum([type(feature) is customwidgets.layer for feature in features])


def particlecount():
    return sum([type(feature) is customwidgets.particle for feature in features])


def update():
    global functionTree
    assert isinstance(layout, QtGui.QVBoxLayout)

    for i in range(layout.count()):
        if layout.itemAt(i) not in features:
            layout.itemAt(i).parent = None

    # layout.addItem(QtGui.QSpacerItem(0,0,vData=QtGui.QSizePolicy.Expanding))

    for i, item in enumerate(features[::-1]):
        layout.insertWidget(i, item)

    if display.viewWidget:
        display.redraw()


def loadform(path):
    guiloader = QUiLoader()
    f = QtCore.QFile(path)
    f.open(QtCore.QFile.ReadOnly)
    form = guiloader.load(f)
    f.close()
    return form


def load():
    global features, functionTree
    layout.setAlignment(QtCore.Qt.AlignBottom)
    addSubstrate()
    addParticle()

