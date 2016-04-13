from PySide.QtUiTools import QUiLoader
from PySide import QtGui
from PySide import QtCore

import customwidgets


features = []
layout = None


def clearFeatures():
    global features
    features = []

def addFunction(function,subfunction):
    global features
    features.append(customwidgets.func(function,subfunction))
    update()


def removeFeature(index):
    global features
    del features[index]
    update()


def update():
    global layout
    assert isinstance(layout, QtGui.QVBoxLayout)

    for i in range(layout.count()):
        layout.itemAt(i).parent = None

    # layout.addItem(QtGui.QSpacerItem(0,0,vData=QtGui.QSizePolicy.Expanding))

    for item in features:
        layout.addWidget(item)


def loadform(path):
    guiloader = QUiLoader()
    f = QtCore.QFile(path)
    f.open(QtCore.QFile.ReadOnly)
    form = guiloader.load(f)
    f.close()
    return form


def load():
    global features, layout
    layout.setAlignment(QtCore.Qt.AlignBottom)
    # addSubstrate()
    # addLayer()
    # addLayer()
    # addParticle()
