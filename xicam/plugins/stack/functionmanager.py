from PySide.QtUiTools import QUiLoader
from PySide import QtGui
from PySide import QtCore
import ui
import customwidgets
import tomopy

functions = []
currentfunction = 0
layout = None


def clearFeatures():
    global functions
    value = QtGui.QMessageBox.question(None, 'Delete all functions?',
                                       'Are you sure you want to clear ALL functions?',
                                       (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))

    if value is QtGui.QMessageBox.Yes:
        for feature in functions:
            feature.deleteLater()
            del feature
        functions = []
        ui.showform(ui.blankform)


def addFunction(function, subfunction, package=tomopy):
    global functions, currentfunction
    currentfunction = len(functions)
    functions.append(customwidgets.func(function, subfunction, package))
    update()


def removeFunction(index):
    global functions
    del functions[index]
    update()


def swapFunctions(idx_1, idx_2):
    global functions, currentfunction
    if idx_2 >= len(functions) or idx_2 < 0:
        return
    functions[idx_1], functions[idx_2] = functions[idx_2], functions[idx_1]
    currentfunction = idx_2
    update()


def update():
    global layout, functions
    assert isinstance(layout, QtGui.QVBoxLayout)

    for i in range(layout.count()):
        layout.itemAt(i).parent = None

    # layout.addItem(QtGui.QSpacerItem(0,0,vData=QtGui.QSizePolicy.Expanding))

    for item in functions:
        layout.addWidget(item)


def loadform(path):
    guiloader = QUiLoader()
    f = QtCore.QFile(path)
    f.open(QtCore.QFile.ReadOnly)
    form = guiloader.load(f)
    f.close()
    return form


def load():
    global functions, layout
    layout.setAlignment(QtCore.Qt.AlignBottom)
