from PySide.QtUiTools import QUiLoader
from PySide import QtGui
from PySide import QtCore
from xicam import threads
import ui
import customwidgets
import tomopy

functions = []
currentindex = 0
layout = None


def clearFeatures():
    global functions
    if len(functions) == 0:
        return

    value = QtGui.QMessageBox.question(None, 'Delete functions',
                                       'Are you sure you want to clear ALL functions?',
                                       (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))

    if value is QtGui.QMessageBox.Yes:
        for feature in functions:
            feature.deleteLater()
            del feature
        functions = []
        ui.showform(ui.blankform)


def addFunction(function, subfunction, package=tomopy):
    global functions, currentindex
    if function in [func.func_name for func in functions]:
        value = QtGui.QMessageBox.question(None, 'Adding duplicate function',
                                           '{} function already in pipeline.\n'
                                           'Are you sure you need another one?'.format(function),
                                           (QtGui.QMessageBox.Yes | QtGui.QMessageBox.No))
        if value is QtGui.QMessageBox.No:
            return

    currentindex = len(functions)
    func = customwidgets.FuncWidget(function, subfunction, package)
    func.sigPreview.connect(runpreview)
    functions.append(func)
    update()


def removeFunction(index):
    global functions
    del functions[index]
    update()


def swapFunctions(idx_1, idx_2):
    global functions, currentindex
    if idx_2 >= len(functions) or idx_2 < 0:
        return
    functions[idx_1], functions[idx_2] = functions[idx_2], functions[idx_1]
    currentindex = idx_2
    update()


def runpreview():
    global currentindex, functions
    func =  functions[currentindex]
    try:
        data = ui.centerwidget.currentWidget().widget
    except AttributeError:
        return

    kwargs = {}

    for arg in func.args_complement:
        if arg in ('arr', 'tomo'):
            kwargs[arg] = data.getdata()[1]
        elif arg in 'flats':
            kwargs[arg] = data.getflats()
        elif arg in 'darks':
            kwargs[arg] = data.getdarks()

    kwargs.update(**func.kwargs_complement)
    print kwargs
    # runnable = threads.RunnableMethod(show, func.partial, **kwargs)
    # threads.queue.put(runnable)


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


def show(data):
    from matplotlib.pyplot import imshow, show
    imshow(data)
    show()
