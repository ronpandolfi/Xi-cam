from PySide.QtUiTools import QUiLoader
from PySide import QtGui, QtCore
from functools import partial
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
    func.sigPreview.connect(runpreviewstack)
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


def runpreviewstack():
    global functions
    try:
        widget = ui.centerwidget.currentWidget().widget
    except AttributeError:
        return

    if len(functions) < 0:
        return

    params = {}
    for i, func in enumerate(functions):
        kwargs = {}
        for arg in func.args_complement:
            if i == 0 and arg in ('arr', 'tomo'):
                kwargs[arg] = widget.getdata()[1]
                print kwargs[arg].shape
                angles = kwargs[arg].shape[0] #TODO have this and COR as inputs to each dataset NOT HERE
            elif arg in 'flats':
                kwargs[arg] = widget.getflats()
            elif arg in 'darks':
                kwargs[arg] = widget.getdarks()

            params[func.subfunc_name] =  func.param_dict
            kwargs.update(**func.param_dict)
            kwargs.update(**func.kwargs_complement)

        if func.func_name == 'Reconstruction':
            kwargs['theta'] = tomopy.angles(angles) #TODO have this and COR as inputs to each dataset NOT HERE

        if i == 0:
            funstack = partial(func.partial, **kwargs)
        else:
            funstack = partial(func.partial, funstack(), **kwargs)

    runnable = threads.RunnableMethod(partial(widget.addPreview, params), funstack)
    threads.queue.put(runnable)


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

def test():
    global functions
    if len(functions) > 0:
        params = {}
        for func in functions:
            params[func.subfunc_name] = func.param_dict
        ui.centerwidget.currentWidget().widget.test(params)