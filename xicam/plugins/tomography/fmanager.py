from PySide.QtUiTools import QUiLoader
from PySide import QtGui, QtCore
from collections import OrderedDict
from functools import partial
from copy import deepcopy
import yamlmod
from xicam import threads
import ui
import fwidgets
import reconpkg
import fdata

functions = []
recon_function = None
currentindex = 0
layout = None


def clear_features():
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


def add_function(function, subfunction, package=reconpkg.tomopy):
    global functions, recon_function, currentindex
    if function in [func.func_name for func in functions]:
        value = QtGui.QMessageBox.question(None, 'Adding duplicate function',
                                           '{} function already in pipeline.\n'
                                           'Are you sure you need another one?'.format(function),
                                           (QtGui.QMessageBox.Yes | QtGui.QMessageBox.No))
        if value is QtGui.QMessageBox.No:
            return

    currentindex = len(functions)
    if function == 'Reconstruction':
        func = fwidgets.ReconFuncWidget(function, subfunction, package)
        recon_function = func
        ui.cor_spinBox.valueChanged.connect(func.setCenterParam)
    else:
        func = fwidgets.FuncWidget(function, subfunction, package)
    functions.append(func)
    update()


def remove_function(index):
    global functions
    del functions[index]
    update()


def swap_functions(idx_1, idx_2):
    global functions, currentindex
    if idx_2 >= len(functions) or idx_2 < 0:
        return
    functions[idx_1], functions[idx_2] = functions[idx_2], functions[idx_1]
    currentindex = idx_2
    update()


def update():
    global layout, functions
    assert isinstance(layout, QtGui.QVBoxLayout)

    for i in range(layout.count()):
        layout.itemAt(i).parent = None

    for item in functions:
        layout.addWidget(item)


def load_form(path):
    guiloader = QUiLoader()
    f = QtCore.QFile(path)
    f.open(QtCore.QFile.ReadOnly)
    form = guiloader.load(f)
    f.close()
    return form


def load():
    global functions, layout
    layout.setAlignment(QtCore.Qt.AlignBottom)


def lock_function_params(boolean):
    global functions
    for func in functions:
        func.allReadOnly(boolean)


def load_function_stack(yaml_file):
    global functions, currentindex
    with open(yaml_file, 'r') as f:
        stack = yamlmod.ordered_load(f)
        # stack = yamlmod.yaml.load(f)
    for func, subfuncs in stack.iteritems():
        for subfunc in subfuncs:
            if func == 'Reconstruction':
                add_function(func, subfunc, package=reconpkg.packages[subfuncs[subfunc][-1]['Package']])
            else:
                add_function(func, subfunc)
            funcWidget = functions[currentindex]
            for param in subfuncs[subfunc]:
                if 'Package' in param:
                    continue
                child = funcWidget.params.child(param['name'])
                child.setValue(param['value'])
                child.setDefault(param['value'])


def construct_pipeline_function():
    global functions
    try:
        widget = ui.centerwidget.currentWidget().widget
    except AttributeError:
        return None, None

    if len(functions) < 0:
        return None, None

    if 'Reconstruction' not in [func.func_name for func in functions]:
        QtGui.QMessageBox.warning(None, 'Reconstruction method required',
                                  'You have to select a reconstruction method to run a preview')
        return None, None

    lock_function_params(True)
    params = OrderedDict()
    init = False
    for i, func in enumerate(functions):
        if not func.previewButton.isChecked() and func.func_name != 'Reconstruction':
            continue

        kwargs = {}
        for arg in func.args_complement:
            if not init and arg in ('arr', 'tomo'):
                kwargs[arg] = deepcopy(widget.getsino())
                angles = kwargs[arg].shape[
                    0]  # TODO have this and COR as inputs to each dataset NOT HERE
            elif arg in 'flats':
                kwargs[arg] = deepcopy(widget.getflats())
            elif arg in 'darks':
                kwargs[arg] = deepcopy(widget.getdarks())

            params[func.subfunc_name] = deepcopy(func.param_dict)
            kwargs.update(**func.param_dict)
            kwargs.update(**func.kwargs_complement)
        if func.func_name == 'Reconstruction':
            kwargs['theta'] = reconpkg.tomopy.angles(
                angles)  # TODO have this and COR as inputs to each dataset NOT HERE

        if not init:
            init = True
            funstack = partial(func.partial, **kwargs)
        else:
            funstack = partial(func.partial, funstack(), **kwargs)
    lock_function_params(False)
    return funstack, partial(widget.addPreview, params)


def run_pipeline(funstack, callback):
    if funstack is not None:
        runnable = threads.RunnableMethod(callback, funstack)
        threads.queue.put(runnable)



                        # def show(data):
#     from matplotlib.pyplot import imshow, show
#     imshow(data)
#     show()


# def test():
#     global functions
#     if len(functions) > 0:
#         params = {}
#         for func in functions:
#             params[func.subfunc_name] = func.param_dict
#         ui.centerwidget.currentWidget().widget.test(params)