from PySide.QtUiTools import QUiLoader
from PySide import QtGui, QtCore
import os
from collections import OrderedDict
from functools import partial
from copy import deepcopy
import yamlmod
from xicam import threads
import ui
import fwidgets
import reconpkg

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


def load_function_pipeline(yaml_file):
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

def open_pipeline_file():
    pipeline_file = QtGui.QFileDialog.getOpenFileName(None, 'Open tomography pipeline file', os.path.expanduser('~'),
                                                      '*.yml')[0]
    if pipeline_file != '':
        load_function_pipeline(pipeline_file)


def construct_preview_pipeline():
    global functions

    if len(functions) < 1:
        return None, None

    if 'Reconstruction' not in [func.func_name for func in functions]:
        QtGui.QMessageBox.warning(None, 'Reconstruction method required',
                                  'You have to select a reconstruction method to run a preview')
        return None, None

    widget = ui.centerwidget.currentWidget().widget
    lock_function_params(True)
    params = OrderedDict()
    funstack = []
    for i, func in enumerate(functions):
        if not func.previewButton.isChecked() and func.func_name != 'Reconstruction':
            continue
        params[func.subfunc_name] = deepcopy(func.param_dict)
        funstack.append(update_function_partial(func.partial, func.func_name, func.args_complement, widget))
    lock_function_params(False)

    return funstack, widget.getsino(), partial(ui.centerwidget.currentWidget().widget.addPreview, params)


def update_function_partial(fpartial, name, argnames, datawidget, data_slc=None, ncore=None):
    kwargs = {}
    args = ()
    for arg in argnames:
        if arg in 'flats':
            kwargs[arg] = datawidget.getflats(slc=data_slc)
        if arg in 'darks':
            kwargs[arg] = datawidget.getdarks(slc=data_slc)
        if arg in 'ncore' and ncore is not None:
            kwargs[arg] = ncore

    if 'Reconstruction' in name:
        angles = datawidget.data.shape[0]
        kwargs['theta'] = reconpkg.tomopy.angles(angles)

    return partial(fpartial, **kwargs)


def set_input_data(fpartial, datawidget, data_slc=None):
    return partial(fpartial, datawidget.getsino(slc=data_slc))


def run_preview_recon(funstack, initializer, callback):
    if funstack is not None:
        runnable = threads.RunnableMethod(callback, reduce, (lambda f1, f2: f2(f1)), funstack, initializer)
        threads.queue.put(runnable)

def run_full_recon(proj, sino, out_name, out_format, nchunk, ncore):
    global functions
    lock_function_params(True)
    # partials = [deepcopy(update_function_partial(f.partial, f.func_name, f.args_complement,
    #                                              data_slc=(sino, proj), ncore=ncore) for f in functions)]
    widget = ui.centerwidget.currentWidget().widget
    partials = [(f.name, deepcopy(f.partial), f.args_complement) for f in functions]

    lock_function_params(False)

    import dxchange as dx
    if out_format == 'TIFF (.tiff)':
        partials.append(('Write to file', partial(dx.write_tiff_stack, fname=out_name), []))
    else:
        print 'Only tiff support right now'
        return

    runnable_it = threads.RunnableIterator(widget.processViewer.log2local, _recon_iter,
                                           widget, partials, proj, sino, nchunk, ncore)
    threads.queue.put(runnable_it)


#TODO use reduce as well
def _recon_iter(datawidget, partials, proj, sino, nchunk, ncore):
    write_start = sino[0]
    total_sino = (sino[1] - sino[0] - 1) // sino[2] + 1
    nsino = (total_sino - 1) // nchunk + 1
    for i in range(nchunk):
        init = True
        start, end = i * nsino + sino[0], (i + 1) * nsino + sino[0]
        for name, partial, argnames in partials:
            partial = update_function_partial(partial, name, argnames, datawidget,
                                              data_slc=(proj, (start, end, sino[2])), ncore=ncore)
            yield 'Running {0} on sinograms {1} to {2} from {3}...\n\n'.format(name, start, end, total_sino)
            if init:
                tomo = partial(datawidget.getsino(slc=(proj, (start, end, sino[2]))))
                init = False
            elif name == 'Write to file':
                partial(tomo, start=write_start)
                write_start += tomo.shape[0]
            else:
                tomo = partial(tomo)






