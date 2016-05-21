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
import fdata
import warnings

functions = []
recon_function = None
currentindex = 0
layout = None


def clear_action():
    global functions
    if len(functions) == 0:
        return

    value = QtGui.QMessageBox.question(None, 'Delete functions',
                                       'Are you sure you want to clear ALL functions?',
                                       (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
    if value is QtGui.QMessageBox.Yes:
        clear_functions()


def clear_functions():
    global functions
    for feature in functions:
        feature.deleteLater()
        del feature
    functions = []
    ui.showform(ui.blankform)


def add_action(function, subfunction, package=reconpkg.packages['tomopy']):
    global functions, recon_function
    if not hasattr(package, fdata.names[subfunction]): return
    if function in [func.func_name for func in functions]:
        value = QtGui.QMessageBox.question(None, 'Adding duplicate function',
                                           '{} function already in pipeline.\n'
                                           'Are you sure you need another one?'.format(function),
                                           (QtGui.QMessageBox.Yes | QtGui.QMessageBox.No))
        if value is QtGui.QMessageBox.No:
            return

    add_function(function, subfunction, package=package)


def add_function(function, subfunction, package=reconpkg.tomopy):
    global functions, recon_function, currentindex
    currentindex = len(functions)
    if function == 'Reconstruction':
        func = fwidgets.ReconFuncWidget(function, subfunction, package)
        recon_function = func
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
    with open(yaml_file, 'r') as y:
        pipeline = yamlmod.ordered_load(y)
    clear_functions()
    for func, subfuncs in pipeline.iteritems():
        for subfunc in subfuncs:
            try:
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
            except (IndexError, AttributeError):
                raise
                # TODO: make this failure more graceful
                warnings.warn('Failed to load subfunction: ' + subfunc)


def create_pipeline_dict():
    d = OrderedDict()
    for f in functions:
        d[f.func_name] = {f.subfunc_name: [{'name': p.name(), 'value': p.value()} for p in f.params.children()]}
        if f.func_name == 'Reconstruction':
            d[f.func_name][f.subfunc_name].append({'Package':f.package})

    return d


def save_function_pipeline(pipeline):
    yaml_file = QtGui.QFileDialog.getSaveFileName(None, 'Save tomography pipeline file as', os.path.expanduser('~'),
                                                      '*.yml', selectedFilter='*.yml')[0]
    if yaml_file != '':
        yaml_file = yaml_file.split('.')[0] + '.yml'
        with open(yaml_file, 'w') as y:
            yamlmod.ordered_dump(pipeline, y)


def open_pipeline_file():
    pipeline_file = QtGui.QFileDialog.getOpenFileName(None, 'Open tomography pipeline file', os.path.expanduser('~'),
                                                      '*.yml')[0]
    if pipeline_file != '':
        load_function_pipeline(pipeline_file)


def pipeline_preview_action(widget, update=True, slc=None):
    global functions

    if len(functions) < 1:
        return None, None, None
    elif 'Reconstruction' not in [func.func_name for func in functions]:
        QtGui.QMessageBox.warning(None, 'Reconstruction method required',
                                  'You have to select a reconstruction method to run a preview')
        return None, None, None

    return construct_preview_pipeline(widget, update=update, slc=slc)


def construct_preview_pipeline(widget, update=True, slc=None):
    global functions

    lock_function_params(True)  # you probably do not need this anymore
    params = OrderedDict()
    funstack = []
    for func in functions:
        if not func.previewButton.isChecked() and func.func_name != 'Reconstruction':
            continue
        params[func.subfunc_name] = deepcopy(func.paramdict(update=update))
        funstack.append(update_function_partial(func.partial, func.func_name, func.args_complement, widget,
                                                input_partials=func.input_partials))
    lock_function_params(False)

    return funstack, widget.getsino(slc), partial(widget.addPreview, params)


def update_function_partial(fpartial, name, argnames, datawidget, input_partials=None, data_slc=None, ncore=None):
    global recon_function
    kwargs = {}
    for arg in argnames:
        if arg in 'flats':
            kwargs[arg] = datawidget.getflats(slc=data_slc)
        if arg in 'darks':
            kwargs[arg] = datawidget.getdarks(slc=data_slc)
        if arg in 'ncore' and ncore is not None:
            kwargs[arg] = ncore

    if input_partials is not None:
        for pname, slices, ipartial in input_partials:
            pargs = []
            if slices is not None:
                map(pargs.append, (map(datawidget.data.fabimage.__getitem__, slices)))
            kwargs[pname] = ipartial(*pargs)
            if pname == 'center':
                recon_function.params.child('center').setValue(kwargs[pname])

    if kwargs:
        return partial(fpartial, **kwargs)
    else:
        return fpartial


def run_preview_recon(funstack, initializer, callback):
    if funstack is not None:
        runnable = threads.RunnableMethod(callback, reduce, (lambda f1, f2: f2(f1)), funstack, initializer)
        runnable.lock = threads.mutex
        threads.queue.put(runnable)


def run_full_recon(widget, proj, sino, out_name, out_format, nchunk, ncore, update_call=None, finish_call=None):
    global functions
    lock_function_params(True)
    partials, params = [], OrderedDict()
    for f in functions:
        params[f.subfunc_name] = deepcopy(f.paramdict(update=update))
        partials.append([f.name, deepcopy(f.partial), f.args_complement, deepcopy(f.input_partials)])
    lock_function_params(False)

    import dxchange as dx
    if out_format == 'TIFF (.tiff)':
        partials.append(('Write to file', partial(dx.write_tiff_stack, fname=out_name), [], None))
    else:
        print 'Only tiff support right now'
        return

    runnable_it = threads.RunnableIterator(update_call, _recon_iter, widget, partials, proj, sino, nchunk, ncore)
    runnable_it.emitter.sigFinished.connect(finish_call)
    threads.queue.put(runnable_it)
    return params
#TODO have current recon parameters in run console or in recon view...


def _recon_iter(datawidget, partials, proj, sino, nchunk, ncore):
    write_start = sino[0]
    total_sino = (sino[1] - sino[0] - 1) // sino[2] + 1
    nsino = (total_sino - 1) // nchunk + 1
    for i in range(nchunk):
        init = True
        start, end = i * nsino + sino[0], (i + 1) * nsino + sino[0]
        for name, fpartial, argnames, ipartials in partials:
            fpartial = update_function_partial(fpartial, name, argnames, datawidget,
                                              data_slc=(slice(*proj), slice(start, end, sino[2])),
                                              ncore=ncore, input_partials=ipartials)
            yield 'Running {0} on sinograms {1} to {2} from {3}...\n\n'.format(name, start, end, total_sino)
            if init:
                tomo = fpartial(datawidget.getsino(slc=(slice(*proj), slice(start, end, sino[2]))))
                init = False
            elif name == 'Write to file':
                fpartial(tomo, start=write_start)
                write_start += tomo.shape[0]
            elif name == 'Reconstruction' and ipartials is not None:
                partials[partials.index([name, fpartial, argnames, ipartials])][3] = None
            else:
                tomo = fpartial(tomo)






